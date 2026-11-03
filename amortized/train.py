"""Train the pyramid generator through the Rust scattering-covariance loss.

    python train.py --target T.npy --out DIR [--nside 64 --j 5 --batch 8 --steps 5000
                     --lr 1e-3 --scatter-weight 1 --device 0 --init DIR/model.pt --freeze K]

The target is a NESTED map at `--nside` (degraded by 4-children means if it
is finer), standardised by its own mean and std as in `stl-synth`; the
generator produces standardised maps.  Every step draws a fresh noise batch
of `--batch` maps and takes one Adam step (cosine schedule) on the
ensemble loss of `stl-synth --ensemble B --weights ergodic --scatter-weight X`.
`--init` loads a generator trained at a lower nside into the coarse levels
(progressive growing); `--freeze K` freezes its first K levels.

Writes DIR/model.pt, DIR/train.json (loss history, timers, configuration).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python", "stl_ensemble"))
from generator import Generator  # noqa: E402
from stl_torch import EnsembleLoss, Stats  # noqa: E402
from stl_port.native import NativeEngine  # noqa: E402


def load_target(path, nside):
    m = np.load(path).astype(np.float64)
    while m.size > 12 * nside * nside:
        m = m.reshape(-1, 4).mean(1)
    assert m.size == 12 * nside * nside, "target nside below --nside"
    return m


def target_statistics(m, nside, j, device):
    e = NativeEngine(nside, j, batch=1, backend="gpu", device=device)
    v, var = e.forward(m[None], moments=True)
    plan = e.plan
    e.close()
    return v[0], var[0], plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--nside0", type=int, default=4)
    p.add_argument("--channels", default=None, help="comma-separated, one per level")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scatter-weight", type=float, default=1.0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None)
    p.add_argument("--freeze", type=int, default=0)
    p.add_argument("--up", default="smooth", choices=["smooth", "nearest"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    j = a.j or int(math.log2(a.nside)) - 1
    dev = torch.device(f"cuda:{a.device}")
    torch.manual_seed(a.seed)

    raw = load_target(a.target, a.nside)
    mu, sd = raw.mean(), raw.std()
    target = (raw - mu) / sd
    t0 = time.time()
    tstats, tvar, plan = target_statistics(target, a.nside, j, a.device)
    engine = NativeEngine(a.nside, j, batch=a.batch, backend="gpu", device=a.device)
    stats = Stats(engine)
    loss_fn = EnsembleLoss(tstats, tvar, plan, a.nside, a.batch, a.scatter_weight, device=dev)
    setup = time.time() - t0

    channels = [int(c) for c in a.channels.split(",")] if a.channels else None
    gen = Generator(a.nside, a.nside0, channels, up=a.up).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        own = gen.state_dict()
        loaded = {k: v for k, v in ck["state"].items() if k in own and own[k].shape == v.shape}
        own.update(loaded)
        gen.load_state_dict(own)
        print(f"init: {len(loaded)} tensors from {a.init}", flush=True)
    for k in range(a.freeze):
        for prm in list(gen.blocks[k].parameters()) + list(gen.heads[k].parameters()):
            prm.requires_grad_(False)
    params = [q for q in gen.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in gen.parameters())
    print(f"nside {a.nside} J {j} coefficients {len(plan)} params {nparams} trainable {sum(q.numel() for q in params)} "
          f"setup {setup:.1f} s", flush=True)

    hist = {"step": [], "loss": [], "mean_term": [], "scatter_term": []}
    timers = {"generator": 0.0, "engine": 0.0, "loss_and_step": 0.0}
    started = time.time()

    def save(tag="model.pt"):
        torch.save({"state": gen.state_dict(), "nside": a.nside, "nside0": a.nside0, "channels": gen.channels, "up": gen.up_mode,
                    "target_mean": float(mu), "target_std": float(sd), "j": j}, os.path.join(a.out, tag))

    for step in range(1, a.steps + 1):
        ts = time.time()
        z = gen.noise(a.batch, device=dev)
        x = gen(z).double()
        torch.cuda.synchronize(dev)
        tg = time.time()
        s = stats(x)
        te = time.time()
        loss, lm, ls = loss_fn(s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        torch.cuda.synchronize(dev)
        tl = time.time()
        timers["generator"] += tg - ts
        timers["engine"] += te - tg
        timers["loss_and_step"] += tl - te  # includes the engine's reverse sweep
        if step % a.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            hist["mean_term"].append(lm.item())
            hist["scatter_term"].append(ls.item())
            print(f"step {step} loss {loss.item():.6g} mean {lm.item():.6g} scatter {ls.item():.6g} "
                  f"{(time.time() - started) / step:.3f} s/step", flush=True)
        if step % a.save_every == 0:
            save()
    save()
    report = {"args": vars(a), "j": j, "coefficients": len(plan), "parameters": nparams, "setup_seconds": setup,
              "seconds": time.time() - started, "timers_seconds": timers, "loss_history": hist,
              "target_mean": float(mu), "target_std": float(sd)}
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
