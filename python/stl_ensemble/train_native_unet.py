#!/usr/bin/env python3
"""Train the Healpix-STL-IA U-Net with the native trainer (NativeUNet) on
the PR3 maps, frozen noise biases (the --real --bias frozen run of stlia
train_denoise.py), with Adam (cosine schedule) or L-BFGS by rounds.

    python train_native_unet.py --data DIR --out RUN --nside 64 --optimizer adam --steps 3000
    python train_native_unet.py --data DIR --out RUN --nside 64 --optimizer lbfgs --rounds 20 --lbfgs-iter 20

Data (as train_denoise.py): DIR/planck353_{Q,U}_n{nside}.npy, the half
missions planck353hm{1,2}_{f}_n{nside}.npy, planck857_I_n{nside}.npy, the
FFP10 noise planck353_noise{,hm1,hm2}_{f}_n{nside}_mc{k}.npy (--sims: the
simulations present for all three are used).

Adam: field f = fields[step % 2], biases refreshed every --bias-every steps
(per field, at the current output). L-BFGS: round r trains field
fields[r % 2] (or both with --lbfgs-fields joint) for --lbfgs-iter
iterations, after a refresh of its biases. With --seconds the training
stops at that wall time. Every --eval-seconds of training (not counted),
both fields are evaluated with the biases of one fixed set of draws: the
common objective of the comparison. Writes RUN/history.json and
RUN/params.npy.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_port.native import NativeUNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--nside", type=int, default=64)
p.add_argument("--j", type=int, default=None)
p.add_argument("--levels", type=int, default=None, help="default: down to nside 8")
p.add_argument("--channels", default=None)
p.add_argument("--loss", default="data=1,l1=1,l2=1,l3=1")
p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
p.add_argument("--steps", type=int, default=3000, help="Adam steps (cosine schedule length)")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bias-every", type=int, default=50)
p.add_argument("--bias-realizations", type=int, default=40)
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--lbfgs-iter", type=int, default=20)
p.add_argument("--lbfgs-history", type=int, default=10)
p.add_argument("--lbfgs-max-eval", type=int, default=None)
p.add_argument("--lbfgs-fields", choices=["alternate", "joint"], default="alternate")
p.add_argument("--sims", default="10-49")
p.add_argument("--seconds", type=float, default=None, help="wall-time budget of the training")
p.add_argument("--eval-seconds", type=float, default=60.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", type=int, default=0)
p.add_argument("--backend", default="gpu")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
n = a.nside
load = lambda name: np.load(os.path.join(a.data, f"{name}_n{n}.npy")).astype(np.float64)
lo, _, hi = a.sims.partition("-")
sims = [k for k in range(int(lo), int(hi or lo) + 1)
        if all(os.path.exists(os.path.join(a.data, f"planck353_{kd}_Q_n{n}_mc{k}.npy")) for kd in ("noise", "noisehm1", "noisehm2"))]
levels = a.levels or int(math.log2(n // 8)) + 1
channels = [int(c) for c in a.channels.split(",")] if a.channels else [32, 64, 96, 128, 128, 128, 128][:levels]
weights = {k: float(v) for k, v in (item.split("=") for item in a.loss.split(","))}
net = NativeUNet(n, channels, cond=1, j=a.j, loss=weights, backend=a.backend, device=a.device, seed=a.seed)
i857 = load("planck857_I")
c = np.log(np.clip(i857, np.percentile(i857[i857 > 0], 0.1), None))
net.set_cond(((c - c.mean()) / c.std()).astype(np.float32)[None])
net.set_common(i857)
fields = ("Q", "U")
for f in fields:
    pools = [np.stack([np.load(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy")) for k in sims])
             for kd in ("noise", "noisehm1", "noisehm2")]
    net.set_field(f, load(f"planck353_{f}"), load(f"planck353hm1_{f}"), load(f"planck353hm2_{f}"), *pools)
rng = np.random.default_rng(a.seed + 777)
eval_draws = (np.arange(a.bias_realizations) % len(sims), np.where(np.arange(a.bias_realizations) % 2, -1.0, 1.0))
print(f"nside {n} levels {levels} channels {channels} params {net.size} sims {len(sims)} loss {weights} "
      f"optimizer {a.optimizer}", flush=True)


def draws():
    return rng.integers(0, len(sims), a.bias_realizations), rng.choice([-1.0, 1.0], a.bias_realizations)


hist = {"train_seconds": [], "eval": [], "step_terms": []}
clock = {"train": 0.0, "next_eval": 0.0}


def evaluate():
    """Both fields, biases from the fixed evaluation draws (restored after)."""
    out = {}
    for f in fields:
        net.estimate_bias(f, *eval_draws)
        out[f] = net.loss(f)
    hist["eval"].append({"train_seconds": clock["train"], **{f: out[f] for f in fields},
                         "total": sum(out[f]["total"] for f in fields)})
    print(f"eval t={clock['train']:.1f}s total {hist['eval'][-1]['total']:.6g} "
          + " ".join(f"{f} {out[f]['total']:.5g}" for f in fields), flush=True)
    clock["next_eval"] = clock["train"] + a.eval_seconds


def out_of_time():
    return a.seconds is not None and clock["train"] >= a.seconds


evaluate()
if a.optimizer == "adam":
    for step in range(1, a.steps + 1):
        f = fields[step % 2]
        t0 = time.perf_counter()
        if (step - 1) % a.bias_every < len(fields):
            net.estimate_bias(f, *draws())
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps))
        terms = net.step(f, lr)
        clock["train"] += time.perf_counter() - t0
        hist["step_terms"].append({"step": step, "field": f, "seconds": clock["train"], **terms})
        if step % 10 == 0 or step == 1:
            print(f"step {step} {f} loss {terms['total']:.6g} {clock['train'] / step:.3f} s/step", flush=True)
        if clock["train"] >= clock["next_eval"]:
            evaluate()
        if out_of_time():
            break
else:
    opts = {"max_iter": a.lbfgs_iter, "history_size": a.lbfgs_history}
    if a.lbfgs_max_eval:
        opts["max_eval"] = a.lbfgs_max_eval
    for r in range(a.rounds):
        names = list(fields) if a.lbfgs_fields == "joint" else [fields[r % 2]]
        t0 = time.perf_counter()
        for f in names:
            net.estimate_bias(f, *draws())
        rep = net.lbfgs(names, opts)
        clock["train"] += time.perf_counter() - t0
        h = rep["loss_history"]
        hist["step_terms"].append({"round": r, "fields": names, "seconds": clock["train"], "evaluations": rep["evaluations"],
                                   "loss_start": h[0], "loss_end": min(h)})
        print(f"round {r} {names} {h[0]:.6g} -> {min(h):.6g} in {rep['evaluations']} evaluations, "
              f"t={clock['train']:.1f}s", flush=True)
        if clock["train"] >= clock["next_eval"]:
            evaluate()
        if out_of_time():
            break
evaluate()
np.save(os.path.join(a.out, "params.npy"), net.get_params())
with open(os.path.join(a.out, "history.json"), "w") as fh:
    json.dump({"args": vars(a), "params": net.size, "sims": sims, **hist}, fh, indent=1)
