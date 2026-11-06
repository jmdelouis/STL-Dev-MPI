#!/usr/bin/env python3
"""Time and memory of one frozen-bias training step (data, l1, l2, l3) of
the Healpix-STL-IA U-Net: the native trainer (NativeUNet, everything on one
GPU) against the torch pipeline of stlia train_denoise.py (torch U-Net on
the GPU, NativeEngine statistics through the host, batches padded as
there: auto to B, cross to 3 + 3B pairs). Random maps of the right sizes.

    python bench_native_unet.py --impl native --nside 256 --levels 6 --steps 5
    python bench_native_unet.py --impl torch  --nside 256 --levels 6 --steps 5   (needs torch, stlia)

Peak memory: the largest device memory of this process seen by nvidia-smi
(sampled every 0.1 s), and torch.cuda.max_memory_allocated for torch.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_port.native import NativeEngine, NativeUNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--impl", choices=["native", "torch"], required=True)
p.add_argument("--nside", type=int, default=256)
p.add_argument("--levels", type=int, default=6)
p.add_argument("--channels", default="32,64,96,128,128,128")
p.add_argument("--j", type=int, default=None)
p.add_argument("--steps", type=int, default=5)
p.add_argument("--realizations", type=int, default=4, help="B of train_denoise.py (torch padding)")
p.add_argument("--bias-draws", type=int, default=40)
p.add_argument("--sims", type=int, default=10)
p.add_argument("--device", type=int, default=0)
a = p.parse_args()
nside, npix = a.nside, 12 * a.nside**2
j = a.j or int(np.log2(nside)) - 1
ch = [int(c) for c in a.channels.split(",")][: a.levels]
rng = np.random.default_rng(0)
sky = rng.normal(size=npix)
n0, n1, n2 = (0.3 * rng.normal(size=(a.sims, npix)) for _ in range(3))
d, d1, d2 = sky + n0[0], sky + n1[0], sky + n2[0]
t857 = sky + rng.normal(size=npix)
cond = rng.normal(size=(1, npix)).astype(np.float32)
draws = (rng.integers(0, a.sims, a.bias_draws), rng.choice([-1.0, 1.0], a.bias_draws))

peak = {"bytes": 0}
stop = threading.Event()


def watch():
    pid = str(os.getpid())
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                q, m = [s.strip() for s in line.split(",")]
                if q == pid:
                    peak["bytes"] = max(peak["bytes"], int(m) << 20)
        except Exception:
            pass
        stop.wait(0.1)


threading.Thread(target=watch, daemon=True).start()
times, bias_s = [], None

if a.impl == "native":
    nat = NativeUNet(nside, ch, cond=1, j=j, loss=dict(data=1, l1=1, l2=1, l3=1), backend="gpu", device=a.device)
    nat.set_cond(cond)
    nat.set_common(t857)
    nat.set_field("Q", d, d1, d2, n0, n1, n2)
    t0 = time.perf_counter()
    nat.estimate_bias("Q", *draws)
    bias_s = time.perf_counter() - t0
    for s in range(a.steps + 1):
        t0 = time.perf_counter()
        terms = nat.step("Q", 1e-3)
        if s:
            times.append(time.perf_counter() - t0)
    nparams = nat.size
else:
    STLIA = os.environ.get("STLIA_PATH")
    if STLIA:
        sys.path.insert(0, STLIA)
    import torch
    from stlia.stl_torch import DenoiseLoss, StatsFunction, as_real
    from stlia.train_denoise import CrossStats
    from stlia.unet import UNet

    dev = torch.device(f"cuda:{a.device}")
    B = a.realizations
    net = UNet(nside, len(ch), ch, cond=1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    one = NativeEngine(nside, j, batch=1, backend="gpu", device=a.device)
    many = NativeEngine(nside, j, batch=B, backend="gpu", device=a.device)
    xeng = NativeEngine(nside, j, batch=3 + 3 * B, backend="gpu", device=a.device, cross=True)
    loss_fn, xloss = DenoiseLoss(one.plan, nside, device=dev), DenoiseLoss(xeng.plan, nside, device=dev)
    xrun = {"engine": xeng, "generation": 0}
    tt = lambda x: torch.as_tensor(x, device=dev)
    D, D1, D2, T, C = tt(d), tt(d1), tt(d2), tt(t857), tt(cond)[None]
    N0, N1, N2 = tt(n0), tt(n1), tt(n2)
    gen = {"g": 0}

    def padded(x, batch):
        return torch.cat([x, x[:1].expand(batch - len(x), -1)]) if len(x) < batch else x

    def auto_stats(x):
        gen["g"] += 1
        return StatsFunction.apply(padded(x, B), many, gen["g"])[: len(x)]

    def cross_stats(x, y):
        n = len(x)
        st = CrossStats.apply(padded(x, 3 + 3 * B), padded(y, 3 + 3 * B), xrun)
        return st[:n], xrun["var"][:n]

    def bias_of(u):
        sims, signs = (torch.as_tensor(v, device=dev) for v in draws)
        M = len(sims)
        bias = {}
        with torch.no_grad():
            acc = 0
            for k0 in range(0, M, B):
                idx, sg = sims[k0:k0 + B], signs[k0:k0 + B, None]
                acc = acc + auto_stats(u + sg * N0[idx]).sum(0)
            bias["data"] = acc / M - auto_stats(u[None])[0]
            st0, _ = cross_stats(torch.stack([u, T]), torch.stack([u, u]))
            acc = {t: 0 for t in ("l1", "l2", "l3")}
            per = (3 + 3 * B) // 3
            for k0 in range(0, M, per):
                idx, sg = sims[k0:k0 + per], signs[k0:k0 + per, None]
                k = len(idx)
                st, _ = cross_stats(torch.cat([u + sg * N1[idx], u + sg * N0[idx], T.expand(k, -1)]),
                                    torch.cat([u + sg * N2[idx], u.expand(k, -1), u + sg * N0[idx]]))
                for t, name in enumerate(("l1", "l2", "l3")):
                    acc[name] = acc[name] + st[t * k:(t + 1) * k].sum(0)
            bias["l1"], bias["l2"], bias["l3"] = acc["l1"] / M - st0[0], acc["l2"] / M - st0[0], acc["l3"] / M - st0[1]
        return bias

    t0 = time.perf_counter()
    bias = bias_of(net(D[None], D.std(), cond=C)[0].detach())
    torch.cuda.synchronize(dev)
    bias_s = time.perf_counter() - t0
    vt, tv = one.forward(d[None], moments=True)
    target, tvar = tt(as_real(vt[0])), tt(tv[0])
    for s in range(a.steps + 1):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        u = net(D[None], D.std(), cond=C)[0]
        terms = {"data": loss_fn((auto_stats(u[None])[0] + bias["data"])[None], target, tvar)}
        st, var = cross_stats(torch.stack([D1, D, T, u, T]), torch.stack([D2, u, D, u, u]))
        model = {"l1": st[3], "l2": st[3], "l3": st[4]}
        for t, name in enumerate(("l1", "l2", "l3")):
            tg = st[t] if name == "l2" else st[t].detach()
            terms[name] = xloss((model[name] + bias[name])[None], tg, var[t])
        loss = sum(terms.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(dev)
        if s:
            times.append(time.perf_counter() - t0)
    terms = {k: float(v) for k, v in terms.items()}
    nparams = sum(q.numel() for q in net.parameters())
    peak["torch_allocated"] = int(torch.cuda.max_memory_allocated(dev))

time.sleep(0.3)
stop.set()
print(json.dumps(dict(impl=a.impl, nside=nside, levels=len(ch), channels=ch, j=j, params=nparams,
                      step_seconds=float(np.median(times)), steps=len(times), bias_seconds=bias_s,
                      bias_draws=a.bias_draws, peak_device_bytes=peak["bytes"],
                      torch_allocated_bytes=peak.get("torch_allocated"), terms=terms)))
