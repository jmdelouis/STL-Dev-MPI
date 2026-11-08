#!/usr/bin/env python3
"""Dump the terms and parameter gradients of a frozen and a live step and the
terms after a bias estimate (--pol ebqu, cl in the trainer, stat and cl masks,
the setup of profile_native_step.py) to compare two builds of the library
bit by bit:

    STL_NATIVE_LIBRARY=old.so python tests/dump_native_step.py --data DIR --nside 64 --out a.npz
    STL_NATIVE_LIBRARY=new.so python tests/dump_native_step.py --data DIR --nside 64 --out b.npz
    python tests/dump_native_step.py --compare a.npz b.npz
"""
import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data")
p.add_argument("--nside", type=int, default=64)
p.add_argument("--channels", default="16,32,48,64")
p.add_argument("--sims", default="10-19")
p.add_argument("--out")
p.add_argument("--compare", nargs=2)
a = p.parse_args()
if a.compare:
    x, y = (np.load(f) for f in a.compare)
    worst = 0.0
    for k in x.files:
        d = np.abs(x[k] - y[k]).max()
        s = np.abs(x[k]).max()
        worst = max(worst, d / s if s else d)
        print(f"{k:24s} max |diff| {d:.3e} (rel {d / s if s else d:.3e}) identical {np.array_equal(x[k], y[k])}")
    print(f"worst relative difference {worst:.3e}")
    sys.exit(0)
from stl_port.native import NativeUNet
from stl_port.denoise import bin_edges, conditioning, galactic_mask, stat_masks
n = a.nside
load = lambda name: np.load(os.path.join(a.data, f"{name}_n{n}.npy")).astype(np.float64)  # noqa: E731
lo, _, hi = a.sims.partition("-")
sims = [k for k in range(int(lo), int(hi or lo) + 1)
        if all(os.path.exists(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy"))
               for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
channels = [int(c) for c in a.channels.split(",")]
net = NativeUNet(n, channels, cond=2, maps=2, j=int(math.log2(n)) - 1, loss=dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0),
                 backend="gpu", engine={"cache_bytes": 1 << 30}, seed=0)
net.set_masks(stat_masks(n, [0.0, 20.0, 30.0, 45.0]))
i857 = load("planck857_I")
net.set_common(i857)
qu = np.stack([load(f"planck353_{f}") for f in "QU"])
hm = [np.stack([load(f"planck353hm{h}_{f}") for f in "QU"]) for h in (1, 2)]
pools = [np.stack([np.stack([np.load(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy")) for k in sims])
                   for f in "QU"], 1).astype(np.float64) for kd in ("noise", "noisehm1", "noisehm2")]
net.set_joint(qu, ("E", "B"), front="eb:3")
fd, fh, fp = net.front(qu), [net.front(h) for h in hm], [net.front(q) for q in pools]
for i, f in enumerate(("E", "B")):
    net.set_field(f, fd[i], fh[0][i], fh[1][i], *[np.ascontiguousarray(q[:, i]) for q in fp])
net.set_cond(conditioning(["log857", "sigma"], i857, [np.mean(np.square(pools[0][:, i]), 0) for i in range(2)]))
lmax = 2 * n - 1
net.set_cl(np.stack([galactic_mask(n, b, 5.0) for b in (0.0, 30.0, 45.0, 60.0)]), bin_edges(10, lmax, 30), lmax=lmax)
rng = np.random.default_rng(0)
bias = (rng.integers(0, len(sims), 40), rng.choice([-1.0, 1.0], 40))
live = [(rng.permutation(len(sims))[:4], rng.choice([-1.0, 1.0], 4)) for _ in range(2)]
out = {}
net.estimate_bias("joint", *bias)
net.zero_grad()
out["frozen_terms"] = np.array(list(net.loss("joint", backward=True).values()))
out["frozen_grads"] = net.grads().astype(np.float64)
net.zero_grad()
out["live_terms"] = np.array(list(net.live("joint", *live, backward=True).values()))
out["live_grads"] = net.grads().astype(np.float64)
np.savez(a.out, **out)
print("dumped", a.out, {k: v[:6] if v.ndim else v for k, v in out.items() if "terms" in k})
