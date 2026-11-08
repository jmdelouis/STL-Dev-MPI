#!/usr/bin/env python3
"""Where the time of one training step goes: the joint network of
train_native_unet.py (--pol ebqu, cl in the trainer, stat and cl masks,
cond log857+sigma) on the PR3 maps, phase by phase from the trainer's
timers (STL_PROFILE=1: every mark waits for the device), for a frozen-bias
step, a live warm-up step and one frozen bias estimate.

    STL_PROFILE=1 python tests/profile_native_step.py --data DIR --nside 256 --out prof.json

Without STL_PROFILE the same script gives the wall time of the steps
without the synchronisations of the timers.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeUNet
from stl_port.denoise import bin_edges, conditioning, galactic_mask, stat_masks

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--nside", type=int, default=256)
p.add_argument("--pol", choices=["eb", "ebqu"], default="ebqu")
p.add_argument("--channels", default="16,32,48,64,64,64")
p.add_argument("--sims", default="10-49")
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--realizations", type=int, default=4)
p.add_argument("--bias-realizations", type=int, default=40)
p.add_argument("--cache-bytes", type=int, default=0, help="GPU geometry cache of the statistics engines (0: from the free memory)")
p.add_argument("--out", default=None)
a = p.parse_args()
n = a.nside
load = lambda name: np.load(os.path.join(a.data, f"{name}_n{n}.npy")).astype(np.float64)  # noqa: E731
lo, _, hi = a.sims.partition("-")
sims = [k for k in range(int(lo), int(hi or lo) + 1)
        if all(os.path.exists(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy"))
               for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
channels = [int(c) for c in a.channels.split(",")]
w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0, cl=1.0)
rep = {"nside": n, "pol": a.pol, "sims": len(sims), "channels": channels, "profile": os.environ.get("STL_PROFILE", "0"), "cache_bytes": a.cache_bytes}
t0 = time.perf_counter()
net = NativeUNet(n, channels, cond=2, maps=2, j=int(math.log2(n)) - 1, loss=w, backend="gpu", engine={"cache_bytes": a.cache_bytes})
net.set_masks(stat_masks(n, [0.0, 20.0, 30.0, 45.0]))
i857 = load("planck857_I")
net.set_common(i857)
qu = np.stack([load(f"planck353_{f}") for f in "QU"])
hm = [np.stack([load(f"planck353hm{h}_{f}") for f in "QU"]) for h in (1, 2)]
pools = [np.stack([np.stack([np.load(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy")) for k in sims])
                   for f in "QU"], 1).astype(np.float64) for kd in ("noise", "noisehm1", "noisehm2")]
if a.pol == "eb":
    from stl_port.denoise import eb_maps
    tf = eb_maps
    x = eb_maps(qu)
    net.set_joint(x, ("E", "B"))
else:
    x = qu
    net.set_joint(x, ("E", "B"), front="eb:3")
    tf = net.front
fd, fh, fp = tf(qu), [tf(h) for h in hm], [tf(q) for q in pools]
for i, f in enumerate(("E", "B")):
    net.set_field(f, fd[i], fh[0][i], fh[1][i], *[np.ascontiguousarray(q[:, i]) for q in fp])
noise_var = [np.mean(np.square(pools[0][:, i]), 0) for i in range(2)]
net.set_cond(conditioning(["log857", "sigma"], i857, noise_var))
del pools, fp
lmax = 2 * n - 1
net.set_cl(np.stack([galactic_mask(n, b, 5.0) for b in (0.0, 30.0, 45.0, 60.0)]), bin_edges(10, lmax, 30), lmax=lmax)
rng = np.random.default_rng(0)
draws = lambda k: (rng.permutation(len(sims))[:k], rng.choice([-1.0, 1.0], k))  # noqa: E731
bias_draws = (rng.integers(0, len(sims), a.bias_realizations), rng.choice([-1.0, 1.0], a.bias_realizations))
# warm-up: engines, caches, cl targets
net.estimate_bias("joint", *bias_draws)
net.step("joint", 1e-6)
net.live("joint", draws(a.realizations), draws(a.realizations), lr=1e-6)
rep["setup_s"] = time.perf_counter() - t0
engine_before = {k: s for k, (s, c) in net.profile().items() if c < 0}  # the engines' timers are cumulative


def section(name, f, k):
    t = time.perf_counter()
    for _ in range(k):
        f()
    wall = (time.perf_counter() - t) / k
    raw = net.profile()
    prof = {key: (s / k, c / k) for key, (s, c) in raw.items() if c >= 0}
    for key, (s, c) in raw.items():
        if c < 0:
            prof[key] = ((s - engine_before.get(key, 0.0)) / k, -1)
            engine_before[key] = s
    rep[name] = {"wall_s": wall, "phases": prof}
    print(f"{name}: {wall:.3f} s", flush=True)


section("frozen_step", lambda: net.step("joint", 1e-6), a.repeat)
section("live_step", lambda: net.live("joint", draws(a.realizations), draws(a.realizations), lr=1e-6), a.repeat)
section("bias_estimate", lambda: net.estimate_bias("joint", *bias_draws), 1)
print(json.dumps(rep, indent=1), flush=True)
if a.out:
    with open(a.out, "w") as fh_:
        json.dump(rep, fh_, indent=1)
