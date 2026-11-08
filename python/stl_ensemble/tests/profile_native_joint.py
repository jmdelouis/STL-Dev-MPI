#!/usr/bin/env python3
"""Profile of one joint step of NativeUNet (--pol eb | spin) on the PR3 maps:
the whole frozen-bias step (network, front end, statistics forward and
reverse), and its parts timed apart: the network alone (forward; forward
and reverse), the eth front end and its transpose (spin), the cl term
(spectra and their gradient, scalar E and B or EE/BB of (Q, U) on the
host), the bias estimate, the one-time setup.

    python tests/profile_native_joint.py --data DIR --nside 64 --pol spin --out prof.json
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
from stl_port.denoise import BinnedSpectra, SpectrumLoss, conditioning, eb_maps

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--nside", type=int, default=64)
p.add_argument("--pol", choices=["eb", "ebqu", "spin"], default="spin")
p.add_argument("--sims", default="10-15")
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--bias-realizations", type=int, default=8)
p.add_argument("--out", default=None)
a = p.parse_args()
n = a.nside
load = lambda name: np.load(os.path.join(a.data, f"{name}_n{n}.npy")).astype(np.float64)  # noqa: E731
lo, _, hi = a.sims.partition("-")
sims = [k for k in range(int(lo), int(hi or lo) + 1)
        if all(os.path.exists(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy"))
               for kd in ("noise", "noisehm1", "noisehm2") for f in ("Q", "U"))]
levels = int(math.log2(n // 8)) + 1
channels = [32, 64, 96, 128, 128, 128, 128][:levels]
w = dict(data=1.0, l1=1.0, l2=1.0, l3=1.0)
rep = {"nside": n, "pol": a.pol, "sims": len(sims), "channels": channels}


def timed(f, k=a.repeat):
    f()  # warm-up (engines, caches)
    t = time.perf_counter()
    for _ in range(k):
        r = f()
    return (time.perf_counter() - t) / k, r


t0 = time.perf_counter()
net = NativeUNet(n, channels, cond=1, maps=2, j=int(math.log2(n)) - 1, loss=w, backend="gpu")
i857 = load("planck857_I")
net.set_common(i857)
qu = np.stack([load(f"planck353_{f}") for f in "QU"])
hm = [np.stack([load(f"planck353hm{h}_{f}") for f in "QU"]) for h in (1, 2)]
pools = [np.stack([np.stack([np.load(os.path.join(a.data, f"planck353_{kd}_{f}_n{n}_mc{k}.npy")) for k in sims])
                   for f in "QU"], 1).astype(np.float64) for kd in ("noise", "noisehm1", "noisehm2")]
t1 = time.perf_counter()
if a.pol == "eb":
    tf = eb_maps
    x = eb_maps(qu)
    net.set_joint(x, ("E", "B"))
elif a.pol == "ebqu":
    x = qu
    net.set_joint(x, ("E", "B"), front="eb")
    tf = net.front
else:
    x = qu
    net.set_joint(x, ("E", "B"), front="eth", weight=0.5)
    tf = net.front
fd, fh, fp = tf(qu), [tf(h) for h in hm], [tf(q) for q in pools]
rep["setup_transform_s"] = time.perf_counter() - t1
for i, f in enumerate(("E", "B")):
    net.set_field(f, fd[i], fh[0][i], fh[1][i], *[np.ascontiguousarray(q[:, i]) for q in fp])
net.set_cond(conditioning(["log857"], i857))
masks = [0.0, 30.0, 45.0, 60.0]
spectra = BinnedSpectra(n, masks=masks, backend="gpu", spin=2 if a.pol == "spin" else 0)
if a.pol == "spin":
    cl = SpectrumLoss(spectra, *hm)
    cl_grad = lambda u: cl(u, grad=True)  # noqa: E731
else:
    cls = [SpectrumLoss(spectra, fh[0][i], fh[1][i]) for i in range(2)]

    def cl_grad(u):
        y = net.front(u) if a.pol == "ebqu" else u
        parts = [c(y[i], grad=True) for i, c in enumerate(cls)]
        g = np.stack([g for _, g in parts])
        return sum(v for v, _ in parts), net.front(g, adjoint=True) if a.pol == "ebqu" else g
rng = np.random.default_rng(0)
draws = (rng.integers(0, len(sims), a.bias_realizations), rng.choice([-1.0, 1.0], a.bias_realizations))
rep["setup_s"] = time.perf_counter() - t0
rep["bias_estimate_s"], _ = timed(lambda: net.estimate_bias("joint", *draws), 2)


def step():
    net.zero_grad()
    terms = net.loss("joint", backward=True)
    u = net.apply(x)
    v, g = cl_grad(u)
    net.apply(x, cot=g)
    net.adam_step(0.0)
    return terms


rep["step_s"], _ = timed(step)
rep["loss_backward_s"], _ = timed(lambda: net.loss("joint", backward=True))
rep["loss_forward_s"], _ = timed(lambda: net.loss("joint"))
rep["network_forward_s"], _ = timed(lambda: net.apply(x))
cot = rng.normal(size=x.shape)
rep["network_forward_backward_s"], _ = timed(lambda: net.apply(x, cot=cot))
u = net.apply(x)
rep["cl_spectra_and_gradient_s"], _ = timed(lambda: cl_grad(u))
if a.pol in ("spin", "ebqu"):
    rep["front_s"], _ = timed(lambda: net.front(x))
    rep["front_adjoint_s"], _ = timed(lambda: net.front(x, adjoint=True))
# statistics forward and reverse: the loss without the network passes and the front end
front = rep.get("front_s", 0.0) + rep.get("front_adjoint_s", 0.0)
rep["statistics_forward_s"] = rep["loss_forward_s"] - rep["network_forward_s"] - rep.get("front_s", 0.0)
rep["statistics_backward_s"] = (rep["loss_backward_s"] - rep["loss_forward_s"]
                                - (rep["network_forward_backward_s"] - rep["network_forward_s"])
                                - rep.get("front_adjoint_s", 0.0))
print(json.dumps(rep, indent=1), flush=True)
if a.out:
    with open(a.out, "w") as fh_:
        json.dump(rep, fh_, indent=1)
