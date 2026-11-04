#!/usr/bin/env python3
"""Timing of SpectrumEngine (forward + vjp) on random maps.

    python bench_spectrum.py --backend gpu --nside 256 --batch 1 --masks 5
"""
import argparse
import json
import time
import numpy as np
from stl_port.native import SpectrumEngine

p = argparse.ArgumentParser()
p.add_argument("--backend", default="cpu")
p.add_argument("--nside", type=int, default=256)
p.add_argument("--lmax", type=int, default=None, help="default 2 nside - 1")
p.add_argument("--batch", type=int, default=1)
p.add_argument("--masks", type=int, default=5)
p.add_argument("--cross", action="store_true")
p.add_argument("--repeat", type=int, default=5)
a = p.parse_args()
lmax = 2 * a.nside - 1 if a.lmax is None else a.lmax
npix = 12 * a.nside**2
rng = np.random.default_rng(0)
shape = (a.batch, 2, npix) if a.cross else (a.batch, npix)
x = rng.normal(size=shape)
w = rng.uniform(size=(a.masks, npix)) if a.masks else None
t0 = time.perf_counter()
e = SpectrumEngine(a.nside, lmax, batch=a.batch, backend=a.backend, cross=a.cross, masks=w)
setup = time.perf_counter() - t0
fwd, bwd = [], []
for i in range(a.repeat + 1):
    t0 = time.perf_counter()
    cl = e.forward(x, generation=i)
    t1 = time.perf_counter()
    e.vjp(np.ones_like(cl), generation=i)
    t2 = time.perf_counter()
    if i:  # the first evaluation warms up
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
print(json.dumps(dict(backend=a.backend, nside=a.nside, lmax=lmax, batch=a.batch, masks=a.masks, cross=a.cross,
                      transforms=a.batch * max(a.masks, 1) * (2 if a.cross else 1), setup_seconds=setup,
                      forward_seconds=float(np.median(fwd)), vjp_seconds=float(np.median(bwd)),
                      info={k: v for k, v in e.info().items() if k in ("version", "transforms", "spin")})))
