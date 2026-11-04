#!/usr/bin/env python3
"""SpectrumEngine against healpy on the same maps and threads: spin 0
(anafast, iter=0) and spin s (map2alm_spin + alm2cl), median of --repeat
evaluations after a warm-up. healpy has no VJP: its column is the forward.

    OMP_NUM_THREADS=N STL_THREADS=N python bench_spectrum_healpy.py --nside 256
"""
import argparse
import json
import time
import healpy as hp
import numpy as np
from stl_port.native import SpectrumEngine

p = argparse.ArgumentParser()
p.add_argument("--nside", type=int, default=256)
p.add_argument("--lmax", type=int, default=None, help="default 2 nside - 1")
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--spins", default="0,1,2")
a = p.parse_args()
lmax = 2 * a.nside - 1 if a.lmax is None else a.lmax
npix = 12 * a.nside**2
rng = np.random.default_rng(0)


def median(f):
    f()
    t = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


rows = []
for spin in map(int, a.spins.split(",")):
    x = rng.normal(size=(npix,) if spin == 0 else (2, npix))
    xr = hp.reorder(x, n2r=True) if spin == 0 else np.stack([hp.reorder(v, n2r=True) for v in x])
    if spin == 0:
        healpy = lambda: hp.anafast(xr, lmax=lmax, iter=0)
    else:
        def healpy():
            e, b = hp.map2alm_spin(list(xr), spin, lmax=lmax)
            return hp.alm2cl(e), hp.alm2cl(b), hp.alm2cl(e, b)
    with SpectrumEngine(a.nside, lmax, spin=spin) as eng:
        state = {"g": 0}

        def forward():
            state["g"] += 1
            return eng.forward(x, generation=state["g"])

        def both():
            cl = forward()
            eng.vjp(np.ones_like(cl), generation=state["g"])

        rows.append(dict(spin=spin, nside=a.nside, lmax=lmax, healpy_forward=median(healpy),
                         stl_forward=median(forward), stl_forward_vjp=median(both)))
    print(json.dumps(rows[-1]), flush=True)
