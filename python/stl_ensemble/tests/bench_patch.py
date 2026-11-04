#!/usr/bin/env python3
"""Timing of PatchEngine: geometry of new discs, cached discs, forward and
VJP for a batch of random centres (python bench_patch.py NSIDE [B] [R] [r] [threads])."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import PatchEngine

nside = int(sys.argv[1]) if len(sys.argv) > 1 else 64
b = int(sys.argv[2]) if len(sys.argv) > 2 else 16
radius = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
core = float(sys.argv[4]) if len(sys.argv) > 4 else 7.0
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 0
j = int(np.log2(nside)) - 1
rng = np.random.default_rng(0)
x = rng.normal(size=12 * nside**2)
with PatchEngine(nside, j, threads=threads) as e:
    for rep in range(3):
        centres = rng.normal(size=(b, 3))
        t0 = time.perf_counter()
        ids = e.select(centres, radius, core)
        t1 = time.perf_counter()
        e._selection = None
        e.select(centres, radius, core)
        t2 = time.perf_counter()
        v, var = e.forward(e.gather(x), generation=rep)
        t3 = time.perf_counter()
        g = e.vjp(np.ones_like(v), generation=rep)
        t4 = time.perf_counter()
        print(f"nside {nside} J {j} B {b} R {radius} r {core} threads {e.info()['threads']}: pixels/disc {np.mean([i.size for i in ids]):.0f}, "
              f"ncoeff {e.ncoeff}; new geometry {t1 - t0:.3f} s, cached {t2 - t1:.4f} s, forward {t3 - t2:.3f} s, vjp {t4 - t3:.3f} s", flush=True)
