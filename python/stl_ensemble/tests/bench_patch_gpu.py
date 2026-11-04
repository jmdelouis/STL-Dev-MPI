#!/usr/bin/env python3
"""Timing of PatchEngine at the operating point of Healpix-STL-IA
(train_patches.py): every step draws D discs among the centres of the
NESTED pixels of nside 8, and evaluates P pairs per disc (D P pairs in one
forward_cross and one vjp_cross).

python bench_patch_gpu.py [backend] [steps] [nside] [D] [P] [R] [core] [cache_bytes]
(defaults: gpu 20 256 16 15 10 4; J = 4, kernel 5, 4 orientations, cross).
"""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port import native
from stl_port.native import PatchEngine

arg = lambda i, d, f=str: f(sys.argv[i]) if len(sys.argv) > i else d
backend = arg(1, "gpu")
steps = arg(2, 20, int)
nside = arg(3, 256, int)
n_discs = arg(4, 16, int)
pairs = arg(5, 15, int)
radius = arg(6, 10.0, float)
core = arg(7, 4.0, float)
cache_bytes = arg(8, None, int)
rng = np.random.default_rng(0)
a_map = rng.normal(size=12 * nside**2)
b_map = rng.normal(size=12 * nside**2)
kw = dict(backend=backend, cache=800)
if cache_bytes is not None:
    kw["cache_bytes"] = cache_bytes
with PatchEngine(nside, 4, cross=True, **kw) as e:
    times = []
    # Unit vectors of the 768 pixel centres of nside 8.
    coarse = np.empty((768, 3))
    ids8 = np.arange(768, dtype=np.int64)
    assert e._lib.stlp_pixel_vectors(8, ids8.ctypes.data_as(native._INT64), 768, native._pointer(coarse)) == 0
    for step in range(steps):
        discs = rng.choice(768, size=n_discs, replace=False)
        centres = np.repeat(coarse[discs], pairs, axis=0)
        t0 = time.perf_counter()
        ids = e.select(centres, radius, core)
        t1 = time.perf_counter()
        a = e.gather(a_map)
        b = e.gather(b_map)
        t2 = time.perf_counter()
        v, var = e.forward_cross(a, b, generation=step)
        t3 = time.perf_counter()
        ga, gb = e.vjp_cross(np.ones_like(v), generation=step)
        t4 = time.perf_counter()
        times.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        print(f"step {step}: select {t1 - t0:.3f} s, gather {t2 - t1:.3f} s, forward {t3 - t2:.3f} s, vjp {t4 - t3:.3f} s, "
              f"total {t4 - t0:.3f} s", flush=True)
    info = e.info()
    t = np.array(times[steps // 2:])
    print(f"{backend} nside {nside} J 4 discs {n_discs} x {pairs} pairs ({len(ids)} patches, {np.mean([i.size for i in ids]):.0f} pixels), "
          f"R {radius} core {core}, ncoeff {e.ncoeff}: steady state (last {len(t)} steps) select {t[:, 0].mean():.3f} s, "
          f"forward {t[:, 2].mean():.3f} s, vjp {t[:, 3].mean():.3f} s, step {t.sum(1).mean():.3f} s")
    print(json.dumps({k: info[k] for k in ("cache", "backend")} | ({"device": info["device"]} if "device" in info else {})))
