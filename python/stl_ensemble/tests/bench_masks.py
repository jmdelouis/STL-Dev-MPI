#!/usr/bin/env python3
"""Time of NativeEngine forward + vjp with K galactic masks against the
whole sky.  Usage: bench_masks.py [backend] [steps] [nside] [J] [cross]
(defaults gpu 10 256 4 0); masks: whole sky and |b| > 0, 20, 30, 45 degrees
about the z axis."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stl_port.native import NativeEngine
from test_masks import latitude_masks

backend = sys.argv[1] if len(sys.argv) > 1 else "gpu"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
nside = int(sys.argv[3]) if len(sys.argv) > 3 else 256
j = int(sys.argv[4]) if len(sys.argv) > 4 else 4
cross = len(sys.argv) > 5 and sys.argv[5] == "1"

masks = latitude_masks(nside, (0.0, 0.0001, 20.0, 30.0, 45.0))
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 12 * nside**2) if cross else (12 * nside**2,))
kw = dict(backend=backend, cross=cross)
for label, m in (("whole sky", None), (f"{len(masks)} masks", masks)):
    with NativeEngine(nside, j, masks=m, **kw) as eng:
        times = []
        for s in range(steps):
            t0 = time.perf_counter()
            v, var = eng.forward(x, generation=s, moments=True)
            t1 = time.perf_counter()
            eng.vjp(np.ones_like(v), generation=s)
            t2 = time.perf_counter()
            times.append((t1 - t0, t2 - t1))
        f, b = np.median(np.array(times[1:]), 0)
        print(f"{backend} nside {nside} J {j} cross {cross} {label}: ncoeff {eng.ncoeff}, values {v.shape}, "
              f"forward {f:.3f} s, vjp {b:.3f} s, step {f + b:.3f} s (median of {steps - 1})", flush=True)
        if m is not None:
            print("pixels at level 0 per mask:", [len(i) for i in m], flush=True)
        if backend == "gpu":
            info = eng.info()
            print("device memory:", info.get("device_memory"), flush=True)
