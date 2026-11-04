"""White-noise initial maps shared by both codes of docs/bench_n128.md.

    python make_inits.py NSIDE B PREFIX     ->  PREFIX_m000.npy ... (standardised units, float64)

Member m is numpy's default_rng(1000 + m).standard_normal(12 nside^2).
"""

import sys

import numpy as np

nside, members, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for m in range(members):
    np.save(f"{prefix}_m{m:03d}.npy", np.random.default_rng(1000 + m).standard_normal(12 * nside**2))
