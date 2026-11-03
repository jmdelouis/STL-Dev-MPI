"""Initial images with the exact histogram of the targets (docs/lagrangian_synthesis.md, step 0b).

    python permuted_init.py --run RUNPREFIX --nbatch B --seed S --out init.npy

Takes the Nyquist-filtered targets RUNPREFIX_target.npy [Bt, C, H, W] and the
standardisation of RUNPREFIX.json (target_mean, target_std per channel),
pools the standardised pixels of every channel over the targets, and fills
B images with a random permutation of them (white spectrum, exact pooled
histogram), in the standardised units of stl-synth2d --init.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--nbatch", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rep = json.load(open(a.run + ".json"))
t = np.load(a.run + "_target.npy")
bt, c, h, w = t.shape
mean, std = np.array(rep["target_mean"]), np.array(rep["target_std"])
z = (t - mean[None, :, None, None]) / std[None, :, None, None]
rng = np.random.default_rng(a.seed)
out = np.empty((a.nbatch, c, h, w))
for ch in range(c):
    pool = z[:, ch].ravel()
    for b in range(a.nbatch):
        out[b, ch] = rng.choice(pool, size=h * w, replace=False).reshape(h, w)
np.save(a.out, out)
print(json.dumps({"shape": out.shape, "mean": float(out.mean()), "std": float(out.std())}))
