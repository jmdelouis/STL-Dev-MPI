"""Tails of the A100 members against r000-r009 at nside 1024: min, max, std, excess kurtosis of kappa,
the excess kurtosis of the 4-children details at levels 1024 and 256 (as cv10.py), and the number of
pixels beyond the extremes of the ten realizations."""
import glob
import sys

import numpy as np

real = sorted(glob.glob(sys.argv[1] + "/r00?_nside1024_nested.npy"))
mem = sorted(glob.glob(sys.argv[2] + "/ens_m00?_physical_nested.npy"))


def kurt(x):
    z = (x - x.mean()) / x.std()
    return (z ** 4).mean() - 3


def detail_kurt(m, nside):
    while m.size > 12 * nside * nside:
        m = m.reshape(-1, 4).mean(1)
    p = m.reshape(-1, 4).mean(1)
    return kurt((m.reshape(-1, 4) - p[:, None]).ravel())


rows = []
lo, hi = np.inf, -np.inf
for f in real + mem:
    m = np.load(f).astype(np.float64)
    m = m - m.mean()
    if f in real:
        lo, hi = min(lo, m.min()), max(hi, m.max())
    rows.append((f.split("/")[-1][:14], m.min(), m.max(), m.std(), kurt(m), detail_kurt(m, 1024), detail_kurt(m, 256), m))
print(f"realizations' extremes after mean removal: min {lo:.4f} max {hi:.4f}")
print(f"{'map':14s} {'min':>9s} {'max':>9s} {'std':>9s} {'kurt':>8s} {'kd1024':>8s} {'kd256':>8s} {'>max':>6s} {'<min':>6s} {'q1e-6':>9s} {'q1-1e-6':>9s}")
for name, a, b, s, k, k1, k2, m in rows:
    print(f"{name:14s} {a:9.4f} {b:9.4f} {s:9.5f} {k:8.1f} {k1:8.1f} {k2:8.1f} {int((m > hi).sum()):6d} {int((m < lo).sum()):6d}"
          f" {np.quantile(m, 1e-6):9.4f} {np.quantile(m, 1 - 1e-6):9.4f}")
