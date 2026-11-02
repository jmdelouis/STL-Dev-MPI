"""Cosmic variance of per-level statistics over 10 Takahashi realizations (r000-r009, zs1) at one
nside, and the position of r000 and of the ensemble members in it.

    python cv10.py NSIDE OUT_PREFIX MAP_r000 MAP_r001 ... [-- LABEL=RUN_DIR ...]

Per map: variance and excess kurtosis of the 4-children details of kappa per level, variance of
the details of the gaussianised map (its own rank transform, for the realizations and the members alike), peaks above 3/5/10 sigma (sigma of
r000), C_l in logarithmic bands.  Writes OUT_PREFIX.txt and OUT_PREFIX.png.
"""
import glob
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

args = sys.argv[1:]
N, prefix = int(args[0]), args[1]
rest = args[2:]
maps = rest[:rest.index("--")] if "--" in rest else rest
specs = rest[rest.index("--") + 1:] if "--" in rest else []


def gauss(m):
    r = np.empty(m.size)
    r[np.argsort(m, kind="stable")] = np.arange(m.size)
    return ndtri((r + 0.5) / m.size)


def details(m, lowest=8):
    out = {}
    nside = int(round((m.size / 12) ** 0.5))
    while nside >= lowest:
        p = m.reshape(-1, 4).mean(1)
        d = (m.reshape(-1, 4) - p[:, None]).ravel()
        v = d.var()
        out[nside] = (v, (d**4).mean() / v**2 - 3.0)
        m, nside = p, nside // 2
    return out


def peaks(m, sig):
    nside = int(round((m.size / 12) ** 0.5))
    nb = hp.get_all_neighbours(nside, np.arange(m.size), nest=True)
    vals = np.where(nb >= 0, m[np.clip(nb, 0, None)], -np.inf)
    pk = m[(m[None, :] > vals).all(0)] / sig
    return np.array([(pk > t).sum() for t in (3, 5, 10)], dtype=float)


lmax = 3 * N - 1
edges = np.unique(np.round(np.geomspace(2, lmax + 1, 20)).astype(int))
lo, hi = edges[:-1], edges[1:]
lc = np.sqrt(lo * (hi - 1).clip(min=lo))


def cl_bands(m):
    c = hp.anafast(hp.reorder(m, n2r=True), lmax=lmax)
    return np.array([c[a:b].mean() for a, b in zip(lo, hi)])


r000 = np.load(maps[0]).astype(np.float64)
sig = r000.std()


def measure(m):
    dk = details(m)
    dg = details(gauss(m))
    lv = sorted(dk)
    return dict(levels=lv, vk=np.array([dk[n][0] for n in lv]), kk=np.array([dk[n][1] for n in lv]),
                vg=np.array([dg[n][0] for n in lv]), pk=peaks(m, sig), cl=cl_bands(m))


real = [measure(np.load(f).astype(np.float64)) for f in maps]
levels = real[0]["levels"]
keys = ("vk", "kk", "vg", "pk", "cl")
R = {k: np.array([r[k] for r in real]) for k in keys}
mean = {k: R[k].mean(0) for k in keys}
std = {k: R[k].std(0, ddof=1) for k in keys}
names = {"vk": "variance kappa", "kk": "kurtosis kappa", "vg": "variance gauss"}
lines = [f"{len(maps)} realizations at nside {N}: mean, relative std over the realizations, r000 deviation from the mean in sigma"]
for k in ("vk", "kk", "vg"):
    lines.append(f"{names[k]}:")
    for i, n in enumerate(levels):
        lines.append(f"  {n:5d}: mean {mean[k][i]:.4g}  rel std {std[k][i] / abs(mean[k][i]):6.1%}  r000 {(R[k][0, i] - mean[k][i]) / std[k][i]:+5.1f}")
lines.append("peaks above 3/5/10 sigma(r000): " + " | ".join(
    f"mean {mean['pk'][j]:.0f} rel std {std['pk'][j] / mean['pk'][j]:.1%} r000 {(R['pk'][0, j] - mean['pk'][j]) / std['pk'][j]:+.1f}" for j in range(3)))
lines.append("C_l bands: l, rel std over realizations, r000 deviation in sigma")
lines += [f"  {a}-{b - 1}: {std['cl'][j] / mean['cl'][j]:6.1%}  {(R['cl'][0, j] - mean['cl'][j]) / std['cl'][j]:+5.1f}" for j, (a, b) in enumerate(zip(lo, hi))]
runs = []
for sp in specs:
    label, run = sp.split("=", 1)
    M = [measure(np.load(k).astype(np.float64)) for k in sorted(glob.glob(f"{run}/synthesis_m[0-9][0-9][0-9]_nested.npy"))]
    Z = {k: (np.array([m[k] for m in M]) - R[k][0]) / std[k] for k in keys}
    runs.append((label, M, Z))
    lines.append(f"== {label}: deviation of the members from r000 in cosmic sigma of the realizations, median [min, max]")
    for i, n in enumerate(levels):
        lines.append(f"  {n:5d} | " + " | ".join(
            f"{names[k]} {np.median(Z[k][:, i]):+5.1f} [{Z[k][:, i].min():+5.1f},{Z[k][:, i].max():+5.1f}]" for k in ("vk", "kk", "vg")))
    lines.append("  peaks 3/5/10: " + " | ".join(
        f"{np.median(Z['pk'][:, j]):+5.1f} [{Z['pk'][:, j].min():+5.1f},{Z['pk'][:, j].max():+5.1f}]" for j in range(3)))
    zc = np.median(Z["cl"], 0)
    sel = lo >= 31
    lines.append(f"  C_l bands l >= 31: median |z| {np.median(np.abs(zc[sel])):.1f}, max |z| {np.abs(zc[sel]).max():.1f}")
open(prefix + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# Figure: every statistic relative to r000, realizations (mean +- 1 sigma band and curves) and runs.
fig, ax = plt.subplots(1, 4, figsize=(24, 5))
cols = ["#eb6834", "#2a78d6", "#3c8a3a", "#8a3c8a"]
titles = ("variance of the details, kappa", "excess kurtosis of the details, kappa", "variance of the details, gaussianised")
for a, k, t in zip(ax[:3], ("vk", "kk", "vg"), titles):
    a.fill_between(levels, (mean[k] - std[k]) / R[k][0], (mean[k] + std[k]) / R[k][0], color="0.85", lw=0,
                   label=f"{len(maps)} realizations, mean +- 1 sigma")
    for r in (R[k] / R[k][0])[1:]:
        a.semilogx(levels, r, color="0.6", lw=0.6)
    for c, (label, M, Z) in zip(cols, runs):
        y = np.array([m[k] for m in M]) / R[k][0]
        a.fill_between(levels, y.min(0), y.max(0), color=c, alpha=0.25, lw=0)
        a.semilogx(levels, np.median(y, 0), "o-", color=c, lw=1.6, label=label)
    a.axhline(1, color="#52514e", ls=":", lw=1)
    a.set(xlabel="nside of the level", ylabel="/ r000", title=t)
    a.legend(frameon=False, fontsize=8)
a = ax[3]
a.fill_between(lc, (mean["cl"] - std["cl"]) / R["cl"][0], (mean["cl"] + std["cl"]) / R["cl"][0], color="0.85", lw=0,
               label=f"{len(maps)} realizations, mean +- 1 sigma")
for r in (R["cl"] / R["cl"][0])[1:]:
    a.semilogx(lc, r, color="0.6", lw=0.6)
for c, (label, M, Z) in zip(cols, runs):
    y = np.array([m["cl"] for m in M]) / R["cl"][0]
    a.fill_between(lc, y.min(0), y.max(0), color=c, alpha=0.25, lw=0)
    a.semilogx(lc, np.median(y, 0), "o-", ms=3, color=c, lw=1.6, label=label)
a.axhline(1, color="#52514e", ls=":", lw=1)
a.set(xlabel="multipole l", ylabel="C_l / C_l(r000)", title=f"kappa C_l, nside {N}", ylim=(0.5, 1.5))
a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(prefix + ".png", dpi=80)
print("figure", prefix + ".png")
