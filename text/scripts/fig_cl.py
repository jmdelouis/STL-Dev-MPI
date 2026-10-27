"""Kappa angular power spectrum of the ensemble members against the 10
Takahashi realizations.

    python fig_cl.py CACHE NSIDE OUT_DIR LABEL=RUN_DIR [...]

Top: l (l + 1) C_l / 2 pi of r000 and of the members (median).  Bottom:
C_l / C_l(r000) in logarithmic bands; grey band: mean +- 1 sigma of the 10
realizations, thin grey lines: r001-r009; coloured: median of the members
and their min-max range.  Writes OUT_DIR/cl_n{NSIDE}.pdf and
cl_n{NSIDE}.txt (per band: realizations' relative std, members' median
deviation from r000 in sigma).
"""

import numpy as np

import common as C

a = C.cli(__doc__)
N = a.nside
plt = C.style()
real, runs = C.load_all(a.cache, N, a.runs)
lo, hi, lc = C.cl_bands(N)
ell = np.arange(3 * N)
band = lambda cl: np.array([C.bin_cl(c, lo, hi) for c in np.atleast_2d(cl)])
R = band(real["cl"])
r0, mean, std = R[0], R.mean(0), R.std(0, ddof=1)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True, gridspec_kw={"height_ratios": (1, 1.3)})
d = ell * (ell + 1) / (2 * np.pi)
ax0.loglog(ell[2:], (d * real["cl"][0])[2:], color=C.INK, lw=1.0, label="r000 (target)")
for (label, data), col in zip(runs, C.RUN_COLORS):
    ax0.loglog(ell[2:], (d * np.median(data["cl"], 0))[2:], color=col, lw=0.8, label=label)
ax0.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
ax0.legend(frameon=False, loc="lower right")
ax1.fill_between(lc, (mean - std) / r0, (mean + std) / r0, color=C.BAND, lw=0, label=r"10 realizations, mean $\pm1\sigma$")
for r in R[1:] / r0:
    ax1.semilogx(lc, r, color="0.65", lw=0.5)
lines = [f"nside {N}: C_l bands, relative std of the realizations, members' median deviation from r000 in sigma"]
for (label, data), col in zip(runs, C.RUN_COLORS):
    y = band(data["cl"]) / r0
    ax1.fill_between(lc, y.min(0), y.max(0), color=col, alpha=0.25, lw=0)
    ax1.semilogx(lc, np.median(y, 0), "o-", ms=2.5, color=col, lw=1.2, label=label)
    z = (np.median(band(data["cl"]), 0) - r0) / std
    sel = lo >= 31
    lines.append(f"{label}: l >= 31 median |z| {np.median(np.abs(z[sel])):.2f} max |z| {np.abs(z[sel]).max():.2f}, "
                 f"rms of C_l / C_l(r000) - 1 {np.sqrt(np.mean((np.median(y, 0)[sel] - 1) ** 2)):.3f}")
    lines += [f"  {a_}-{b_ - 1}: real {s / m:6.1%}  z {zz:+5.2f}" for a_, b_, s, m, zz in zip(lo, hi, std, mean, z)]
ax1.axhline(1, color=C.INK, ls=":", lw=0.8)
ax1.set(xlabel=r"multipole $\ell$", ylabel=r"$C_\ell / C_\ell^{\rm r000}$", ylim=(0.5, 1.5), xlim=(2, 3 * N))
ax1.legend(frameon=False, loc="lower left")
fig.savefig(f"{a.out}/cl_n{N}.pdf")
open(f"{a.out}/cl_n{N}.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
