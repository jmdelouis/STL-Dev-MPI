"""Minkowski functionals V0, V1, V2 of kappa against the 10 realizations.

    python fig_minkowski.py CACHE NSIDE OUT_DIR LABEL=RUN_DIR [...]

One figure per smoothing scale (common.mf_fwhms: 4 pixels at this nside,
and 30 arcmin).  Top row: V_k(nu) of r000 and of the members (median), the
Gaussian expectation for r000's tau (dotted).  Bottom row: (V_k - V_k(r000))
/ sigma_k(nu), sigma the std of the realizations; grey band +- 1.  Writes
OUT_DIR/minkowski_n{NSIDE}_s{i}.pdf and minkowski_n{NSIDE}.txt (rms over the
thresholds of the members' median deviation, in sigma).
"""

import warnings

import numpy as np

import common as C

warnings.simplefilter("ignore", RuntimeWarning)

a = C.cli(__doc__)
N = a.nside
plt = C.style()
real, runs = C.load_all(a.cache, N, a.runs)
nu = C.NU
lines = [f"nside {N}: Minkowski functionals, members' median deviation from r000 in sigma of the realizations, rms over -2 <= nu <= 5"]
sel = (nu >= -2) & (nu <= 5)
for i, fwhm in enumerate(C.mf_fwhms(N)):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    for k in range(3):
        key = f"mf{i}_v{k}"
        r0, _, std = C.cosmic(real, key)
        top, bot = axes[0, k], axes[1, k]
        top.fill_between(nu, r0 - std, r0 + std, color=C.BAND, lw=0, label=r"realizations, $\pm1\sigma$ around r000")
        top.plot(nu, r0, color=C.INK, lw=1.0, label="r000 (target)")
        top.plot(nu, real[f"mf{i}_g{k}"][0], color=C.INK, ls=":", lw=0.9, label=r"Gaussian, $\tau$ of r000")
        bot.axhspan(-1, 1, color=C.BAND, lw=0)
        for r in (real[key][1:] - r0) / std:
            bot.plot(nu, r, color="0.65", lw=0.5)
        for (label, data), col in zip(runs, C.RUN_COLORS):
            z = (data[key] - r0) / std
            top.plot(nu, np.median(data[key], 0), color=col, lw=1.0, label=label)
            zm = np.median(z, 0)
            bot.fill_between(nu, z.min(0), z.max(0), color=col, alpha=0.25, lw=0)
            bot.plot(nu, zm, color=col, lw=1.2)
            lines.append(f"fwhm {fwhm:g}' V{k} {label}: rms {np.sqrt(np.nanmean(zm[sel] ** 2)):.2f}, "
                         f"max |median| {np.nanmax(np.abs(zm[sel])):.2f}")
        top.set_title(f"$V_{k}$")
        bot.axhline(0, color=C.INK, ls=":", lw=0.8)
        bot.set(xlabel=r"threshold $\nu$", ylim=(-5, 5))
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    axes[1, 0].set_ylabel(r"$\Delta V_k / \sigma_k$")
    fig.suptitle(f"nside {N}, smoothing FWHM {fwhm:g} arcmin", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{a.out}/minkowski_n{N}_s{i}.pdf")
open(f"{a.out}/minkowski_n{N}.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
