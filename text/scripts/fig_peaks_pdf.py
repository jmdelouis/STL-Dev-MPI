"""Peak counts and one-point PDF of kappa.

    python fig_peaks_pdf.py CACHE NSIDE OUT_DIR LABEL=RUN_DIR [...]

Top: number of peaks per bin of height (units of the std of r000 at that
nside) at the map's nside and at nside / 4, divided by the mean of the
realizations.  Bottom: PDF of kappa / std(r000) of the maps degraded to
nside / 4 and nside / 16 (at the map's own nside the members have r000's
histogram by construction), divided by the mean of the realizations.  Grey
band: +- 1 sigma of the realizations; dashed: r000; coloured: median of the
members and their min-max range.  Writes OUT_DIR/peaks_pdf_n{NSIDE}.pdf.
"""

import warnings

import numpy as np

import common as C

warnings.simplefilter("ignore", RuntimeWarning)

a = C.cli(__doc__)
N = a.nside
plt = C.style()
real, runs = C.load_all(a.cache, N, a.runs)
pk_c = 0.5 * (C.PEAK_EDGES[1:] + C.PEAK_EDGES[:-1])
pdf_c = 0.5 * (C.PDF_EDGES[1:] + C.PDF_EDGES[:-1])
panels = [(f"peaks_hist_{f}", pk_c, f"peaks, nside {N // f}", "peak height / std(r000)") for f in (1, 4)]
panels += [(f"pdf_{f}", pdf_c, f"PDF, nside {N // f}", r"$\kappa$ / std(r000)") for f in C.PDF_FACTORS]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
for ax, (key, x, title, xlabel) in zip(axes.ravel(), panels):
    R = real[key].astype(np.float64)
    mean, std = R.mean(0), R.std(0, ddof=1)
    # Bins where the realizations average fewer than 10 peaks or 30 pixels are left out.
    if key.startswith("peaks"):
        ok = mean >= 10
    else:
        factor = int(key.split("_")[1])
        ok = mean * np.diff(C.PDF_EDGES) * 12 * (N // factor) ** 2 >= 30
    if ok.sum() < 3:
        ok = mean > 0
    rel = lambda y: np.where(ok, y / np.where(ok, mean, 1), np.nan)
    ax.fill_between(x, rel(mean - std), rel(mean + std), color=C.BAND, lw=0, label=r"realizations, $\pm1\sigma$")
    ax.plot(x, rel(R[0]), color=C.INK, ls="--", lw=0.9, label="r000 (target)")
    for (label, data), col in zip(runs, C.RUN_COLORS):
        y = np.array([rel(v) for v in data[key].astype(np.float64)])
        ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), color=col, alpha=0.25, lw=0)
        ax.plot(x, np.nanmedian(y, 0), color=col, lw=1.2, label=label)
    ax.axhline(1, color=C.INK, ls=":", lw=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel="/ mean of the realizations", ylim=(0.5, 1.5))
    ax.set_xlim(x[ok].min(), x[ok].max())
axes[0, 0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{a.out}/peaks_pdf_n{N}.pdf")
print(f"{a.out}/peaks_pdf_n{N}.pdf")
