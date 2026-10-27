"""Per-level statistics of the 4-children pyramid details: variance and
excess kurtosis of kappa, variance in gaussianised space.

    python fig_levels.py CACHE NSIDE OUT_DIR LABEL=RUN_DIR [...]

Every statistic is divided by r000's.  Grey band: mean +- 1 sigma of the 10
realizations; thin grey lines: r001-r009; coloured: median of the members
and their min-max range.  Writes OUT_DIR/levels_n{NSIDE}.pdf.
"""

import numpy as np

import common as C

a = C.cli(__doc__)
N = a.nside
plt = C.style()
real, runs = C.load_all(a.cache, N, a.runs)
levels = real["levels"][0]
titles = {"vk": r"variance of the details, $\kappa$", "kk": r"excess kurtosis of the details, $\kappa$",
          "vg": "variance of the details, gaussianised"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, key in zip(axes, ("vk", "kk", "vg")):
    r0, mean, std = C.cosmic(real, key)
    ax.fill_between(levels, (mean - std) / r0, (mean + std) / r0, color=C.BAND, lw=0, label=r"realizations, mean $\pm1\sigma$")
    for r in real[key][1:] / r0:
        ax.semilogx(levels, r, color="0.65", lw=0.5)
    for (label, data), col in zip(runs, C.RUN_COLORS):
        y = data[key] / r0
        ax.fill_between(levels, y.min(0), y.max(0), color=col, alpha=0.25, lw=0)
        ax.semilogx(levels, np.median(y, 0), "o-", ms=2.5, color=col, lw=1.2, label=label)
    ax.axhline(1, color=C.INK, ls=":", lw=0.8)
    ax.set(xlabel="nside of the level", title=titles[key])
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(n) for n in levels], rotation=90)
axes[0].set_ylabel("/ r000")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{a.out}/levels_n{N}.pdf")
print(f"{a.out}/levels_n{N}.pdf")
