"""Table of the deviations of the ensemble members from r000, in units of
the standard deviation of the 10 Takahashi realizations.

    python table_zscores.py CACHE NSIDE OUT_DIR LABEL=RUN_DIR [...]

Rows: per level, variance and excess kurtosis of the kappa details and
variance of the gaussianised details; peak counts above 3, 5, 10 sigma; the
rms over the C_l bands with l >= 31 and over the Minkowski thresholds
-2 <= nu <= 5 (4-pixel smoothing) of each member's deviation.  Columns: the
realizations' relative std, r000's deviation from their mean, then per run
the median [min, max] over the members.  Writes OUT_DIR/table_zscores_n{NSIDE}.tex
(a table* for A&A) and .txt.
"""

import warnings

import numpy as np

import common as C

warnings.simplefilter("ignore", RuntimeWarning)

a = C.cli(__doc__)
N = a.nside
real, runs = C.load_all(a.cache, N, a.runs)
levels = real["levels"][0]


def fmt(x):
    s = f"{x:+.1f}".replace("-", "$-$")
    return rf"\textbf{{{s}}}" if abs(x) > 2 else s


def cell(z):
    return f"{fmt(np.median(z))} [{fmt(z.min())}, {fmt(z.max())}]"


def rms_cell(z):
    return f"{np.median(z):.1f} [{z.min():.1f}, {z.max():.1f}]"


rows = []  # (group, label, rel std, r000 z, [cells per run], [txt per run])
names = {"vk": r"variance, $\kappa$", "kk": r"excess kurtosis, $\kappa$", "vg": "variance, gaussianised"}
for key in ("vk", "kk", "vg"):
    r0, mean, std = C.cosmic(real, key)
    for i, n in enumerate(levels):
        zs = [C.zscores(real, d, key)[:, i] for _, d in runs]
        rows.append((names[key], str(n), f"{std[i] / abs(mean[i]):.1%}".replace("%", r"\%"), fmt((r0[i] - mean[i]) / std[i]),
                     [cell(z) for z in zs]))
r0, mean, std = C.cosmic(real, "peaks_count_1")
for j, t in enumerate(C.PEAK_THRESHOLDS):
    zs = [C.zscores(real, d, "peaks_count_1")[:, j] for _, d in runs]
    rows.append(("peaks", rf"$>{t:g}\sigma$", f"{std[j] / mean[j]:.1%}".replace("%", r"\%"), fmt((r0[j] - mean[j]) / std[j]),
                 [cell(z) for z in zs]))
lo, hi, _ = C.cl_bands(N)
sel = lo >= 31
band = lambda cl: np.array([C.bin_cl(c, lo, hi) for c in cl])
R = band(real["cl"])
rcl = lambda cl: np.sqrt(np.mean(((band(cl) - R[0]) / R.std(0, ddof=1))[:, sel] ** 2, 1))
rows.append((r"$C_\ell$", r"rms, $\ell\geq31$", "", "", [rms_cell(rcl(d["cl"])) for _, d in runs]))
nusel = (C.NU >= -2) & (C.NU <= 5)
for k in range(3):
    key = f"mf0_v{k}"
    rms = lambda d: np.sqrt(np.nanmean(C.zscores(real, d, key)[:, nusel] ** 2, 1))
    rows.append(("Minkowski", rf"$V_{k}$, rms", "", "", [rms_cell(rms(d)) for _, d in runs]))

ncol = 4 + len(runs)
fwhm = C.mf_fwhms(N)[0]
tex = [r"\begin{table*}",
       rf"\caption{{Deviation of the ensemble members from the target r000 at nside {N}, in units of the standard deviation "
       r"$\sigma$ of the 10 Takahashi realizations: median [min, max] over the members (bold: $|z| > 2$). "
       r"$\sigma/\mu$ is the realizations' relative standard deviation and the r000 column the target's own deviation from their mean. "
       r"Details of level $n$: the map at nside $n$ minus the mean of each group of 4 children. "
       rf"The $C_\ell$ and Minkowski rows give the rms over the bands ($\ell \geq 31$) or thresholds ($-2\leq\nu\leq5$, FWHM {fwhm:g}$'$) "
       rf"of each member's deviation.}}",
       rf"\label{{tab:zscores_n{N}}}", r"\centering\small",
       r"\begin{tabular}{ll" + "r" * 2 + "c" * len(runs) + "}", r"\hline\hline",
       " & ".join(["Statistic", "Level", r"$\sigma/\mu$", "r000"] + [label for label, _ in runs]) + r" \\", r"\hline"]
txt = [f"nside {N}: deviation of the members from r000 in sigma of the realizations, median [min, max]"]
group = None
for g, lab, rel, z0, cells in rows:
    if g != group:
        if group is not None:
            tex.append(r"\hline")
        tex.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{g}}}}} \\")
        group = g
    tex.append(" & ".join(["", lab, rel, z0] + cells) + r" \\")
    clean = lambda s: s.replace("$", "").replace(r"\textbf{", "").replace("}", "").replace("\\", "")
    txt.append(f"{clean(g):28s} {clean(lab):14s} {clean(rel):8s} {clean(z0):6s} " + " | ".join(clean(c) for c in cells))
tex += [r"\hline", r"\end{tabular}", r"\end{table*}"]
open(f"{a.out}/table_zscores_n{N}.tex", "w").write("\n".join(tex) + "\n")
open(f"{a.out}/table_zscores_n{N}.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
