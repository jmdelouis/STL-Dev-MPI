"""Minkowski functionals per level, C_l and reduced scattering coefficients
of the target population (r000-r009) and of the ensembles: mean +- std of
each population, and the residual (mean - mean of the target) / std of the
target.

    python plot_populations.py CACHE STATS_JSON OUT_DIR REAL_DIR LABEL=RUN_DIR [...]

CACHE: measure_populations.py (one npz per map, CACHE/real and
CACHE/{slug of LABEL}); STATS_JSON: stl-stats over the maps of REAL_DIR and
of every RUN_DIR, standardised by the moments of r000 (--norm).  Reduced
coefficients: S1(j) and S2(j) averaged over the orientations; S3(j1, j3)
and S4(j1, j2, j3) the real part averaged over the orientations, divided by
sqrt(S2(j1) S2(j3)) and sqrt(S2(j1) S2(j2)).  j = 0 is the finest scale
(nside N), j the level of nside N / 2^j.  Sizes for A&A: Minkowski and
scattering two columns wide, C_l one column.  Writes
OUT_DIR/{mf,cl,sc}_populations_n{N}.png, .pdf (vector) and a .txt summary.
"""

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../../../text/scripts", "../../../tools"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, d)))
import common as C  # noqa: E402

ONE, TWO = 3.46, 7.0  # inches: one and two A&A columns (88 and 180 mm)
cache, stats_json, out, real_dir = sys.argv[1:5]
runs = C.parse_runs(sys.argv[5:])
os.makedirs(out, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)
plt = C.style()
from matplotlib.gridspec import GridSpec  # noqa: E402


def load(slug):
    files = sorted(f for f in os.listdir(f"{cache}/{slug}") if f.endswith(".npz"))
    data = [dict(np.load(f"{cache}/{slug}/{f}")) for f in files]
    return {k: np.array([d[k] for d in data]) for k in data[0]}


real = load("real")
pops = [(label, load(slug)) for label, slug, _ in runs]
N = int(round(((len(real["cl"][0]) + 1) / 3)))
NU = real["nu"][0]
LEVELS = [int(n) for n in real["levels"][0]]
summary = [f"nside {N}: populations against the target population r000-r009 (mean and std over the 10 realizations, ddof 1)",
           "residual z = (mean of the population - mean of the target) / std of the target; columns: rms and max |z|",
           "r000 alone is the target of the recipe (Campeti's method targets the ten realizations)"]


def stats_of(x):
    return x.mean(0), x.std(0, ddof=1)


def residual_panel(ax, xs, rmean, rstd, values, r0, xlabel=None, step=False):
    """Residuals of every population's mean, +- its std, and of r000 (dashed), in units of the target's std."""
    ax.axhspan(-1, 1, color=C.BAND, lw=0)
    ax.axhline(0, color=C.INK, lw=0.6, ls=":")
    ax.plot(xs, (r0 - rmean) / rstd, color=C.INK, lw=0.7, ls="--", drawstyle="steps-mid" if step else "default")
    lim = 3.0
    for (label, v), col in zip(values, C.RUN_COLORS):
        m, s = stats_of(v)
        z, w = (m - rmean) / rstd, s / rstd
        kw = {"drawstyle": "steps-mid"} if step else {}
        ax.fill_between(xs, z - w, z + w, color=col, alpha=0.18, lw=0, step="mid" if step else None)
        ax.plot(xs, z, color=col, lw=1.0, **kw)
        lim = max(lim, np.nanmax(np.abs(z)) * 1.1)
    lim = min(lim, 12.0)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel(r"$\Delta/\sigma_{\rm target}$")
    if xlabel:
        ax.set_xlabel(xlabel)


def value_panel(ax, xs, target, values, step=False):
    rmean, rstd = stats_of(target)
    ax.fill_between(xs, rmean - rstd, rmean + rstd, color="0.75", lw=0, label="target r000-r009, mean $\\pm\\sigma$",
                    step="mid" if step else None)
    ax.plot(xs, rmean, color=C.INK, lw=0.8, drawstyle="steps-mid" if step else "default")
    ax.plot(xs, target[0], color=C.INK, lw=0.7, ls="--", label="r000 (the recipe's target)",
            drawstyle="steps-mid" if step else "default")
    for (label, v), col in zip(values, C.RUN_COLORS):
        m, s = stats_of(v)
        ax.fill_between(xs, m - s, m + s, color=col, alpha=0.18, lw=0, step="mid" if step else None)
        ax.plot(xs, m, color=col, lw=1.0, label=f"{label}, mean $\\pm\\sigma$", drawstyle="steps-mid" if step else "default")
    return rmean, rstd


def save(fig, name):
    fig.savefig(f"{out}/{name}_n{N}.png", dpi=170)
    fig.savefig(f"{out}/{name}_n{N}.pdf")
    plt.close(fig)


def summarise(name, rmean, rstd, values, r0):
    row = f"  {name:28s}"
    for label, v in [("r000", r0[None])] + values:
        z = (v.mean(0) - rmean) / rstd
        row += f" | {label}: {np.sqrt(np.nanmean(z**2)):5.2f} {np.nanmax(np.abs(z)):6.2f}"
    summary.append(row)


# Minkowski functionals: one row per level (values, residuals), one column per functional.
fig = plt.figure(figsize=(TWO, 2.0 * len(LEVELS) + 0.4))
gs = GridSpec(2 * len(LEVELS), 3, figure=fig, height_ratios=[2.2, 1.0] * len(LEVELS), hspace=0.08, wspace=0.32)
summary.append("Minkowski functionals of kappa degraded to each level (4-children means, no smoothing), nu in units of the level's std")
for i, n in enumerate(LEVELS):
    for k in range(3):
        key = f"l{n}_v{k}"
        top = fig.add_subplot(gs[2 * i, k])
        bot = fig.add_subplot(gs[2 * i + 1, k], sharex=top)
        rmean, rstd = value_panel(top, NU, real[key], [(label, d[key]) for label, d in pops])
        # Thresholds where the target barely varies (V = 0 or 1 for every map) are left out.
        rstd = np.where(rstd > 0.05 * np.nanmax(rstd), rstd, np.nan)
        residual_panel(bot, NU, rmean, rstd, [(label, d[key]) for label, d in pops], real[key][0],
                       xlabel=r"threshold $\nu$" if i == len(LEVELS) - 1 else None)
        summarise(f"level {n} V{k}", rmean, rstd, [(label, d[key]) for label, d in pops], real[key][0])
        top.tick_params(labelbottom=False)
        if i < len(LEVELS) - 1:
            bot.tick_params(labelbottom=False)
        if i == 0:
            top.set_title(f"$V_{k}$")
        if k == 0:
            top.set_ylabel(f"nside {n}\n$V_{k}$")
        else:
            bot.set_ylabel("")
fig.axes[0].legend(frameon=False, fontsize=6, loc="upper right")
save(fig, "mf_populations")

# C_l in logarithmic bands.
lo, hi, centre = C.cl_bands(N, 41)
ell = np.arange(3 * N)


def banded(cl):
    d = cl * ell * (ell + 1) / (2 * np.pi)
    return np.array([[row[a:b].mean() for a, b in zip(lo, hi)] for row in d])


fig = plt.figure(figsize=(ONE, 3.6))
gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1.0], hspace=0.06)
top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
bot.sharex(top)
values = [(label, banded(d["cl"])) for label, d in pops]
target = banded(real["cl"])
rmean, rstd = value_panel(top, centre, target, values)
top.set_xscale("log")
top.set_yscale("log")
top.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
top.tick_params(labelbottom=False)
top.legend(frameon=False, fontsize=6, loc="upper left")
residual_panel(bot, centre, rmean, rstd, values, target[0], xlabel=r"multipole $\ell$")
summary.append("C_l in 40 logarithmic bands, l = 2 ... 3 nside - 1")
summarise("C_l bands", rmean, rstd, values, target[0])
high = centre >= 31
summarise("C_l bands l >= 31", rmean[high], rstd[high], [(label, v[:, high]) for label, v in values], target[0][high])
save(fig, "cl_populations")

# Reduced scattering coefficients.
st = json.load(open(stats_json))
plan = st["plan"]
fam = np.array([c["family"] for c in plan])
idx = {k: np.array([-1 if c[k] is None else c[k] for c in plan]) for k in ("j1", "j2", "j3")}
J = st["J"]


def reduced(re):
    re = np.asarray(re)
    s1 = np.array([re[(fam == "S1") & (idx["j1"] == j)].mean() for j in range(J)])
    s2 = np.array([re[(fam == "S2") & (idx["j1"] == j)].mean() for j in range(J)])
    pairs = [(a, b) for b in range(J) for a in range(b + 1)]
    s3 = np.array([re[(fam == "S3") & (idx["j1"] == a) & (idx["j3"] == b)].mean() / np.sqrt(s2[a] * s2[b]) for a, b in pairs])
    triples = [(a, b, c) for c in range(J) for b in range(c + 1) for a in range(b + 1)]
    s4 = np.array([re[(fam == "S4") & (idx["j1"] == a) & (idx["j2"] == b) & (idx["j3"] == c)].mean() / np.sqrt(s2[a] * s2[b])
                   for a, b, c in triples])
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}, pairs, triples


def population(directory):
    d = os.path.normpath(directory)
    rows = [reduced(m["re"])[0] for m in st["maps"] if os.path.normpath(os.path.dirname(m["file"])) == d]
    if not rows:
        raise SystemExit(f"no map of {directory} in {stats_json}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


_, pairs, triples = reduced(st["maps"][0]["re"])
sc_real = population(real_dir)
sc_pops = [(label, population(run)) for label, _, run in runs]
fig = plt.figure(figsize=(TWO, 8.6))
# Rows: S1 and S2, S3, S4 (values, residuals), with empty rows for the axis labels between them.
gs = GridSpec(8, 2, figure=fig, height_ratios=[2.0, 1.0, 0.55, 2.0, 1.0, 0.55, 2.0, 1.0], hspace=0.1, wspace=0.3)
summary.append(f"reduced scattering coefficients (J = {J}, L = {st['L']}, kernel {st['kernel_size']}), maps standardised by r000")
for col, key in enumerate(("S1", "S2")):
    top, bot = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
    bot.sharex(top)
    xs = np.arange(J)
    rmean, rstd = value_panel(top, xs, sc_real[key], [(label, p[key]) for label, p in sc_pops])
    top.set_yscale("log")
    top.set_ylabel(f"{key}(j)")
    top.tick_params(labelbottom=False)
    residual_panel(bot, xs, rmean, rstd, [(label, p[key]) for label, p in sc_pops], sc_real[key][0],
                   xlabel=r"scale $j$ (nside %d / $2^j$)" % N)
    summarise(key, rmean, rstd, [(label, p[key]) for label, p in sc_pops], sc_real[key][0])
fig.axes[0].legend(frameon=False, fontsize=6, loc="lower left")
for row, (key, groups, name) in enumerate((("S3", pairs, r"$(j_1, j_3)$"), ("S4", triples, r"$(j_1, j_2, j_3)$"))):
    top, bot = fig.add_subplot(gs[3 + 3 * row, :]), fig.add_subplot(gs[4 + 3 * row, :])
    bot.sharex(top)
    xs = np.arange(len(groups))
    rmean, rstd = value_panel(top, xs, sc_real[key], [(label, p[key]) for label, p in sc_pops], step=True)
    top.set_ylabel(f"reduced {key}")
    top.tick_params(labelbottom=False)
    residual_panel(bot, xs, rmean, rstd, [(label, p[key]) for label, p in sc_pops], sc_real[key][0], step=True,
                   xlabel=f"{name}, sorted by $j_3$ (ticks: first coefficient of each $j_3$)")
    starts = [i for i, g in enumerate(groups) if i == 0 or g[-1] != groups[i - 1][-1]]
    bot.set_xticks(starts)
    # A label only where the previous one leaves room (the first scales hold a few coefficients).
    shown, last = [], -len(groups)
    for i in starts:
        shown.append(f"$j_3$={groups[i][-1]}" if i - last >= len(groups) / 25 else "")
        last = i if shown[-1] else last
    bot.set_xticklabels(shown, fontsize=6.5)
    for ax in (top, bot):
        for i in starts[1:]:
            ax.axvline(i - 0.5, color="0.8", lw=0.5)
    summarise(key, rmean, rstd, [(label, p[key]) for label, p in sc_pops], sc_real[key][0])
save(fig, "sc_populations")
open(f"{out}/populations_n{N}.txt", "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
