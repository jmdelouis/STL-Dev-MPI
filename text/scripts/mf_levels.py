"""Minkowski functionals of kappa per pyramid level: the maps degraded by
4-children averaging to the nside of each level, without further smoothing,
against the 10 realizations.

    python mf_levels.py CACHE NSIDE OUT_DIR REAL_DIR [LABEL=RUN_DIR ...] [--levels 64,128,256,512] [--procs P] [--threads T]

Per map and level: V0, V1, V2 at the thresholds nu = -2 ... 5 (steps of
0.2, in units of the degraded map's std), cached in
CACHE/n{NSIDE}/{set}/{map}_mfl.npz.  Per run, level and functional:
z(nu) = (V - V(r000)) / sigma(nu), sigma the std of r000-r009; the
summary gives the median and max of |z| over the members and thresholds,
and the rms over the thresholds of the members' median z (the systematic
part).  The same numbers for r001-r009 against r000 are the reference: an
ensemble within cosmic variance looks like them.  Thresholds where the
realizations' std is below 5 % of its maximum are left out.  At the map's
own nside the members have r000's histogram, so V0 there tests only the
harmonic reconstruction.  Writes
OUT_DIR/mf_levels_n{NSIDE}.txt, .tex and .pdf.
"""

import argparse
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

import common as C

NU = np.round(np.arange(-2.0, 5.0001, 0.2), 10)


def measure(task):
    src, dst, levels = task
    from minkowski import minkowski_map

    t0 = time.time()
    m = np.load(src).astype(np.float64)
    nside = int(round((m.size / 12) ** 0.5))
    out = {"levels": np.array(levels)}
    for n in levels:
        r = minkowski_map(C.degrade(m, (nside // n) ** 2), nest=True, nu=NU)
        for k in range(3):
            out[f"l{n}_v{k}"] = r[f"v{k}"]
        out[f"l{n}_tau"] = r["tau"]
    tmp = dst + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return f"{src} ({time.time() - t0:.0f} s)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("nside", type=int)
    p.add_argument("out")
    p.add_argument("real_dir")
    p.add_argument("runs", nargs="*")
    p.add_argument("--levels", default=None)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    N = a.nside
    levels = [int(x) for x in a.levels.split(",")] if a.levels else [n for n in (64, 128, 256, 512, 1024, 2048) if n <= N]
    sets = [("real", "real", [(f, os.path.basename(f)[:4]) for f in C.realization_files(a.real_dir, N)])]
    for label, slug, run in C.parse_runs(a.runs):
        sets.append((label, slug, [(f, os.path.basename(f).split("_")[1]) for f in C.member_files(run)]))
    tasks = []
    for _, slug, files in sets:
        d = f"{a.cache}/n{N}/{slug}"
        os.makedirs(d, exist_ok=True)
        for src, name in files:
            dst = f"{d}/{name}_mfl.npz"
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                tasks.append((src, dst, levels))
    print(f"{len(tasks)} maps to measure, levels {levels}", flush=True)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for line in pool.imap_unordered(measure, tasks):
            print(line, flush=True)

    load = lambda slug, files: [dict(np.load(f"{a.cache}/n{N}/{slug}/{name}_mfl.npz")) for _, name in files]
    real = load("real", sets[0][2])
    runs = [(label, load(slug, files)) for label, slug, files in sets[1:]]
    os.makedirs(a.out, exist_ok=True)
    warnings.simplefilter("ignore", RuntimeWarning)

    def z_of(maps, key):
        # Thresholds where the realizations barely differ (beyond the map's
        # range, where V is 0 or 1 for every map) are left out: sigma below
        # 5 % of its maximum over the thresholds.
        R = np.array([r[key] for r in real])
        s = R.std(0, ddof=1)
        s = np.where(s > 0.05 * s.max(), s, np.nan)
        return (np.array([m[key] for m in maps]) - R[0]) / s

    lines = [f"nside {N}: Minkowski functionals of kappa degraded to each level (4-children means, no smoothing), "
             f"nu = {NU[0]:g} ... {NU[-1]:g}; deviation from r000 in units of the std of r000-r009",
             "columns: median |z| and max |z| over members x thresholds; rms over nu of the members' median z",
             "reference: r001-r009 against r000 (what an ensemble within cosmic variance gives)"]
    tex_rows = []
    groups = [("r001--r009", real[1:])] + runs
    for n in levels:
        for k in range(3):
            key = f"l{n}_v{k}"
            cells, row = [], f"  level {n:5d} V{k}"
            for label, maps in groups:
                z = z_of(maps, key)
                med = np.nanmedian(np.abs(z))
                mx = np.nanmax(np.abs(z))
                sysrms = np.sqrt(np.nanmean(np.nanmedian(z, 0) ** 2))
                row += f" | {label}: {med:4.2f} {mx:5.2f} {sysrms:4.2f}"
                cells.append(f"{med:.1f} / {mx:.1f} / {sysrms:.1f}")
            lines.append(row)
            tex_rows.append(f"{n} & $V_{k}$ & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{table*}",
           rf"\caption{{Minkowski functionals of $\kappa$ degraded to each level (nside $n$, means of 4 children, no smoothing), "
           rf"thresholds ${NU[0]:g}\leq\nu\leq{NU[-1]:g}$: deviation from r000 in units of the standard deviation of the 10 realizations. "
           r"Each cell: median $|z|$ and maximum $|z|$ over the members and the thresholds, and rms over the thresholds of the members' median $z$. "
           r"The first column applies the same measure to the realizations r001--r009, the reference for an ensemble within cosmic variance.}",
           rf"\label{{tab:mf_levels_n{N}}}", r"\centering\small",
           r"\begin{tabular}{ll" + "c" * len(groups) + "}", r"\hline\hline",
           " & ".join(["Level", ""] + [g for g, _ in groups]) + r" \\", r"\hline", *tex_rows, r"\hline", r"\end{tabular}", r"\end{table*}"]
    open(f"{a.out}/mf_levels_n{N}.txt", "w").write("\n".join(lines) + "\n")
    open(f"{a.out}/mf_levels_n{N}.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(lines))

    plt = C.style()
    fig, axes = plt.subplots(len(levels), 3, figsize=(7.2, 1.6 * len(levels) + 0.4), sharex=True, squeeze=False)
    for i, n in enumerate(levels):
        for k in range(3):
            ax = axes[i, k]
            key = f"l{n}_v{k}"
            ax.axhspan(-1, 1, color=C.BAND, lw=0)
            for r in z_of(real[1:], key):
                ax.plot(NU, r, color="0.65", lw=0.5)
            for (label, maps), col in zip(runs, C.RUN_COLORS):
                z = z_of(maps, key)
                ax.fill_between(NU, np.nanmin(z, 0), np.nanmax(z, 0), color=col, alpha=0.25, lw=0)
                ax.plot(NU, np.nanmedian(z, 0), color=col, lw=1.1, label=label if i == 0 and k == 0 else None)
            ax.axhline(0, color=C.INK, ls=":", lw=0.7)
            ax.set_ylim(-6, 6)
            if i == 0:
                ax.set_title(f"$V_{k}$")
            if k == 0:
                ax.set_ylabel(f"level {n}\n" + r"$\Delta V/\sigma$")
    for ax in axes[-1]:
        ax.set_xlabel(r"threshold $\nu$")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{a.out}/mf_levels_n{N}.pdf")


if __name__ == "__main__":
    main()
