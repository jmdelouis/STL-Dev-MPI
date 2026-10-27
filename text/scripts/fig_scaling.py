"""Strong scaling of one evaluation (statistics + gradient) on CPUs and GPUs,
from the reports kept in the repository.

    python fig_scaling.py OUT_DIR

(a) nside 64, L = 2: Datarmor (job 7696400, Intel MPI) and the hpc cluster
(job 1523174, OpenMPI 5), 1 to 128 ranks.  (b) nside 1024, L = 2: hpc (job
1523296, one node, 24 to 192 ranks) and Irene Rome (docs/USAGE.md section 7).
(c) one T4 to three T4s (garoupe, benchmarks/garoupe/gpu_scaling_2026-10-20),
L = 4.  Dashed: ideal 1 / P from the first point.  Writes OUT_DIR/scaling.pdf
and scaling.txt.
"""

import glob
import json
import os
import sys

import numpy as np
from matplotlib import ticker

import common as C

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.normpath(os.path.join(C.HERE, "..", ".."))
plt = C.style()


def reports(pattern, key="ranks"):
    pts = []
    for f in glob.glob(os.path.join(ROOT, pattern)):
        j = json.load(open(f))
        if j.get("block_pixels") == 16384 and not j.get("conv_chunk"):
            pts.append((j[key], j["seconds_per_evaluation"]))
    return sorted(pts)


series = {
    "a": [("Datarmor (Xeon E5-2680)", reports("benchmarks/datarmor/job7696400_kap_nside64_reduced/p*.json")),
          ("hpc (EPYC 9655)", reports("benchmarks/hpc/job1523174_kap_nside64/p*.json"))],
    "b": [("hpc, 1 node (EPYC 9655)", reports("benchmarks/hpc/job1523296_kap_nside1024/p*.json")),
          # Irene Rome (EPYC 7H12), OpenMPI 4.1: docs/USAGE.md section 7.
          ("Irene (EPYC 7H12)", [(128, 30.2), (512, 8.9)])],
    "c": [],
}
for n in (256, 1024, 2048):
    pts = []
    for f in glob.glob(os.path.join(ROOT, f"benchmarks/garoupe/gpu_scaling_2026-10-20/bench_n{n}_L4_P*.json")):
        j = json.load(open(f))
        pts.append((j["ranks"], j["seconds_per_evaluation"]))
    series["c"].append((f"nside {n}", sorted(pts)))

titles = {"a": "(a) nside 64, $L=2$, CPU", "b": "(b) nside 1024, $L=2$, CPU", "c": "(c) $L=4$, Tesla T4 GPUs"}
xlabels = {"a": "MPI ranks", "b": "MPI ranks", "c": "GPUs (one per rank)"}
lines = []
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, key in zip(axes, "abc"):
    for (label, pts), col in zip(series[key], C.RUN_COLORS):
        if not pts:
            continue
        p, t = np.array(pts, dtype=float).T
        ax.loglog(p, t, "o-", ms=3, color=col, lw=1.2, label=label)
        ax.loglog(p, t[0] * p[0] / p, ls="--", color=col, lw=0.7)
        lines.append(f"{titles[key]} {label}: " + ", ".join(f"{int(a)}: {b:g} s" for a, b in pts)
                     + f"; efficiency at {int(p[-1])}: {t[0] * p[0] / (t[-1] * p[-1]):.2f}")
    ax.set(title=titles[key], xlabel=xlabels[key])
    ax.legend(frameon=False, fontsize=6.5, loc={"a": "lower left", "b": "upper right", "c": "center"}[key])
    ranks = []
    for r in sorted({int(r) for _, pts in series[key] for r, _ in pts}):
        if not ranks or r >= 1.45 * ranks[-1]:
            ranks.append(r)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ranks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(r) for r in ranks]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
axes[0].set_ylabel("s per evaluation")
fig.tight_layout()
fig.savefig(f"{OUT}/scaling.pdf")
open(f"{OUT}/scaling.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
