"""C_l and Minkowski functionals per pyramid level of the realizations and
of the ensemble members, one cache file per map, for plot_populations.py.

    python measure_populations.py CACHE REAL_DIR [LABEL=RUN_DIR ...] [--levels 64,128,256,512,1024] [--procs P] [--threads T]

REAL_DIR holds rXXX_nside{N}_nested.npy (r000-r009); every RUN_DIR holds
synthesis_mMMM_nested.npy.  Per map: the C_l of healpy's anafast (lmax =
3 nside - 1) and V0, V1, V2 of the map degraded to each level by 4-children
means, without smoothing, at the thresholds nu = -2 ... 5 by 0.2 in units of
the degraded map's std (as text/scripts/mf_levels.py).  Writes
CACHE/{slug}/{name}.npz; existing files newer than their map are kept.
Needs text/scripts/common.py and tools/minkowski.py on the path.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../../../text/scripts", "../../../tools"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, d)))
import common as C  # noqa: E402

NU = np.round(np.arange(-2.0, 5.0001, 0.2), 10)


def measure(task):
    src, dst, levels = task
    import healpy as hp
    from minkowski import minkowski_map

    t0 = time.time()
    m = np.load(src).astype(np.float64)
    nside = int(round((m.size / 12) ** 0.5))
    out = {"levels": np.array(levels), "nu": NU, "cl": hp.anafast(hp.reorder(m, n2r=True), lmax=3 * nside - 1)}
    for n in levels:
        r = minkowski_map(C.degrade(m, (nside // n) ** 2), nest=True, nu=NU)
        for k in range(3):
            out[f"l{n}_v{k}"] = r[f"v{k}"]
    tmp = dst + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return f"{src} ({time.time() - t0:.0f} s)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("real_dir")
    p.add_argument("runs", nargs="*")
    p.add_argument("--levels", default="64,128,256,512,1024")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    levels = [int(x) for x in a.levels.split(",")]
    reals = sorted(glob.glob(f"{a.real_dir}/r00[0-9]_nside*_nested.npy"))
    sets = [("real", [(f, os.path.basename(f)[:4]) for f in reals])]
    for _, slug, run in C.parse_runs(a.runs):
        sets.append((slug, [(f, os.path.basename(f).split("_")[1]) for f in C.member_files(run)]))
    tasks = []
    for slug, files in sets:
        os.makedirs(f"{a.cache}/{slug}", exist_ok=True)
        for src, name in files:
            dst = f"{a.cache}/{slug}/{name}.npz"
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                tasks.append((src, dst, levels))
        print(f"{slug}: {len(files)} maps", flush=True)
    print(f"{len(tasks)} maps to measure, levels {levels}", flush=True)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for line in pool.imap_unordered(measure, tasks):
            print(line, flush=True)


if __name__ == "__main__":
    main()
