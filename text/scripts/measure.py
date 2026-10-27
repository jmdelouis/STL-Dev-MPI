"""Measure every statistic of the paper on the realizations and the ensemble
members at one nside, one cache file per map (see common.py for the
definitions).

    python measure.py CACHE NSIDE REAL_DIR [LABEL=RUN_DIR ...] [--procs P] [--threads T] [--force]

REAL_DIR holds rXXX_nside{NSIDE}_nested.npy (r000 = the synthesis target,
r001-r009 the other realizations).  Every RUN_DIR holds the kappa members
synthesis_mMMM_nested.npy.  Writes CACHE/n{NSIDE}/real/rXXX.npz and
CACHE/n{NSIDE}/<slug of LABEL>/mMMM.npz; existing caches newer than their map
are kept unless --force.  Heights and PDFs are in units of the std of r000
at the same nside (degraded by the same factor).
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import common as C


def measure_one(task):
    src, dst, nside, scales = task
    import healpy as hp

    from minkowski import minkowski_map

    t0 = time.time()
    m = np.load(src).astype(np.float64)
    assert m.size == 12 * nside**2, f"{src}: {m.size} pixels, expected nside {nside}"
    out = {"mean": m.mean(), "std": m.std()}
    dk, dg = C.details(m), C.details(C.gaussianise(m))
    levels = sorted(dk)
    out["levels"] = np.array(levels)
    out["vk"] = np.array([dk[n][0] for n in levels])
    out["kk"] = np.array([dk[n][1] for n in levels])
    out["vg"] = np.array([dg[n][0] for n in levels])
    out["cl"] = hp.anafast(hp.reorder(m, n2r=True), lmax=3 * nside - 1)
    for factor in (1, 4):
        pk = C.peak_heights(C.degrade(m, factor)) / scales[factor]
        out[f"peaks_hist_{factor}"] = np.histogram(pk, C.PEAK_EDGES)[0]
        out[f"peaks_count_{factor}"] = np.array([(pk > t).sum() for t in C.PEAK_THRESHOLDS])
    for factor in C.PDF_FACTORS:
        out[f"pdf_{factor}"] = np.histogram(C.degrade(m, factor) / scales[factor], C.PDF_EDGES, density=True)[0]
    for i, fwhm in enumerate(C.mf_fwhms(nside)):
        r = minkowski_map(m, nest=True, fwhm_arcmin=fwhm, nu=C.NU)
        for k in ("v0", "v1", "v2", "g0", "g1", "g2"):
            out[f"mf{i}_{k}"] = r[k]
        out[f"mf{i}_tau"], out[f"mf{i}_fwhm"] = r["tau"], fwhm
    tmp = dst + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return f"{src} -> {dst} ({time.time() - t0:.0f} s)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("nside", type=int)
    p.add_argument("real_dir")
    p.add_argument("runs", nargs="*")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    N = a.nside
    reals = C.realization_files(a.real_dir, N)
    r000 = np.load(reals[0]).astype(np.float64)
    scales = {f: C.degrade(r000, f).std() for f in sorted({1, 4, *C.PDF_FACTORS})}
    del r000
    sets = [("real", [(f, os.path.basename(f)[:4]) for f in reals if os.path.exists(f)])]
    for label, slug, run in C.parse_runs(a.runs):
        files = C.member_files(run)
        if not files:
            raise SystemExit(f"{label}: no synthesis_mMMM_nested.npy in {run}")
        sets.append((slug, [(f, os.path.basename(f).split("_")[1]) for f in files]))
    tasks = []
    for slug, files in sets:
        d = f"{a.cache}/n{N}/{slug}"
        os.makedirs(d, exist_ok=True)
        for src, name in files:
            dst = f"{d}/{name}.npz"
            if a.force or not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                tasks.append((src, dst, N, scales))
        print(f"{slug}: {len(files)} maps")
    print(f"{len(tasks)} maps to measure on {a.procs} processes x {a.threads} threads")
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for line in pool.imap_unordered(measure_one, tasks):
            print(line, flush=True)


if __name__ == "__main__":
    main()
