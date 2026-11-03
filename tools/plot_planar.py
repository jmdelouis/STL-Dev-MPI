"""Figures and tables of the planar runs (docs/planar_2d.md).

    python tools/plot_planar.py --run PREFIX --fig DIR     # a stl-synth2d synthesis
    python tools/plot_planar.py --bench DIR                # benchmark table (markdown)

--run reads PREFIX.json, PREFIX_maps.npy, PREFIX_target.npy and the
feature files, and writes DIR/<name>_maps.png (targets and syntheses on
one colour scale), DIR/<name>_stats.png (per family: target mean and
scatter, synthesis mean; pulls (synthesis - target mean) / target scatter;
pixel PDF; Minkowski functionals; loss history) and prints a summary.
"""

import argparse
import glob
import json
import os

import numpy as np


def run_figures(prefix, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = os.path.basename(prefix)
    rep = json.load(open(prefix + ".json"))
    maps = np.load(prefix + "_maps.npy")[:, 0]
    target = np.load(prefix + "_target.npy")[:, 0]
    feats = {k: np.load(f"{prefix}_features_{k}.npy") for k in ("target", "init", "final")}
    feats = {k: v[..., 0] + 1j * v[..., 1] for k, v in feats.items()}
    fam = np.array(rep["families"])
    mf = {k: np.load(f"{prefix}_mf_{k}.npy") for k in ("target", "init", "final")} if os.path.exists(prefix + "_mf_target.npy") else None

    # Images.
    k = min(5, len(target), len(maps))
    lo, hi = np.percentile(target, [0.5, 99.5])
    fig, ax = plt.subplots(2, k, figsize=(2.6 * k, 5.6))
    for i in range(k):
        ax[0, i].imshow(target[i], cmap="plasma", vmin=lo, vmax=hi)
        ax[0, i].set_title(f"target {i}")
        ax[1, i].imshow(maps[i], cmap="plasma", vmin=lo, vmax=hi)
        ax[1, i].set_title(f"synthesis {i}")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"{name}: {rep['nbatch']} images {rep['shape'][1]}x{rep['shape'][2]}, mode {rep['mode']}, {rep['evaluations']} evaluations")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + "_maps.png"), dpi=110)
    plt.close(fig)

    # Statistics.
    families = [f for f in ("S1", "S2", "S3", "S4", "PS") if (fam == f).any()]
    fig, ax = plt.subplots(3, len(families) + 1, figsize=(3.6 * (len(families) + 1), 9.5))
    summary = {}
    for c, f in enumerate(families):
        sel = fam == f
        t = feats["target"][:, sel]
        tm, ts = t.mean(0), t.std(0, ddof=1)
        fm, im = feats["final"][:, sel].mean(0), feats["init"][:, sel].mean(0)
        part = np.real if f in ("S1", "S2", "PS") else np.abs
        x = np.arange(sel.sum())
        a = ax[0, c]
        a.fill_between(x, part(tm) - np.abs(ts), part(tm) + np.abs(ts), color="0.8", label="target ±1σ")
        a.plot(x, part(tm), "k-", lw=0.8, label="target mean")
        a.plot(x, part(im), ".", ms=2, color="tab:blue", alpha=0.5, label="initial")
        a.plot(x, part(fm), ".", ms=2, color="tab:red", label="synthesis")
        a.set_yscale("log" if (part(tm) > 0).all() else "linear")
        a.set_title(f"{f} ({'Re' if part is np.real else '|.|'})")
        if c == 0:
            a.legend(fontsize=7)
        pull = np.abs(fm - tm) / np.where(ts > 0, ts, np.inf)
        pull0 = np.abs(im - tm) / np.where(ts > 0, ts, np.inf)
        b = ax[1, c]
        bins = np.logspace(-3, 3, 50)
        b.hist(np.clip(pull0, 1e-3, 1e3), bins=bins, color="tab:blue", alpha=0.5, label="initial")
        b.hist(np.clip(pull, 1e-3, 1e3), bins=bins, color="tab:red", alpha=0.7, label="synthesis")
        b.axvline(1, color="k", lw=0.8)
        b.set_xscale("log")
        b.set_title(f"{f}: |mean - target| / σ_target")
        summary[f] = dict(count=int(sel.sum()), median_pull=float(np.median(pull)), frac_within_1sigma=float((pull < 1).mean()), median_pull_init=float(np.median(pull0)))
        vr = np.abs(feats["final"][:, sel].std(0, ddof=1)) / np.where(ts > 0, ts, np.inf)
        summary[f]["median_std_ratio"] = float(np.median(vr))
        ax[2, c].hist(np.clip(vr, 1e-2, 1e2), bins=np.logspace(-2, 2, 40), color="tab:green")
        ax[2, c].axvline(1, color="k", lw=0.8)
        ax[2, c].set_xscale("log")
        ax[2, c].set_title(f"{f}: σ_synthesis / σ_target")
    a = ax[0, -1]
    bins = np.linspace(min(target.min(), maps.min()), max(target.max(), maps.max()), 80)
    a.hist(target.ravel(), bins=bins, density=True, histtype="step", color="k", label="targets")
    a.hist(maps.ravel(), bins=bins, density=True, histtype="step", color="tab:red", label="syntheses")
    a.set_yscale("log")
    a.legend(fontsize=7)
    a.set_title("pixel PDF")
    a = ax[1, -1]
    if mf is not None:
        nu = np.linspace(-4, 4, 65)
        for v, col in zip(range(3), ("k", "tab:red", "tab:blue")):
            tt = mf["target"][:, v * 65 : (v + 1) * 65]
            ff = mf["final"][:, v * 65 : (v + 1) * 65]
            sc = np.abs(tt.mean(0)).max()
            a.fill_between(nu, (tt.mean(0) - tt.std(0, ddof=1)) / sc, (tt.mean(0) + tt.std(0, ddof=1)) / sc, color=col, alpha=0.2)
            a.plot(nu, ff.mean(0) / sc, color=col, lw=1, label=f"V{v} synthesis")
        a.legend(fontsize=7)
        a.set_title("Minkowski (band: targets ±1σ)")
        a.set_xlabel("ν")
    else:
        a.axis("off")
    a = ax[2, -1]
    a.semilogy(rep["loss_history"])
    a.set_title("loss")
    a.set_xlabel("evaluation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + "_stats.png"), dpi=110)
    plt.close(fig)
    out = dict(name=name, evaluations=rep["evaluations"], seconds_per_evaluation=rep["seconds_per_evaluation"], loss_first=rep["loss_first"], loss_last=rep["loss_last"], families=summary)
    print(json.dumps(out, indent=1))


def bench_table(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        name = os.path.basename(p)[:-5]
        if "failed" in r:
            rows.append(f"| {name} | failed: {r['failed'][:80]} | | |")
            continue
        sec = r.get("bench_seconds", r.get("seconds"))
        mem = r.get("peak_device_bytes", r.get("peak_rss_bytes"))
        rows.append(f"| {name} | {np.median(sec[1:] if len(sec) > 1 else sec):.3f} | {r.get('peak_rss_bytes', 0) / 2**30:.2f} | {mem / 2**30:.2f} |")
    print("| configuration | s / evaluation | peak RSS (GiB) | peak (RSS or device, GiB) |")
    print("|---|---|---|---|")
    print("\n".join(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--fig", default=".")
    p.add_argument("--bench")
    a = p.parse_args()
    if a.run:
        run_figures(a.run, a.fig)
    if a.bench:
        bench_table(a.bench)


if __name__ == "__main__":
    main()
