"""Diagnostics of generated NESTED maps against a target, as in the STL
synthesis work.

    python diffusion/evaluate.py [--nside N] OUT_PREFIX TARGET.npy LABEL=FILE[,FILE...] [LABEL=...]

Every FILE is a NESTED map `[npix]` or a stack of maps `[S, npix]`; the maps
of one LABEL form an ensemble.  With --nside, the target and all maps are
first averaged down to N.  Writes OUT_PREFIX.json and OUT_PREFIX.png.

Per pyramid level (fine nside n, from the map's nside down to 8):
  - detail variance mean((x_n - up(x_{n+1}))^2) over the target's;
  - skewness and excess kurtosis of x_n and of the details d_n;
  - independence: Pearson correlation of d_n between the members of the
    ensemble (all pairs) and between every member and the target.
Angular power: healpy anafast (RING) of the finest map, averaged in
logarithmic l bands (two per octave), over the target's.
Histogram of the finest map (log counts).
"""

import json
import sys

import healpy as hp
import numpy as np

import healpix_faces as hf


def load_set(spec):
    maps = []
    for path in spec.split(","):
        a = np.load(path).astype(np.float64)
        maps.extend(a if a.ndim == 2 else [a])
    return np.array(maps)


def moments(v):
    v = v - v.mean()
    s2 = np.mean(v**2)
    return float(np.mean(v**3) / s2**1.5), float(np.mean(v**4) / s2**2 - 3)


def level_stats(m, nside_min=8):
    """Per level: nside, detail variance, map/detail skewness and kurtosis, details."""
    pyr = hf.pyramid(m, nside_min)
    out = []
    for n in range(len(pyr) - 1):
        d = pyr[n] - hf.up(pyr[n + 1])
        sk, ku = moments(pyr[n])
        dsk, dku = moments(d)
        out.append(dict(nside=hf.nside_of(pyr[n].size), detail_var=float(np.mean(d**2)),
                        map_skew=sk, map_kurt=ku, detail_skew=dsk, detail_kurt=dku, d=d))
    return out


def band_edges(lmax):
    e = np.unique(np.round(2 ** np.arange(1, np.log2(lmax + 1) + 0.01, 0.5)).astype(int))
    return e[e <= lmax + 1]


def band_power(m, edges):
    nside = hf.nside_of(m.size)
    cl = hp.anafast(hp.reorder(m, n2r=True), lmax=3 * nside - 1)
    ell = np.arange(cl.size)
    return np.array([np.mean(cl[(ell >= a) & (ell < b)]) for a, b in zip(edges[:-1], edges[1:])])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def evaluate(target, sets):
    nside = hf.nside_of(target.size)
    edges = band_edges(3 * nside - 1)
    t_levels = level_stats(target)
    t_bands = band_power(target, edges)
    report = dict(nside=nside, l_edges=edges.tolist(), target=dict(
        levels=[{k: v for k, v in s.items() if k != "d"} for s in t_levels], band_power=t_bands.tolist(),
        std=float(target.std())), sets={})
    hist_edges = np.linspace(-6, 6, 121)
    report["hist_edges"] = hist_edges.tolist()
    report["target"]["hist"] = np.histogram(target, hist_edges)[0].tolist()
    for label, maps in sets.items():
        per = [level_stats(m) for m in maps]
        bands = np.array([band_power(m, edges) for m in maps])
        r = dict(members=len(maps), levels=[])
        for n, t in enumerate(t_levels):
            ratio = np.array([p[n]["detail_var"] / t["detail_var"] for p in per])
            entry = dict(nside=t["nside"], detail_var_ratio_mean=float(ratio.mean()),
                         detail_var_ratio_std=float(ratio.std()), detail_var_ratio=ratio.tolist())
            for key in ("map_skew", "map_kurt", "detail_skew", "detail_kurt"):
                v = np.array([p[n][key] for p in per])
                entry[key + "_mean"], entry[key + "_std"] = float(v.mean()), float(v.std())
            entry["corr_with_target"] = [corr(p[n]["d"], t["d"]) for p in per]
            entry["corr_between_members"] = [corr(per[i][n]["d"], per[j][n]["d"])
                                             for i in range(len(per)) for j in range(i + 1, len(per))]
            r["levels"].append(entry)
        ratio = bands / t_bands
        r["band_power_ratio_mean"] = ratio.mean(axis=0).tolist()
        r["band_power_ratio_std"] = ratio.std(axis=0).tolist()
        r["hist"] = np.mean([np.histogram(m, hist_edges)[0] for m in maps], axis=0).tolist()
        r["std"] = [float(m.std()) for m in maps]
        report["sets"][label] = r
    return report


# Reference categorical order (dataviz skill palette, light mode); target in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    labels = list(report["sets"])
    colour = {lab: SERIES[i % len(SERIES)] for i, lab in enumerate(labels)}
    nsides = [lev["nside"] for lev in report["target"]["levels"]]
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = axes[0, 0]
    ax.axhline(1, color=MUTED, lw=1)
    for lab in labels:
        lv = report["sets"][lab]["levels"]
        ax.errorbar(nsides, [e["detail_var_ratio_mean"] for e in lv], [e["detail_var_ratio_std"] for e in lv],
                    color=colour[lab], lw=2, marker="o", ms=5, capsize=3, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fine nside of the level")
    ax.set_ylabel("detail variance / target")
    ax.set_title("Pyramid detail variance per level")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    e = np.array(report["l_edges"])
    lc = np.sqrt(e[:-1] * np.maximum(e[1:] - 1, e[:-1]))
    cv = cosmic_scatter(e)
    ax.fill_between(lc, 1 - cv, 1 + cv, color=GRID, label="one sky / another (1 sigma)")
    ax.axhline(1, color=MUTED, lw=1)
    for lab in labels:
        s = report["sets"][lab]
        ax.errorbar(lc, s["band_power_ratio_mean"], s["band_power_ratio_std"], color=colour[lab], lw=2,
                    marker="o", ms=4, capsize=2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("multipole l (band centre)")
    ax.set_ylabel("C_l / target")
    ax.set_title("Angular power in log bands (anafast)")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    h = np.array(report["hist_edges"])
    hc = 0.5 * (h[1:] + h[:-1])
    ax.step(hc, np.maximum(report["target"]["hist"], 0.5), color=INK, lw=2, where="mid", label="target")
    for lab in labels:
        ax.step(hc, np.maximum(report["sets"][lab]["hist"], 0.5), color=colour[lab], lw=1.5, where="mid", label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value (finest level)")
    ax.set_ylabel("count")
    ax.set_title("Pixel histogram")
    ax.legend(frameon=False)

    for ax, key, title in ((axes[1, 0], "detail_skew", "Skewness of the details"),
                           (axes[1, 1], "detail_kurt", "Excess kurtosis of the details"),
                           (axes[1, 2], "map_kurt", "Excess kurtosis of the map x_n")):
        ax.plot(nsides, [lev[key] for lev in report["target"]["levels"]], color=INK, lw=2, marker="s", ms=5,
                label="target")
        for lab in labels:
            lv = report["sets"][lab]["levels"]
            ax.errorbar(nsides, [x[key + "_mean"] for x in lv], [x[key + "_std"] for x in lv], color=colour[lab],
                        lw=1.5, marker="o", ms=4, capsize=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("fine nside of the level")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.savefig(path, dpi=110)


def cosmic_scatter(edges):
    """Relative scatter of the band power of one Gaussian sky over another's:
    2 / sqrt(number of modes in the band)."""
    modes = np.array([np.sum(2 * np.arange(a, b) + 1) for a, b in zip(edges[:-1], edges[1:])])
    return 2 / np.sqrt(modes)


def summary(report):
    e = np.array(report["l_edges"])
    lines = []
    for lab, s in report["sets"].items():
        dv = " ".join(f"{x['nside']}:{x['detail_var_ratio_mean']:.3f}" for x in s["levels"])
        bp = np.array(s["band_power_ratio_mean"])
        high = e[:-1] >= 16
        bands = " ".join(f"{a}:{r:.3f}" for a, r in zip(e[:-1], bp))
        ct = " ".join(f"{x['nside']}:{np.mean(np.abs(x['corr_with_target'])):.3f}" for x in s["levels"])
        cm = " ".join(f"{x['nside']}:{np.mean(np.abs(x['corr_between_members'])):.3f}"
                      for x in s["levels"] if x["corr_between_members"])
        sk = " ".join(f"{x['nside']}:{x['detail_skew_mean']:.3f}/{x['detail_kurt_mean']:.3f}" for x in s["levels"])
        lines.append(f"{lab} ({s['members']} maps)\n  detail var / target: {dv}\n"
                     f"  C_l bands / target (band start:ratio): {bands}\n"
                     f"  max |C_l ratio - 1| for l >= 16: {np.max(np.abs(bp[high] - 1)):.3f}\n"
                     f"  detail skew/kurt: {sk}\n"
                     f"  |corr| with target: {ct}\n  |corr| between members: {cm or '-'}")
    t = " ".join(f"{x['nside']}:{x['detail_skew']:.3f}/{x['detail_kurt']:.3f}" for x in report["target"]["levels"])
    lines.append(f"target detail skew/kurt: {t}")
    return "\n".join(lines)


def to_nside(m, nside):
    while hf.nside_of(m.shape[-1]) > nside:
        m = hf.down(m)
    return m


if __name__ == "__main__":
    args = sys.argv[1:]
    nside = None
    if args[0] == "--nside":
        nside, args = int(args[1]), args[2:]
    out, target_path, *specs = args
    target = np.load(target_path).astype(np.float64)
    nside = nside or hf.nside_of(target.size)
    target = to_nside(target, nside)
    sets = {}
    for spec in specs:
        label, files = spec.split("=", 1)
        sets[label] = to_nside(load_set(files), nside)
    report = evaluate(target, sets)
    with open(out + ".json", "w") as f:
        json.dump(report, f, indent=1)
    plot(report, out + ".png")
    print(summary(report))
