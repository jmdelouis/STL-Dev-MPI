"""Small figures for the examples (numpy, matplotlib, healpix_geo).

    python examples/plotting.py maps OUT.png [--norm hist|linear] [--range P1,P2] TITLE=MAP.npy [TITLE=MAP.npy ...]
    python examples/plotting.py loss OUT.png LABEL=REPORT.json [LABEL=REPORT.json ...]
    python examples/plotting.py stats OUT.png STATS.json
    python examples/plotting.py summary REPORT.json [REPORT.json ...]

maps     Mollweide views of NESTED maps and one base face of each (face 4,
         on the equator), on one colour scale for all maps.  --norm hist
         (default, healpy's mollview norm='hist'): histogram equalisation by
         the first map's pixel distribution, which balances the colours of
         skewed fields such as the lensing convergence; --norm linear: the
         [P1, P2] percentiles of the first map (default 0.5, 99.5).
loss     loss against evaluation for stl-synth / stl-synth2d / stl-ensemble
         reports (field `loss_history`).
stats    stl-stats output: S1 and S2 per scale (mean over orientations) and
         S3, S4 of every map against those of the first map.
summary  one line per report: evaluations, seconds per evaluation, first
         and last loss, peak memory.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mollweide_index(nside, width=600):
    """NESTED pixel of every point of a Mollweide grid (-1 outside the ellipse)."""
    from healpix_geo.nested import lonlat_to_healpix

    height = width // 2
    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), width)
    y = np.linspace(np.sqrt(2), -np.sqrt(2), height)
    X, Y = np.meshgrid(x, y)
    inside = (X / (2 * np.sqrt(2))) ** 2 + (Y / np.sqrt(2)) ** 2 <= 1
    theta = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
    lat = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
    lon = -np.pi * X / (2 * np.sqrt(2) * np.cos(theta).clip(1e-12))
    idx = np.full(X.shape, -1, dtype=np.int64)
    idx[inside] = np.asarray(lonlat_to_healpix(np.degrees(lon[inside]) % 360, np.degrees(lat[inside]), int(np.log2(nside))), dtype=np.int64)
    return idx


def face_image(m, face):
    """One HEALPix base face of a NESTED map as an nside x nside image (bits of x and y interleaved)."""
    nside = int(round((m.size / 12) ** 0.5))
    x, y = np.meshgrid(np.arange(nside), np.arange(nside))
    ids = np.zeros_like(x)
    for b in range(int(np.log2(nside))):
        ids |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1)
    return m[face * nside * nside + ids][::-1]


def hist_norm(reference):
    """Histogram equalisation by the pixel distribution of `reference` (healpy's
    norm='hist'): a value maps to its rank among the reference pixels, in [0, 1].
    The same mapping is applied to every map, so their colours stay comparable."""
    ref = np.sort(reference)
    return lambda v: np.searchsorted(ref, v, side="left") / (ref.size - 1.0)


def split_specs(args):
    return [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args]


def plot_maps(out, specs, prange=(0.5, 99.5), face=4, norm="hist"):
    """Mollweide views (top row) and base face `face` of every map (bottom row),
    on the colour scale of the first map."""
    maps = [(t, np.load(p).astype(np.float64).reshape(-1)) for t, p in specs]
    nside = int(round((maps[0][1].size / 12) ** 0.5))
    if norm == "hist":
        f = hist_norm(maps[0][1])
        maps = [(t, f(m)) for t, m in maps]
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(maps[0][1], prange)
    idx = mollweide_index(nside)
    n = len(maps)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), squeeze=False, gridspec_kw={"height_ratios": [1, 1.6]})
    for k, (title, m) in enumerate(maps):
        img = np.where(idx >= 0, m[np.maximum(idx, 0)], np.nan)
        axes[0, k].imshow(img, cmap="plasma", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0, k].set_title(title, fontsize=10)
        axes[1, k].imshow(face_image(m, face), cmap="plasma", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[1, k].set_title(f"base face {face} ({nside} x {nside} pixels)", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("figure", out)


def plot_loss(out, specs):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for label, path in specs:
        rep = json.load(open(path))
        ax.semilogy(np.arange(1, len(rep["loss_history"]) + 1), rep["loss_history"], lw=1, label=label)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("figure", out)


def plot_stats(out, path):
    rep = json.load(open(path))
    plan = rep["plan"]
    fam = np.array([c["family"] for c in plan])
    j1 = np.array([c["j1"] for c in plan])
    maps = rep["maps"]
    names = [m["file"].rsplit("/", 1)[-1].replace("_nested.npy", "").replace(".npy", "") for m in maps]
    vals = [np.array(m["re"]) + 1j * np.array(m["im"]) for m in maps]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    for k, f in enumerate(("S1", "S2")):
        sel = fam == f
        js = np.unique(j1[sel])
        for name, v in zip(names, vals):
            ax[k].semilogy(js, [np.real(v[sel & (j1 == j)]).mean() for j in js], "o-", ms=3, lw=1, label=name)
        ax[k].set_xlabel("scale j")
        ax[k].set_title(f"{f} (mean over orientations)")
    ax[0].legend(fontsize=7)
    for k, f in enumerate(("S3", "S4"), start=2):
        sel = fam == f
        ref = vals[0][sel]
        lim = np.abs(ref).max() * 1.05
        for name, v in zip(names[1:], vals[1:]):
            ax[k].plot(np.real(ref), np.real(v[sel]), ".", ms=2, label=f"{name} (Re)")
        ax[k].plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
        ax[k].set_xlabel(f"{names[0]}")
        ax[k].set_title(f"{f}: {sel.sum()} coefficients")
        if len(vals) > 1:
            ax[k].legend(fontsize=7, markerscale=4)
    fig.suptitle(f"nside {rep['nside']}, J {rep['J']}, L {rep['L']}, {len(plan)} coefficients", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("figure", out)


def summary(paths):
    for p in paths:
        r = json.load(open(p))
        hist = r.get("loss_history", [])
        mem = r.get("memory_peak_bytes_per_rank") or r.get("peak_rss_bytes")
        mem = max(mem) if isinstance(mem, list) else mem
        dev = r.get("gpu", {}) if isinstance(r.get("gpu"), dict) else {}
        dev = dev.get("memory_peak_bytes_per_rank") or r.get("peak_device_bytes") or (r.get("device_memory") or {}).get("peak_bytes")
        dev = max(dev) if isinstance(dev, list) else dev
        fmt = lambda b: f"{b / 2**30:.2f} GiB" if b else "-"
        print(
            f"{p}: {r.get('evaluations')} evaluations, {r.get('seconds_per_evaluation')} s/eval, "
            f"{r.get('seconds', 0):.1f} s total, loss {hist[0] if hist else '-':.4g} -> {hist[-1] if hist else '-':.4g}, "
            f"host peak {fmt(mem)}, device peak {fmt(dev)}"
        )


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "maps":
        prange, norm = (0.5, 99.5), "hist"
        if "--norm" in rest:
            i = rest.index("--norm")
            norm = rest[i + 1]
            rest = rest[:i] + rest[i + 2 :]
        if "--range" in rest:
            i = rest.index("--range")
            prange = tuple(float(v) for v in rest[i + 1].split(","))
            rest = rest[:i] + rest[i + 2 :]
        plot_maps(rest[0], split_specs(rest[1:]), prange, norm=norm)
    elif cmd == "loss":
        plot_loss(rest[0], split_specs(rest[1:]))
    elif cmd == "stats":
        plot_stats(rest[0], rest[1])
    elif cmd == "summary":
        summary(rest)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
