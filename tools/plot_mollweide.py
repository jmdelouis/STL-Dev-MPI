"""Mollweide views of NESTED HEALPix maps on a common colour scale.

    python tools/plot_mollweide.py OUT.png TITLE=MAP.npy [TITLE=MAP.npy ...] [--cmap NAME] [--range P1,P2]

Each map is sampled on a Mollweide grid (nearest pixel, healpix_geo), as
healpy's mollview does; the colour scale is the [P1, P2] percentiles of
the first map (default 0.5, 99.5).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from healpix_geo.nested import lonlat_to_healpix


def mollweide_index(nside, width=1600):
    """NESTED pixel of every point of a Mollweide grid (-1 outside the ellipse)."""
    height = width // 2
    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), width)
    y = np.linspace(np.sqrt(2), -np.sqrt(2), height)
    X, Y = np.meshgrid(x, y)
    inside = (X / (2 * np.sqrt(2))) ** 2 + (Y / np.sqrt(2)) ** 2 <= 1
    theta = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
    lat = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
    lon = -np.pi * X / (2 * np.sqrt(2) * np.cos(theta).clip(1e-12))  # east to the left, as mollview
    idx = np.full(X.shape, -1, dtype=np.int64)
    sel = inside
    idx[sel] = np.asarray(lonlat_to_healpix(np.degrees(lon[sel]) % 360, np.degrees(lat[sel]), int(np.log2(nside))), dtype=np.int64)
    return idx


def main():
    argv = sys.argv[1:]
    opts = {a.split("=", 1)[0].lstrip("-"): a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
    pos = [a for a in argv if not a.startswith("--")]
    out, specs = pos[0], pos[1:]
    maps = [(s.rsplit("=", 1)[0], np.load(s.rsplit("=", 1)[1]).astype(np.float64).reshape(-1)) for s in specs]
    nside = int(round((maps[0][1].size / 12) ** 0.5))
    p1, p2 = (float(v) for v in opts.get("range", "0.5,99.5").split(","))
    vmin, vmax = np.percentile(maps[0][1], [p1, p2])
    idx = mollweide_index(nside)
    n = len(maps)
    fig, axes = plt.subplots(n, 1, figsize=(11, 5.6 * n))
    for ax, (title, m) in zip(np.atleast_1d(axes), maps):
        img = np.where(idx >= 0, m[np.maximum(idx, 0)], np.nan)
        im = ax.imshow(img, cmap=opts.get("cmap", "plasma"), vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("figure", out)


if __name__ == "__main__":
    main()
