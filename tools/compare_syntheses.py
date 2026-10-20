"""Compare a target HEALPix map with one or more syntheses (NESTED).

    python tools/compare_syntheses.py OUT.png TARGET.fits LABEL=SYNTH.npy [LABEL=SYNTH.npy ...]

Figure: full sky and a 20 x 20 deg zoom for the target and every
synthesis, pixel histograms, and the anisotropy of the small scales.

Anisotropy: 30 gnomonic patches (256 x 256 samples, 1.5 pixel spacing,
centres at latitudes -40 to 40 deg), Hann window, 2-D power spectrum;
ratio of the mean power of east-west variations (|kx| > 2 |ky|) to that
of north-south variations (|ky| > 2 |kx|) at small scales (0.15 < k <
0.45 cycles per sample).  About 1 for an isotropic field (white noise
1.01, WebSky kappa 1.04 at nside 1024); below 1 when structures are
elongated east-west.  Mean and standard error over the patches.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from healpix_geo.nested import healpix_to_lonlat, lonlat_to_healpix


def load(path):
    if path.endswith((".fits", ".fit")):
        with fits.open(path) as h:
            return np.asarray(h[1].data.field(0), dtype=np.float64).reshape(-1)
    return np.load(path).astype(np.float64).reshape(-1)


def raster_index(nside, lon0, lon1, lat0, lat1, height):
    lon, lat = healpix_to_lonlat(np.arange(12 * nside**2, dtype=np.uint64), int(np.log2(nside)))
    lon, lat = np.asarray(lon), np.asarray(lat)
    width = int(height * (lon1 - lon0) / (lat1 - lat0))
    sel = (lon >= lon0) & (lon < lon1) & (lat >= lat0) & (lat < lat1)
    row = np.clip(((lat1 - lat[sel]) / (lat1 - lat0) * height).astype(int), 0, height - 1)
    col = np.clip(((lon[sel] - lon0) / (lon1 - lon0) * width).astype(int), 0, width - 1)
    return sel, row, col, (height, width)


def raster(v, index):
    sel, row, col, shape = index
    a, c = np.zeros(shape), np.zeros(shape)
    np.add.at(a, (row, col), v[sel])
    np.add.at(c, (row, col), 1)
    return np.where(c > 0, a / np.maximum(c, 1), np.nan)


def gnomonic_patches(nside, n=256, spacing=1.5):
    """NESTED pixels of 30 gnomonic patches `[n * n]` (x east, y north)."""
    depth = int(np.log2(nside))
    step = np.radians(spacing * np.degrees(np.sqrt(4 * np.pi / (12 * nside**2))))
    x = (np.arange(n) - n / 2) * step
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    c = np.arctan(rho)
    out = []
    for lat0 in (-40, -20, 0, 20, 40):
        for lon0 in range(0, 360, 60):
            l0, p0 = np.radians(lon0), np.radians(lat0)
            lat = np.arcsin(np.cos(c) * np.sin(p0) + np.where(rho > 0, Y * np.sin(c) * np.cos(p0) / np.maximum(rho, 1e-300), 0))
            lon = l0 + np.arctan2(X * np.sin(c), rho * np.cos(p0) * np.cos(c) - Y * np.sin(p0) * np.sin(c))
            out.append(np.asarray(lonlat_to_healpix(np.degrees(lon).ravel() % 360, np.degrees(lat).ravel(), depth), dtype=np.int64))
    return out


def anisotropy(m, patches, n=256):
    """Power of east-west over north-south small-scale variations: (mean, error)."""
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    k = np.hypot(kx, ky)
    small = (k > 0.15) & (k < 0.45)
    ew, ns = small & (np.abs(kx) > 2 * np.abs(ky)), small & (np.abs(ky) > 2 * np.abs(kx))
    win = np.outer(np.hanning(n), np.hanning(n))
    r = []
    for ix in patches:
        p = m[ix].reshape(n, n)
        power = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
        r.append(power[ew].mean() / power[ns].mean())
    r = np.array(r)
    return float(r.mean()), float(r.std() / np.sqrt(r.size))


def main():
    out, target_path, *specs = sys.argv[1:]
    target = load(target_path)
    nside = int(round((target.size / 12) ** 0.5))
    # LABEL=PATH, split on the last '=' (labels may contain one, paths do not).
    syntheses = [(s.rsplit("=", 1)[0], load(s.rsplit("=", 1)[1])) for s in specs]
    maps = [("target", target)] + syntheses
    # Raster cells no finer than the pixels (about 58.6 deg / nside), zoom
    # of at least ~20 pixels a side.
    full = raster_index(nside, 0, 360, -90, 90, min(512, nside))
    span = min(80.0, 20.0 * max(1.0, 256.0 / nside))
    zlon0, zlat0 = 100.0, 20.0 - span / 2
    zoom_extent = [zlon0, zlon0 + span, zlat0, zlat0 + span]
    zoom = raster_index(nside, *zoom_extent, max(16, min(340, int(nside * span / 117.0))))
    # Patches of at most ~45 deg: 256 samples from nside 512, fewer below.
    n_patch = 256 if nside >= 512 else max(16, nside // 2)
    patches = gnomonic_patches(nside, n=n_patch)
    print(f"nside {nside}")
    print(f"{'map':28s} {'mean':>11s} {'std':>10s} {'skew':>7s} {'max':>9s}  P(E-W)/P(N-S), small scales")
    ratios = []
    for name, m in maps:
        skew = ((m - m.mean()) ** 3).mean() / m.std() ** 3
        r, err = anisotropy(m, patches, n=n_patch)
        ratios.append((r, err))
        print(f"{name:28s} {m.mean():+.4e} {m.std():.4e} {skew:+.3f} {m.max():+.3e}  {r:.3f} +- {err:.3f}")
    vmin, vmax = np.percentile(target, [0.5, 99.5])
    n = len(maps)
    fig, ax = plt.subplots(3, n, figsize=(6.5 * n, 14), squeeze=False)
    for k, (name, m) in enumerate(maps):
        for row, (index, extent, what) in enumerate(((full, [0, 360, -90, 90], "full sky"), (zoom, zoom_extent, f"{span:.0f} x {span:.0f} deg"))):
            im = ax[row, k].imshow(raster(m, index), cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent,
                                   aspect="auto", interpolation="nearest")
            ax[row, k].set_title(f"{name}, {what}")
            fig.colorbar(im, ax=ax[row, k], fraction=0.03)
    bins = np.linspace(np.percentile(target, 0.01), np.percentile(target, 99.99), 200)
    for name, m in maps:
        ax[2, 0].hist(m, bins=bins, histtype="step", density=True, label=name, lw=1.4)
    ax[2, 0].set_yscale("log")
    ax[2, 0].legend()
    ax[2, 0].set_title(f"pixel histograms, nside {nside}")
    if n > 1:
        ax[2, 1].bar([name for name, _ in maps], [r for r, _ in ratios], yerr=[e for _, e in ratios], capsize=6,
                     color=["0.4"] + [f"C{i}" for i in range(n - 1)])
        ax[2, 1].axhline(1.0, color="k", lw=0.8, ls="--")
        lo = min(r - 3 * e for r, e in ratios)
        ax[2, 1].set_ylim(min(0.8, lo), max(1.15, max(r + 3 * e for r, e in ratios)))
        ax[2, 1].set_title("small-scale power, east-west / north-south variations\n(1 = isotropic, < 1 = elongated east-west)")
    for a in ax[2, 2:]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=85)
    print("figure", out)


if __name__ == "__main__":
    main()
