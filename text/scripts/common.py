"""Shared definitions of the paper's statistics and plotting style.

Every statistic of a map (kappa, NESTED, full sky) is computed once by
measure.py and kept in a cache file per map; the figure and table scripts
read only the caches.  Definitions:

- details of level n: the map at nside n minus the mean of each group of 4
  children (4-children pyramid, NESTED), from the map's nside down to 8;
  their variance and excess kurtosis <d^4> / <d^2>^2 - 3, for kappa and for
  the map's own rank gaussianisation norm.ppf((rank + 0.5) / N);
- C_l: healpy anafast of the map (RING), lmax = 3 nside - 1;
- peaks: pixels higher than their 8 neighbours, at the map's nside and at
  nside / 4 (mean of the children), heights in units of the r000 map's std
  at that nside;
- PDF: histograms of kappa / std(r000) at nside / 4 and nside / 16 (at the
  map's own nside the members are histogram-matched to r000, so their PDF
  is r000's by construction);
- Minkowski functionals V0, V1, V2 of the map smoothed by a Gaussian beam
  (tools/minkowski.py), thresholds in units of the smoothed map's std.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
if os.path.isdir(TOOLS):
    sys.path.insert(0, TOOLS)

# Thresholds of the peak counts, the PDF bins and the Minkowski functionals.
PEAK_THRESHOLDS = (3.0, 5.0, 10.0)
PEAK_EDGES = np.linspace(-1.0, 30.0, 63)
PDF_EDGES = np.linspace(-2.0, 25.0, 109)
PDF_FACTORS = (4, 16)
NU = np.linspace(-3.0, 6.0, 46)
LOWEST_LEVEL = 8


def mf_fwhms(nside):
    """FWHMs (arcmin) of the Minkowski functionals: 4 pixels at this nside, and 30 arcmin for every nside."""
    pix = np.degrees(np.sqrt(4 * np.pi / (12 * nside**2))) * 60.0
    return (round(4 * pix, 2), 30.0)


def details(m, lowest=LOWEST_LEVEL):
    """{nside: (variance, excess kurtosis)} of the 4-children details of a NESTED map."""
    out = {}
    nside = int(round((m.size / 12) ** 0.5))
    while nside >= lowest:
        p = m.reshape(-1, 4).mean(1)
        d = (m.reshape(-1, 4) - p[:, None]).ravel()
        v = d.var()
        out[nside] = (v, (d**4).mean() / v**2 - 3.0)
        m, nside = p, nside // 2
    return out


def gaussianise(m):
    from scipy.special import ndtri

    r = np.empty(m.size)
    r[np.argsort(m, kind="stable")] = np.arange(m.size)
    return ndtri((r + 0.5) / m.size)


def degrade(m, factor):
    return m.reshape(-1, factor).mean(1) if factor > 1 else m


def peak_heights(m, chunk=1 << 22):
    """Heights of the local maxima (higher than their 8 neighbours) of a NESTED map."""
    import healpy as hp

    nside = int(round((m.size / 12) ** 0.5))
    out = []
    for s in range(0, m.size, chunk):
        ids = np.arange(s, min(s + chunk, m.size))
        nb = hp.get_all_neighbours(nside, ids, nest=True)
        vals = np.where(nb >= 0, m[np.clip(nb, 0, None)], -np.inf)
        out.append(m[ids][(m[ids][None, :] > vals).all(0)])
    return np.concatenate(out)


def cl_bands(nside, nbands=20):
    """Logarithmic multipole bands [lo, hi) from 2 to 3 nside, and their centres."""
    lmax = 3 * nside - 1
    edges = np.unique(np.round(np.geomspace(2, lmax + 1, nbands)).astype(int))
    lo, hi = edges[:-1], edges[1:]
    return lo, hi, np.sqrt(lo * (hi - 1).clip(min=lo))


def bin_cl(cl, lo, hi):
    return np.array([cl[a:b].mean() for a, b in zip(lo, hi)])


def realization_files(directory, nside, count=10):
    return [f"{directory}/r{r:03d}_nside{nside}_nested.npy" for r in range(count)]


def member_files(run, kind="synthesis"):
    return sorted(glob.glob(f"{run}/{kind}_m[0-9][0-9][0-9]_nested.npy"))


def load_set(cache, nside, label):
    """All cache files of a set (realizations or a run), as a dict of stacked arrays."""
    files = sorted(glob.glob(f"{cache}/n{nside}/{label}/*.npz"))
    if not files:
        raise FileNotFoundError(f"no cache for {label} at nside {nside} in {cache}")
    data = [dict(np.load(f)) for f in files]
    return {k: np.array([d[k] for d in data]) for k in data[0]}, [os.path.basename(f)[:-4] for f in files]


def parse_runs(specs):
    """LABEL=RUN_DIR arguments -> [(label, slug, dir)]."""
    out = []
    for sp in specs:
        label, run = sp.split("=", 1)
        slug = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
        out.append((label, slug, run))
    return out


def load_all(cache, nside, specs):
    """Realizations (r000 first) and runs [(label, data)] from the caches."""
    real, names = load_set(cache, nside, "real")
    assert names[0] == "r000", names
    runs = [(label, load_set(cache, nside, slug)[0]) for label, slug, _ in parse_runs(specs)]
    return real, runs


def cosmic(real, key):
    """r000, and mean and std (ddof 1) over the realizations of a statistic (std NaN where it is 0)."""
    x = real[key].astype(np.float64)
    s = x.std(0, ddof=1)
    return x[0], x.mean(0), np.where(s > 0, s, np.nan)


def zscores(real, data, key):
    """Deviation of every member from r000 in units of the realizations' std."""
    r0, _, s = cosmic(real, key)
    return (data[key] - r0) / s


def cli(doc):
    """CACHE NSIDE OUT_DIR [LABEL=RUN_DIR ...] arguments of the figure scripts."""
    import argparse

    p = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    p.add_argument("cache")
    p.add_argument("nside", type=int)
    p.add_argument("out")
    p.add_argument("runs", nargs="*")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    return a


# Plot style: one ink colour, a grey band for the realizations, a few run colours.
INK = "#52514e"
BAND = "0.85"
RUN_COLORS = ["#2a78d6", "#eb6834", "#3c8a3a", "#8a3c8a"]


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
    })
    return plt
