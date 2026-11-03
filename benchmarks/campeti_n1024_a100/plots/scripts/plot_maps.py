"""Maps of three realizations and of three members of every run, at the
same colour scale: Mollweide views and gnomonic zooms on one patch.

    python plot_maps.py OUT_DIR REAL_DIR LABEL=RUN_DIR [...] [--members 0,1,2] [--centre LON,LAT] [--size DEG]

One row per population: r000-r002, then the members of each run.  The
colours of u = kappa / std(r000) are histogram-equalised (healpy's
norm='hist'), so that both the voids and the rare high peaks of the
lensing field show.  To keep one scale for every panel, a single
HistEqNorm is shared by all of them: min and max of r000, and the
equalisation from the distribution of r000 (its quantiles), instead of
each panel's own histogram.  Values outside the range of r000 take the end
colours.  Two A&A columns wide.  Writes OUT_DIR/maps_mollweide_n{N}.png and
.pdf, and maps_gnomonic_n{N}.png and .pdf.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../../../text/scripts", "../../../tools"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, d)))
import common as C  # noqa: E402

WIDTH = 7.0  # inches: two A&A columns (180 mm)

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("real_dir")
p.add_argument("runs", nargs="+")
p.add_argument("--members", default="0,1,2")
p.add_argument("--centre", default="20,-27.5")
p.add_argument("--size", type=float, default=10.0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
plt = C.style()
import healpy as hp  # noqa: E402
from healpy.projaxes import HistEqNorm  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

members = [int(k) for k in a.members.split(",")]
first = sorted(glob.glob(f"{a.real_dir}/r000_nside*_nested.npy"))[0]
N = int(os.path.basename(first).split("_nside")[1].split("_")[0])
reals = [f"{a.real_dir}/r{k:03d}_nside{N}_nested.npy" for k in range(len(members))]
r000 = np.load(reals[0]).astype(np.float64)
sig = r000.std()
u000 = r000 / sig
vmin, vmax = u000.min(), u000.max()
# norm='hist' for every panel with one mapping: healpy builds a HistEqNorm per
# panel from that panel's own pixels; this one is set once from the CDF of
# r000 (2001 quantiles: finer than healpy's linear bins in the bulk) and
# given to every mollview and gnomview (healpy keeps a norm whose table is set).
norm = HistEqNorm(vmin=vmin, vmax=vmax, clip=False)
norm.yval = np.linspace(0.0, 1.0, 2001)
norm.xval = np.quantile(u000, norm.yval)
# Colour-bar ticks at quantiles of r000, evenly spaced on the equalised bar.
TICK_QUANTILES = (0.02, 0.2, 0.4, 0.6, 0.8, 0.95)
ticks = np.quantile(u000, TICK_QUANTILES)
del r000, u000

rows = [("Takahashi", [(f"r{i:03d}", np.load(f).astype(np.float64) / sig) for i, f in enumerate(reals)])]
for label, _, run in C.parse_runs(a.runs):
    files = C.member_files(run)
    rows.append((label, [(f"member {k}", np.load(files[k]).astype(np.float64) / sig) for k in members]))
cmap = "inferno"


def colourbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{u:.2g}" for u in ticks])
    cb.set_label(r"$\kappa/\sigma_{\rm r000}$, colours equalised on r000 (ticks at its 2, 20, 40, 60, 80, 95 % quantiles)")


nr, nc = len(rows), len(members)
# Two A&A columns (180 mm); the map images at about 210 dpi of that width.
fig = plt.figure(figsize=(WIDTH, 1.5 * nr))
for i, (label, panels) in enumerate(rows):
    for j, (name, m) in enumerate(panels):
        hp.mollview(m, nest=True, fig=fig.number, sub=(nr, nc, i * nc + j + 1), title=f"{label}, {name}", min=vmin, max=vmax,
                    norm=norm, cmap=cmap, cbar=False, xsize=500, notext=True)
colourbar(fig, [0.3, -0.03, 0.4, 0.025])
fig.savefig(f"{a.out}/maps_mollweide_n{N}.png", dpi=150)
fig.savefig(f"{a.out}/maps_mollweide_n{N}.pdf", dpi=300)
plt.close(fig)

lon, lat = map(float, a.centre.split(","))
fig = plt.figure(figsize=(WIDTH, 2.45 * nr))
for i, (label, panels) in enumerate(rows):
    for j, (name, m) in enumerate(panels):
        hp.gnomview(m, nest=True, rot=(lon, lat), xsize=450, reso=a.size * 60 / 450, fig=fig.number,
                    sub=(nr, nc, i * nc + j + 1), title=f"{label}, {name}", min=vmin, max=vmax, norm=norm, cmap=cmap,
                    cbar=False, notext=True)
colourbar(fig, [0.25, -0.02, 0.5, 0.015])
fig.suptitle(f"gnomonic zooms, {a.size:g}$^\\circ$ around (lon, lat) = ({lon:g}, {lat:g})", y=1.03)
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.png", dpi=130)
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.pdf", dpi=300)
print(f"{a.out}/maps_mollweide_n{N}.png {a.out}/maps_gnomonic_n{N}.png, norm='hist' shared, "
      f"range of u = kappa / std(r000): {vmin:.3f} to {vmax:.3f}")
