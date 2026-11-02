"""Maps of three realizations and of three members of every run, at the
same colour scale: Mollweide views and gnomonic zooms on one patch.

    python plot_maps.py OUT_DIR REAL_DIR LABEL=RUN_DIR [...] [--members 0,1,2] [--centre LON,LAT] [--size DEG]

One row per population: r000-r002, then the members of each run.  As in
text/scripts/fig_maps.py, u = kappa / std(r000) is displayed as
log10(u - min(u_r000) + 0.1), so that the texture of a field with rare high
peaks shows; one scale for every panel (0.3 and 99.97 percentiles of r000);
values below the minimum of r000 at the bottom of the scale.  Two A&A
columns wide.  Writes OUT_DIR/maps_mollweide_n{N}.png and .pdf, and
maps_gnomonic_n{N}.png and .pdf.
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

EPS = 0.1
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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

members = [int(k) for k in a.members.split(",")]
first = sorted(glob.glob(f"{a.real_dir}/r000_nside*_nested.npy"))[0]
N = int(os.path.basename(first).split("_nside")[1].split("_")[0])
reals = [f"{a.real_dir}/r{k:03d}_nside{N}_nested.npy" for k in range(len(members))]
r000 = np.load(reals[0]).astype(np.float64)
sig = r000.std()
u_min = (r000 / sig).min()


def show(m):
    return np.log10(np.maximum(m / sig - u_min + EPS, EPS))


rows = [("Takahashi", [(f"r{i:03d}", show(np.load(f).astype(np.float64))) for i, f in enumerate(reals)])]
for label, _, run in C.parse_runs(a.runs):
    files = C.member_files(run)
    rows.append((label, [(f"member {k}", show(np.load(files[k]).astype(np.float64))) for k in members]))
vmin, vmax = np.percentile(rows[0][1][0][1], [0.3, 99.97])
cmap = "inferno"
norm = Normalize(vmin, vmax)


def colourbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    ticks = [u for u in (-1, 0, 1, 3, 10, 30, 100) if u - u_min + EPS > 0 and vmin <= np.log10(u - u_min + EPS) <= vmax]
    cb.set_ticks([np.log10(u - u_min + EPS) for u in ticks])
    cb.set_ticklabels([f"{u:g}" for u in ticks])
    cb.set_label(r"$\kappa/\sigma_{\rm r000}$ (logarithmic scale)")


nr, nc = len(rows), len(members)
# Two A&A columns (180 mm); the map images at about 250 dpi of that width.
fig = plt.figure(figsize=(WIDTH, 1.5 * nr))
for i, (label, panels) in enumerate(rows):
    for j, (name, m) in enumerate(panels):
        hp.mollview(m, nest=True, fig=fig.number, sub=(nr, nc, i * nc + j + 1), title=f"{label}, {name}", min=vmin, max=vmax,
                    cmap=cmap, cbar=False, xsize=600, notext=True)
colourbar(fig, [0.3, -0.03, 0.4, 0.025])
fig.savefig(f"{a.out}/maps_mollweide_n{N}.png", dpi=150)
fig.savefig(f"{a.out}/maps_mollweide_n{N}.pdf", dpi=300)
plt.close(fig)

lon, lat = map(float, a.centre.split(","))
fig = plt.figure(figsize=(WIDTH, 2.45 * nr))
for i, (label, panels) in enumerate(rows):
    for j, (name, m) in enumerate(panels):
        hp.gnomview(m, nest=True, rot=(lon, lat), xsize=450, reso=a.size * 60 / 450, fig=fig.number,
                    sub=(nr, nc, i * nc + j + 1), title=f"{label}, {name}", min=vmin, max=vmax, cmap=cmap, cbar=False,
                    notext=True)
colourbar(fig, [0.25, -0.02, 0.5, 0.015])
fig.suptitle(f"gnomonic zooms, {a.size:g}$^\\circ$ around (lon, lat) = ({lon:g}, {lat:g})", y=1.03)
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.png", dpi=130)
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.pdf", dpi=300)
print(f"{a.out}/maps_mollweide_n{N}.png {a.out}/maps_gnomonic_n{N}.png, min(u_r000) = {u_min:.3f}, "
      f"colour range {vmin:.3f} to {vmax:.3f} (log10 units)")
