"""Maps: Mollweide views and gnomonic zooms of the target, another
realization and two ensemble members.

    python fig_maps.py NSIDE OUT_DIR REAL_DIR LABEL=RUN_DIR [--members 0,1] [--centre LON,LAT]

Maps: r000 (the target), r001 (an independent realization of the same
simulation suite), and members of the run given (the last if several).
Colour: kappa / std(r000), the same scale for every panel (0.5 and 99.7
percentiles of r000).  Gnomonic zooms of 20, 6 and 2 degrees around the same
centre.  Writes OUT_DIR/maps_mollweide_n{NSIDE}.png and
maps_gnomonic_n{NSIDE}.png.
"""

import argparse
import os

import numpy as np

import common as C

p = argparse.ArgumentParser()
p.add_argument("nside", type=int)
p.add_argument("out")
p.add_argument("real_dir")
p.add_argument("runs", nargs="+")
p.add_argument("--members", default="0,1")
p.add_argument("--centre", default="20,-27.5")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
N = a.nside
plt = C.style()
import healpy as hp  # noqa: E402

label, _, run = C.parse_runs(a.runs)[-1]
files = C.member_files(run)
reals = C.realization_files(a.real_dir, N, 2)
r000 = np.load(reals[0]).astype(np.float64)
sig = r000.std()
panels = [("r000 (target)", r000 / sig), ("r001 (another realization)", np.load(reals[1]).astype(np.float64) / sig)]
panels += [(f"{label}, member {k}", np.load(files[k]).astype(np.float64) / sig) for k in map(int, a.members.split(","))]
vmin, vmax = np.percentile(panels[0][1], [0.5, 99.7])
cmap = "inferno"

fig = plt.figure(figsize=(7.2, 4.4))
for i, (name, m) in enumerate(panels):
    hp.mollview(m, nest=True, fig=fig.number, sub=(2, 2, i + 1), title=name, min=vmin, max=vmax, cmap=cmap,
                cbar=False, xsize=1600, notext=True)
fig.savefig(f"{a.out}/maps_mollweide_n{N}.png", dpi=220)
plt.close(fig)

lon, lat = map(float, a.centre.split(","))
sizes = (20.0, 6.0, 2.0)
fig = plt.figure(figsize=(7.2, 7.2 * len(panels) / len(sizes) * 1.05))
for i, (name, m) in enumerate(panels):
    for j, size in enumerate(sizes):
        hp.gnomview(m, nest=True, rot=(lon, lat), xsize=600, reso=size * 60 / 600, fig=fig.number,
                    sub=(len(panels), len(sizes), i * len(sizes) + j + 1), title=f"{name}, {size:g}$^\\circ$",
                    min=vmin, max=vmax, cmap=cmap, cbar=False, notext=True)
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.png", dpi=160)
print(f"{a.out}/maps_mollweide_n{N}.png {a.out}/maps_gnomonic_n{N}.png, colour range {vmin:.2f} to {vmax:.2f} std")
