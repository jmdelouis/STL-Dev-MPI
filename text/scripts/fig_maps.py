"""Maps: Mollweide views and gnomonic zooms of the target, another
realization, a Gaussian field and ensemble members.

    python fig_maps.py NSIDE OUT_DIR REAL_DIR LABEL=RUN_DIR [--members 0,1,2] [--centre LON,LAT] [--seed S]

Maps: r000 (the target), r001 (an independent realization of the same
simulation suite), a Gaussian field with the angular power spectrum and the
mean of r000 (seed S), and members of the run given (the last if several).
As in Mousset et al. (2024), the colour is logarithmic so that the texture
of a field with rare high peaks shows: u = kappa / std(r000) is displayed as
log10(u - min(u_r000) + EPS), the same scale for every panel (0.3 and 99.97
percentiles of r000), values below the minimum of r000 (Gaussian field,
other realization) at the bottom of the scale; the colour bar is labelled in
u.  Gnomonic zooms of 20, 6 and 2 degrees around the same centre.  Writes
OUT_DIR/maps_mollweide_n{NSIDE}.png and maps_gnomonic_n{NSIDE}.png.
"""

import argparse
import os

import numpy as np

import common as C

EPS = 0.1  # offset of the logarithm, in units of std(r000)

p = argparse.ArgumentParser()
p.add_argument("nside", type=int)
p.add_argument("out")
p.add_argument("real_dir")
p.add_argument("runs", nargs="+")
p.add_argument("--members", default="0,1,2")
p.add_argument("--centre", default="20,-27.5")
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
N = a.nside
plt = C.style()
import healpy as hp  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

label, _, run = C.parse_runs(a.runs)[-1]
files = C.member_files(run)
reals = C.realization_files(a.real_dir, N, 2)
r000 = np.load(reals[0]).astype(np.float64)
mu, sig = r000.mean(), r000.std()

# Gaussian field with the spectrum and the mean of r000 (RING for the
# harmonic transforms, back to NESTED).
cl = hp.anafast(hp.reorder(r000 - mu, n2r=True), lmax=3 * N - 1)
np.random.seed(a.seed)
gauss = hp.reorder(hp.synfast(cl, N, lmax=3 * N - 1, new=True), r2n=True) + mu

u_min = (r000 / sig).min()


def show(m):
    return np.log10(np.maximum(m / sig - u_min + EPS, EPS))


members = [int(k) for k in a.members.split(",")]
panels = [("r000 (target)", show(r000)),
          ("r001 (another realization)", show(np.load(reals[1]).astype(np.float64))),
          ("Gaussian, same $C_\\ell$ as r000", show(gauss))]
panels += [(f"{label}, member {k}", show(np.load(files[k]).astype(np.float64))) for k in members]
vmin, vmax = np.percentile(panels[0][1], [0.3, 99.97])
cmap = "inferno"
norm = Normalize(vmin, vmax)


def colourbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    ticks = [u for u in (-1, 0, 1, 3, 10, 30, 100) if u - u_min + EPS > 0 and vmin <= np.log10(u - u_min + EPS) <= vmax]
    cb.set_ticks([np.log10(u - u_min + EPS) for u in ticks])
    cb.set_ticklabels([f"{u:g}" for u in ticks])
    cb.set_label(r"$\kappa/\sigma_{\rm r000}$ (logarithmic scale)")


cols = 3
rows = (len(panels) + cols - 1) // cols
fig = plt.figure(figsize=(7.2, 1.5 * rows))
for i, (name, m) in enumerate(panels):
    hp.mollview(m, nest=True, fig=fig.number, sub=(rows, cols, i + 1), title=name, min=vmin, max=vmax, cmap=cmap,
                cbar=False, xsize=1600, notext=True)
colourbar(fig, [0.3, -0.04, 0.4, 0.03])
fig.savefig(f"{a.out}/maps_mollweide_n{N}.png", dpi=240)
plt.close(fig)

lon, lat = map(float, a.centre.split(","))
sizes = (20.0, 6.0, 2.0)
zoom = panels[:5]  # r000, r001, Gaussian, two members
fig = plt.figure(figsize=(7.2, 7.2 * len(zoom) / len(sizes) * 1.05))
for i, (name, m) in enumerate(zoom):
    for j, size in enumerate(sizes):
        hp.gnomview(m, nest=True, rot=(lon, lat), xsize=600, reso=size * 60 / 600, fig=fig.number,
                    sub=(len(zoom), len(sizes), i * len(sizes) + j + 1), title=f"{name}, {size:g}$^\\circ$",
                    min=vmin, max=vmax, cmap=cmap, cbar=False, notext=True)
colourbar(fig, [0.25, -0.02, 0.5, 0.012])
fig.savefig(f"{a.out}/maps_gnomonic_n{N}.png", dpi=160)
print(f"{a.out}/maps_mollweide_n{N}.png {a.out}/maps_gnomonic_n{N}.png, min(u_r000) = {u_min:.3f}, "
      f"colour range {vmin:.3f} to {vmax:.3f} (log10 units)")
