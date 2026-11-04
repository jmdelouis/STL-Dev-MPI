"""Maps side by side: the target, then one map of every run (NESTED .npy),
full sky (mollview) and a 20-degree zoom (gnomview), `norm='hist'` and the
target's min / max as the common scale.

    python plot_maps.py OUT.png TARGET.npy LABEL=MAP.npy [LABEL=MAP.npy ...]
"""

import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, target = sys.argv[1], sys.argv[2]
panels = [("target r000", np.load(target).astype(np.float64))]
for spec in sys.argv[3:]:
    label, path = spec.split("=", 1)
    panels.append((label, np.load(path).astype(np.float64)))
nside = hp.npix2nside(panels[0][1].size)
for k, (label, m) in enumerate(panels):
    while m.size > 12 * nside * nside:
        m = m.reshape(-1, 4).mean(1)
    panels[k] = (label, m)
lo, hi = panels[0][1].min(), panels[0][1].max()
n = len(panels)
fig = plt.figure(figsize=(4 * n, 6))
for k, (label, m) in enumerate(panels):
    hp.mollview(m, nest=True, norm="hist", min=lo, max=hi, title=label, fig=fig.number, sub=(2, n, k + 1), cbar=False)
    hp.gnomview(m, nest=True, norm="hist", min=lo, max=hi, rot=(30, 30), xsize=400, reso=3.0,
                title=f"{label}, 20 deg", fig=fig.number, sub=(2, n, n + k + 1), cbar=False)
plt.savefig(out, dpi=110, bbox_inches="tight")
print("figure", out)
