"""Face-edge seams of generated maps: statistics of the finest pyramid
details against the distance to the nearest base-face edge.

    python diffusion/seams.py TARGET.npy LABEL=FILE[,FILE...] [LABEL=...]

For the details `d = x - up(down(x))` of the finest level, in bands of
distance (in fine pixels) to the nearest face edge, separately for the
equatorial faces (4-7) and the polar ones, and within a few pixels of the
8 vertices where 3 faces meet: variance over the target's in the same band,
and excess kurtosis.  A seam shows as a band near the edges whose variance
departs from the one of the face interiors.
"""

import sys

import numpy as np

import healpix_faces as hf


def masks(nside):
    """Boolean maps (NESTED) of the distance bands and of the 3-face vertices."""
    n = nside
    i = np.arange(n)
    d1 = np.minimum(i, n - 1 - i)
    dist = np.minimum(d1[:, None], d1[None, :])
    dist = hf.from_faces(np.broadcast_to(dist, (12, n, n)))
    face = np.arange(12 * n * n) // (n * n)
    polar = (face < 4) | (face >= 8)
    # 3-face vertices: the east and west corners of the polar faces
    # (ix = 0, iy = n - 1 and ix = n - 1, iy = 0) and the north and south
    # corners of the equatorial faces.
    ix, iy = np.meshgrid(i, i, indexing="ij")
    k = 4
    corner_img = np.zeros((12, n, n), bool)
    e_w = ((ix < k) & (iy >= n - k)) | ((ix >= n - k) & (iy < k))
    n_s = ((ix < k) & (iy < k)) | ((ix >= n - k) & (iy >= n - k))
    corner_img[[0, 1, 2, 3, 8, 9, 10, 11]] = e_w
    corner_img[[4, 5, 6, 7]] = n_s
    # North-pole and south-pole corners of the polar faces are 4-face vertices: not included.
    return dist, polar, hf.from_faces(corner_img)


def kurt(v):
    v = v - v.mean()
    return float(np.mean(v**4) / np.mean(v**2) ** 2 - 3)


def stats(m, dist, polar, corner):
    d = m - hf.up(hf.down(m))
    out = {}
    for name, sel_face in (("equatorial", ~polar), ("polar", polar)):
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 10**9)):
            s = sel_face & (dist >= lo) & (dist < hi) & ~corner
            out[(name, lo)] = (np.mean(d[s] ** 2), kurt(d[s]))
    out[("3-face vertices", 0)] = (np.mean(d[corner] ** 2), kurt(d[corner]))
    return out


if __name__ == "__main__":
    target_path, *specs = sys.argv[1:]
    target = np.load(target_path).astype(np.float64)
    dist, polar, corner = masks(hf.nside_of(target.size))
    t = stats(target, dist, polar, corner)
    print(f"{'set':22s} {'faces':16s} {'band':>8s}  var/target  kurt (target)")
    for spec in specs:
        label, files = spec.split("=", 1)
        maps = []
        for f in files.split(","):
            a = np.load(f).astype(np.float64)
            maps.extend(a if a.ndim == 2 else [a])
        per = [stats(m, dist, polar, corner) for m in maps]
        for key in t:
            ratio = np.array([p[key][0] / t[key][0] for p in per])
            k = np.mean([p[key][1] for p in per])
            band = f"{key[1]}-" if key[0] != "3-face vertices" else "<4 px"
            print(f"{label:22s} {key[0]:16s} {band:>8s}  {ratio.mean():.3f} +- {ratio.std():.3f}  {k:.2f} ({t[key][1]:.2f})")
