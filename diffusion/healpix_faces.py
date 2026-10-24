"""HEALPix NESTED maps as 12 face images, face-aware padding, and the
4-children pyramid with its Haar details.

Face images: `img[f, ix, iy]` is the pixel `(ix, iy)` of base face `f`
(HEALPix `nest2xyf`: `ix` in the even bits of the in-face NESTED index,
`iy` in the odd bits).  The 4 children of the parent `(f, i, j)` are
`(f, 2i + bx, 2j + by)` with NESTED child number `bx + 2 by`, so the
NESTED parent mean is the 2 x 2 average pooling of the face image.

Padding: a ring of `p` pixels around every face, filled with the pixels of
the neighbouring faces through the base-face adjacency of the HEALPix C++
library (`T_Healpix_Base::neighbors`: offset, wrap, flip and swap of the
face frame).  Exact on the 4 edges and on the corners where 4 faces meet;
the 8 vertices where only 3 faces meet have no corner face and take the
nearest edge-neighbour pixel instead (`p * p` duplicated pixels there).

Haar details of one level (orthonormal, on the parent grid):

    hx = (c1 + c3 - c0 - c2) / 2      (x)
    hy = (c2 + c3 - c0 - c1) / 2      (y)
    hd = (c0 + c3 - c1 - c2) / 2      (diagonal)

with `c_k` the child `k`; with `m` the parent mean, the children are
`c = m + (+-hx +-hy +-hd) / 2`.  The detail field of the pyramid,
`d_n = x_n - up(x_{n+1})`, has `mean(d_n^2) = (hx^2 + hy^2 + hd^2) / 4`
per parent.
"""

import numpy as np

# HEALPix C++ base-face tables (healpix_base.cc).
NB_FACE = np.array([
    [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],   # S   (dx -1, dy -1)
    [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],       # SE  (dx  0, dy -1)
    [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],   # E   (dx +1, dy -1)
    [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],       # SW  (dx -1, dy  0)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],         # centre
    [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],           # NE  (dx +1, dy  0)
    [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],   # W   (dx -1, dy +1)
    [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],           # NW  (dx  0, dy +1)
    [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],       # N   (dx +1, dy +1)
])
NB_SWAP = np.array([[0, 0, 3], [0, 0, 6], [0, 0, 0], [0, 0, 5], [0, 0, 0],
                    [5, 0, 0], [0, 0, 0], [6, 0, 0], [3, 0, 0]])


def nside_of(npix):
    nside = int(round(np.sqrt(npix / 12)))
    assert 12 * nside * nside == npix, f"{npix} pixels is not a HEALPix map"
    return nside


def _spread_bits(v):
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for k in range(31):
        out |= ((v >> k) & 1) << (2 * k)
    return out


def xyf2nest(nside, ix, iy, face):
    return np.asarray(face, dtype=np.int64) * nside * nside + (_spread_bits(ix) | (_spread_bits(iy) << 1))


def face_nest(nside):
    """`[12, nside, nside]` NESTED id of the face-image pixel `[f, ix, iy]`."""
    ix, iy = np.meshgrid(np.arange(nside), np.arange(nside), indexing="ij")
    return xyf2nest(nside, ix[None], iy[None], np.arange(12)[:, None, None])


def to_faces(m):
    """NESTED map(s) `[..., 12 nside^2]` -> face images `[..., 12, nside, nside]`."""
    m = np.asarray(m)
    nside = nside_of(m.shape[-1])
    return m[..., face_nest(nside)]


def from_faces(img):
    """Face images `[..., 12, nside, nside]` -> NESTED map(s) `[..., 12 nside^2]`."""
    img = np.asarray(img)
    nside = img.shape[-1]
    out = np.empty(img.shape[:-3] + (12 * nside * nside,), dtype=img.dtype)
    out[..., face_nest(nside).ravel()] = img.reshape(img.shape[:-3] + (-1,))
    return out


def pad_nest(nside, p, with_bits=False):
    """`[12, nside + 2p, nside + 2p]` NESTED ids of every face padded by `p`
    pixels of its neighbours (`p <= nside`).  With `with_bits`, also the
    frame change of every padded pixel (`NB_SWAP` bits: 1 flips x, 2 flips
    y, 4 then swaps x and y; 0 inside the face), for the oriented channels
    (`orient_haar`)."""
    assert 0 <= p <= nside
    n = nside
    r = np.arange(-p, n + p)
    ix, iy = np.meshgrid(r, r, indexing="ij")
    ix = np.broadcast_to(ix, (12, n + 2 * p, n + 2 * p)).copy()
    iy = np.broadcast_to(iy, (12, n + 2 * p, n + 2 * p)).copy()
    face = np.broadcast_to(np.arange(12)[:, None, None], ix.shape).copy()
    dx = np.where(ix < 0, -1, np.where(ix >= n, 1, 0))
    dy = np.where(iy < 0, -1, np.where(iy >= n, 1, 0))
    # 3-face vertices: no corner face, fall back to the x-edge neighbour
    # with y clamped to the face.
    missing = NB_FACE[4 + dx + 3 * dy, face] < 0
    iy = np.where(missing, np.clip(iy, 0, n - 1), iy)
    dy = np.where(missing, 0, dy)
    nb = 4 + dx + 3 * dy
    f = NB_FACE[nb, face]
    bits = NB_SWAP[nb, face >> 2]
    x, y = ix - dx * n, iy - dy * n
    x = np.where(bits & 1, n - x - 1, x)
    y = np.where(bits & 2, n - y - 1, y)
    x, y = np.where(bits & 4, y, x), np.where(bits & 4, x, y)
    ids = xyf2nest(n, x, y, f)
    return (ids, bits) if with_bits else ids


def pad_flat(nside, p, with_bits=False):
    """Like `pad_nest`, as indices into the flattened face images
    `img.reshape(..., 12 nside^2)`."""
    flat_of_nest = np.empty(12 * nside * nside, dtype=np.int64)
    flat_of_nest[face_nest(nside).ravel()] = np.arange(12 * nside * nside)
    ids, bits = pad_nest(nside, p, with_bits=True)
    return (flat_of_nest[ids], bits) if with_bits else flat_of_nest[ids]


def orient_haar(h, bits):
    """Haar details `[..., 3, *pix]` gathered from a neighbouring face,
    expressed in the frame of the padded face.  The neighbour's frame is
    reached by flipping x (bit 1) and y (bit 2), then swapping x and y (bit
    4), so along the padded face's x the neighbour's details read
    `sx * (hy if swap else hx)`, along y `sy * (hx if swap else hy)`, and the
    diagonal `sx * sy * hd`.  Works on numpy arrays and torch tensors."""
    sx = 1 - 2 * (bits & 1 != 0)
    sy = 1 - 2 * (bits & 2 != 0)
    swap = bits & 4 != 0
    hx, hy, hd = h[..., 0, :], h[..., 1, :], h[..., 2, :]
    if isinstance(h, np.ndarray):
        a, b = np.where(swap, hy, hx), np.where(swap, hx, hy)
        return np.stack([sx * a, sy * b, sx * sy * hd], axis=-2)
    import torch

    a, b = torch.where(swap, hy, hx), torch.where(swap, hx, hy)
    return torch.stack([sx * a, sy * b, sx * sy * hd], dim=-2)


def down(m):
    """NESTED parent means: `[..., 12 n^2] -> [..., 3 n^2]`."""
    m = np.asarray(m)
    return m.reshape(m.shape[:-1] + (-1, 4)).mean(axis=-1)


def up(m):
    """Piecewise-constant NESTED up-sampling: every child takes its parent value."""
    return np.repeat(np.asarray(m), 4, axis=-1)


def pyramid(m, nside_min=1):
    """`[x_0, x_1, ...]`: the map and its NESTED means down to `nside_min`."""
    out = [np.asarray(m, dtype=np.float64)]
    while nside_of(out[-1].shape[-1]) > nside_min:
        out.append(down(out[-1]))
    return out


def haar(m):
    """NESTED map `[..., 12 n^2]` -> (parent means `[..., 3 n^2/4]`,
    details `[..., 3, 3 n^2/4]` in the order hx, hy, hd)."""
    c = np.asarray(m).reshape(np.shape(m)[:-1] + (-1, 4))
    c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    mean = (c0 + c1 + c2 + c3) / 4
    hx = (c1 + c3 - c0 - c2) / 2
    hy = (c2 + c3 - c0 - c1) / 2
    hd = (c0 + c3 - c1 - c2) / 2
    return mean, np.stack([hx, hy, hd], axis=-2)


def unhaar(mean, h):
    """Inverse of `haar`."""
    hx, hy, hd = h[..., 0, :], h[..., 1, :], h[..., 2, :]
    c = np.stack([mean + (-hx - hy + hd) / 2, mean + (hx - hy - hd) / 2,
                  mean + (-hx + hy - hd) / 2, mean + (hx + hy + hd) / 2], axis=-1)
    return c.reshape(c.shape[:-2] + (-1,))
