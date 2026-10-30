"""Checks of the face geometry against healpy.

    python diffusion/test_healpix_faces.py
"""

import healpy as hp
import numpy as np

import healpix_faces as hf


def test_face_layout(nside):
    fn = hf.face_nest(nside)
    x, y, f = hp.pix2xyf(nside, fn.ravel(), nest=True)
    ix, iy = np.meshgrid(np.arange(nside), np.arange(nside), indexing="ij")
    assert np.array_equal(x.reshape(fn.shape), np.broadcast_to(ix, fn.shape))
    assert np.array_equal(y.reshape(fn.shape), np.broadcast_to(iy, fn.shape))
    assert np.array_equal(f.reshape(fn.shape), np.broadcast_to(np.arange(12)[:, None, None], fn.shape))
    m = np.random.default_rng(0).normal(size=12 * nside**2)
    assert np.array_equal(hf.from_faces(hf.to_faces(m)), m)
    # 2 x 2 pooling of the faces is the NESTED parent mean.
    img = hf.to_faces(m)
    pooled = img.reshape(12, nside // 2, 2, nside // 2, 2).mean(axis=(2, 4))
    assert np.allclose(hf.from_faces(pooled), hf.down(m))


def test_pad_neighbours(nside):
    pad = hf.pad_nest(nside, 1)
    xoff = [-1, -1, 0, 1, 1, 1, 0, -1]
    yoff = [0, 1, 1, 1, 0, -1, -1, -1]
    pix = hf.face_nest(nside)
    nb = hp.get_all_neighbours(nside, pix.ravel(), nest=True).reshape((8,) + pix.shape)
    ix, iy = np.meshgrid(np.arange(nside), np.arange(nside), indexing="ij")
    checked = 0
    for k in range(8):
        got = pad[:, ix + 1 + xoff[k], iy + 1 + yoff[k]]
        ok = nb[k] >= 0
        assert np.array_equal(got[ok], nb[k][ok]), f"nside {nside} direction {k}"
        checked += ok.sum()
    assert np.array_equal(pad[:, 1:-1, 1:-1], pix)
    return checked


def test_wide_pad(nside, p):
    """Padding of width p: outside the corner squares, consecutive padded
    pixels are neighbours on the sphere (steps below 2 pixel sizes)."""
    pad = hf.pad_nest(nside, p)
    vec = np.array(hp.pix2vec(nside, pad.ravel(), nest=True)).T.reshape(pad.shape + (3,))
    pix = hp.nside2resol(nside)
    r = np.arange(-p, nside + p)
    inside = (r >= 0) & (r < nside)
    worst = 0.0
    for axis in (1, 2):
        a = np.take(vec, range(1, len(r)), axis=axis)
        b = np.take(vec, range(len(r) - 1), axis=axis)
        step = np.arccos(np.clip((a * b).sum(-1), -1, 1)) / pix
        pair = inside[1:] & inside[:-1]
        edge = (pair[:, None] | inside[None, :]) if axis == 1 else (inside[:, None] | pair[None, :])
        worst = max(worst, step[:, edge].max())
    assert worst < 2, f"max step {worst:.2f} pixels"
    centre = vec[:, p:-p, p:-p]
    assert np.allclose(centre, np.array(hp.pix2vec(nside, hf.face_nest(nside).ravel(), nest=True)).T.reshape(centre.shape))


def test_haar(nside):
    m = np.random.default_rng(1).normal(size=12 * nside**2)
    mean, h = hf.haar(m)
    assert np.allclose(mean, hf.down(m))
    assert np.allclose(hf.unhaar(mean, h), m)
    d = m - hf.up(mean)
    assert np.isclose(np.mean(d**2), np.mean((h**2).sum(axis=-2)) / 4)


def test_orient_haar(nside, h):
    """Haar details of the padded fine faces (computed in the padded face's
    frame) equal the padded parent-grid details re-oriented by orient_haar,
    outside the corner squares of the 3-face vertices."""
    fine = np.random.default_rng(2).normal(size=12 * (2 * nside) ** 2)
    img = fine[hf.pad_nest(2 * nside, 2 * h)]
    c0, c1, c2, c3 = img[:, 0::2, 0::2], img[:, 1::2, 0::2], img[:, 0::2, 1::2], img[:, 1::2, 1::2]
    direct = np.stack([(c1 + c3 - c0 - c2) / 2, (c2 + c3 - c0 - c1) / 2, (c0 + c3 - c1 - c2) / 2], axis=1)
    _, det = hf.haar(fine)
    flat = hf.to_faces(det).reshape(3, -1)
    ids, bits = hf.pad_flat(nside, h, with_bits=True)
    m = nside + 2 * h
    gathered = flat[:, ids.reshape(12, -1)].transpose(1, 0, 2)  # [12, 3, m*m]
    oriented = hf.orient_haar(gathered, bits.reshape(12, -1)).reshape(12, 3, m, m)
    r = np.arange(-h, nside + h)
    out = (r < 0) | (r >= nside)
    corner = out[:, None] & out[None, :]
    err = np.abs(oriented - direct)[:, :, ~corner].max()
    unoriented = np.abs(gathered.reshape(12, 3, m, m) - direct)[:, :, ~corner].max()
    assert err < 1e-12, f"max error {err}"
    return unoriented


if __name__ == "__main__":
    for nside, h in ((4, 2), (8, 8), (32, 12)):
        u = test_orient_haar(nside, h)
        print(f"nside {nside} halo {h}: re-oriented padded details equal the details of the padded fine faces "
              f"(without re-orientation: max error {u:.2f})")
    for nside in (2, 4, 8, 32):
        test_face_layout(nside)
        n = test_pad_neighbours(nside)
        test_haar(nside)
        print(f"nside {nside}: layout, pooling, haar ok; {n} neighbours equal to healpy")
    for nside, p in ((8, 4), (16, 16), (32, 8)):
        test_wide_pad(nside, p)
        print(f"nside {nside} pad {p}: padded steps below 2 pixels outside the corners")
