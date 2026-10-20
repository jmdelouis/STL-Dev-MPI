"""Degrade a full-sky HEALPix map (.npy, RING or NESTED, any nside) to
NESTED maps at lower nsides by averaging the sub-pixels.

    python3 tools/prepare_healpix_target.py IN.npy OUT_PREFIX NSIDE [NSIDE ...] [--ordering ring|nested|auto]

Writes OUT_PREFIX_nside<N>_nested.npy for every requested NSIDE.  With
--ordering auto (default) the ordering is detected: the map is degraded
to nside 64 both as NESTED and after RING -> NESTED reordering, and the
hypothesis that keeps more large-scale variance wins (averaging the
sub-pixels of the true ordering keeps the structures, the wrong one
averages scattered pixels).  Pure numpy, memory mapped, by chunks.
"""

import sys

import numpy as np

JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)


def spread_bits(v):
    v = v & 0x00000000FFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    return (v | (v << 1)) & 0x5555555555555555


def isqrt(v):
    r = np.sqrt(v.astype(np.float64)).astype(np.int64)
    r = np.where(r * r > v, r - 1, r)
    return np.where((r + 1) * (r + 1) <= v, r + 1, r)


def ring2nest(pix, nside):
    """RING -> NESTED (healpix_base ring2xyf + xyf2nest), vectorised."""
    order = int(np.log2(nside))
    npix, ncap, nl2 = 12 * nside * nside, 2 * nside * (nside - 1), 2 * nside
    pix = np.asarray(pix, dtype=np.int64)
    north, south = pix < ncap, pix >= npix - ncap
    iring = np.empty_like(pix); iphi = np.empty_like(pix); kshift = np.zeros_like(pix)
    nr = np.empty_like(pix); face = np.empty_like(pix)
    # North cap.
    p = pix[north]; ir = (1 + isqrt(1 + 2 * p)) >> 1; ip = p + 1 - 2 * ir * (ir - 1)
    iring[north], iphi[north], nr[north], face[north] = ir, ip, ir, (ip - 1) // np.maximum(ir, 1)
    # Equatorial belt.
    eq = ~(north | south)
    p = pix[eq] - ncap; tmp = p >> (order + 2); ir = tmp + nside; ip = p - tmp * 4 * nside + 1
    ire = ir - nside + 1; irm = nl2 + 2 - ire
    ifm = (ip - ire // 2 + nside - 1) >> order; ifp = (ip - irm // 2 + nside - 1) >> order
    iring[eq], iphi[eq], kshift[eq], nr[eq] = ir, ip, (ir + nside) & 1, nside
    face[eq] = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    # South cap.
    p = npix - pix[south]; ir = (1 + isqrt(2 * p - 1)) >> 1; ip = 4 * ir + 1 - (p - 2 * ir * (ir - 1))
    iring[south], iphi[south], nr[south], face[south] = 2 * nl2 - ir, ip, ir, 8 + (ip - 1) // np.maximum(ir, 1)
    irt = iring - JRLL[face] * nside + 1
    ipt = 2 * iphi - JPLL[face] * nr - kshift - 1
    ipt = np.where(ipt >= nl2, ipt - 8 * nside, ipt)
    ix, iy = (ipt - irt) >> 1, (-ipt - irt) >> 1
    return (face << (2 * order)) + spread_bits(ix) + (spread_bits(iy) << 1)


def to_nested(m, nside, chunk=1 << 24):
    """RING map -> NESTED map (new array), by chunks of pixels."""
    out = np.empty(m.size, dtype=np.float64)
    for s in range(0, m.size, chunk):
        e = min(s + chunk, m.size)
        out[ring2nest(np.arange(s, e), nside)] = m[s:e]
    return out


def degrade(nest, nside_out):
    f = nest.size // (12 * nside_out**2)
    return nest.reshape(-1, f).mean(axis=1)


def moments(m):
    s = (m - m.mean()) / m.std()
    return f"mean {m.mean():+.4e} std {m.std():.4e} skew {(s**3).mean():+.3f} kurt {(s**4).mean() - 3:+.3f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    src, prefix, nsides = args[0], args[1], [int(a) for a in args[2:]]
    m = np.load(src, mmap_mode="r").reshape(-1)
    nside = int(round((m.size / 12) ** 0.5))
    assert 12 * nside * nside == m.size, f"{m.size} pixels is not a full-sky map"
    print(f"{src}: nside {nside}, {m.dtype}, {moments(np.asarray(m[: 1 << 22], dtype=np.float64))} (first 4M pixels)")
    ordering = opts.get("ordering", "auto")
    ring_nest = None
    if ordering in ("auto", "ring"):
        ring_nest = to_nested(m, nside)
    if ordering == "auto":
        test = min(64, nside)
        v_nested = degrade(np.asarray(m, dtype=np.float64), test).var()
        v_ring = degrade(ring_nest, test).var()
        print(f"variance at nside {test}: as NESTED {v_nested:.4e}, as RING {v_ring:.4e}")
        ordering = "nested" if v_nested > v_ring else "ring"
        print(f"detected ordering: {ordering.upper()} (ratio {max(v_nested, v_ring) / min(v_nested, v_ring):.1f})")
    nest = ring_nest if ordering == "ring" else np.asarray(m, dtype=np.float64)
    del ring_nest
    for ns in nsides:
        out = f"{prefix}_nside{ns}_nested.npy"
        d = degrade(nest, ns)
        np.save(out, d)
        print(f"{out}: {moments(d)}")


if __name__ == "__main__":
    main()
