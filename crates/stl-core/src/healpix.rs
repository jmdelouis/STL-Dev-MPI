//! HEALPix pixelisation: NESTED <-> (x, y, face), pixel centres, RING ->
//! NESTED, and the bilinear interpolation of the HEALPix C++ library.
//!
//! Line-by-line port of `STL_main/TorchGeometry.py` (itself the healpix_base
//! reference formulas), keeping its floating-point operation order so that
//! the results agree with STL-Dev to the last few ulps.  Pixel numbers are
//! `i64` like the torch tensors.

use std::f64::consts::PI;

/// `180 / pi` and `pi / 180` as used by `torch.rad2deg` / `torch.deg2rad`.
pub const RAD2DEG: f64 = 57.295_779_513_082_320_876_798_154_814_105_170_332_405_472_466_564;
/// `pi / 180`, see [`RAD2DEG`].
pub const DEG2RAD: f64 = 0.017_453_292_519_943_295_769_236_907_684_886_127_134_428_718_885_417;

const JRLL: [i64; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL: [i64; 12] = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/// Radians to degrees, `x * RAD2DEG` (`torch.rad2deg`).
#[inline]
pub fn rad2deg(x: f64) -> f64 {
    x * RAD2DEG
}

/// Degrees to radians, `x * DEG2RAD` (`torch.deg2rad`).
#[inline]
pub fn deg2rad(x: f64) -> f64 {
    x * DEG2RAD
}

/// `torch.remainder` / Python `%` for floats (result has the sign of `b`).
#[inline]
pub fn remainder(a: f64, b: f64) -> f64 {
    let mut m = a % b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        m += b;
    }
    m
}

/// Order (log2) of a power-of-two `nside`.
pub fn order_of(nside: i64) -> u32 {
    assert!(nside > 0 && nside & (nside - 1) == 0, "nside must be a power of two");
    nside.trailing_zeros()
}

/// Number of pixels of a full sphere at `nside`.
#[inline]
pub fn npix(nside: i64) -> i64 {
    12 * nside * nside
}

#[inline]
fn compress_bits(v: i64) -> i64 {
    let mut v = v & 0x5555_5555_5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v >> 8)) & 0x0000_FFFF_0000_FFFF;
    (v | (v >> 16)) & 0x0000_0000_FFFF_FFFF
}

#[inline]
fn spread_bits(v: i64) -> i64 {
    let mut v = v & 0x0000_0000_FFFF_FFFF;
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    (v | (v << 1)) & 0x5555_5555_5555_5555
}

/// NESTED pixel -> `(ix, iy, face)`.
#[inline]
pub fn nest2xyf(pix: i64, order: u32) -> (i64, i64, i64) {
    let face = pix >> (2 * order);
    let local = pix & ((1i64 << (2 * order)) - 1);
    (compress_bits(local), compress_bits(local >> 1), face)
}

/// `(ix, iy, face)` -> NESTED pixel.
#[inline]
pub fn xyf2nest(ix: i64, iy: i64, face: i64, order: u32) -> i64 {
    (face << (2 * order)) + spread_bits(ix) + (spread_bits(iy) << 1)
}

/// Colatitude and longitude (radians) of a pixel centre (healpix_base `pix2loc`).
pub fn xyf2ang(ix: i64, iy: i64, face: i64, nside: i64) -> (f64, f64) {
    let f = face as usize;
    let jr = JRLL[f] * nside - ix - iy - 1;
    let ns = nside as f64;
    let fact2 = 4.0 / (12.0 * ns * ns);
    let fact1 = 2.0 / (3.0 * ns);
    let north = jr < nside;
    let south = jr > 3 * nside;
    let nr = if north {
        jr
    } else if south {
        4 * nside - jr
    } else {
        nside
    };
    let kshift = if north || south { 0 } else { (jr - nside) & 1 };
    let nrf = nr as f64;
    let theta = if north || south {
        let tmp = nrf * nrf * fact2;
        let z_cap = 1.0 - tmp;
        let sth_cap = (tmp * (2.0 - tmp)).sqrt();
        let theta_cap = sth_cap.atan2(z_cap);
        if north {
            theta_cap
        } else {
            PI - theta_cap
        }
    } else {
        ((2.0 * ns - jr as f64) * fact1).acos()
    };
    let mut jp = (JPLL[f] * nr + ix - iy + 1 + kshift).div_euclid(2);
    if jp > 4 * nr {
        jp -= 4 * nr;
    }
    if jp < 1 {
        jp += 4 * nr;
    }
    let phi = (jp as f64 - (kshift as f64 + 1.0) * 0.5) * (0.5 * PI / nrf);
    (theta, phi)
}

/// Colatitude and longitude of a NESTED pixel centre.
#[inline]
pub fn pix2ang_nest(pix: i64, nside: i64) -> (f64, f64) {
    let (ix, iy, face) = nest2xyf(pix, order_of(nside));
    xyf2ang(ix, iy, face, nside)
}

/// Longitude and latitude (radians) of a NESTED pixel centre, rounded
/// through degrees like the healpix-geo output used by the library
/// (`TorchGeometry.centres_lonlat`).
#[inline]
pub fn centre_lonlat(pix: i64, nside: i64) -> (f64, f64) {
    let (theta, phi) = pix2ang_nest(pix, nside);
    (deg2rad(rad2deg(phi)), deg2rad(90.0 - rad2deg(theta)))
}

fn isqrt(v: i64) -> i64 {
    let mut r = (v as f64).sqrt() as i64;
    if r * r > v {
        r -= 1;
    }
    if (r + 1) * (r + 1) <= v {
        r += 1;
    }
    r
}

/// RING -> NESTED pixel number (healpix_base `ring2xyf` then `xyf2nest`).
pub fn ring2nest(pix: i64, order: u32) -> i64 {
    let nside = 1i64 << order;
    let npix = npix(nside);
    let ncap = 2 * nside * (nside - 1);
    let nl2 = 2 * nside;
    let (iring, iphi, kshift, nr, face);
    if pix < ncap {
        let ir = (1 + isqrt(1 + 2 * pix)) >> 1;
        let ip = pix + 1 - 2 * ir * (ir - 1);
        iring = ir;
        iphi = ip;
        kshift = 0;
        nr = ir;
        face = (ip - 1).div_euclid(ir.max(1));
    } else if pix < npix - ncap {
        let ip = pix - ncap;
        let tmp = ip >> (order + 2);
        let ir = tmp + nside;
        let ip2 = ip - tmp * 4 * nside + 1;
        let ire = ir - nside + 1;
        let irm = nl2 + 2 - ire;
        let ifm = (ip2 - ire.div_euclid(2) + nside - 1) >> order;
        let ifp = (ip2 - irm.div_euclid(2) + nside - 1) >> order;
        iring = ir;
        iphi = ip2;
        kshift = (ir + nside) & 1;
        nr = nside;
        face = if ifp == ifm {
            ifp | 4
        } else if ifp < ifm {
            ifp
        } else {
            ifm + 8
        };
    } else {
        let ip = npix - pix;
        let ir = (1 + isqrt(2 * ip - 1)) >> 1;
        let ip2 = 4 * ir + 1 - (ip - 2 * ir * (ir - 1));
        iring = 2 * nl2 - ir;
        iphi = ip2;
        kshift = 0;
        nr = ir;
        face = 8 + (ip2 - 1).div_euclid(ir.max(1));
    }
    let f = face as usize;
    let irt = iring - JRLL[f] * nside + 1;
    let mut ipt = 2 * iphi - JPLL[f] * nr - kshift - 1;
    if ipt >= nl2 {
        ipt -= 8 * nside;
    }
    let ix = (ipt - irt) >> 1;
    let iy = (-ipt - irt) >> 1;
    xyf2nest(ix, iy, face, order)
}

// ---------------------------------------------------------------------------
// Bilinear interpolation (RING scheme), port of T_Healpix_Base::get_interpol
// ---------------------------------------------------------------------------

fn ring_above(z: f64, nside: i64) -> i64 {
    let az = z.abs();
    let ns = nside as f64;
    if az <= 2.0 / 3.0 {
        (ns * (2.0 - 1.5 * z)).trunc() as i64
    } else {
        let iring = (ns * (3.0 * (1.0 - az)).sqrt()).trunc() as i64;
        if z > 0.0 {
            iring
        } else {
            4 * nside - iring - 1
        }
    }
}

/// `(startpix, ringpix, theta, shifted)` of a RING ring.
fn ring_info(ring: i64, nside: i64) -> (i64, i64, f64, bool) {
    let ncap = 2 * nside * (nside - 1);
    let npix = npix(nside);
    let ns = nside as f64;
    let fact1 = 2.0 / (3.0 * ns);
    let fact2 = 1.0 / (3.0 * ns * ns);
    let northring = if ring > 2 * nside { 4 * nside - ring } else { ring };
    let cap = northring < nside;
    let (mut theta, ringpix, shifted, mut startpix);
    if cap {
        let nrf = northring as f64;
        let tmp = nrf * nrf * fact2;
        theta = (tmp * (2.0 - tmp)).sqrt().atan2(1.0 - tmp);
        ringpix = 4 * northring;
        shifted = true;
        startpix = 2 * northring * (northring - 1);
    } else {
        theta = ((2 * nside - northring) as f64 * fact1).acos();
        ringpix = 4 * nside;
        shifted = ((northring - nside) & 1) == 0;
        startpix = ncap + (northring - nside) * 4 * nside;
    }
    if northring != ring {
        theta = PI - theta;
        startpix = npix - startpix - ringpix;
    }
    (startpix, ringpix, theta, shifted)
}

/// Two neighbouring pixels of `ring` around `phi`, their weights and the ring colatitude.
fn ring_pair(phi: f64, ring: i64, nside: i64) -> (i64, i64, f64, f64, f64) {
    let (sp, nr, theta_ring, shifted) = ring_info(ring, nside);
    let dphi = (2.0 * PI) / nr as f64;
    let half = if shifted { 0.5 } else { 0.0 };
    let tmp = phi / dphi - half;
    let mut i1 = tmp.trunc() as i64;
    if tmp < 0.0 {
        i1 -= 1;
    }
    let w1 = (phi - (i1 as f64 + half) * dphi) / dphi;
    let mut i2 = i1 + 1;
    if i1 < 0 {
        i1 += nr;
    }
    if i2 >= nr {
        i2 -= nr;
    }
    (sp + i1, sp + i2, 1.0 - w1, w1, theta_ring)
}

/// Four RING pixels and weights of the direction `(theta, phi)`.
pub fn get_interpol_ring(theta: f64, phi: f64, nside: i64) -> ([i64; 4], [f64; 4]) {
    let npix = npix(nside);
    let z = theta.cos();
    let ir1 = ring_above(z, nside);
    let ir2 = ir1 + 1;
    let mut pix = [0i64; 4];
    let mut wgt = [0f64; 4];
    let (p0, p1, w0, w1, theta1) = ring_pair(phi, ir1.max(1), nside);
    if ir1 > 0 {
        pix[0] = p0;
        pix[1] = p1;
        wgt[0] = w0;
        wgt[1] = w1;
    }
    let (p2, p3, w2, w3, theta2) = ring_pair(phi, ir2.min(4 * nside - 1), nside);
    if ir2 < 4 * nside {
        pix[2] = p2;
        pix[3] = p3;
        wgt[2] = w2;
        wgt[3] = w3;
    }
    let north = ir1 == 0;
    let south = ir2 == 4 * nside && !north;
    if north {
        let wn = theta / theta2;
        let facn = (1.0 - wn) * 0.25;
        (
            [(pix[2] + 2) & 3, (pix[3] + 2) & 3, pix[2], pix[3]],
            [facn, facn, wgt[2] * wn + facn, wgt[3] * wn + facn],
        )
    } else if south {
        let ws = (theta - theta1) / (PI - theta1);
        let facs = ws * 0.25;
        (
            [pix[0], pix[1], ((pix[0] + 2) & 3) + npix - 4, ((pix[1] + 2) & 3) + npix - 4],
            [wgt[0] * (1.0 - ws) + facs, wgt[1] * (1.0 - ws) + facs, facs, facs],
        )
    } else {
        let wm = (theta - theta1) / (theta2 - theta1);
        (pix, [wgt[0] * (1.0 - wm), wgt[1] * (1.0 - wm), wgt[2] * wm, wgt[3] * wm])
    }
}

/// Euclidean norm with the summation order of `torch.linalg.vector_norm` on 3 values.
#[inline]
pub fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Four NESTED pixels and weights of a direction (sphere).
///
/// Follows `TorchGeometry.interp_weights_nested`: direction -> (theta, phi)
/// -> degrees -> radians again, so that the rounding path is the library's.
pub fn interp_weights_nested(v: [f64; 3], order: u32) -> ([i64; 4], [f64; 4]) {
    let n = norm3(v).max(1e-12);
    let vn = [v[0] / n, v[1] / n, v[2] / n];
    let theta = vn[2].clamp(-1.0, 1.0).acos();
    let phi = remainder(vn[1].atan2(vn[0]), 2.0 * PI);
    let lon_deg = rad2deg(phi);
    let lat_deg = 90.0 - rad2deg(theta);
    let theta2 = deg2rad(90.0 - lat_deg);
    let phi2 = remainder(deg2rad(lon_deg), 2.0 * PI);
    let (pix, w) = get_interpol_ring(theta2, phi2, 1i64 << order);
    (pix.map(|p| ring2nest(p, order)), w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nest_xyf_roundtrip() {
        for order in [0u32, 1, 3, 10, 20] {
            let nside = 1i64 << order;
            for pix in [0, 1, npix(nside) / 3, npix(nside) - 1] {
                let (ix, iy, f) = nest2xyf(pix, order);
                assert_eq!(xyf2nest(ix, iy, f, order), pix);
            }
        }
    }
}
