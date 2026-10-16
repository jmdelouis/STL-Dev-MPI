//! Milestone 1: the geometry must reproduce STL-Dev's `TorchGeometry`
//! (fixtures `fixtures/geometry`, see `tools/dump_reference.py`).

mod common;

use common::{cases, max_abs_diff};
use std::collections::BTreeMap;
use stl_core::conv::{convolution_forward, ConvGeometry, Kernel};
use stl_core::decimation::{smooth_decimation, DecimationParams};
use stl_core::healpix::{
    centre_lonlat, get_interpol_ring, interp_weights_nested, nest2xyf, order_of, ring2nest, xyf2ang,
};
use stl_core::neighbours::kth_neighbourhood;
use stl_core::stencil::{local_kernel_grid, rotation_matrices, Gauge};

/// Weights to 1e-12 (CLAUDE.md, validation 1).
const TOL: f64 = 1e-12;

#[test]
fn centres() {
    for case in cases("geometry", "centres") {
        let nside = case.int("nside");
        let order = order_of(nside);
        let ids = case.i64("ids");
        let (ix, iy, face) = (case.i64("ix"), case.i64("iy"), case.i64("face"));
        let (theta, phi) = (case.f64("theta"), case.f64("phi"));
        let (lon, lat) = (case.f64("lon"), case.f64("lat"));
        let mut err = [0f64; 4];
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(nest2xyf(id, order), (ix[i], iy[i], face[i]), "{} id {id}", case.name());
            let (t, p) = xyf2ang(ix[i], iy[i], face[i], nside);
            let (lo, la) = centre_lonlat(id, nside);
            for (e, d) in err.iter_mut().zip([t - theta[i], p - phi[i], lo - lon[i], la - lat[i]]) {
                *e = e.max(d.abs());
            }
        }
        assert!(err.iter().all(|&e| e <= 1e-14), "{}: {err:?}", case.name());
    }
}

#[test]
fn ring_to_nest() {
    for case in cases("geometry", "ring2nest") {
        let order = case.int("order") as u32;
        let (ring, nest) = (case.i64("ring"), case.i64("nest"));
        for (&r, &n) in ring.iter().zip(&nest) {
            assert_eq!(ring2nest(r, order), n, "{} ring {r}", case.name());
        }
    }
}

/// Bilinear weights as an operator: pixel -> summed weight.  Directions on
/// a pixel centre or a ring boundary may put a unit (or zero) weight in
/// another slot depending on rounding while the operator is the same.
fn as_operator(pix: &[i64], w: &[f64]) -> BTreeMap<i64, f64> {
    let mut m = BTreeMap::new();
    for (&p, &v) in pix.iter().zip(w) {
        *m.entry(p).or_insert(0.0) += v;
    }
    m
}

/// Largest weight difference over the union of the pixels of two operators.
fn same_operator(a: &BTreeMap<i64, f64>, b: &BTreeMap<i64, f64>, tol: f64) -> bool {
    let get = |m: &BTreeMap<i64, f64>, p: &i64| m.get(p).copied().unwrap_or(0.0);
    a.keys().chain(b.keys()).all(|p| (get(a, p) - get(b, p)).abs() <= tol)
}

/// Interpolation weights are linear in the direction over one pixel, so a
/// one-ulp difference of `acos`/`atan2` (glibc here, SLEEF in torch) moves
/// them by about `4e-16 * nside`: 1e-12, relaxed above nside 100.
fn weight_tol(nside: i64) -> f64 {
    TOL.max(1e-14 * nside as f64)
}

/// Rounding noise of a convolution output: the direction's colatitude goes
/// through `acos(z)`, whose ulp error is amplified by `1 / sin(theta)`, and
/// the weights are linear in it over one pixel (`~nside`); the output sums
/// `P` taps of O(1) data and kernel.  The worst fixture case (nside 1024,
/// equatorial corner of face 4, 1.7e-11) is 0.45 of it; a porting error
/// shows up as orders of magnitude.
fn conv_noise(id: i64, nside: i64) -> f64 {
    let (theta, _) = stl_core::healpix::pix2ang_nest(id, nside);
    let s = theta.sin().max(1.0 / nside as f64);
    TOL.max(f64::EPSILON * 128.0 * nside as f64 / s)
}

#[test]
fn interpolation() {
    for case in cases("geometry", "interp") {
        let nside = case.int("nside");
        let order = order_of(nside);
        let vecs = case.f64("vecs");
        let (theta, phi) = (case.f64("theta"), case.f64("phi"));
        let (pr, wr) = (case.i64("pix_ring"), case.f64("w_ring"));
        let (pn, wn) = (case.i64("pix_nest"), case.f64("w_nest"));
        let tol = weight_tol(nside);
        let (mut slot_diff, mut bad) = (0, Vec::new());
        for i in 0..theta.len() {
            // Same (theta, phi): only the ring arithmetic is tested.
            let (p, w) = get_interpol_ring(theta[i], phi[i], nside);
            let (rp, rw) = (&pr[4 * i..4 * i + 4], &wr[4 * i..4 * i + 4]);
            if p[..] != *rp || max_abs_diff(&w, rw) > TOL {
                slot_diff += 1;
                if !same_operator(&as_operator(&p, &w), &as_operator(rp, rw), TOL) {
                    bad.push(format!("ring {i}: {p:?} {w:?} vs {rp:?} {rw:?}"));
                }
            }
            let v = [vecs[3 * i], vecs[3 * i + 1], vecs[3 * i + 2]];
            let (p, w) = interp_weights_nested(v, order);
            let (rp, rw) = (&pn[4 * i..4 * i + 4], &wn[4 * i..4 * i + 4]);
            if p[..] != *rp || max_abs_diff(&w, rw) > TOL {
                slot_diff += 1;
                if !same_operator(&as_operator(&p, &w), &as_operator(rp, rw), tol) {
                    bad.push(format!("nest {i} {v:?}: {p:?} {w:?} vs {rp:?} {rw:?}"));
                }
            }
        }
        assert!(bad.is_empty(), "{}: {} mismatches\n{}", case.name(), bad.len(), bad.join("\n"));
        if slot_diff > 0 {
            eprintln!("{}: {slot_diff} directions with permuted slots (same operator)", case.name());
        }
    }
}

#[test]
fn rotations() {
    for case in cases("geometry", "rotation") {
        let g = case.int("G") as usize;
        let gauge = Gauge::parse(case.str("gauge")).unwrap();
        let (theta, phi, r) = (case.f64("theta"), case.f64("phi"), case.f64("R"));
        let mut err = 0f64;
        for k in 0..theta.len() {
            for (gi, m) in rotation_matrices(theta[k], phi[k], g, gauge).iter().enumerate() {
                let flat: Vec<f64> = m.iter().flatten().copied().collect();
                err = err.max(max_abs_diff(&flat, &r[(k * g + gi) * 9..][..9]));
            }
        }
        assert!(err <= 1e-14, "{}: {err}", case.name());
    }
}

#[test]
fn kernel_grid() {
    for case in cases("geometry", "kernel_grid") {
        let vec = local_kernel_grid(case.int("kernel_sz") as usize, case.int("nside"));
        let flat: Vec<f64> = vec.iter().flatten().copied().collect();
        let err = max_abs_diff(&flat, &case.f64("vec"));
        assert!(err <= 1e-15, "{}: {err}", case.name());
    }
}

#[test]
fn convolution_geometry_and_forward() {
    for case in cases("geometry", "conv") {
        let nside = case.int("nside");
        let ksz = case.int("kernel_sz") as usize;
        let g = case.int("G") as usize;
        let gauge = Gauge::parse(case.str("gauge")).unwrap();
        let ids = case.i64("ids");
        let geo = ConvGeometry::new(&ids, nside, ksz, g, gauge);
        let complete = case.bool("complete");
        let mismatched = (0..ids.len()).filter(|&k| geo.complete[k] != complete[k]).count();
        assert_eq!(mismatched, 0, "{}: complete flags differ on {mismatched} rows", case.name());

        let p = ksz * ksz;
        let ker = Kernel::shared(&case.f64("weights"), g, 1, 2, p);
        assert_eq!(max_abs_diff(&ker.w, &case.f64("weight_f32")), 0.0, "{}: kernel", case.name());

        if case.has("pos_safe") {
            // Raw binding, as operators per (gauge, pixel, tap) on complete rows.
            let (pos, w) = (case.i64("pos_safe"), case.f64("w_norm"));
            for k in (0..ids.len()).filter(|&k| complete[k]) {
                for gi in 0..g {
                    for t in 0..p {
                        let at: Vec<usize> = (0..4).map(|n| geo.at(n, gi, k, t)).collect();
                        let mine: Vec<i64> = at.iter().map(|&a| geo.pos[a] as i64).collect();
                        let mw: Vec<f64> = at.iter().map(|&a| geo.w[a]).collect();
                        let rp: Vec<i64> = at.iter().map(|&a| pos[a]).collect();
                        let rw: Vec<f64> = at.iter().map(|&a| w[a]).collect();
                        assert!(
                            same_operator(&as_operator(&mine, &mw), &as_operator(&rp, &rw), weight_tol(nside)),
                            "{}: k {k} g {gi} tap {t}: {mine:?} {mw:?} vs {rp:?} {rw:?}",
                            case.name()
                        );
                    }
                }
            }
        }

        let (x, y) = (case.f64("x"), case.f64("y"));
        let mine = convolution_forward(&geo, &ker, &x, 1);
        let kk = ids.len();
        let (mut err, mut ratio) = (0f64, 0f64);
        for row in 0..g * 2 {
            for k in (0..kk).filter(|&k| complete[k]) {
                let e = (mine[row * kk + k] - y[row * kk + k]).abs();
                err = err.max(e);
                ratio = ratio.max(e / conv_noise(ids[k], nside));
            }
        }
        eprintln!("{}: convolution max error {err:.2e}, max error / noise {ratio:.2}", case.name());
        assert!(ratio <= 1.0, "{}: convolution max error {err:e} ({ratio} x noise)", case.name());
    }
}

#[test]
fn decimation_weights() {
    for case in cases("geometry", "down") {
        let nside = case.int("nside");
        let ids = case.i64("ids");
        let (coarse, csr) = smooth_decimation(&ids, nside, &DecimationParams::default());
        assert_eq!(coarse, case.i64("coarse_ids"), "{}: coarse ids", case.name());
        let (rows, cols, vals) = (case.i64("rows"), case.i64("cols"), case.f64("values"));
        let mut mine = Vec::new();
        for r in 0..csr.n_rows {
            let (c, v) = csr.row(r);
            mine.extend(c.iter().zip(v).map(|(&c, &v)| (r as i64, c as i64, v)));
        }
        assert_eq!(mine.len(), vals.len(), "{}: number of entries", case.name());
        let mut err = 0f64;
        for (i, &(r, c, v)) in mine.iter().enumerate() {
            assert_eq!((r, c), (rows[i], cols[i]), "{}: entry {i}", case.name());
            err = err.max((v - vals[i]).abs());
        }
        assert!(err <= TOL, "{}: weights max error {err:e}", case.name());
        let t = csr.transpose();
        assert_eq!(t.transpose(), csr);
    }
}

#[test]
fn neighbourhoods() {
    for case in cases("geometry", "neighbours") {
        let nside = case.int("nside");
        let k = case.int("k") as u32;
        let ids = case.i64("ids");
        let nb = case.i64("neighbours");
        let width = case.shape("neighbours")[1];
        for (i, &id) in ids.iter().enumerate() {
            let expected: Vec<i64> = nb[i * width..(i + 1) * width].iter().copied().filter(|&v| v >= 0).collect();
            assert_eq!(kth_neighbourhood(id, nside, k), expected, "{} id {id}", case.name());
        }
    }
}
