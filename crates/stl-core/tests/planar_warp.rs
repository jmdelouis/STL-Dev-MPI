//! Lagrangian warp of a periodic plane (`stl_core::planar::warp`): identity,
//! divergence-free stream part, VJPs against central finite differences,
//! rank re-projection.

use stl_core::planar::spectrum::Fft2;
use stl_core::planar::warp::{reproject, Flow, Warp};

fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn zero_displacement_is_identity() {
    let (h, w) = (16, 24);
    let x0 = noise(h * w, 1);
    for flow in [Flow::Stream, Flow::Potential, Flow::Both] {
        let wp = Warp::new(h, w, flow, 2.0, 1.0, 1.5);
        let (x, _) = wp.forward(&x0, &vec![0.0; wp.len()]);
        assert!(x.iter().zip(&x0).all(|(a, b)| (a - b).abs() < 1e-15), "{flow:?}");
    }
}

#[test]
fn stream_part_is_divergence_free_and_rms_is_set() {
    let (h, w) = (32, 32);
    let wp = Warp::new(h, w, Flow::Stream, 2.0, 1.0, 1.5);
    let th = noise(wp.len(), 2).iter().map(|v| v * 3f64.sqrt()).collect::<Vec<_>>(); // unit variance
    let (ur, uc) = wp.displacement(&th);
    let fft = Fft2::new(h, w);
    let d = fft.derivatives(&ur)[0].iter().zip(&fft.derivatives(&uc)[1]).map(|(a, b)| a + b).collect::<Vec<_>>();
    let umax = ur.iter().chain(&uc).fold(0f64, |m, v| m.max(v.abs()));
    assert!(d.iter().all(|v| v.abs() < 1e-12 * umax.max(1.0)), "divergence");
    let rms = ((dot(&ur, &ur) + dot(&uc, &uc)) / (h * w) as f64).sqrt();
    assert!(rms > 0.8 && rms < 2.2, "rms {rms}");
}

#[test]
fn displacement_vjp_is_the_adjoint() {
    let (h, w) = (12, 20);
    for flow in [Flow::Stream, Flow::Potential, Flow::Both] {
        let wp = Warp::new(h, w, flow, 1.5, 0.5, 1.0);
        let th = noise(wp.len(), 3);
        let (gr, gc) = (noise(h * w, 4), noise(h * w, 5));
        let (ur, uc) = wp.displacement(&th);
        let lhs = dot(&ur, &gr) + dot(&uc, &gc);
        let rhs = dot(&th, &wp.displacement_vjp(&gr, &gc));
        assert!((lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0), "{flow:?}: {lhs} vs {rhs}");
    }
}

#[test]
fn warp_vjp_matches_finite_differences() {
    let (h, w) = (16, 16);
    let x0 = noise(h * w, 6);
    for flow in [Flow::Stream, Flow::Both] {
        let wp = Warp::new(h, w, flow, 2.0, 1.0, 1.3);
        let th = noise(wp.len(), 7);
        let g = noise(h * w, 8);
        let (_, st) = wp.forward(&x0, &th);
        let grad = wp.vjp(&st, &g);
        let dir = noise(wp.len(), 9);
        let e = 1e-6;
        let f = |s: f64| {
            let t: Vec<f64> = th.iter().zip(&dir).map(|(a, b)| a + s * b).collect();
            dot(&wp.forward(&x0, &t).0, &g)
        };
        let fd = (f(e) - f(-e)) / (2.0 * e);
        let an = dot(&grad, &dir);
        // Bilinear interpolation is C0 across pixel edges: a crossing in the
        // +-e interval costs O(e) relative error at most.
        assert!((fd - an).abs() < 1e-5 * an.abs().max(1.0), "{flow:?}: fd {fd} vs {an}");
    }
}

#[test]
fn reprojection_takes_the_pool_quantiles_in_rank_order() {
    let x0 = noise(1000, 10);
    let mut pool: Vec<f64> = noise(4000, 11).iter().map(|v| v.powi(3)).collect();
    pool.sort_by(f64::total_cmp);
    let mut x = x0.clone();
    reproject(&mut x, &pool);
    for i in 0..x.len() {
        for j in 0..x.len() {
            if x0[i] < x0[j] {
                assert!(x[i] <= x[j]);
            }
        }
    }
    let mut s = x.clone();
    s.sort_by(f64::total_cmp);
    for (i, v) in s.iter().enumerate() {
        assert_eq!(*v, pool[(i * 4 + 2).min(3999)]);
    }
}
