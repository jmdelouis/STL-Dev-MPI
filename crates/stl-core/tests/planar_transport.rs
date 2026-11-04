//! `stl_core::planar::transport`: bicubic transport, its VJP and
//! derivative in `u`, and the multi-scale smoother.

use stl_core::planar::transport::{bands, bicubic, bicubic_du, bicubic_vjp, down, down_t, gradient, Smoother};

const H: usize = 16;
const W: usize = 24;

fn field(seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed;
    (0..H * W)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            scale * (((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn zero_and_integer_displacements_are_exact() {
    let x = field(1, 2.0);
    let z = vec![0.0; H * W];
    assert_eq!(bicubic(&x, H, W, &z, &z), x);
    let (two, minus) = (vec![2.0; H * W], vec![-3.0; H * W]);
    let y = bicubic(&x, H, W, &two, &minus);
    for i in 0..H {
        for j in 0..W {
            assert!((y[i * W + j] - x[((i + 2) % H) * W + (j + W - 3) % W]).abs() < 1e-14);
        }
    }
}

#[test]
fn vjp_is_the_adjoint() {
    let (x, g) = (field(2, 1.0), field(3, 1.0));
    let (ur, uc) = (field(4, 5.0), field(5, 5.0));
    let lhs = dot(&bicubic(&x, H, W, &ur, &uc), &g);
    let rhs = dot(&x, &bicubic_vjp(&g, H, W, &ur, &uc));
    assert!((lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0), "{lhs} {rhs}");
}

#[test]
fn derivative_in_u_matches_finite_differences() {
    let x = field(6, 1.0);
    let (ur, uc) = (field(7, 3.0), field(8, 3.0));
    let (dr, dc) = bicubic_du(&x, H, W, &ur, &uc);
    let e = 1e-6;
    let shift = |v: &[f64], d: f64| v.iter().map(|a| a + d).collect::<Vec<_>>();
    let fr: Vec<f64> = bicubic(&x, H, W, &shift(&ur, e), &uc).iter().zip(bicubic(&x, H, W, &shift(&ur, -e), &uc)).map(|(a, b)| (a - b) / (2.0 * e)).collect();
    let fc: Vec<f64> = bicubic(&x, H, W, &ur, &shift(&uc, e)).iter().zip(bicubic(&x, H, W, &ur, &shift(&uc, -e))).map(|(a, b)| (a - b) / (2.0 * e)).collect();
    for p in 0..H * W {
        assert!((dr[p] - fr[p]).abs() < 1e-7 && (dc[p] - fc[p]).abs() < 1e-7, "pixel {p}");
    }
    // At u = 0 the derivative is the centred gradient.
    let z = vec![0.0; H * W];
    let (d0r, d0c) = bicubic_du(&x, H, W, &z, &z);
    let (gr, gc) = gradient(&x, H, W);
    for p in 0..H * W {
        assert!((d0r[p] - gr[p]).abs() < 1e-14 && (d0c[p] - gc[p]).abs() < 1e-14);
    }
}

#[test]
fn smoother_is_symmetric_positive_and_keeps_constants() {
    let (a, b) = (field(9, 1.0), field(10, 1.0));
    let lo = field(11, 1.0)[..H * W / 4].to_vec();
    assert!((dot(&down(&a, H, W), &lo) - dot(&a, &down_t(&lo, H, W))).abs() < 1e-13);
    let k = Smoother::new(H, W, 4, -0.5);
    assert_eq!(k.weights.len(), 3);
    let (ka, kb) = (k.apply(&a), k.apply(&b));
    assert!((dot(&ka, &b) - dot(&a, &kb)).abs() < 1e-12);
    assert!(dot(&ka, &a) > 0.0);
    let total: f64 = k.weights.iter().sum();
    for v in k.apply(&vec![1.0; H * W]) {
        assert!((v - total).abs() < 1e-12);
    }
}

#[test]
fn bands_sum_to_the_image() {
    let x = field(12, 1.0);
    let b = bands(&x, H, W, 2);
    assert_eq!(b.len(), 3);
    for p in 0..H * W {
        let s: f64 = b.iter().map(|v| v[p]).sum();
        assert!((s - x[p]).abs() < 1e-13);
    }
    // The coarsest band of a constant is the constant.
    let c = bands(&vec![3.0; H * W], H, W, 2);
    assert!(c[2].iter().all(|v| (v - 3.0).abs() < 1e-13) && c[0].iter().all(|v| v.abs() < 1e-13));
}
