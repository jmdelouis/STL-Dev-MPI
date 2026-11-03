//! Tangent-plane stencil and its rotation to every pixel and gauge
//! (ports of `TorchGeometry.local_kernel_grid` and `rotation_matrices`,
//! themselves `_build_rotation_matrices` of healpix-analyse).

use std::f64::consts::PI;

/// 3 x 3 matrix, row-major (`m[i][j]`, row `i`).
pub type Mat3 = [[f64; 3]; 3];

/// Orientation convention of the rotated stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gauge {
    /// Orientation measured from the local meridian, mirrored in the south.
    Cosmo,
    /// Orientation measured from the local meridian everywhere.
    Phi,
}

impl Gauge {
    /// Gauge of its STL-Dev name (`cosmo`, `phi`), if any.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cosmo" => Some(Gauge::Cosmo),
            "phi" => Some(Gauge::Phi),
            _ => None,
        }
    }
}

/// Unit vectors `[P, 3]` of the `kernel_sz x kernel_sz` stencil around the
/// North Pole, spaced by the mean pixel size; tap `p = i * kernel_sz + j`
/// sits at `(x, y) = (j - kernel_sz / 2, i - kernel_sz / 2)`.
pub fn local_kernel_grid(kernel_sz: usize, nside: i64) -> Vec<[f64; 3]> {
    let half = (kernel_sz / 2) as i64;
    let ns = nside as f64;
    let alpha_pix = (4.0 * PI / (12.0 * ns * ns)).sqrt();
    let mut out = Vec::with_capacity(kernel_sz * kernel_sz);
    for i in 0..kernel_sz as i64 {
        for j in 0..kernel_sz as i64 {
            let (x, y) = (j - half, i - half);
            let dtheta = ((x * x + y * y) as f64).sqrt() * alpha_pix;
            let dphi = (y as f64).atan2(x as f64);
            out.push([dtheta.sin() * dphi.cos(), dtheta.sin() * dphi.sin(), dtheta.cos()]);
        }
    }
    out
}

/// Rotations `[G]` taking the North-Pole stencil to the pixel `(theta, phi)`
/// for each of the `G` gauges (orientations `g * pi / G`).
pub fn rotation_matrices(theta: f64, phi: f64, g_count: usize, gauge: Gauge) -> Vec<Mat3> {
    let (ct, st) = (theta.cos(), theta.sin());
    let (cp, sp) = (phi.cos(), phi.sin());
    let base: Mat3 = [
        [cp * ct, -sp, cp * st],
        [sp * ct, cp, sp * st],
        [-st, 0.0, ct],
    ];
    let n0 = [base[0][2], base[1][2], base[2][2]];
    let nn = crate::healpix::norm3(n0).max(1e-12);
    let n = [n0[0] / nn, n0[1] / nn, n0[2] / nn];
    let (alpha_base, sign_g) = match gauge {
        Gauge::Cosmo => {
            if theta > PI / 2.0 {
                (phi, -1.0)
            } else {
                (-phi, 1.0)
            }
        }
        Gauge::Phi => (0.0, 1.0),
    };
    let skew: Mat3 = [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]];
    let step = PI / g_count as f64;
    (0..g_count)
        .map(|g| {
            let alpha = alpha_base + sign_g * (g as f64 * step);
            let (ca, sa) = (alpha.cos(), alpha.sin());
            let mut rg = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let eye = if i == j { 1.0 } else { 0.0 };
                    rg[i][j] = eye * ca + skew[i][j] * sa + n[i] * n[j] * (1.0 - ca);
                }
            }
            matmul(&rg, &base)
        })
        .collect()
}

/// Matrix product `a b`.
#[inline]
pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// Matrix-vector product `r v`.
#[inline]
pub fn apply(r: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}
