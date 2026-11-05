//! Lagrangian parameterisation of a periodic plane (`docs/lagrangian_synthesis.md`,
//! section 3): `x(p) = x0(p + u(p))`, the pull-back of an anchor plane `x0`
//! by a smooth displacement `u`, with bilinear interpolation (periodic).
//!
//! The displacement is built in Fourier from white variables:
//! a stream function `omega = K theta_omega` (incompressible part,
//! `u = grad^perp omega`: `u_r = -d_c omega`, `u_c = d_r omega`) and, if
//! asked, a potential `psi = K theta_psi` (compressive part, `u = grad psi`).
//! The multiplier `K(k) = A exp(-(|k| s)^2 / 2) |k|^-(slope + 1)` (zero at
//! `k = 0`) gives a displacement spectrum `|u(k)| ~ exp(-(|k| s)^2 / 2)
//! |k|^-slope`; `A` makes the rms displacement of unit white variables equal
//! to `rms` pixels for each part.  The spectral derivatives are zero at the
//! Nyquist frequencies (`Fft2::multiply`), so the discrete divergence of the
//! incompressible part is exactly zero.  Rows are axis `r`, columns `c`, in
//! pixel units.

use super::spectrum::{integer_frequency, Fft2};
use num_complex::Complex64;

type C64 = Complex64;

/// Which parts of the displacement are variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// `u = grad^perp omega`: one variable per pixel, area preserving to first order.
    Stream,
    /// `u = grad psi`: one variable per pixel, packs and spreads.
    Potential,
    /// Both: two variables per pixel, `[theta_omega, theta_psi]`.
    Both,
}

impl Flow {
    /// Variables per pixel: 2 for [`Flow::Both`], else 1.
    pub fn parts(self) -> usize {
        if self == Flow::Both {
            2
        } else {
            1
        }
    }
}

/// The displacement operator of `h x w` periodic planes: white variables
/// to `(u_r, u_c)` through the multiplier `K`, and the warp of an anchor plane.
pub struct Warp {
    /// Rows of a plane.
    pub h: usize,
    /// Columns of a plane.
    pub w: usize,
    /// Which parts of the displacement are variables.
    pub flow: Flow,
    fft: Fft2,
    /// `K(k)` on the unshifted frequency grid `[H, W]`.
    k: Vec<f64>,
}

/// Displacement `(u_r, u_c)` and the interpolation state of one plane.
pub struct WarpState {
    /// `dx/dq_r`, `dx/dq_c` at every pixel (derivatives of the bilinear interpolant).
    pub dr: Vec<f64>,
    /// `dx/dq_c` at every pixel.
    pub dc: Vec<f64>,
    /// Row displacement `u_r` (pixels).
    pub ur: Vec<f64>,
    /// Column displacement `u_c` (pixels).
    pub uc: Vec<f64>,
}

fn ik(k: i64, n: usize) -> C64 {
    if 2 * k.unsigned_abs() as usize == n {
        C64::default()
    } else {
        C64::new(0.0, 2.0 * std::f64::consts::PI * k as f64 / n as f64)
    }
}

impl Warp {
    /// `scale` (pixels) and `slope` shape the displacement spectrum, `rms`
    /// its amplitude for unit white variables.
    pub fn new(h: usize, w: usize, flow: Flow, scale: f64, slope: f64, rms: f64) -> Self {
        let tau = 2.0 * std::f64::consts::PI;
        let mut k = vec![0.0; h * w];
        let mut power = 0.0;
        for r in 0..h {
            let kr = tau * integer_frequency(r, h) as f64 / h as f64;
            for c in 0..w {
                let kc = tau * integer_frequency(c, w) as f64 / w as f64;
                let m2 = kr * kr + kc * kc;
                if m2 > 0.0 {
                    let m = m2.sqrt();
                    let v = (-(m * scale).powi(2) / 2.0).exp() * m.powf(-(slope + 1.0));
                    k[r * w + c] = v;
                    power += v * v * m2;
                }
            }
        }
        let a = rms / (power / (h * w) as f64).sqrt();
        k.iter_mut().for_each(|v| *v *= a);
        Warp { h, w, flow, fft: Fft2::new(h, w), k }
    }

    /// Variables per image.
    pub fn len(&self) -> usize {
        self.flow.parts() * self.h * self.w
    }

    /// No variable (an empty plane).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn filter(&self, x: &[f64], m: impl Fn(i64, i64) -> C64) -> Vec<f64> {
        let (h, w) = (self.h, self.w);
        // Fft2::multiply passes integer frequencies; recover the grid index for K.
        self.fft.multiply(x, |a, b| {
            let r = a.rem_euclid(h as i64) as usize;
            let c = b.rem_euclid(w as i64) as usize;
            m(a, b) * self.k[r * w + c]
        })
    }

    /// `(u_r, u_c)` of the variables `theta` of one image.
    pub fn displacement(&self, theta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (h, w, n) = (self.h, self.w, self.h * self.w);
        assert_eq!(theta.len(), self.len());
        let (mut ur, mut uc) = (vec![0.0; n], vec![0.0; n]);
        let mut add = |t: &[f64], mr: &dyn Fn(i64, i64) -> C64, mc: &dyn Fn(i64, i64) -> C64| {
            ur.iter_mut().zip(self.filter(t, mr)).for_each(|(a, v)| *a += v);
            uc.iter_mut().zip(self.filter(t, mc)).for_each(|(a, v)| *a += v);
        };
        let mut off = 0;
        if self.flow != Flow::Potential {
            add(&theta[..n], &|_, b| -ik(b, w), &|a, _| ik(a, h));
            off = n;
        }
        if self.flow != Flow::Stream {
            add(&theta[off..off + n], &|a, _| ik(a, h), &|_, b| ik(b, w));
        }
        (ur, uc)
    }

    /// Adjoint of [`Warp::displacement`]: cotangent of `theta` from those of `(u_r, u_c)`.
    pub fn displacement_vjp(&self, gr: &[f64], gc: &[f64]) -> Vec<f64> {
        let (h, w) = (self.h, self.w);
        let mut out = Vec::with_capacity(self.len());
        let sum = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
        // conj(i k) = -i k.
        if self.flow != Flow::Potential {
            out.extend(sum(self.filter(gr, |_, b| ik(b, w)), self.filter(gc, |a, _| -ik(a, h))));
        }
        if self.flow != Flow::Stream {
            out.extend(sum(self.filter(gr, |a, _| -ik(a, h)), self.filter(gc, |_, b| -ik(b, w))));
        }
        out
    }

    /// `x(p) = x0(p + u(p))` (periodic bilinear) and the state of its VJP.
    pub fn forward(&self, x0: &[f64], theta: &[f64]) -> (Vec<f64>, WarpState) {
        let (h, w) = (self.h, self.w);
        let (ur, uc) = self.displacement(theta);
        let mut x = vec![0.0; h * w];
        let (mut dr, mut dc) = (vec![0.0; h * w], vec![0.0; h * w]);
        for r in 0..h {
            for c in 0..w {
                let p = r * w + c;
                let (qr, qc) = (r as f64 + ur[p], c as f64 + uc[p]);
                let (fr, fc) = (qr.floor(), qc.floor());
                let (t, s) = (qr - fr, qc - fc);
                let i0 = (fr as i64).rem_euclid(h as i64) as usize;
                let j0 = (fc as i64).rem_euclid(w as i64) as usize;
                let (i1, j1) = ((i0 + 1) % h, (j0 + 1) % w);
                let (a, b, cc, d) = (x0[i0 * w + j0], x0[i0 * w + j1], x0[i1 * w + j0], x0[i1 * w + j1]);
                x[p] = (1.0 - t) * ((1.0 - s) * a + s * b) + t * ((1.0 - s) * cc + s * d);
                dr[p] = (1.0 - s) * (cc - a) + s * (d - b);
                dc[p] = (1.0 - t) * (b - a) + t * (d - cc);
            }
        }
        (x, WarpState { dr, dc, ur, uc })
    }

    /// Cotangent of `theta` from the cotangent `g` of the warped plane.
    pub fn vjp(&self, st: &WarpState, g: &[f64]) -> Vec<f64> {
        let gr: Vec<f64> = g.iter().zip(&st.dr).map(|(a, b)| a * b).collect();
        let gc: Vec<f64> = g.iter().zip(&st.dc).map(|(a, b)| a * b).collect();
        self.displacement_vjp(&gr, &gc)
    }
}

/// Rank matching of `x` onto the sorted `pool` (Brenier's monotone
/// rearrangement on the line): the `i`-th smallest value of `x` becomes the
/// quantile `(i + 1/2) / n` of the pool.  Ties keep their index order.
pub fn reproject(x: &mut [f64], sorted_pool: &[f64]) {
    let n = x.len();
    let m = sorted_pool.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    for (rank, &i) in idx.iter().enumerate() {
        let q = ((2 * rank + 1) * m) / (2 * n);
        x[i] = sorted_pool[q.min(m - 1)];
    }
}
