//! Per-member features of the physical map and their map cotangents: the
//! 16 log bands of its C_l (`lp_windows`), its cloud-in-cell PDF
//! (`stl_port/pdf.py`) and its Minkowski functionals with harmonic
//! derivatives (`stl_port/minkowski.py`).  Full-sky NESTED maps.

use super::lut::linspace;
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};
use stl_core::planar::spectrum::Fft2;
use stl_core::sht::Sht;

/// Host threads: `STL_THREADS`, else the CPUs available to the process.
pub fn host_threads() -> usize {
    std::env::var("STL_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1)
}

/// `f(chunk)` over the chunks `[lo, hi)` of `0..n` on the host threads,
/// results in chunk order (sums independent of the number of threads).
pub fn chunked<T: Send>(n: usize, chunk: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let chunks = n.div_ceil(chunk).max(1);
    let t = host_threads().min(chunks);
    let run = |c: usize| f(c * chunk, ((c + 1) * chunk).min(n));
    if t == 1 {
        return (0..chunks).map(run).collect();
    }
    let next = AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..t)
            .map(|_| {
                s.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= chunks {
                            break;
                        }
                        v.push((c, run(c)));
                    }
                    v
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    out.sort_by_key(|(c, _)| *c);
    out.into_iter().map(|(_, v)| v).collect()
}

/// `f(lo, part)` over contiguous parts of `out` (`part = out[lo..]`), one
/// per host thread: for element-wise passes (no sums).
pub fn par_chunks(out: &mut [f64], f: impl Fn(usize, &mut [f64]) + Sync) {
    let per = out.len().div_ceil(host_threads().min(out.len().div_ceil(CHUNK)).max(1)).max(1);
    std::thread::scope(|s| {
        for (k, part) in out.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || f(k * per, part));
        }
    });
}

/// [`par_chunks`] over the same parts of two slices of one length.
pub fn par_zip(a: &mut [f64], b: &mut [f64], f: impl Fn(usize, &mut [f64], &mut [f64]) + Sync) {
    assert_eq!(a.len(), b.len());
    let per = a.len().div_ceil(host_threads().min(a.len().div_ceil(CHUNK)).max(1)).max(1);
    std::thread::scope(|s| {
        for (k, (pa, pb)) in a.chunks_mut(per).zip(b.chunks_mut(per)).enumerate() {
            let f = &f;
            s.spawn(move || f(k * per, pa, pb));
        }
    });
}

const CHUNK: usize = 1 << 14;

/// The 16 unnormalised cosine-squared windows of the C_l bands and their
/// centres (`lp_windows`): centres log-spaced from 2 to `lmax`, evaluated
/// at `log(l + 0.5)`, zero below l = 2.
pub fn lp_windows(lmax: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let nb = 16;
    let logc = linspace(2f64.ln(), (lmax as f64).ln(), nb, true);
    let delta = (logc[nb - 1] - logc[0]) / (nb - 1) as f64;
    let windows = logc
        .iter()
        .map(|c| {
            (0..=lmax)
                .map(|l| {
                    let x = (((l as f64) + 0.5).ln() - c) / delta;
                    if l >= 2 && x.abs() <= 1.0 {
                        (std::f64::consts::PI * x / 2.0).cos().powi(2)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    (windows, logc.iter().map(|c| c.exp()).collect())
}

/// Band powers `W C_l` from the `C_l`.
pub fn window_sums(windows: &[Vec<f64>], cl: &[f64]) -> Vec<f64> {
    windows.iter().map(|w| w.iter().zip(cl).map(|(a, b)| a * b).sum()).collect()
}

/// Cotangent of the `C_l` (`l <= lmax`) from that of the band powers.
pub fn window_cotangent(windows: &[Vec<f64>], cot: &[f64], lmax: usize) -> Vec<f64> {
    (0..=lmax).map(|l| windows.iter().zip(cot).map(|(w, c)| w[l] * c).sum()).collect()
}

/// Band powers `W C_l` of a map, and the analysis kept for the cotangent.
pub fn band_powers(sht: &Sht, windows: &[Vec<f64>], x: &[f64]) -> (Vec<f64>, Vec<Complex64>) {
    let alm = sht.analysis(x);
    (window_sums(windows, &sht.power(&alm)), alm)
}

/// Map cotangent of the band powers.
pub fn band_powers_vjp(sht: &Sht, windows: &[Vec<f64>], alm: &[Complex64], cot: &[f64]) -> Vec<f64> {
    let dcl = window_cotangent(windows, cot, sht.lmax);
    sht.analysis_vjp(&sht.power_vjp(alm, &dcl))
}

/// Cloud-in-cell density on `bins` centres from `lo` to `hi` (mass outside
/// the centres partly dropped, not renormalised), per pixel and bin width.
pub struct Cic {
    pub lo: f64,
    pub hi: f64,
    pub bins: usize,
    dx: f64,
}

impl Cic {
    pub fn new(lo: f64, hi: f64, bins: usize) -> Self {
        Cic { lo, hi, bins, dx: (hi - lo) / (bins - 1) as f64 }
    }

    /// Spacing of the centres.
    pub fn dx(&self) -> f64 {
        self.dx
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let parts = chunked(x.len(), CHUNK, |a, b| {
            let mut h = vec![0.0; self.bins];
            for &v in &x[a..b] {
                let pos = (v - self.lo) / self.dx;
                let i0 = pos.floor();
                let frac = pos - i0;
                let i0 = i0 as i64;
                if i0 >= 0 && i0 < self.bins as i64 {
                    h[i0 as usize] += 1.0 - frac;
                }
                if i0 + 1 >= 0 && i0 + 1 < self.bins as i64 {
                    h[(i0 + 1) as usize] += frac;
                }
            }
            h
        });
        let mut h = vec![0.0; self.bins];
        for p in parts {
            h.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        let norm = x.len() as f64 * self.dx;
        h.iter_mut().for_each(|v| *v /= norm);
        h
    }

    pub fn vjp(&self, x: &[f64], cot: &[f64]) -> Vec<f64> {
        let norm = x.len() as f64 * self.dx * self.dx;
        x.iter()
            .map(|&v| {
                let i = ((v - self.lo) / self.dx).floor() as i64;
                let mut g = 0.0;
                if i >= 0 && i < self.bins as i64 {
                    g -= cot[i as usize];
                }
                if i + 1 >= 0 && i + 1 < self.bins as i64 {
                    g += cot[(i + 1) as usize];
                }
                g / norm
            })
            .collect()
    }
}

/// First and second derivatives `[x_1, x_2, x_11, x_22, x_12]` of a map
/// and their adjoint: harmonic covariant derivatives of a full-sky map,
/// spectral derivatives of a periodic image (pixel units).
pub trait Derivatives {
    fn map_derivatives(&self, x: &[f64]) -> [Vec<f64>; 5];
    fn map_derivatives_vjp(&self, g: [&[f64]; 5]) -> Vec<f64>;
}

impl Derivatives for Sht {
    fn map_derivatives(&self, x: &[f64]) -> [Vec<f64>; 5] {
        self.derivatives(&self.analysis(x))
    }

    fn map_derivatives_vjp(&self, g: [&[f64]; 5]) -> Vec<f64> {
        self.analysis_vjp(&self.derivatives_vjp(g))
    }
}

impl Derivatives for Fft2 {
    fn map_derivatives(&self, x: &[f64]) -> [Vec<f64>; 5] {
        self.derivatives(x)
    }

    fn map_derivatives_vjp(&self, g: [&[f64]; 5]) -> Vec<f64> {
        self.derivatives_vjp(g)
    }
}

/// Soft Minkowski functionals V0, V1, V2 of the standardised map at 65
/// thresholds in [-4, 4] (width 0.125), with the harmonic covariant
/// derivatives (spectral ones on the plane) and the detached curvature
/// floor of the source.
pub struct Minkowski {
    pub nu: Vec<f64>,
    pub eps: f64,
}

/// What the cotangent needs of a forward evaluation.
pub struct MinkowskiState {
    u: Vec<f64>,
    sigma: f64,
    d: [Vec<f64>; 5],
    floor: f64,
}

/// Normal CDF; exactly 1 in double from z = 8.5 on, 0 below -38.5.
fn ndtr(z: f64) -> f64 {
    if z >= 8.5 {
        1.0
    } else if z <= -38.5 {
        0.0
    } else {
        0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
    }
}

impl Default for Minkowski {
    fn default() -> Self {
        Minkowski { nu: linspace(-4.0, 4.0, 65, true), eps: 0.125 }
    }
}

impl Minkowski {
    pub fn count(&self) -> usize {
        3 * self.nu.len()
    }

    /// `f(t, z_t, exp(-z_t^2 / 2))` for every threshold, `z_t = (u - nu_t) /
    /// eps`; the Gaussian by recurrence away from the threshold nearest to
    /// `u` (equally spaced thresholds: two products instead of an exp).
    #[inline(always)]
    fn each_threshold(&self, u: f64, mut f: impl FnMut(usize, f64, f64)) {
        let k = self.nu.len();
        let h = self.nu[1] - self.nu[0];
        let d = h / self.eps;
        let q = (-d * d).exp();
        let c = ((u - self.nu[0]) / h).round().clamp(0.0, (k - 1) as f64) as usize;
        let zc = (u - self.nu[c]) / self.eps;
        let gc = (-0.5 * zc * zc).exp();
        f(c, zc, gc);
        let (mut g, mut r) = (gc, (d * zc - 0.5 * d * d).exp());
        for t in c + 1..k {
            g *= r;
            r *= q;
            f(t, (u - self.nu[t]) / self.eps, g);
        }
        let (mut g, mut s) = (gc, (-d * zc - 0.5 * d * d).exp());
        for t in (0..c).rev() {
            g *= s;
            s *= q;
            f(t, (u - self.nu[t]) / self.eps, g);
        }
    }

    /// Functionals `[3, 65]` (flattened) and the state.
    pub fn forward<D: Derivatives + ?Sized>(&self, deriv: &D, x: &[f64]) -> (Vec<f64>, MinkowskiState) {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let sigma = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n + 1e-12).sqrt();
        let u: Vec<f64> = x.iter().map(|v| (v - mean) / sigma).collect();
        let d = deriv.map_derivatives(&u);
        let floor = 1e-4 * d[0].iter().zip(&d[1]).map(|(a, b)| a * a + b * b).sum::<f64>() / n + 1e-12;
        let k = self.nu.len();
        let c = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * self.eps);
        let parts = chunked(u.len(), CHUNK, |a, b| {
            let mut v = vec![0.0; 3 * k];
            for p in a..b {
                let (d1, d2, d11, d22, d12) = (d[0][p], d[1][p], d[2][p], d[3][p], d[4][p]);
                let q = d1 * d1 + d2 * d2;
                let mag = (q + 1e-30).sqrt();
                let curv = (2.0 * d1 * d2 * d12 - d1 * d1 * d22 - d2 * d2 * d11) / (q + floor);
                self.each_threshold(u[p], |t, z, g| {
                    let delta = g * c;
                    v[t] += ndtr(z);
                    v[k + t] += 0.25 * delta * mag;
                    v[2 * k + t] += delta * curv / (2.0 * std::f64::consts::PI);
                });
            }
            v
        });
        let mut v = vec![0.0; 3 * k];
        for p in parts {
            v.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        v.iter_mut().for_each(|a| *a /= n);
        (v, MinkowskiState { u, sigma, d, floor })
    }

    /// Map cotangent from the cotangent `[3, 65]` of the functionals (the
    /// floor held fixed).
    pub fn vjp<D: Derivatives + ?Sized>(&self, deriv: &D, s: &MinkowskiState, cot: &[f64]) -> Vec<f64> {
        let k = self.nu.len();
        let npix = s.u.len();
        let n = npix as f64;
        let c = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * self.eps);
        let tau = 2.0 * std::f64::consts::PI;
        // Per chunk: the cotangent of u and of the five derivative maps.
        let parts = chunked(npix, CHUNK, |a, b| {
            let mut gu = vec![0.0; b - a];
            let mut gd: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; b - a]);
            for p in a..b {
                let (d1, d2, d11, d22, d12) = (s.d[0][p], s.d[1][p], s.d[2][p], s.d[3][p], s.d[4][p]);
                let q = d1 * d1 + d2 * d2;
                let mag = (q + 1e-30).sqrt();
                let den = q + s.floor;
                let num = 2.0 * d1 * d2 * d12 - d1 * d1 * d22 - d2 * d2 * d11;
                let curv = num / den;
                let (mut aa, mut bb, mut g) = (0.0, 0.0, 0.0);
                self.each_threshold(s.u[p], |t, z, e| {
                    let delta = e * c;
                    let dprime = -z * delta / self.eps;
                    g += cot[t] * delta + cot[k + t] * dprime * mag / 4.0 + cot[2 * k + t] * dprime * curv / tau;
                    aa += cot[k + t] * delta / 4.0;
                    bb += cot[2 * k + t] * delta / tau;
                });
                let (aa, bb) = (aa / n, bb / n);
                let i = p - a;
                gu[i] = g / n;
                gd[0][i] = aa * d1 / mag + bb * ((2.0 * d2 * d12 - 2.0 * d1 * d22) / den - 2.0 * d1 * num / (den * den));
                gd[1][i] = aa * d2 / mag + bb * ((2.0 * d1 * d12 - 2.0 * d2 * d11) / den - 2.0 * d2 * num / (den * den));
                gd[2][i] = -bb * d2 * d2 / den;
                gd[3][i] = -bb * d1 * d1 / den;
                gd[4][i] = bb * 2.0 * d1 * d2 / den;
            }
            (gu, gd)
        });
        let mut gu = Vec::with_capacity(npix);
        let mut gd: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(npix));
        for (u, d) in parts {
            gu.extend(u);
            for (a, b) in gd.iter_mut().zip(d) {
                a.extend(b);
            }
        }
        let back = deriv.map_derivatives_vjp([&gd[0], &gd[1], &gd[2], &gd[3], &gd[4]]);
        gu.iter_mut().zip(back).for_each(|(a, b)| *a += b);
        // u = (x - mean) / sqrt(mean((x - mean)^2) + 1e-12).
        let mg = gu.iter().sum::<f64>() / n;
        let mu = gu.iter().zip(&s.u).map(|(g, u)| g * u).sum::<f64>() / n;
        gu.iter().zip(&s.u).map(|(g, u)| (g - mg - u * mu) / s.sigma).collect()
    }
}
