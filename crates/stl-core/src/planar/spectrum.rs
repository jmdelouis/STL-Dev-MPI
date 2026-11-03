//! Fourier tools of the planar objective on a periodic `H x W` grid: the
//! 2D FFT, the Nyquist prefilter of STL-Dev (`Synthesis.apply_nyquist_filter`),
//! the power spectrum in Gaussian log rings (`CS_operator_2D_Kernel_Torch`,
//! `power_spectrum_method="gaussian_rings"`, `pbc=True`) and the spectral
//! derivatives of the planar Minkowski functionals, each with its VJP.

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

type C64 = Complex64;

/// Unnormalised complex 2D FFT of `[H, W]` arrays.
pub struct Fft2 {
    /// Number of rows `H`.
    pub h: usize,
    /// Number of columns `W`.
    pub w: usize,
    rows: [Arc<dyn Fft<f64>>; 2],
    cols: [Arc<dyn Fft<f64>>; 2],
}

/// `torch.fft.fftfreq(n) * n`: `0, 1, .., n/2 - 1, -n/2, .., -1`.
pub fn integer_frequency(k: usize, n: usize) -> i64 {
    if k < n.div_ceil(2) {
        k as i64
    } else {
        k as i64 - n as i64
    }
}

impl Fft2 {
    /// Plans the forward and inverse transforms of `h x w` arrays.
    pub fn new(h: usize, w: usize) -> Self {
        let mut p = FftPlanner::new();
        Fft2 { h, w, rows: [p.plan_fft_forward(w), p.plan_fft_inverse(w)], cols: [p.plan_fft_forward(h), p.plan_fft_inverse(h)] }
    }

    fn run(&self, x: &mut [C64], dir: usize) {
        assert_eq!(x.len(), self.h * self.w);
        self.rows[dir].process(x);
        let mut t = vec![C64::default(); x.len()];
        for r in 0..self.h {
            for c in 0..self.w {
                t[c * self.h + r] = x[r * self.w + c];
            }
        }
        self.cols[dir].process(&mut t);
        for r in 0..self.h {
            for c in 0..self.w {
                x[r * self.w + c] = t[c * self.h + r];
            }
        }
    }

    /// `X[k] = sum_x x e^{-2 pi i k.x / N}`.
    pub fn forward(&self, x: &mut [C64]) {
        self.run(x, 0)
    }

    /// `x = sum_k X e^{+2 pi i k.x / N}` (no `1 / N`).
    pub fn inverse(&self, x: &mut [C64]) {
        self.run(x, 1)
    }

    /// Orthonormal transform of a real plane (`norm="ortho"`).
    pub fn ortho(&self, x: &[f64]) -> Vec<C64> {
        let mut v: Vec<C64> = x.iter().map(|&a| C64::new(a, 0.0)).collect();
        self.forward(&mut v);
        let s = 1.0 / ((self.h * self.w) as f64).sqrt();
        v.iter_mut().for_each(|a| *a *= s);
        v
    }

    /// Real part of the orthonormal inverse transform: the cotangent of a
    /// real plane from that of its [`Fft2::ortho`] transform.
    pub fn ortho_adjoint(&self, g: &[C64]) -> Vec<f64> {
        let mut v = g.to_vec();
        self.inverse(&mut v);
        let s = 1.0 / ((self.h * self.w) as f64).sqrt();
        v.iter().map(|a| a.re * s).collect()
    }

    /// `Re ifft(H fft x)` for a real plane: a real linear operator when the
    /// multiplier is Hermitian.
    fn multiply(&self, x: &[f64], hk: impl Fn(i64, i64) -> C64) -> Vec<f64> {
        let mut v: Vec<C64> = x.iter().map(|&a| C64::new(a, 0.0)).collect();
        self.forward(&mut v);
        for r in 0..self.h {
            let kr = integer_frequency(r, self.h);
            for c in 0..self.w {
                v[r * self.w + c] *= hk(kr, integer_frequency(c, self.w));
            }
        }
        self.inverse(&mut v);
        let s = 1.0 / (self.h * self.w) as f64;
        v.iter().map(|a| a.re * s).collect()
    }

    /// `Synthesis.apply_nyquist_filter` of one plane: the modes with
    /// `k_r^2 + k_c^2 > (min(H, W) / 2)^2` removed.
    pub fn nyquist_filter(&self, x: &[f64]) -> Vec<f64> {
        let r2 = (self.h.min(self.w) as f64 / 2.0).powi(2);
        self.multiply(x, |a, b| if ((a * a + b * b) as f64) <= r2 { C64::new(1.0, 0.0) } else { C64::default() })
    }

    /// Multiplier of the derivative along the rows (`axis = 0`) or the
    /// columns, in pixel units; zero at the Nyquist frequency.
    fn ik(k: i64, n: usize) -> C64 {
        if 2 * k.unsigned_abs() as usize == n {
            C64::default()
        } else {
            C64::new(0.0, 2.0 * std::f64::consts::PI * k as f64 / n as f64)
        }
    }

    fn kk(k: i64, n: usize) -> f64 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        -t * t
    }

    /// `[x_1, x_2, x_11, x_22, x_12]` of a periodic plane (1 = rows, 2 =
    /// columns), spectral derivatives.
    pub fn derivatives(&self, x: &[f64]) -> [Vec<f64>; 5] {
        let (h, w) = (self.h, self.w);
        [
            self.multiply(x, |a, _| Self::ik(a, h)),
            self.multiply(x, |_, b| Self::ik(b, w)),
            self.multiply(x, |a, _| C64::new(Self::kk(a, h), 0.0)),
            self.multiply(x, |_, b| C64::new(Self::kk(b, w), 0.0)),
            self.multiply(x, |a, b| Self::ik(a, h) * Self::ik(b, w)),
        ]
    }

    /// Adjoint of [`Fft2::derivatives`] (conjugate multipliers).
    pub fn derivatives_vjp(&self, g: [&[f64]; 5]) -> Vec<f64> {
        let (h, w) = (self.h, self.w);
        let parts = [
            self.multiply(g[0], |a, _| Self::ik(a, h).conj()),
            self.multiply(g[1], |_, b| Self::ik(b, w).conj()),
            self.multiply(g[2], |a, _| C64::new(Self::kk(a, h), 0.0)),
            self.multiply(g[3], |_, b| C64::new(Self::kk(b, w), 0.0)),
            self.multiply(g[4], |a, b| (Self::ik(a, h) * Self::ik(b, w)).conj()),
        ];
        (0..h * w).map(|i| parts.iter().map(|p| p[i]).sum()).collect()
    }
}

/// `torch.linspace(start, end, steps)` in float64 (CPU kernel: from the
/// start below the middle, from the end above it).
fn torch_linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    let step = (end - start) / (steps - 1) as f64;
    let half = steps / 2;
    (0..steps).map(|i| if i < half { start + step * i as f64 } else { end - step * (steps - i - 1) as f64 }).collect()
}

/// Ring weights below `exp(-CUT^2 / 2)` (about 5e-32 at 12 widths) are
/// not computed: their contribution is far below the rounding of a bin.
pub const CUT: f64 = 12.0;

/// Pixels per chunk of the threaded passes (sums added in chunk order).
const CHUNK: usize = 1 << 15;

/// `f(lo, hi)` over the chunks of `0..n` on the available threads, results
/// in chunk order.
fn chunks<T: Send>(n: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let count = n.div_ceil(CHUNK).max(1);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(count);
    let per = count.div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let f = &f;
                s.spawn(move || (t * per..((t + 1) * per).min(count)).map(|c| f(c * CHUNK, ((c + 1) * CHUNK).min(n))).collect::<Vec<T>>())
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Cross spectra in Gaussian log rings of the radial frequency: bins
/// log-spaced from `1 / 2^J` to `0.5 / 2^Jmin` cycles per pixel,
/// `PS[c1, c2, j] = sum_k m_j(k) X_c1(k) conj X_c2(k) / sum_k m_j(k)` with
/// the orthonormal transform `X` and
/// `m_j(k) = exp(-(log f_k - log c_j)^2 / (2 w_j^2))` (zero at `f = 0`).
/// STL-Dev stores the `n_bins x H x W` masks (4 GB at 2048^2); here the
/// weights of a pixel are computed when needed, for the bins within
/// `CUT` widths.
pub struct GaussianRings {
    /// 2D FFT of the grid.
    pub fft: Fft2,
    /// Number of radial bins.
    pub n_bins: usize,
    /// Bin centres (cycles per pixel).
    pub centers: Vec<f64>,
    lc: Vec<f64>,
    ls: Vec<f64>,
    /// `log f` per pixel (unshifted layout), NaN at `f = 0`.
    lr: Vec<f64>,
    norm: Vec<f64>,
}

impl GaussianRings {
    /// The radial frequency is computed in float32 and cast, as STL-Dev
    /// does with torch's default dtype.
    pub fn new(h: usize, w: usize, n_bins: usize, j: usize, jmin: usize) -> Self {
        assert!(jmin < j, "Jmin must be below J");
        let min_freq = 1.0 / 2f64.powi(j as i32);
        let max_freq = 0.5 / 2f64.powi(jmin as i32);
        let edges = torch_linspace(min_freq.ln(), max_freq.ln(), n_bins + 1);
        let lc: Vec<f64> = (0..n_bins).map(|i| 0.5 * (edges[i] + edges[i + 1])).collect();
        let ls: Vec<f64> = (0..n_bins).map(|i| (edges[i + 1] - edges[i]).abs()).map(|s| if s > 0.0 { s } else { 1.0 }).collect();
        let freq = |k: usize, n: usize| integer_frequency(k, n) as f32 * (1.0 / n as f32);
        let mut lr = Vec::with_capacity(h * w);
        for r in 0..h {
            let fy = freq(r, h);
            for c in 0..w {
                let fx = freq(c, w);
                let radial = (fx * fx + fy * fy).sqrt() as f64;
                lr.push(if radial == 0.0 { f64::NAN } else { radial.ln() });
            }
        }
        let mut g = GaussianRings { fft: Fft2::new(h, w), n_bins, centers: lc.iter().map(|c| c.exp()).collect(), lc, ls, lr, norm: Vec::new() };
        let parts = chunks(h * w, |a, b| {
            let mut s = vec![0.0; n_bins];
            (a..b).for_each(|p| g.weights(p, |k, m| s[k] += m));
            s
        });
        g.norm = vec![0.0; n_bins];
        for s in parts {
            g.norm.iter_mut().zip(s).for_each(|(a, b)| *a += b);
        }
        g
    }

    /// `f(bin, m_bin(p))` for the bins of pixel `p` within `CUT` widths.
    #[inline]
    fn weights(&self, p: usize, mut f: impl FnMut(usize, f64)) {
        let lr = self.lr[p];
        if lr.is_nan() {
            return;
        }
        let wmax = self.ls.iter().fold(0f64, |m, &v| m.max(v));
        let lo = self.lc.partition_point(|&c| c < lr - CUT * wmax);
        for k in lo..self.n_bins {
            let d = lr - self.lc[k];
            if d < -CUT * wmax {
                break;
            }
            if d.abs() <= CUT * self.ls[k] {
                f(k, ((-0.5 * (d * d)) / (self.ls[k] * self.ls[k])).exp());
            }
        }
    }

    /// `(log centres, log widths, log f per pixel, bin norms)`, for a device
    /// implementation of the same rings.
    pub fn tables(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (&self.lc, &self.ls, &self.lr, &self.norm)
    }

    /// Number of values for `channels` channels: pairs `c1 <= c2`, bins.
    pub fn count(&self, channels: usize) -> usize {
        channels * (channels + 1) / 2 * self.n_bins
    }

    /// Spectra of `x` `[C, H, W]`, pairs `c1 <= c2` then bins, and the
    /// transforms kept for the VJP.
    pub fn forward(&self, x: &[f64], channels: usize) -> (Vec<C64>, Vec<Vec<C64>>) {
        let n = self.fft.h * self.fft.w;
        assert_eq!(x.len(), channels * n);
        let xs: Vec<Vec<C64>> = (0..channels).map(|c| self.fft.ortho(&x[c * n..][..n])).collect();
        let mut out = Vec::with_capacity(self.count(channels));
        for c1 in 0..channels {
            for c2 in c1..channels {
                let parts = chunks(n, |a, b| {
                    let mut s = vec![C64::default(); self.n_bins];
                    for p in a..b {
                        let v = xs[c1][p] * xs[c2][p].conj();
                        self.weights(p, |k, m| s[k] += m * v);
                    }
                    s
                });
                let mut s = vec![C64::default(); self.n_bins];
                for part in parts {
                    s.iter_mut().zip(part).for_each(|(a, b)| *a += b);
                }
                out.extend(s.iter().zip(&self.norm).map(|(v, nrm)| v / nrm));
            }
        }
        (out, xs)
    }

    /// Map cotangent `[C, H, W]` of `Re sum conj(g) PS`.
    pub fn vjp(&self, xs: &[Vec<C64>], g: &[C64]) -> Vec<f64> {
        let channels = xs.len();
        let n = self.fft.h * self.fft.w;
        let mut gx = vec![vec![C64::default(); n]; channels];
        let mut k = 0;
        for c1 in 0..channels {
            for c2 in c1..channels {
                let gb: Vec<C64> = (0..self.n_bins).map(|b| g[k + b] / self.norm[b]).collect();
                k += self.n_bins;
                let parts = chunks(n, |a, b| {
                    let mut out = Vec::with_capacity(2 * (b - a));
                    for p in a..b {
                        let mut gk = C64::default();
                        self.weights(p, |j, m| gk += gb[j] * m);
                        out.push((gk * xs[c2][p], gk.conj() * xs[c1][p]));
                    }
                    out
                });
                for (p, (u, v)) in parts.into_iter().flatten().enumerate() {
                    gx[c1][p] += u;
                    gx[c2][p] += v;
                }
            }
        }
        gx.iter().flat_map(|v| self.fft.ortho_adjoint(v)).collect()
    }
}
