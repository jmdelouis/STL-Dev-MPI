//! Coarse-to-fine initial images with detail drawn conditionally on the
//! coarser scale (`docs/lagrangian_synthesis.md`, section 6.4; port of
//! `tools/lagrangian/multiscale.py`).
//!
//! From level `n` to level `2n` the synthesis is upsampled by Fourier zero
//! padding ([`resample`]) and a detail field is added: for every pixel, a
//! value drawn uniformly from the targets' detail pixels (`t_2n -
//! resample(t_n)`) that share its bin of the joint (coarse value, coarse
//! gradient modulus) quantiles ([`ConditionalDetail`]).  Planes are square,
//! periodic, row-major.

use super::spectrum::{integer_frequency, Fft2};
use num_complex::Complex64;

/// Periodic Fourier resampling of one `n x n` plane to `m x m`: the modes
/// with `|k_r|, |k_c| <= min(n, m) / 2 - 1` are kept (the Nyquist rows of
/// the smaller grid are dropped), pixel means are kept.
pub fn resample(x: &[f64], n: usize, m: usize) -> Vec<f64> {
    assert_eq!(x.len(), n * n);
    let mut v: Vec<Complex64> = x.iter().map(|&a| Complex64::new(a, 0.0)).collect();
    Fft2::new(n, n).forward(&mut v);
    let kmax = (n.min(m) / 2) as i64 - 1;
    let mut y = vec![Complex64::default(); m * m];
    for r in 0..m {
        let kr = integer_frequency(r, m);
        if kr.abs() > kmax {
            continue;
        }
        for c in 0..m {
            let kc = integer_frequency(c, m);
            if kc.abs() <= kmax {
                y[r * m + c] = v[kr.rem_euclid(n as i64) as usize * n + kc.rem_euclid(n as i64) as usize];
            }
        }
    }
    Fft2::new(m, m).inverse(&mut y);
    let s = 1.0 / (n * n) as f64;
    y.iter().map(|a| a.re * s).collect()
}

/// `|(x[r+1] - x[r-1], x[c+1] - x[c-1])|` of a periodic `n x n` plane.
pub fn gradient_modulus(x: &[f64], n: usize) -> Vec<f64> {
    let at = |r: usize, c: usize| x[(r % n) * n + c % n];
    let mut g = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let a = at(r + 1, c) - at(r + n - 1, c);
            let b = at(r, c + 1) - at(r, c + n - 1);
            g[r * n + c] = a.hypot(b);
        }
    }
    g
}

/// `numpy.quantile` (linear interpolation) of the sorted values at `p`.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let pos = p * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let f = pos - i as f64;
    if i + 1 < sorted.len() {
        sorted[i] + f * (sorted[i + 1] - sorted[i])
    } else {
        sorted[i]
    }
}

/// `numpy.searchsorted(edges, v)` (side left): the number of edges `< v`.
fn bucket(edges: &[f64], v: f64) -> usize {
    edges.partition_point(|&e| e < v)
}

/// SplitMix64, as `stl_cli::planar::white_noise`.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.next_f64();
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// The targets' detail pixels grouped by bin of (coarse value, coarse
/// gradient modulus), `q` quantile bins per feature.
pub struct ConditionalDetail {
    pub n: usize,
    pub q: usize,
    edges: [Vec<f64>; 2],
    start: Vec<usize>,
    count: Vec<usize>,
    pool: Vec<f64>,
}

impl ConditionalDetail {
    /// `coarse` and `detail`: the targets' upsampled coarse level and detail,
    /// concatenated `n x n` planes.
    pub fn new(coarse: &[f64], detail: &[f64], n: usize, q: usize) -> Self {
        assert_eq!(coarse.len(), detail.len());
        let grad: Vec<f64> = coarse.chunks(n * n).flat_map(|p| gradient_modulus(p, n)).collect();
        let edges = [coarse, &grad[..]].map(|f| {
            let mut s = f.to_vec();
            s.sort_by(f64::total_cmp);
            (1..q).map(|k| quantile(&s, k as f64 / q as f64)).collect::<Vec<f64>>()
        });
        let mut me = ConditionalDetail { n, q, edges, start: vec![0; q * q], count: vec![0; q * q], pool: Vec::new() };
        let bins: Vec<usize> = coarse.iter().zip(&grad).map(|(&c, &g)| me.bin(c, g)).collect();
        let mut order: Vec<usize> = (0..bins.len()).collect();
        order.sort_by_key(|&i| bins[i]);
        me.pool = order.iter().map(|&i| detail[i]).collect();
        for &b in &bins {
            me.count[b] += 1;
        }
        for b in 1..q * q {
            me.start[b] = me.start[b - 1] + me.count[b - 1];
        }
        me
    }

    pub fn bin(&self, value: f64, grad: f64) -> usize {
        bucket(&self.edges[0], value) * self.q + bucket(&self.edges[1], grad)
    }

    /// Detail for the upsampled synthesis `coarse` (concatenated planes).
    pub fn draw(&self, coarse: &[f64], rng: &mut SplitMix) -> Vec<f64> {
        let n = self.n;
        coarse
            .chunks(n * n)
            .flat_map(|p| {
                let g = gradient_modulus(p, n);
                p.iter().zip(g).map(|(&c, g)| self.bin(c, g)).collect::<Vec<_>>()
            })
            .map(|b| {
                let u = rng.next_f64();
                if self.count[b] > 0 {
                    self.pool[self.start[b] + ((u * self.count[b] as f64) as usize).min(self.count[b] - 1)]
                } else {
                    self.pool[((u * self.pool.len() as f64) as usize).min(self.pool.len() - 1)]
                }
            })
            .collect()
    }
}
