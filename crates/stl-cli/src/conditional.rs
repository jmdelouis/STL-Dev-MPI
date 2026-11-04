//! `--init-noise conditional`: the details added to a coarse `--init` drawn
//! conditionally on the coarse map, the HEALPix form of the planar
//! coarse-to-fine start (`docs/lagrangian_synthesis.md` 6.4,
//! `stl_core::planar::cascade`).
//!
//! The init (NESTED, `nside_c = nside / sqrt(factor)`) is copied on its
//! children, as for `--init-noise auto`, and every child gets a detail drawn
//! from the target's details (`target - mean of its parent`) that fall in the
//! same bin of two features of the parent: its value and its local variation
//! `sqrt(mean_q (x_q - x_p)^2)` over its 8 neighbours at `nside_c` (the
//! sphere's form of the gradient modulus).  Each feature is cut into `q`
//! quantile bins of the target's parents; the details of every joint bin
//! are kept as a histogram of [`HIST`] cells on a global range.  Bin edges and
//! histograms are global (all-reduced), and every pixel's draw depends only
//! on the seed and its NESTED index: the result does not depend on the ranks.

use stl_core::neighbours::kth_neighbourhood;
use stl_mpi::Comm;

/// Cells of the detail histogram of a bin.
pub const HIST: usize = 4096;
/// Cells of the feature histograms that give the quantile edges.
const FEATURE_HIST: usize = 1 << 16;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Two uniforms in (0, 1) of `(seed, id)`.
fn uniforms(seed: u64, id: i64) -> (f64, f64) {
    let a = splitmix(seed ^ splitmix(id as u64 ^ 0xc0d1_7105_u64));
    let b = splitmix(a);
    (((a >> 11) as f64 + 0.5) / (1u64 << 53) as f64, ((b >> 11) as f64 + 0.5) / (1u64 << 53) as f64)
}

/// `sqrt(mean_q (x_q - x_p)^2)` over the neighbours `q` of `p` at `nside`.
pub fn local_variation(full: &[f64], nside: i64, p: usize) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for q in kth_neighbourhood(p as i64, nside, 1) {
        if q >= 0 && q as usize != p {
            let d = full[q as usize] - full[p];
            s += d * d;
            n += 1;
        }
    }
    (s / n.max(1) as f64).sqrt()
}

fn global_range<C: Comm>(comm: &C, v: &[f64]) -> (f64, f64) {
    let hi = comm.allreduce_max(v.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x)));
    let lo = -comm.allreduce_max(v.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(-x)));
    (lo, hi)
}

fn cell(v: f64, lo: f64, width: f64, cells: usize) -> usize {
    if width <= 0.0 {
        0
    } else {
        (((v - lo) / width) as usize).min(cells - 1)
    }
}

/// Quantile edges `k / q`, `k = 1..q`, of the values held by all ranks,
/// from a global histogram (linear inside a cell).
fn global_quantiles<C: Comm>(comm: &C, v: &[f64], q: usize) -> Vec<f64> {
    let (lo, hi) = global_range(comm, v);
    let width = (hi - lo) / FEATURE_HIST as f64;
    let mut h = vec![0.0; FEATURE_HIST];
    for &x in v {
        h[cell(x, lo, width, FEATURE_HIST)] += 1.0;
    }
    comm.allreduce_sum(&mut h);
    let total: f64 = h.iter().sum();
    let mut edges = Vec::with_capacity(q - 1);
    let (mut cum, mut c) = (0.0, 0);
    for k in 1..q {
        let want = total * k as f64 / q as f64;
        while c < FEATURE_HIST - 1 && cum + h[c] < want {
            cum += h[c];
            c += 1;
        }
        let f = if h[c] > 0.0 { ((want - cum) / h[c]).clamp(0.0, 1.0) } else { 0.0 };
        edges.push(lo + (c as f64 + f) * width);
    }
    edges
}

/// The target's details per joint bin of (parent value, parent local variation).
pub struct Conditional {
    pub q: usize,
    pub factor: usize,
    pub nside_c: i64,
    edges: [Vec<f64>; 2],
    lo: f64,
    width: f64,
    /// Cumulative histograms `[q * q + 1][HIST]` (the last one: all bins), normalised.
    cdf: Vec<Vec<f64>>,
}

impl Conditional {
    /// `target`: this rank's standardised target slice `[s, s + len)` at the
    /// fine nside, `s` and the length multiples of `factor`.
    pub fn new<C: Comm>(comm: &C, target: &[f64], s: usize, factor: usize, nside_c: i64, q: usize) -> Self {
        Self::pooled(comm, &[target], None, s, factor, nside_c, q)
    }

    /// As [`Conditional::new`], with the parents and details of several
    /// target maps pooled (the same slice `[s, s + len)` of each).  With
    /// `bases`, the details are `target - base` (a smooth upsampling of the
    /// target's parents, [`upsample`]) instead of `target - parent mean`.
    pub fn pooled<C: Comm>(comm: &C, targets: &[&[f64]], bases: Option<&[&[f64]]>, s: usize, factor: usize, nside_c: i64, q: usize) -> Self {
        let n_c = 12 * (nside_c * nside_c) as usize;
        let (mut means, mut var, mut details) = (Vec::new(), Vec::new(), Vec::new());
        for (k, target) in targets.iter().enumerate() {
            let (p0, p1) = (s / factor, (s + target.len()) / factor);
            let m: Vec<f64> = target.chunks(factor).map(|c| c.iter().sum::<f64>() / factor as f64).collect();
            let mut full = vec![0.0; n_c];
            full[p0..p1].copy_from_slice(&m);
            comm.allreduce_sum(&mut full);
            var.extend((p0..p1).map(|p| local_variation(&full, nside_c, p)));
            match bases {
                Some(b) => details.extend(target.iter().zip(b[k].iter()).map(|(v, u)| v - u)),
                None => details.extend(target.iter().zip(m.iter().flat_map(|&v| std::iter::repeat(v).take(factor))).map(|(v, mean)| v - mean)),
            }
            means.extend(m);
        }
        let edges = [global_quantiles(comm, &means, q), global_quantiles(comm, &var, q)];
        let (lo, hi) = global_range(comm, &details);
        let width = (hi - lo) / HIST as f64;
        let nb = q * q;
        let mut h = vec![0.0; (nb + 1) * HIST];
        let bin = |p: usize| bucket(&edges[0], means[p]) * q + bucket(&edges[1], var[p]);
        for (i, &d) in details.iter().enumerate() {
            let c = cell(d, lo, width, HIST);
            h[bin(i / factor) * HIST + c] += 1.0;
            h[nb * HIST + c] += 1.0;
        }
        comm.allreduce_sum(&mut h);
        let cdf = h
            .chunks(HIST)
            .map(|row| {
                let total: f64 = row.iter().sum();
                let mut acc = 0.0;
                row.iter()
                    .map(|v| {
                        acc += v;
                        if total > 0.0 { acc / total } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        Conditional { q, factor, nside_c, edges, lo, width, cdf }
    }

    pub fn bin(&self, value: f64, variation: f64) -> usize {
        bucket(&self.edges[0], value) * self.q + bucket(&self.edges[1], variation)
    }

    fn sample(&self, bin: usize, (u, v): (f64, f64)) -> f64 {
        let row = if *self.cdf[bin].last().unwrap() > 0.0 { &self.cdf[bin] } else { self.cdf.last().unwrap() };
        let c = row.partition_point(|&x| x < u).min(HIST - 1);
        self.lo + (c as f64 + v) * self.width
    }

    /// Initial slice `[s, e)` at the fine nside: the coarse map `coarse`
    /// (full, standardised, at `nside_c`) on its children plus the drawn details.
    pub fn draw(&self, coarse: &[f64], s: usize, e: usize, seed: u64) -> Vec<f64> {
        self.draw_on(coarse, None, s, e, seed)
    }

    /// As [`Conditional::draw`], the details added to `base` (the slice
    /// `[s, e)` of a smooth upsampling of `coarse`) when given.
    pub fn draw_on(&self, coarse: &[f64], base: Option<&[f64]>, s: usize, e: usize, seed: u64) -> Vec<f64> {
        let f = self.factor;
        let mut out = Vec::with_capacity(e - s);
        let mut p = usize::MAX;
        let mut b = 0;
        for id in s..e {
            if id / f != p {
                p = id / f;
                b = self.bin(coarse[p], local_variation(coarse, self.nside_c, p));
            }
            let at = base.map_or(coarse[p], |u| u[id - s]);
            out.push(at + self.sample(b, uniforms(seed, id as i64)));
        }
        out
    }
}

/// Jacobi iterations of the analysis in [`upsample`] (the pixel sum alone
/// leaks 40 % of a dipole into high l at `lmax = 2 nside`; 10 iterations
/// bring it to 1e-9).
const ANALYSIS_ITERATIONS: usize = 10;

/// Harmonic upsampling of a NESTED map from `nside_c` to `nside`: its
/// `a_lm` up to `l = 2 nside_c` (iterated analysis at `nside_c`) synthesised at `nside`.
pub fn upsample(coarse: &[f64], nside_c: usize, nside: usize) -> Vec<f64> {
    let lmax = 2 * nside_c;
    let sht = stl_core::sht::Sht::new(nside_c, lmax);
    let mut alm = sht.analysis(coarse);
    for _ in 0..ANALYSIS_ITERATIONS {
        let back = sht.synthesis(&alm);
        let residual: Vec<f64> = coarse.iter().zip(&back).map(|(x, y)| x - y).collect();
        alm.iter_mut().zip(sht.analysis(&residual)).for_each(|(a, d)| *a += d);
    }
    stl_core::sht::Sht::new(nside, lmax).synthesis(&alm)
}

/// `numpy.searchsorted(edges, v)` (side left).
fn bucket(edges: &[f64], v: f64) -> usize {
    edges.partition_point(|&e| e < v)
}
