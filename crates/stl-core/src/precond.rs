//! Per-scale preconditioners of L-BFGS with the map as the variable.
//!
//! `Pi_j` is the mean over the groups of `4^j` consecutive NESTED pixels
//! (the children of one pixel of level `j`), broadcast back to the pixels:
//! an orthogonal projector, and `Pi_0 = I >= Pi_1 >= ... >= Pi_K` are
//! nested.  The bands `B_j = Pi_j - Pi_{j+1}` (`j < K`, the details of level
//! `j`) and `B_K = Pi_K` (the means of level `K`) are therefore orthogonal
//! projectors summing to the identity, and
//!
//! ```text
//! P = sum_j alpha_j B_j = alpha_0 I + sum_{j >= 1} (alpha_j - alpha_{j-1}) Pi_j
//! ```
//!
//! is symmetric positive definite for any `alpha_j > 0`: L-BFGS uses it as
//! its initial inverse Hessian `H0` (a change of metric), so that its
//! updates and the strong Wolfe line search stay valid, and the map stays
//! the variable (STL-Dev's Laplacian bands as variables stall L-BFGS).
//!
//! The variable is a set of rows (members, channels) of owned NESTED
//! ranges whose ends are multiples of `4^K` on every rank: a group never
//! crosses a rank or a row, so `P` is local; only the band dot products of
//! the adaptive form are summed over the ranks.
//!
//! Summation orders: a mean adds its 4 children in order then scales by
//! 1/4; `P g` adds the levels in increasing order per pixel; local dot
//! products add the groups in order.

use crate::lbfgs::{Preconditioner, Reduce};

/// How the factors `alpha_j` are chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum BandFactors {
    /// `alpha_j = 4^(gamma j)`.
    PowerLaw(f64),
    /// `alpha_j = e_j / e_0`, `e_j` the energy per degree of freedom of
    /// band `j` of the target (`|B_j t|^2 / dim B_j`).
    TargetPower,
    /// Per-band Barzilai-Borwein scaling from the last L-BFGS pair,
    /// `alpha_j = (s . B_j y) / (y . B_j y)` (the scalar `s.y / y.y` of
    /// torch per band), within `[h / CLAMP, CLAMP h]` of the scalar `h`.
    Curvature,
}

/// Bound of the adaptive factors relative to the scalar scaling.
pub const CLAMP: f64 = 1e3;

/// `P = sum_j alpha_j B_j` on rows whose length is a multiple of `4^depth`.
#[derive(Clone, Debug)]
pub struct Bands {
    /// Number of levels `K` of the projectors (groups of `4^K` pixels).
    pub depth: usize,
    /// Factors `alpha_0 .. alpha_K` (`K + 1` values).
    pub alpha: Vec<f64>,
    /// How the factors are chosen.
    pub factors: BandFactors,
}

/// Means of the groups of 4 of `v` (in order, then times 1/4).
fn quad_means(v: &[f64]) -> Vec<f64> {
    v.chunks_exact(4).map(|c| (((c[0] + c[1]) + c[2]) + c[3]) * 0.25).collect()
}

/// `[m_1, ..., m_depth]`, `m_j` the means of the groups of `4^j` of `v`.
fn pyramid(v: &[f64], depth: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(depth);
    for j in 0..depth {
        let m = quad_means(if j == 0 { v } else { &out[j - 1] });
        out.push(m);
    }
    out
}

/// Top-level groups per thread chunk.
fn chunk_groups(n_groups: usize) -> usize {
    let threads = crate::lbfgs::threads();
    n_groups.div_ceil(threads).max(1)
}

impl Bands {
    /// The deepest `K <= max_depth` such that `4^K` divides `start` and
    /// `end` of every rank (`r.max` of the local shortfall).
    pub fn aligned_depth(r: &dyn Reduce, start: usize, end: usize, max_depth: usize) -> usize {
        let mut k = 0;
        while k < max_depth {
            let g = 1usize << (2 * (k + 1));
            if start % g != 0 || end % g != 0 {
                break;
            }
            k += 1;
        }
        // Smallest depth over the ranks.
        (max_depth as f64 - r.max((max_depth - k) as f64)) as usize
    }

    /// Power-law factors `alpha_j = 4^(gamma j)`.
    pub fn power_law(depth: usize, gamma: f64) -> Self {
        let alpha = (0..=depth).map(|j| 4f64.powf(gamma * j as f64)).collect();
        Bands { depth, alpha, factors: BandFactors::PowerLaw(gamma) }
    }

    /// Factors from the band energies of the target slice `t`.
    pub fn target_power(r: &dyn Reduce, t: &[f64], depth: usize) -> Self {
        let pij = Self::projections(r, t, t, depth);
        let n = r.sum(t.len() as f64);
        let dim = |j: usize| n / 4f64.powi(j as i32) - if j < depth { n / 4f64.powi(j as i32 + 1) } else { 0.0 };
        let energy: Vec<f64> =
            (0..=depth).map(|j| (if j < depth { pij[j] - pij[j + 1] } else { pij[j] }).max(0.0) / dim(j)).collect();
        let e0 = energy[0];
        let alpha = energy.iter().map(|&e| if e > 0.0 && e0 > 0.0 { e / e0 } else { 1.0 }).collect();
        Bands { depth, alpha, factors: BandFactors::TargetPower }
    }

    /// Curvature factors ([`BandFactors::Curvature`]), all 1 initially.
    pub fn curvature(depth: usize) -> Self {
        Bands { depth, alpha: vec![1.0; depth + 1], factors: BandFactors::Curvature }
    }

    /// `[a . Pi_j b]`, `j = 0..=depth`, summed over the ranks.
    pub fn projections(r: &dyn Reduce, a: &[f64], b: &[f64], depth: usize) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        let top = 1usize << (2 * depth);
        assert_eq!(a.len() % top, 0, "rows not aligned on groups of 4^{depth}");
        let per = chunk_groups(a.len() / top) * top;
        let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
            let handles: Vec<_> = a
                .chunks(per)
                .zip(b.chunks(per))
                .map(|(ac, bc)| {
                    s.spawn(move || {
                        let (pa, pb) = (pyramid(ac, depth), pyramid(bc, depth));
                        let mut out = vec![ac.iter().zip(bc).map(|(x, y)| x * y).sum::<f64>()];
                        for j in 0..depth {
                            let w = 4f64.powi(j as i32 + 1);
                            out.push(w * pa[j].iter().zip(&pb[j]).map(|(x, y)| x * y).sum::<f64>());
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        (0..=depth).map(|j| r.sum(parts.iter().map(|p| p[j]).sum())).collect()
    }
}

impl Preconditioner for Bands {
    fn apply(&self, g: &[f64]) -> Vec<f64> {
        let depth = self.depth;
        let top = 1usize << (2 * depth);
        assert_eq!(g.len() % top, 0, "rows not aligned on groups of 4^{depth}");
        let a0 = self.alpha[0];
        let c: Vec<f64> = (1..=depth).map(|j| self.alpha[j] - self.alpha[j - 1]).collect();
        let mut out = vec![0.0; g.len()];
        let per = chunk_groups(g.len() / top) * top;
        std::thread::scope(|s| {
            for (gc, oc) in g.chunks(per).zip(out.chunks_mut(per)) {
                let c = &c;
                s.spawn(move || {
                    let m = pyramid(gc, depth);
                    for (i, o) in oc.iter_mut().enumerate() {
                        let mut v = a0 * gc[i];
                        for j in 0..depth {
                            v += c[j] * m[j][i >> (2 * (j + 1))];
                        }
                        *o = v;
                    }
                });
            }
        });
        out
    }

    fn update(&mut self, r: &dyn Reduce, s: &[f64], y: &[f64]) {
        if self.factors != BandFactors::Curvature {
            return;
        }
        let depth = self.depth;
        let sy = Self::projections(r, s, y, depth);
        let yy = Self::projections(r, y, y, depth);
        let h = sy[0] / yy[0];
        for j in 0..=depth {
            let (num, den) = if j < depth { (sy[j] - sy[j + 1], yy[j] - yy[j + 1]) } else { (sy[j], yy[j]) };
            let a = num / den;
            self.alpha[j] = if num > 0.0 && den > 0.0 && a.is_finite() { a.clamp(h / CLAMP, h * CLAMP) } else { h };
        }
    }

    fn scaled(&self) -> bool {
        self.factors != BandFactors::Curvature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lbfgs::Local;

    fn vector(n: usize, seed: u64) -> Vec<f64> {
        (0..n).map(|i| (((i as u64 * 7919 + seed * 104729) % 1009) as f64) / 97.0 - 5.0).collect()
    }

    /// `P` is symmetric, positive, has eigenvalue `alpha_j` on band `j`,
    /// and `a . P b` agrees with the band projections.
    #[test]
    fn bands_are_spd_with_the_given_factors() {
        let depth = 3;
        let n = 3 * 64 * 5;
        let p = Bands { depth, alpha: vec![0.5, 3.0, 0.2, 7.0], factors: BandFactors::PowerLaw(0.0) };
        let (a, b) = (vector(n, 1), vector(n, 2));
        let (pa, pb) = (p.apply(&a), p.apply(&b));
        let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
        assert!((dot(&a, &pb) - dot(&pa, &b)).abs() < 1e-9 * dot(&a, &pa).abs());
        assert!(dot(&a, &pa) > 0.0);
        // a . P b from the projections.
        let pij = Bands::projections(&Local, &a, &b, depth);
        let want: f64 = (0..=depth).map(|j| p.alpha[j] * if j < depth { pij[j] - pij[j + 1] } else { pij[j] }).sum();
        assert!((dot(&a, &pb) - want).abs() < 1e-9 * want.abs());
        // A detail of level 1 (zero mean over 16, constant over 4) is an eigenvector of alpha_1.
        let mut v = vec![0.0; n];
        v[..16].iter_mut().enumerate().for_each(|(i, x)| *x = if i < 4 { 3.0 } else { -1.0 });
        let pv = p.apply(&v);
        assert!(pv.iter().zip(&v).all(|(x, y)| (x - 3.0 * y).abs() < 1e-12));
        // Constants per group of 64 (band depth): alpha_depth.
        let w: Vec<f64> = (0..n).map(|i| (i / 64) as f64).collect();
        assert!(p.apply(&w).iter().zip(&w).all(|(x, y)| (x - 7.0 * y).abs() < 1e-12));
    }

    /// Equal factors: `P = alpha I`, exactly.
    #[test]
    fn flat_factors_are_a_multiple_of_identity() {
        let a = vector(4 * 256, 3);
        let p = Bands::power_law(4, 0.0);
        assert_eq!(p.apply(&a), a);
    }

    #[test]
    fn curvature_factors_are_bounded_and_positive() {
        let n = 16 * 32;
        let s = vector(n, 4);
        let y: Vec<f64> = s.iter().zip(vector(n, 5)).map(|(a, b)| 2.0 * a + 0.3 * b).collect();
        let mut p = Bands::curvature(2);
        p.update(&Local, &s, &y);
        let h = Bands::projections(&Local, &s, &y, 2)[0] / Bands::projections(&Local, &y, &y, 2)[0];
        assert!(h > 0.0);
        assert!(p.alpha.iter().all(|&a| a >= h / CLAMP && a <= h * CLAMP), "{:?} vs {h}", p.alpha);
    }
}
