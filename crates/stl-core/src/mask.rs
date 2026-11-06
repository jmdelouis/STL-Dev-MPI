//! Masked averages of the full-sky statistics: every coefficient (a pixel
//! mean at its native level `n`) is averaged over the pixels of a mask at
//! that level instead of the whole sky, the fields (wavelets, moduli,
//! stored decimated moduli) being those of the full sky.  The mask of level
//! 0 is a set of NESTED pixels; that of level `n + 1` holds the parents of
//! the level-`n` mask (a parent is kept when at least one of its children
//! is), the definition of the core of a disc patch ([`crate::disc`]): with
//! the support the whole sky, a disc whose core is the mask has the same
//! statistics (to the summation order).  Several masks are evaluated in one
//! sweep; values, variances and cotangents are `[B, K, len(plan)]`.

use crate::healpix::npix;

/// `K` masks on every level of the pyramid.
#[derive(Clone, Debug)]
pub struct Masks {
    /// `weights[k][n]`: 1 on the pixels of mask `k` at level `n`, 0
    /// elsewhere, full-sky NESTED `[12 (nside / 2^n)^2]`.
    pub weights: Vec<Vec<Vec<f64>>>,
    /// `counts[k][n]`: pixels of mask `k` at level `n`.
    pub counts: Vec<Vec<usize>>,
}

impl Masks {
    /// Masks of the level-0 pixels `ids[k]` (NESTED, any order) on `levels`
    /// levels from `nside0`.
    pub fn new(nside0: i64, levels: usize, ids: &[Vec<i64>]) -> Self {
        let n0 = npix(nside0) as usize;
        let rows: Vec<Vec<f64>> = ids
            .iter()
            .map(|set| {
                let mut w = vec![0.0; n0];
                for &p in set {
                    assert!(p >= 0 && (p as usize) < n0, "mask pixel {p} out of range");
                    w[p as usize] = 1.0;
                }
                w
            })
            .collect();
        Self::from_weights(nside0, levels, rows)
    }

    /// Masks of the level-0 indicators `w0[k]` (`[12 nside0^2]`, nonzero:
    /// in the mask).
    pub fn from_weights(nside0: i64, levels: usize, w0: Vec<Vec<f64>>) -> Self {
        assert!(levels >= 1 && (nside0 >> (levels - 1)) >= 1, "too many levels for nside");
        let n0 = npix(nside0) as usize;
        let mut weights = Vec::with_capacity(w0.len());
        let mut counts = Vec::with_capacity(w0.len());
        for w in w0 {
            assert_eq!(w.len(), n0, "mask length must be 12 nside^2");
            assert!(w.iter().all(|v| v.is_finite()), "mask weights must be finite");
            let mut level: Vec<f64> = w.iter().map(|&v| if v != 0.0 { 1.0 } else { 0.0 }).collect();
            let mut per_level = Vec::with_capacity(levels);
            let mut c = Vec::with_capacity(levels);
            for n in 0..levels {
                if n > 0 {
                    level = level.chunks(4).map(|q| if q.iter().any(|&v| v != 0.0) { 1.0 } else { 0.0 }).collect();
                }
                let k = level.iter().filter(|&&v| v != 0.0).count();
                assert!(k > 0, "empty mask");
                c.push(k);
                per_level.push(level.clone());
            }
            weights.push(per_level);
            counts.push(c);
        }
        assert!(!weights.is_empty(), "no mask");
        Masks { weights, counts }
    }

    /// Masks given on every level, `weights[k][n]` (nonzero: in the mask),
    /// e.g. the pixels of the spin front end that read valid data only
    /// ([`crate::spin::SpinPyramid::valid_rows`]).
    pub fn from_levels(weights: Vec<Vec<Vec<f64>>>) -> Self {
        assert!(!weights.is_empty(), "no mask");
        let weights: Vec<Vec<Vec<f64>>> =
            weights.into_iter().map(|lv| lv.into_iter().map(|w| w.iter().map(|&v| if v != 0.0 { 1.0 } else { 0.0 }).collect()).collect()).collect();
        let counts: Vec<Vec<usize>> = weights.iter().map(|lv| lv.iter().map(|w| w.iter().filter(|&&v| v != 0.0).count()).collect()).collect();
        assert!(counts.iter().flatten().all(|&c| c > 0), "empty mask at some level");
        Masks { weights, counts }
    }

    /// Number of masks `K`.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True without masks (never: [`Masks::new`] refuses it).
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Positions in `block` (NESTED ids of level `n`) of the pixels of mask `k`.
    pub fn pick(&self, k: usize, n: usize, block: &[i64]) -> Vec<usize> {
        let w = &self.weights[k][n];
        block.iter().enumerate().filter(|(_, &id)| w[id as usize] != 0.0).map(|(i, _)| i).collect()
    }
}
