//! Power per level of the 4-children pyramid of the map (`stl-synth
//! --level-power-weight`).  With `x_0` the map and `x_{n+1}` the means of the
//! groups of 4 children of `x_n` (NESTED: consecutive ids), the details of
//! level n are `d_n = x_n - x_{n+1}` on the children, and `V_n = <d_n^2>`.
//! The term `w / B sum_m sum_n (V_n(map_m) / V_n(target) - 1)^2` holds the
//! power of every level at the target's, which the scattering statistics
//! leave loose at the finest level.

use stl_mpi::Comm;

/// Details and sum of their squares for `levels` levels of one row (its
/// length a multiple of 4^levels).
pub(crate) fn pyramid(row: &[f64], levels: usize) -> Vec<(Vec<f64>, f64)> {
    let mut cur = row.to_vec();
    let mut out = Vec::with_capacity(levels);
    for _ in 0..levels {
        let parents: Vec<f64> = cur.chunks(4).map(|c| c.iter().sum::<f64>() / 4.0).collect();
        let d: Vec<f64> = cur.chunks(4).zip(&parents).flat_map(|(c, &m)| c.iter().map(move |v| v - m)).collect();
        let s = d.iter().map(|v| v * v).sum();
        out.push((d, s));
        cur = parents;
    }
    out
}

pub struct LevelPower {
    pub weight: f64,
    /// `V_n` of the target, finest level first.
    pub target: Vec<f64>,
    pub levels: usize,
    npix: usize,
    members: usize,
}

impl LevelPower {
    /// Levels of the term: from the finest down to the details of nside 8,
    /// as far as every rank's range `[s, e)` stays aligned on the groups.
    pub fn levels<C: Comm>(nside: usize, s: usize, e: usize, comm: &C) -> usize {
        let wanted = (nside.trailing_zeros() as usize).saturating_sub(2);
        let mut aligned = 0;
        while aligned < wanted && s % (1 << (2 * (aligned + 1))) == 0 && e % (1 << (2 * (aligned + 1))) == 0 {
            aligned += 1;
        }
        (-comm.allreduce_max(-(aligned as f64))) as usize
    }

    /// `target`: owned slice of the target (standardised), `comm`: the ranks sharing the sky.
    pub fn new<C: Comm>(weight: f64, target: &[f64], levels: usize, npix: usize, members: usize, comm: &C) -> Self {
        let mut sums: Vec<f64> = pyramid(target, levels).iter().map(|(_, s)| *s).collect();
        comm.allreduce_sum(&mut sums);
        let target = sums.iter().enumerate().map(|(n, s)| s / (npix >> (2 * n)) as f64).collect();
        LevelPower { weight, target, levels, npix, members }
    }

    /// `V_n` of every row of `x` (rows of `len` owned values), `[row, level]`.
    fn variances<C: Comm>(&self, x: &[f64], len: usize, comm: &C) -> (Vec<Vec<(Vec<f64>, f64)>>, Vec<f64>) {
        let pyr: Vec<_> = x.chunks(len).map(|row| pyramid(row, self.levels)).collect();
        let mut v: Vec<f64> = pyr.iter().flat_map(|p| p.iter().map(|(_, s)| *s)).collect();
        comm.allreduce_sum(&mut v);
        for (k, val) in v.iter_mut().enumerate() {
            *val /= (self.npix >> (2 * (k % self.levels))) as f64;
        }
        (pyr, v)
    }

    /// The term and its gradient on the owned slice `x` of this group's
    /// members; `comm` shares the sky, `cross` links the groups.
    pub fn loss_and_grad<C: Comm>(&self, x: &[f64], len: usize, comm: &C, cross: Option<&C>) -> (f64, Vec<f64>) {
        let (pyr, v) = self.variances(x, len, comm);
        let w = self.weight / self.members as f64;
        let mut loss = 0.0;
        let mut grad = vec![0.0; x.len()];
        for (row, (p, g)) in pyr.iter().zip(grad.chunks_mut(len)).enumerate() {
            for (n, (d, _)) in p.iter().enumerate() {
                let r = v[row * self.levels + n] / self.target[n] - 1.0;
                loss += w * r * r;
                // d V_n / d x_p = 2 d_n[ancestor of p] / N_0 (the details of a group sum to zero).
                let coef = w * 2.0 * r / self.target[n] * 2.0 / self.npix as f64;
                let k = 1usize << (2 * n);
                for (i, gi) in g.iter_mut().enumerate() {
                    *gi += coef * d[i / k];
                }
            }
        }
        let mut total = [loss];
        if let Some(c) = cross {
            c.allreduce_sum(&mut total);
        }
        (total[0], grad)
    }

    /// Mean over all the members of `V_n / V_n(target)`, finest level first.
    pub fn ratios<C: Comm>(&self, x: &[f64], len: usize, comm: &C, cross: Option<&C>) -> Vec<f64> {
        let (_, v) = self.variances(x, len, comm);
        let mut sum = vec![0.0; self.levels];
        for (k, val) in v.iter().enumerate() {
            sum[k % self.levels] += val / self.target[k % self.levels];
        }
        if let Some(c) = cross {
            c.allreduce_sum(&mut sum);
        }
        sum.iter().map(|s| s / self.members as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stl_mpi::threads::run_ranks;

    fn field(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|p| ((p as f64 * 0.731 + seed).sin() + 0.4 * (p as f64 * 0.0137 + seed).cos()) * (1.0 + 0.3 * ((p % 7) as f64))).collect()
    }

    /// Gradient of the term against central differences, 2 members on 2 ranks.
    #[test]
    fn gradient_matches_finite_differences() {
        let nside = 16usize;
        let npix = 12 * nside * nside;
        let target = field(npix, 0.3);
        let maps = [field(npix, 1.1), field(npix, 2.7)];
        run_ranks(2, |comm| {
            let (s, e) = (comm.rank() * npix / 2, (comm.rank() + 1) * npix / 2);
            let levels = LevelPower::levels(nside, s, e, comm);
            assert_eq!(levels, 2, "nside 16: details of 16 and 8");
            let lp = LevelPower::new(0.7, &target[s..e], levels, npix, 2, comm);
            let len = e - s;
            let x: Vec<f64> = maps.iter().flat_map(|m| m[s..e].iter().copied()).collect();
            let (_, g) = lp.loss_and_grad(&x, len, comm, None);
            for &k in &[0, 5, len - 1, len + 9, 2 * len - 3] {
                let h = 1e-6;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                if comm.rank() == 0 {
                    xp[k] += h;
                    xm[k] -= h;
                }
                let (lp_, _) = lp.loss_and_grad(&xp, len, comm, None);
                let (lm_, _) = lp.loss_and_grad(&xm, len, comm, None);
                if comm.rank() == 0 {
                    let fd = (lp_ - lm_) / (2.0 * h);
                    let rel = (fd - g[k]).abs() / g[k].abs().max(1e-12);
                    assert!(rel <= 1e-5, "pixel {k}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
                }
            }
        });
    }
}
