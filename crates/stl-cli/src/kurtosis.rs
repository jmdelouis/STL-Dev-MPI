//! Kurtosis per level of the 4-children pyramid (`stl-synth
//! --kurtosis-weight`).  With the details `d_n` of level n as in
//! `level_power`, `K_n = <d_n^4> / <d_n^2>^2`, and the term
//! `w / B sum_m sum_{n in levels} (K_n(map_m) / K_n(target) - 1)^2` over the
//! selected levels.  With `--kappa-table` it acts on kappa = f(g): the
//! statistics of g and the S1, S2 of kappa leave the tails of the kappa
//! details too heavy at intermediate scales (excess kurtosis 3 times the
//! target's at nside 64 in a 4-member ensemble at nside 256).

use crate::level_power::pyramid;
use stl_mpi::Comm;

pub struct Kurtosis {
    pub weight: f64,
    /// Levels of the term (0: finest) and the target's `K_n` there.
    pub levels: Vec<usize>,
    pub target: Vec<f64>,
    pub(crate) npix: usize,
    pub(crate) members: usize,
}

/// Per row and level of the term: the details and the global `V_n`, `K_n`.
type Moments = (Vec<Vec<(Vec<f64>, f64)>>, Vec<(f64, f64)>);

impl Kurtosis {
    /// `target`: owned slice of the target (standardised), `comm`: the ranks
    /// sharing the sky; `levels` within `LevelPower::levels`.
    pub fn new<C: Comm>(weight: f64, target: &[f64], levels: Vec<usize>, npix: usize, members: usize, comm: &C) -> Self {
        let mut k = Kurtosis { weight, levels, target: Vec::new(), npix, members };
        k.target = k.moments(target, target.len(), comm).1.iter().map(|&(_, kn)| kn).collect();
        k
    }

    /// `V_n` and `K_n` of every row of `x` (rows of `len` owned values) and
    /// level of the term, `[row, level]`.
    fn moments<C: Comm>(&self, x: &[f64], len: usize, comm: &C) -> Moments {
        let depth = self.levels.iter().max().map_or(0, |m| m + 1);
        let pyr: Vec<_> = x.chunks(len).map(|row| pyramid(row, depth)).collect();
        let mut sums: Vec<f64> = pyr
            .iter()
            .flat_map(|p| self.levels.iter().flat_map(move |&n| [p[n].1, p[n].0.iter().map(|v| v * v * v * v).sum()]))
            .collect();
        comm.allreduce_sum(&mut sums);
        let nl = self.levels.len();
        let vk = sums
            .chunks(2)
            .enumerate()
            .map(|(k, s)| {
                let count = (self.npix >> (2 * self.levels[k % nl])) as f64;
                let v = s[0] / count;
                (v, s[1] / count / (v * v))
            })
            .collect();
        (pyr, vk)
    }

    /// The term and its gradient on the owned slice `x` of this group's
    /// members; `comm` shares the sky, `cross` links the groups.
    pub fn loss_and_grad<C: Comm>(&self, x: &[f64], len: usize, comm: &C, cross: Option<&C>) -> (f64, Vec<f64>) {
        let (pyr, vk) = self.moments(x, len, comm);
        let nl = self.levels.len();
        let w = self.weight / self.members as f64;
        let mut loss = 0.0;
        let mut grad = vec![0.0; x.len()];
        for (row, (p, g)) in pyr.iter().zip(grad.chunks_mut(len)).enumerate() {
            for (t, &n) in self.levels.iter().enumerate() {
                let (v, kn) = vk[row * nl + t];
                let r = kn / self.target[t] - 1.0;
                loss += w * r * r;
                // With c_i the mean of d^3 over the 4 siblings of detail i (ancestor of p):
                // d K_n / d x_p = 4 (d_i^3 - c_i - K_n V_n d_i) / (N_0 V_n^2).
                let coef = w * 2.0 * r / self.target[t] * 4.0 / (self.npix as f64 * v * v);
                let d = &p[n].0;
                let gd: Vec<f64> = d
                    .chunks(4)
                    .flat_map(|sib| {
                        let c = sib.iter().map(|di| di * di * di).sum::<f64>() / 4.0;
                        sib.iter().map(move |di| coef * (di * di * di - c - kn * v * di))
                    })
                    .collect();
                let k = 1usize << (2 * n);
                for (i, gi) in g.iter_mut().enumerate() {
                    *gi += gd[i / k];
                }
            }
        }
        let mut total = [loss];
        if let Some(c) = cross {
            c.allreduce_sum(&mut total);
        }
        (total[0], grad)
    }

    /// Mean over all the members of `K_n`, levels of the term.
    pub fn values<C: Comm>(&self, x: &[f64], len: usize, comm: &C, cross: Option<&C>) -> Vec<f64> {
        let (_, vk) = self.moments(x, len, comm);
        let nl = self.levels.len();
        let mut sum = vec![0.0; nl];
        for (k, &(_, kn)) in vk.iter().enumerate() {
            sum[k % nl] += kn;
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

    /// Heavy-tailed test field: a smooth part and sparse peaks.
    fn field(n: usize, seed: f64) -> Vec<f64> {
        (0..n)
            .map(|p| {
                let s = ((p as f64 * 0.731 + seed).sin() + 0.4 * (p as f64 * 0.0137 + seed).cos()) * (1.0 + 0.3 * ((p % 7) as f64));
                s + if (p * 37 + seed as usize) % 101 == 0 { 6.0 + (p % 5) as f64 } else { 0.0 }
            })
            .collect()
    }

    /// Kurtosis of the details of one level of a whole map.
    fn reference(m: &[f64], n: usize) -> f64 {
        let (d, s) = &pyramid(m, n + 1)[n];
        let v = s / d.len() as f64;
        d.iter().map(|x| x.powi(4)).sum::<f64>() / d.len() as f64 / (v * v)
    }

    /// Target kurtosis as on one whole map; gradient of the term against
    /// central differences, 2 members on 2 ranks, levels 0 and 2.
    #[test]
    fn gradient_matches_finite_differences() {
        let nside = 16usize;
        let npix = 12 * nside * nside;
        let target = field(npix, 0.3);
        let maps = [field(npix, 1.1), field(npix, 2.7)];
        run_ranks(2, |comm| {
            let (s, e) = (comm.rank() * npix / 2, (comm.rank() + 1) * npix / 2);
            let ku = Kurtosis::new(0.7, &target[s..e], vec![0, 2], npix, 2, comm);
            for (t, &n) in [0, 2].iter().enumerate() {
                let r = reference(&target, n);
                assert!((ku.target[t] / r - 1.0).abs() < 1e-12, "level {n}: {} vs {r}", ku.target[t]);
            }
            let len = e - s;
            let x: Vec<f64> = maps.iter().flat_map(|m| m[s..e].iter().copied()).collect();
            let (_, g) = ku.loss_and_grad(&x, len, comm, None);
            for &k in &[0, 5, 37, len - 1, len + 9, len + 101, 2 * len - 3] {
                let h = 1e-6;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                if comm.rank() == 0 {
                    xp[k] += h;
                    xm[k] -= h;
                }
                let (lp, _) = ku.loss_and_grad(&xp, len, comm, None);
                let (lm, _) = ku.loss_and_grad(&xm, len, comm, None);
                if comm.rank() == 0 {
                    let fd = (lp - lm) / (2.0 * h);
                    let rel = (fd - g[k]).abs() / g[k].abs().max(1e-12);
                    assert!(rel <= 1e-5, "pixel {k}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
                }
            }
        });
    }
}
