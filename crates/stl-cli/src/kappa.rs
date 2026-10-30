//! Statistics of a fixed monotone transform of the map (`stl-synth
//! --kappa-table`).  The synthesis runs on the gaussianised map `g`; this
//! term matches the ScatCov statistics of `kappa = f(g)` to those of
//! `f(g_target)`, `f` the monotone cubic (PCHIP) interpolation of a table of
//! knots `(g, kappa)` (`tools/gaussianize.py knots`).  The gaussianised
//! statistics barely weigh the extreme tail of `g` (the halos), which carries
//! most of the power of kappa at small scales: this term weighs it as kappa
//! does.  Both `g` and `kappa` are standardised with the target's moments.

use stl_mpi::Comm;

/// Monotone piecewise-cubic Hermite interpolation (Fritsch-Butland
/// derivatives, as scipy's `PchipInterpolator`), linear beyond the ends.
/// (A constant extrapolation leaves the pixels pushed beyond the last knot
/// without any gradient from the kappa term: they pile up at the maximum.)
pub struct Monotone {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) d: Vec<f64>,
}

impl Monotone {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        let n = x.len();
        if n < 3 || y.len() != n {
            return Err(format!("monotone table: {n} knots, {} values (at least 3 each)", y.len()));
        }
        if x.windows(2).any(|w| w[1] <= w[0]) || y.windows(2).any(|w| w[1] < w[0]) {
            return Err("monotone table: knots must increase and values must not decrease".into());
        }
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let s: Vec<f64> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
        let mut d = vec![0.0; n];
        for k in 1..n - 1 {
            if s[k - 1] > 0.0 && s[k] > 0.0 {
                let (w1, w2) = (2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]);
                d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
            }
        }
        // One-sided three-point ends, kept monotone.
        let edge = |h0: f64, h1: f64, s0: f64, s1: f64| {
            let e = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
            if e.signum() != s0.signum() || s0 == 0.0 {
                0.0
            } else if s0.signum() != s1.signum() && e.abs() > 3.0 * s0.abs() {
                3.0 * s0
            } else {
                e
            }
        };
        d[0] = edge(h[0], h[1], s[0], s[1]);
        d[n - 1] = edge(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
        Ok(Monotone { x, y, d })
    }

    /// Value and derivative at `t`.
    pub fn eval(&self, t: f64) -> (f64, f64) {
        let n = self.x.len();
        if t <= self.x[0] {
            return (self.y[0] + self.d[0] * (t - self.x[0]), self.d[0]);
        }
        if t >= self.x[n - 1] {
            return (self.y[n - 1] + self.d[n - 1] * (t - self.x[n - 1]), self.d[n - 1]);
        }
        let k = self.x.partition_point(|&v| v <= t) - 1;
        let h = self.x[k + 1] - self.x[k];
        let u = (t - self.x[k]) / h;
        let (y0, y1, d0, d1) = (self.y[k], self.y[k + 1], self.d[k] * h, self.d[k + 1] * h);
        let (u2, u3) = (u * u, u * u * u);
        let v = (2.0 * u3 - 3.0 * u2 + 1.0) * y0 + (u3 - 2.0 * u2 + u) * d0 + (3.0 * u2 - 2.0 * u3) * y1 + (u3 - u2) * d1;
        let dv = (6.0 * u2 - 6.0 * u) * (y0 - y1) + (3.0 * u2 - 4.0 * u + 1.0) * d0 + (3.0 * u2 - 2.0 * u) * d1;
        (v, dv / h)
    }
}

/// `x` (standardised g) -> standardised `kappa = f(x std_g + mean_g)`.
pub struct Transform {
    pub(crate) f: Monotone,
    pub(crate) g: (f64, f64),
    pub(crate) kappa: (f64, f64),
}

impl Transform {
    /// `table`: the knots then the values (`2 K` numbers); `target`: owned
    /// slice of the standardised target, `(mean, std)` its moments in the
    /// units of the table; `comm`: the ranks sharing the sky.
    pub fn new<C: Comm>(table: &[f64], target: &[f64], mean: f64, std: f64, npix: usize, comm: &C) -> Result<Self, String> {
        if table.len() % 2 != 0 {
            return Err(format!("--kappa-table: {} numbers, expected knots then values", table.len()));
        }
        let k = table.len() / 2;
        let f = Monotone::new(table[..k].to_vec(), table[k..].to_vec())?;
        let mut tf = Transform { f, g: (mean, std), kappa: (0.0, 1.0) };
        let raw: Vec<f64> = target.iter().map(|&x| tf.f.eval(x * std + mean).0).collect();
        let mut sums = [raw.iter().sum::<f64>(), raw.iter().map(|v| v * v).sum::<f64>()];
        comm.allreduce_sum(&mut sums);
        let m = sums[0] / npix as f64;
        tf.kappa = (m, (sums[1] / npix as f64 - m * m).max(1e-300).sqrt());
        Ok(tf)
    }

    /// Standardised kappa and its derivative with respect to `x`, per pixel.
    pub fn apply(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ((gm, gs), (km, ks)) = (self.g, self.kappa);
        x.iter()
            .map(|&v| {
                let (k, dk) = self.f.eval(v * gs + gm);
                ((k - km) / ks, dk * gs / ks)
            })
            .unzip()
    }

    /// Mean and std of kappa on the target, in the units of the table.
    pub fn kappa_moments(&self) -> (f64, f64) {
        self.kappa
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::path::PathBuf;
    use stl_mpi::check::Fixture;
    use stl_mpi::synthesis::{ergodic_weights, Ensemble, Objective};
    use stl_mpi::threads::run_ranks;

    /// A convex, lognormal-like table: kappa = exp(0.9 g) - 1 + 0.05 g on 41 knots.
    fn table() -> Vec<f64> {
        let x: Vec<f64> = (0..41).map(|k| -5.0 + 0.25 * k as f64).collect();
        let y: Vec<f64> = x.iter().map(|g| (0.9 * g).exp() - 1.0 + 0.05 * g).collect();
        [x, y].concat()
    }

    #[test]
    fn interpolation_is_exact_on_knots_and_differentiable() {
        let t = table();
        let f = Monotone::new(t[..41].to_vec(), t[41..].to_vec()).unwrap();
        for k in 0..41 {
            assert!((f.eval(t[k]).0 - t[41 + k]).abs() < 1e-14);
        }
        for &v in &[-6.0, -4.9, -1.13, 0.0, 0.37, 2.5, 4.9, 5.5] {
            let h = 1e-6;
            let fd = (f.eval(v + h).0 - f.eval(v - h).0) / (2.0 * h);
            let d = f.eval(v).1;
            assert!((fd - d).abs() <= 1e-6 * d.abs().max(1.0), "at {v}: derivative {d} vs finite difference {fd}");
            assert!(d >= 0.0);
        }
    }

    fn fixture() -> Fixture {
        let root = match std::env::var_os("STL_FIXTURES") {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
        }
        .join("synthesis");
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
        Fixture::open(root.join(index[0]["name"].as_str().unwrap()))
    }

    /// A plan restricted to the S3 of one level (`--kappa-s3-nside`) gives the
    /// values of those coefficients in the full plan, and the gradient of the
    /// full plan's loss with zero weights elsewhere.
    #[test]
    fn s3_of_one_level_matches_the_full_plan() {
        let fx = fixture();
        let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
        let (target_map, initial) = (fx.f64("target_map"), fx.f64("initial_map"));
        let keep: Vec<usize> = (0..plan.len()).filter(|&i| plan[i].family == stl_core::plan::Family::S3 && plan[i].native_dg == 1).collect();
        assert!(!keep.is_empty());
        let sub: Vec<_> = keep.iter().map(|&i| plan[i].clone()).collect();
        run_ranks(2, |comm| {
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            let stats = probe.statistics(&probe.slice(&target_map));
            let mut full = Objective::new(&geo, &cfg, &plan, &stats, 1, 1, comm);
            full.weights = (0..plan.len()).map(|i| if keep.contains(&i) { 1.0 + 0.1 * i as f64 } else { 0.0 }).collect();
            let sub_stats: Vec<Complex64> = keep.iter().map(|&i| stats[i]).collect();
            let mut part = Objective::new(&geo, &cfg, &sub, &sub_stats, 1, 1, comm);
            part.weights = keep.iter().map(|&i| 1.0 + 0.1 * i as f64).collect();
            let x = full.slice(&initial);
            let (vf, vp) = (full.statistics(&x), part.statistics(&x));
            for (k, &i) in keep.iter().enumerate() {
                assert!((vf[i] - vp[k]).norm() <= 1e-13 * vf[i].norm().max(1e-300), "coefficient {i}");
            }
            let ((lf, gf), (lp, gp)) = (full.loss_and_grad(&x), part.loss_and_grad(&x));
            assert!((lf - lp).abs() <= 1e-12 * lf.abs());
            let scale = gf.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (a, b) in gf.iter().zip(&gp) {
                assert!((a - b).abs() <= 1e-12 * scale);
            }
        });
    }

    /// Gradient of the kappa term (statistics of two members with the ergodic
    /// weights of the transformed target, chain rule through the table)
    /// against central differences, on 2 ranks.
    #[test]
    fn gradient_matches_finite_differences() {
        let fx = fixture();
        let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
        let (target_map, initial) = (fx.f64("target_map"), fx.f64("initial_map"));
        let npix = initial.len();
        let std = |m: &[f64]| {
            let mean = m.iter().sum::<f64>() / m.len() as f64;
            let s = (m.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / m.len() as f64).sqrt();
            m.iter().map(|v| (v - mean) / s).collect::<Vec<_>>()
        };
        let (target_map, initial) = (std(&target_map), std(&initial));
        let second: Vec<f64> = initial.iter().enumerate().map(|(p, v)| 0.8 * v + 0.4 * (0.37 * p as f64).sin()).collect();
        let both = [initial.clone(), second].concat();
        run_ranks(2, |comm| {
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            let tf = Transform::new(&table(), &probe.slice(&target_map), 0.1, 1.2, npix, comm).unwrap();
            let (stats, var) = probe.statistics_moments(&tf.apply(&probe.slice(&target_map)).0);
            let rows = stats.repeat(2);
            let mut obj = Objective::new(&geo, &cfg, &plan, &rows, 2, 1, comm);
            obj.weights = ergodic_weights(&geo, &plan, &var);
            let ens = Ensemble { obj: &obj, cross: None, members: 2, target: &stats, scatter: None };
            let term = |x: &[f64]| {
                let (k, dk) = tf.apply(x);
                let (l, g) = ens.loss_and_grad(&k);
                (l, g.iter().zip(&dk).map(|(a, b)| a * b).collect::<Vec<_>>())
            };
            let x = obj.slice(&both);
            let (_, g) = term(&x);
            let len = x.len() / 2;
            for &k in &[0, 7, len / 2, len - 1, len + 3, 2 * len - 5] {
                let h = 1e-5;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                if comm.rank() == 0 {
                    xp[k] += h;
                    xm[k] -= h;
                }
                let (lp, _) = term(&xp);
                let (lm, _) = term(&xm);
                if comm.rank() == 0 {
                    let fd = (lp - lm) / (2.0 * h);
                    let rel = (fd - g[k]).abs() / g[k].abs().max(1e-8);
                    assert!(rel <= 1e-5, "pixel {k}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
                }
            }
        });
    }
}
