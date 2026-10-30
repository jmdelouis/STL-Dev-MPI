//! The objective of `stl-synth --gpu` evaluated on the device: the synthesis
//! and kappa sweeps read and leave their maps and gradients on the device,
//! and the level power, the pixel marginal, the kappa transform and the
//! kurtosis per level run there too (`stl_gpu::terms`).  The host keeps the
//! statistics' losses, the reductions of the terms' sums over pixels and
//! their communicators.  The same objective as the host closure of `run`:
//! element-wise expressions in the host's order, sums over pixels by chunks
//! in a fixed tree (equal to the host's sequential sums to rounding).
//!
//! Nothing but the engine's fields is on the device during a sweep, so the
//! engine's memory budget is the host path's: the synthesis gradient waits
//! on the host during the kappa sweep, then the maps go up again for
//! kappa's derivative (recomputed, the same values).  Transfers go through
//! a page-locked buffer: maps up twice, the gradient down twice and up once
//! per evaluation with the kappa statistics, else up once and down once.
//!
//! With several ranks per group of members, every rank works on its owned
//! slice (copied to and from the engine's layout on the device); the
//! moments of the pyramid are summed over the ranks, and the marginal sorts
//! every member's slice on the device and merges the ranks' orders on the
//! host (the distributed sample sort of `marginal`, without its local
//! sort).  Not with `--minkowski-weight` or `--freeze-nside`.

use crate::kappa::Transform;
use crate::kurtosis::Kurtosis;
use crate::level_power::LevelPower;
use crate::marginal::Marginal;
use std::sync::Arc;
use std::time::Instant;
use stl_gpu::terms::{add_product, copy_rows, marginal_row, DeviceTable, Pyramid, RadixSort};
use stl_gpu::{CudaSlice, Gpu, GpuEvaluator, PinnedHostSlice};
use stl_mpi::synthesis::{Ensemble, Objective};
use stl_mpi::Comm;

/// The kappa term: its transform, table on the device, evaluator and loss.
pub struct KappaPart<'o, 'k, 'a, C: Comm> {
    pub transform: &'o Transform,
    pub ensemble: Option<&'o Ensemble<'o, 'k, C>>,
    pub evaluator: Option<GpuEvaluator<'a>>,
}

pub struct DeviceObjective<'o, 's, 'k, 'a, C: Comm> {
    gpu: Arc<Gpu>,
    obj: &'o Objective<'s, C>,
    /// The ensemble loss (`None`: one member, the objective's own loss).
    ens: Option<&'o Ensemble<'o, 's, C>>,
    synthesis: GpuEvaluator<'a>,
    kappa: Option<(KappaPart<'o, 'k, 'a, C>, DeviceTable)>,
    level_power: Option<&'o LevelPower>,
    /// The marginal term and, on the device, the sorted target (one rank)
    /// or the identity order of the slice (several ranks).
    marginal: Option<(&'o Marginal, Marginal1)>,
    kurtosis: Option<&'o Kurtosis>,
    cross: Option<&'o C>,
    rows: usize,
    len: usize,
    /// First owned cell (global).
    start: usize,
    lay: Rows,
    /// Page-locked buffer of the members' maps for the transfers.
    stage: PinnedHostSlice<f64>,
}

/// A rank's owned rows `[B, len]` in the engine's layout `[B, held]` (owned
/// cells from position `p0`).
#[derive(Clone, Copy)]
struct Rows {
    rows: usize,
    len: usize,
    held: usize,
    p0: usize,
}

impl Rows {
    /// Owned rows -> layout (halo zero).
    fn to_layout(self, gpu: &Gpu, x: CudaSlice<f64>) -> CudaSlice<f64> {
        if self.held == self.len {
            return x;
        }
        let mut out = gpu.zeros::<f64>(self.rows * self.held);
        copy_rows(gpu, &x, self.len, 0, &mut out, self.held, self.p0, self.len, self.rows);
        out
    }

    /// Layout -> owned rows.
    fn from_layout(self, gpu: &Gpu, x: CudaSlice<f64>) -> CudaSlice<f64> {
        if self.held == self.len {
            return x;
        }
        let mut out = gpu.zeros::<f64>(self.rows * self.len);
        copy_rows(gpu, &x, self.held, self.p0, &mut out, self.len, 0, self.len, self.rows);
        out
    }
}

/// What the marginal keeps on the device.
enum Marginal1 {
    /// The whole sorted target (one rank: matched on the device).
    Sorted(CudaSlice<f64>),
    /// Positions `0..len` (several ranks: the target values come from the host).
    Identity(CudaSlice<u32>),
}

/// Whether the device objective applies: no Minkowski term, no frozen
/// scales.
pub fn applies(minkowski: bool, freeze: bool) -> bool {
    !minkowski && !freeze
}

impl<'o, 's, 'k, 'a, C: Comm> DeviceObjective<'o, 's, 'k, 'a, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        obj: &'o Objective<'s, C>,
        ens: Option<&'o Ensemble<'o, 's, C>>,
        synthesis: GpuEvaluator<'a>,
        kappa: Option<KappaPart<'o, 'k, 'a, C>>,
        level_power: Option<&'o LevelPower>,
        marginal: Option<&'o Marginal>,
        kurtosis: Option<&'o Kurtosis>,
        cross: Option<&'o C>,
    ) -> Self {
        let gpu = synthesis.gpu();
        let (s, e) = obj.owned();
        let len = (e - s) as usize;
        let layout = &obj.part.layouts[0];
        let (held, p0) = (layout.len(), if len > 0 { layout.pos(s) } else { 0 });
        let kappa = kappa.map(|k| {
            let f = &k.transform.f;
            let table = DeviceTable::new(&gpu, &f.x, &f.y, &f.d);
            (k, table)
        });
        let marginal = marginal.map(|m| {
            if obj.comm.size() == 1 {
                (m, Marginal1::Sorted(gpu.upload(&m.chunk)))
            } else {
                let identity: Vec<u32> = (0..len as u32).collect();
                (m, Marginal1::Identity(gpu.upload(&identity)))
            }
        });
        let stage = gpu.pinned(obj.batch * len);
        let rows = obj.batch;
        let lay = Rows { rows, len, held, p0 };
        DeviceObjective { gpu, obj, ens, synthesis, kappa, level_power, marginal, kurtosis, cross, rows, len, start: s as usize, lay, stage }
    }

    fn cross_sum(&self, v: f64) -> f64 {
        let mut total = [v];
        if let Some(c) = self.cross {
            c.allreduce_sum(&mut total);
        }
        total[0]
    }

    /// Loss and gradient (owned slices of this group's members) of the map
    /// `p`; `tick` takes the wall time of the phases.
    pub fn loss_and_grad(&mut self, p: &[f64], tick: &dyn Fn(&'static str, Instant)) -> (f64, Vec<f64>) {
        let gpu = self.gpu.clone();
        let obj = self.obj;
        let ex = obj.exchange();
        let comm = obj.comm;
        let (rows, len, lay) = (self.rows, self.len, self.lay);
        let t = Instant::now();
        let x = gpu.upload_staged(p, &mut self.stage);
        tick("device_transfers", t);

        // Synthesis statistics and their gradient.
        let t = Instant::now();
        let x = lay.to_layout(&gpu, x);
        let (values, _) = self.synthesis.forward_device(ex, x, false);
        let (mut loss, cotangent) = match self.ens {
            Some(ens) => ens.loss_cotangent(&values),
            None => obj.loss_cotangent(&values),
        };
        let g = self.synthesis.gradient_device(ex, &cotangent);
        let mut g = lay.from_layout(&gpu, g);
        let x = self.synthesis.take_map();
        let x = lay.from_layout(&gpu, x);
        tick("host_synthesis_objective", t);

        if let Some(lp) = self.level_power {
            let t = Instant::now();
            let levels = lp.levels;
            let pyr = Pyramid::new(&gpu, &x, rows, len, levels);
            let mut v: Vec<f64> = pyr.moments(&gpu, &x).iter().map(|m| m[0]).collect();
            comm.allreduce_sum(&mut v);
            for (k, val) in v.iter_mut().enumerate() {
                *val /= (lp.npix >> (2 * (k % levels))) as f64;
            }
            let w = lp.weight / lp.members as f64;
            let mut l = 0.0;
            let mut coef = vec![0.0; rows * levels];
            for row in 0..rows {
                for n in 0..levels {
                    let r = v[row * levels + n] / lp.target[n] - 1.0;
                    l += w * r * r;
                    coef[row * levels + n] = w * 2.0 * r / lp.target[n] * 2.0 / lp.npix as f64;
                }
            }
            pyr.level_power_grad(&gpu, &x, &coef, &mut g);
            loss += self.cross_sum(l);
            tick("host_level_power", t);
        }

        if let Some((mg, kept)) = &self.marginal {
            let t = Instant::now();
            let n = mg.npix as f64;
            let w = mg.weight / mg.members as f64;
            let coef = 2.0 * w / n;
            let mut sorter = RadixSort::new(&gpu, len);
            let mut sum = 0.0;
            for row in 0..rows {
                let perm = sorter.sort(&gpu, gpu.at(&x, row * len));
                sum += match kept {
                    Marginal1::Sorted(sorted) => marginal_row(&gpu, gpu.at(&x, row * len), perm, sorted, coef, gpu.at(&g, row * len)),
                    Marginal1::Identity(identity) => {
                        // The ranks' orders merged on the host: the target
                        // value of every owned pixel, in pixel order.
                        let perm = gpu.download(perm);
                        let t = gpu.upload(&mg.matched_sorted(&p[row * len..(row + 1) * len], &perm, self.start, comm));
                        marginal_row(&gpu, gpu.at(&x, row * len), identity, &t, coef, gpu.at(&g, row * len))
                    }
                };
            }
            let mut sums = [sum];
            comm.allreduce_sum(&mut sums);
            loss += self.cross_sum(w / n * sums[0]);
            tick("host_marginal", t);
        }

        // Terms on kappa = f(g): cotangent on kappa, chained by f'(g).
        let (source, dk) = match &mut self.kappa {
            Some((part, table)) => {
                let t = Instant::now();
                let (tg, tk) = (part.transform.g, part.transform.kappa);
                let (mut k, mut dk) = table.apply(&gpu, &x, tg, tk);
                tick("host_kappa_transform", t);
                if let (Some(ens), Some(ev)) = (part.ensemble, part.evaluator.as_mut()) {
                    // The kappa sweep with nothing else on the device: the
                    // gradient waits on the host, x and dk are dropped.
                    let t = Instant::now();
                    drop((x, dk));
                    let g_host = gpu.download_staged(&g, &mut self.stage);
                    drop(g);
                    tick("device_transfers", t);
                    let t = Instant::now();
                    let (values, _) = ev.forward_device(ex, lay.to_layout(&gpu, k), false);
                    let (lk, cotangent) = ens.loss_cotangent(&values);
                    let gk = lay.from_layout(&gpu, ev.gradient_device(ex, &cotangent));
                    k = lay.from_layout(&gpu, ev.take_map());
                    loss += lk;
                    tick("host_kappa_objective", t);
                    let t = Instant::now();
                    let x = gpu.upload_staged(p, &mut self.stage);
                    g = gpu.upload_staged(&g_host, &mut self.stage);
                    drop(g_host);
                    tick("device_transfers", t);
                    let t = Instant::now();
                    dk = table.apply(&gpu, &x, tg, tk).1;
                    add_product(&gpu, &mut g, &gk, Some(&dk));
                    tick("host_kappa_transform", t);
                } else {
                    drop(x);
                }
                (k, Some(dk))
            }
            None => (x, None),
        };

        if let Some(ku) = self.kurtosis {
            let t = Instant::now();
            let depth = ku.levels.iter().max().map_or(0, |m| m + 1);
            let pyr = Pyramid::new(&gpu, &source, rows, len, depth);
            let mom = pyr.moments(&gpu, &source);
            let nl = ku.levels.len();
            let mut sums: Vec<f64> = Vec::with_capacity(rows * nl * 2);
            for row in 0..rows {
                for &n in &ku.levels {
                    sums.extend(mom[row * depth + n]);
                }
            }
            comm.allreduce_sum(&mut sums);
            let w = ku.weight / ku.members as f64;
            let mut l = 0.0;
            let mut cf = vec![0.0; rows * nl * 3];
            for (k, s) in sums.chunks(2).enumerate() {
                let count = (ku.npix >> (2 * ku.levels[k % nl])) as f64;
                let v = s[0] / count;
                let kn = s[1] / count / (v * v);
                let r = kn / ku.target[k % nl] - 1.0;
                l += w * r * r;
                let coef = w * 2.0 * r / ku.target[k % nl] * 4.0 / (ku.npix as f64 * v * v);
                cf[k * 3..k * 3 + 3].copy_from_slice(&[coef, kn, v]);
            }
            pyr.kurtosis_grad(&gpu, &source, &ku.levels, &cf, dk.as_ref(), &mut g);
            loss += self.cross_sum(l);
            tick("host_kurtosis", t);
        }
        drop((source, dk));

        let t = Instant::now();
        let out = gpu.download_staged(&g, &mut self.stage);
        tick("device_transfers", t);
        (loss, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::path::PathBuf;
    use stl_core::plan::{build_plan, Family};
    use stl_gpu::{GpuEngine, GpuOptions};
    use stl_mpi::check::Fixture;
    use stl_mpi::synthesis::ergodic_weights;
    use stl_mpi::threads::run_ranks;

    fn fixture() -> Fixture {
        let root = match std::env::var_os("STL_FIXTURES") {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
        }
        .join("synthesis");
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
        Fixture::open(root.join(index[0]["name"].as_str().unwrap()))
    }

    /// kappa = exp(0.9 g) - 1 + 0.05 g on 41 knots.
    fn table() -> Vec<f64> {
        let x: Vec<f64> = (0..41).map(|k| -5.0 + 0.25 * k as f64).collect();
        let y: Vec<f64> = x.iter().map(|g| (0.9 * g).exp() - 1.0 + 0.05 * g).collect();
        [x, y].concat()
    }

    fn standardise(m: &[f64]) -> Vec<f64> {
        let mean = m.iter().sum::<f64>() / m.len() as f64;
        let s = (m.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / m.len() as f64).sqrt();
        m.iter().map(|v| (v - mean) / s).collect()
    }

    /// The objective on the device against the host terms on the same device
    /// sweeps, every term (ensemble with scatter, level power, marginal,
    /// kappa statistics, kurtosis of kappa), 2 members, nside 64: the kappa
    /// transform bit for bit, the loss to 1e-13 and the gradient to 1e-12 of
    /// its largest value (sums over pixels in chunks), on 1 and 3 ranks.
    #[test]
    fn device_objective_matches_host_terms() {
        if let Err(e) = Gpu::new(0) {
            eprintln!("no device, test skipped: {e}");
            return;
        }
        let fx = fixture();
        let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
        let (target, initial) = (standardise(&fx.f64("target_map")), standardise(&fx.f64("initial_map")));
        let npix = target.len();
        let second: Vec<f64> = initial.iter().enumerate().map(|(p, v)| 0.8 * v + 0.4 * (0.37 * p as f64).sin()).collect();
        let x = [initial.clone(), second].concat();
        let moved: Vec<f64> = x.iter().enumerate().map(|(p, v)| v + 0.05 * (0.11 * p as f64).cos()).collect();
        let kplan = build_plan(cfg.j_count, geo.l, &cfg.j_to_dg, 1, None, &Family::ALL[..4]);
        for ranks in [1, 3] {
        run_ranks(ranks, |comm| {
            let members = 2;
            let gpu = Arc::new(Gpu::new(0).unwrap());
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            let opts = GpuOptions { block_rows: 3000, ..Default::default() };
            let dev = GpuEvaluator::new(GpuEngine::new(gpu.clone(), &geo, &cfg, probe.part.clone(), members, 1, &opts), &plan);
            probe.set_evaluator(Box::new(dev.share(&plan, 1)));
            let (s, e) = probe.owned();
            let (s, len) = (s as usize, (e - s) as usize);
            let target = probe.slice(&target);
            let (stats, var) = probe.statistics_moments(&target);
            let rows = stats.repeat(members);
            let mut obj = Objective::with_partition(&geo, &cfg, &plan, &rows, members, 1, comm, probe.dec.clone(), probe.part.clone());
            obj.weights = ergodic_weights(&geo, &plan, &var).iter().map(|w| w * members as f64).collect();
            obj.set_evaluator(Box::new(dev.share(&plan, members)));
            let sigma2: Vec<f64> = plan.iter().enumerate().map(|(i, c)| (var[i] / geo.pixels(c.native_dg) as f64).max(1e-300)).collect();
            let ens = Ensemble { obj: &obj, cross: None, members, target: &stats, scatter: Some((1.0, &sigma2)) };
            let lp = LevelPower::new(10.0, &target, LevelPower::levels(64, s, s + len, comm), npix, members, comm);
            let mg = Marginal::new(100.0, &target, s, npix, members, comm);
            let tf = Transform::new(&table(), &target, 0.01, 1.3, npix, comm).unwrap();
            let kt = tf.apply(&target).0;
            let kdummy = vec![Complex64::new(0.0, 0.0); kplan.len()];
            let mut kprobe = Objective::with_partition(&geo, &cfg, &kplan, &kdummy, 1, 1, comm, probe.dec.clone(), probe.part.clone());
            kprobe.set_evaluator(Box::new(dev.share(&kplan, 1)));
            let (kstats, kvar) = kprobe.statistics_moments(&kt);
            let krows = kstats.repeat(members);
            let mut kobj = Objective::with_partition(&geo, &cfg, &kplan, &krows, members, 1, comm, probe.dec.clone(), probe.part.clone());
            kobj.weights = ergodic_weights(&geo, &kplan, &kvar).iter().map(|w| w * members as f64 * 0.1).collect();
            kobj.set_evaluator(Box::new(dev.share(&kplan, members)));
            let kens = Ensemble { obj: &kobj, cross: None, members, target: &kstats, scatter: None };
            let ku = Kurtosis::new(1.0, &kt, (0..lp.levels).step_by(2).collect(), npix, members, comm);
            let (x, moved) = (obj.slice(&x), obj.slice(&moved));

            // The kappa transform, bit for bit.
            let dt = DeviceTable::new(&gpu, &tf.f.x, &tf.f.y, &tf.f.d);
            let (k, dk) = tf.apply(&x);
            let (dkk, ddk) = dt.apply(&gpu, &gpu.upload(&x), tf.g, tf.kappa);
            assert!(gpu.download(&dkk) == k && gpu.download(&ddk) == dk, "kappa transform");

            let host = |p: &[f64]| {
                let (mut loss, mut g) = ens.loss_and_grad(p);
                let (l2, g2) = lp.loss_and_grad(p, len, comm, None);
                loss += l2;
                g.iter_mut().zip(&g2).for_each(|(a, b)| *a += b);
                let (lm, gm) = mg.loss_and_grad(p, len, s, comm, None);
                loss += lm;
                g.iter_mut().zip(&gm).for_each(|(a, b)| *a += b);
                let (k, dk) = tf.apply(p);
                let (lk, gk) = kens.loss_and_grad(&k);
                loss += lk;
                g.iter_mut().zip(gk.iter().zip(&dk)).for_each(|(a, (b, d))| *a += b * d);
                let (l4, g4) = ku.loss_and_grad(&k, len, comm, None);
                loss += l4;
                g.iter_mut().zip(g4.iter().zip(&dk)).for_each(|(a, (b, d))| *a += b * d);
                (loss, g)
            };
            let kappa = KappaPart { transform: &tf, ensemble: Some(&kens), evaluator: Some(dev.share(&kplan, members)) };
            let mut dobj = DeviceObjective::new(&obj, Some(&ens), dev.share(&plan, members), Some(kappa), Some(&lp), Some(&mg), Some(&ku), None);
            for p in [&x, &moved] {
                let (lh, gh) = host(p);
                let (ld, gd) = dobj.loss_and_grad(p, &|_, _| {});
                let scale = gh.iter().fold(0f64, |m, v| m.max(v.abs()));
                let err = gh.iter().zip(&gd).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
                eprintln!("{ranks} ranks, rank {}: loss {lh:.15e} vs {ld:.15e} (rel {:.1e}), gradient {err:.1e} of its maximum", comm.rank(), ((lh - ld) / lh).abs());
                assert!((lh - ld).abs() <= 1e-13 * lh.abs(), "loss {lh} vs {ld}");
                assert!(err <= 1e-12, "gradient {err:e}");
            }
        });
        }
    }
}
