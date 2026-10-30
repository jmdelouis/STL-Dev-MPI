//! Distributed synthesis: the map is the variable (STL-Dev
//! `variables="map"`), the loss is `sum w |S(x) - S_target|^2`, minimised
//! by the L-BFGS / strong Wolfe port of `stl_core::lbfgs`.  Every rank holds
//! the owned slice `[B, C, end - start]` of level 0.

use crate::comm::Comm;
use crate::decomposition::{build_partition_for, Decomposition};
use crate::halo::HaloExchange;
use num_complex::Complex64;
use std::time::Instant;
use stl_core::engine::{EngineConfig, Evaluator, LocalEngine, Partition};
use stl_core::lbfgs::{minimize, LbfgsOptions, Reduce};
use stl_core::local::LocalGeometry;
use stl_core::plan::Coefficient;

/// Global sums and maxima for L-BFGS, in rank order on every rank.
pub struct CommReduce<'c, C: Comm>(pub &'c C);

impl<C: Comm> Reduce for CommReduce<'_, C> {
    fn sum(&self, local: f64) -> f64 {
        let mut v = [local];
        self.0.allreduce_sum(&mut v);
        v[0]
    }

    fn max(&self, local: f64) -> f64 {
        self.0.allreduce_max(local)
    }
}

/// Loss, statistics and map gradient on the processes of `comm`.
pub struct Objective<'a, C: Comm> {
    pub geo: &'a LocalGeometry,
    pub cfg: &'a EngineConfig,
    pub plan: &'a [Coefficient],
    pub target: &'a [Complex64],
    /// Weight per coefficient (`len(plan)`), real and nonnegative.
    pub weights: Vec<f64>,
    pub batch: usize,
    pub channels: usize,
    pub comm: &'a C,
    pub dec: Decomposition,
    pub part: Partition,
    halo: HaloExchange<'a, C>,
    /// Seconds per phase, summed over the evaluations.
    pub timers: std::cell::RefCell<std::collections::BTreeMap<&'static str, f64>>,
    /// Backend replacing the CPU sweeps (a device engine), if any.
    evaluator: Option<std::cell::RefCell<Box<dyn Evaluator + 'a>>>,
}

impl<'a, C: Comm> Objective<'a, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        plan: &'a [Coefficient],
        target: &'a [Complex64],
        batch: usize,
        channels: usize,
        comm: &'a C,
    ) -> Self {
        let dec = Decomposition::new(geo, cfg.levels(), comm.size());
        let part = build_partition_for(geo, cfg, &dec, comm.rank());
        Self::with_partition(geo, cfg, plan, target, batch, channels, comm, dec, part)
    }

    /// [`new`](Self::new) on a decomposition and partition already built
    /// (by another objective of the same problem): `build_partition` is the
    /// longest part of the setup at large nside.
    #[allow(clippy::too_many_arguments)]
    pub fn with_partition(
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        plan: &'a [Coefficient],
        target: &'a [Complex64],
        batch: usize,
        channels: usize,
        comm: &'a C,
        dec: Decomposition,
        part: Partition,
    ) -> Self {
        assert_eq!(target.len(), batch * plan.len());
        let halo = HaloExchange::new(comm, &dec, &part);
        Objective {
            geo,
            cfg,
            plan,
            target,
            weights: vec![1.0; plan.len()],
            batch,
            channels,
            comm,
            dec,
            part,
            halo,
            timers: Default::default(),
            evaluator: None,
        }
    }

    /// Evaluates with `evaluator` (built on this objective's partition and
    /// plan) instead of the CPU sweeps.
    pub fn set_evaluator(&mut self, evaluator: Box<dyn Evaluator + 'a>) {
        self.evaluator = Some(std::cell::RefCell::new(evaluator));
    }

    pub fn take_evaluator(&mut self) -> Option<Box<dyn Evaluator + 'a>> {
        self.evaluator.take().map(|e| e.into_inner())
    }

    /// Phase timers of the evaluator since the last call, and its device memory.
    #[allow(clippy::type_complexity)]
    pub fn evaluator_report(&self) -> (Vec<(&'static str, f64)>, Option<(usize, usize)>) {
        match &self.evaluator {
            Some(e) => {
                let mut e = e.borrow_mut();
                (e.take_timers(), e.device_memory())
            }
            None => (Vec::new(), None),
        }
    }

    /// `(neighbour exchanges, global sums)` issued by the sweeps so far.
    pub fn exchange_calls(&self) -> (u64, u64) {
        self.halo.calls()
    }

    /// Owned range of level 0.
    pub fn owned(&self) -> (i64, i64) {
        self.part.owned[0]
    }

    pub fn owned_len(&self) -> usize {
        let (s, e) = self.owned();
        self.batch * self.channels * (e - s) as usize
    }

    /// Owned slice of a full-sky map `[B, C, N_0]`.
    pub fn slice(&self, full: &[f64]) -> Vec<f64> {
        let (s, e) = self.owned();
        let n0 = self.geo.pixels(0);
        full.chunks(n0).flat_map(|row| row[s as usize..e as usize].iter().copied()).collect()
    }

    /// Owned slice -> layout of level 0 (halo zero, filled by the engine).
    fn to_layout(&self, x: &[f64]) -> Vec<f64> {
        let layout = &self.part.layouts[0];
        let (s, e) = self.owned();
        let len = (e - s) as usize;
        let mut out = vec![0.0; self.batch * self.channels * layout.len()];
        for r in 0..self.batch * self.channels {
            for i in 0..len {
                out[r * layout.len() + layout.pos(s + i as i64)] = x[r * len + i];
            }
        }
        out
    }

    fn tick(&self, name: &'static str, since: Instant) {
        *self.timers.borrow_mut().entry(name).or_default() += since.elapsed().as_secs_f64();
    }

    /// Statistics `[B, len(plan)]` of the map whose owned slice is `x`.
    pub fn statistics(&self, x: &[f64]) -> Vec<Complex64> {
        if let Some(ev) = &self.evaluator {
            return ev.borrow_mut().forward(&self.halo, &self.to_layout(x), false).0;
        }
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        engine.forward(self.plan)
    }

    /// Statistics and, per coefficient, the spatial variance of the terms
    /// it averages ([`LocalEngine::forward_moments`]).
    pub fn statistics_moments(&self, x: &[f64]) -> (Vec<Complex64>, Vec<f64>) {
        if let Some(ev) = &self.evaluator {
            return ev.borrow_mut().forward(&self.halo, &self.to_layout(x), true);
        }
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        engine.forward_moments(self.plan)
    }

    /// Loss and its cotangent on the statistics.
    fn loss_cotangent(&self, values: &[Complex64]) -> (f64, Vec<Complex64>) {
        let n = self.plan.len();
        let mut loss = 0.0;
        let cotangent: Vec<Complex64> = values
            .iter()
            .zip(self.target)
            .enumerate()
            .map(|(k, (v, t))| {
                let w = self.weights[k % n];
                let residual = v - t;
                let a = residual.norm();
                loss += w * (a * a);
                2.0 * w * residual
            })
            .collect();
        (loss, cotangent)
    }

    /// Loss and owned slice of its gradient (`loss_and_grad_local`, map variable).
    pub fn loss_and_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        self.loss_and_grad_with(x, |values| self.loss_cotangent(values))
    }

    /// [`loss_and_grad`](Self::loss_and_grad) with another loss: `loss`
    /// maps the statistics `[B, len(plan)]` to the loss and its cotangent.
    pub fn loss_and_grad_with(&self, x: &[f64], loss: impl FnOnce(&[Complex64]) -> (f64, Vec<Complex64>)) -> (f64, Vec<f64>) {
        let start = Instant::now();
        if let Some(ev) = &self.evaluator {
            let mut ev = ev.borrow_mut();
            let layout = self.to_layout(x);
            self.tick("host_to_layout", start);
            let (values, _) = ev.forward(&self.halo, &layout, false);
            drop(layout);
            self.tick("forward", start);
            let (loss, cotangent) = loss(&values);
            let start = Instant::now();
            let g0 = ev.gradient(&self.halo, &cotangent);
            self.tick("reverse", start);
            let (s, e) = self.owned();
            if s == e {
                return (loss, Vec::new());
            }
            let start = Instant::now();
            let layout = &self.part.layouts[0];
            let len = (e - s) as usize;
            let p0 = layout.pos(s);
            let owned: Vec<f64> = g0.chunks(layout.len()).flat_map(|row| row[p0..p0 + len].iter().copied()).collect();
            drop(g0);
            self.tick("host_owned_gradient", start);
            return (loss, owned);
        }
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        let values = engine.forward(self.plan);
        self.tick("forward", start);
        let (loss, cotangent) = loss(&values);
        let start = Instant::now();
        let mg = engine.reverse(self.plan, &cotangent);
        let g0 = engine.map_cotangent(mg);
        self.tick("reverse", start);
        (loss, engine.owned_values(0, &g0))
    }

    /// Bytes of the per-rank state that does not depend on the block size:
    /// maps and their cotangents, stored moduli and their cotangents.
    pub fn memory_floor_bytes(&self, history_size: usize) -> serde_json::Value {
        let bc = self.batch * self.channels;
        let held: Vec<usize> = self.part.layouts.iter().map(|l| l.len()).collect();
        let maps: usize = held.iter().sum::<usize>() * bc * 8;
        let levels = held.len();
        let moduli: usize = (0..levels).map(|n| n * held[n]).sum::<usize>() * bc * self.geo.l * 8;
        serde_json::json!({
            "maps_and_gradients": 2 * maps,
            "stored_moduli_and_cotangents": 2 * moduli,
            "lbfgs_history": 2 * (history_size + 2) * self.owned_len() * 8,
            "cells_held_per_level": held,
        })
    }
}

/// Ergodic weights `1 / (n sigma_i^2)` with `sigma_i^2 = V_i / N_i`: `V_i`
/// the spatial variance of the terms averaged by coefficient `i` on the
/// target, `N_i` the pixels of its native level (variance of a mean of
/// independent pixels), `n` the number of coefficients: the loss is the
/// reduced chi^2, the mean of `|S_i - S_target,i|^2 / sigma_i^2`.  (A
/// normalisation by the mean weight would be dominated by the most precise
/// coefficients and shrink the loss and its gradient below the L-BFGS
/// tolerances.)  A zero variance (an exactly constant coefficient) gets the
/// largest finite weight.
pub fn ergodic_weights(geo: &LocalGeometry, plan: &[Coefficient], variance: &[f64]) -> Vec<f64> {
    let mut w: Vec<f64> = plan
        .iter()
        .zip(variance)
        .map(|(c, &v)| {
            let n = geo.pixels(c.native_dg) as f64;
            if v > 0.0 {
                n / v
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let max_finite = w.iter().copied().filter(|v| v.is_finite()).fold(0.0, f64::max);
    w.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = max_finite);
    let n = w.len() as f64;
    w.iter_mut().for_each(|v| *v /= n);
    w
}

/// Run the synthesis from the owned slice `x` (updated in place); `report`
/// is called after every evaluation with `(index, loss, seconds)`.
pub fn run_lbfgs<C: Comm>(
    obj: &Objective<C>,
    x: &mut [f64],
    opts: &LbfgsOptions,
    report: impl FnMut(usize, f64, f64),
) -> Vec<f64> {
    run_lbfgs_with(obj.comm, x, opts, |p| obj.loss_and_grad(p), report)
}

/// L-BFGS on any loss of the owned slice `x`, the dot products summed over
/// `comm` (every rank of the problem).
pub fn run_lbfgs_with<C: Comm>(
    comm: &C,
    x: &mut [f64],
    opts: &LbfgsOptions,
    mut loss_and_grad: impl FnMut(&[f64]) -> (f64, Vec<f64>),
    mut report: impl FnMut(usize, f64, f64),
) -> Vec<f64> {
    let started = Instant::now();
    let mut history = Vec::new();
    let reduce = CommReduce(comm);
    let mut f = |p: &[f64]| {
        let (loss, g) = loss_and_grad(p);
        history.push(loss);
        report(history.len() - 1, loss, started.elapsed().as_secs_f64());
        (loss, g)
    };
    minimize(x, opts, &reduce, &mut f);
    history
}

/// Ensemble synthesis: `members` maps whose mean statistics match the
/// target, `loss = sum_i w_i |(1/B) sum_m S_{m,i} - T_i|^2`, so that the
/// ergodic scatter of the target goes into the scatter between the members.
/// The members are split over groups of ranks: the batch of `obj` holds the
/// members of this rank's group, `obj.comm` the ranks of the group (which
/// share the sky), `cross` the ranks of the same position in every group
/// (None with one group).  Every member gets the same cotangent
/// `2 w_i ((1/B) sum_m S_{m,i} - T_i) / B`.
///
/// `scatter = Some((lambda, sigma2))` adds the second moment of the
/// ensemble: with `D_i = sum_m |S_{m,i} - mean_i|^2 / (B - 1)` the scatter of
/// the members (unbiased) and `sigma2_i` its ergodic expectation,
/// `lambda / n sum_i (D_i / sigma2_i - 1)^2`; member k gets the extra
/// cotangent `4 lambda / n (D_i / sigma2_i - 1) / sigma2_i (S_{k,i} - mean_i) / (B - 1)`.
pub struct Ensemble<'o, 'a, C: Comm> {
    pub obj: &'o Objective<'a, C>,
    pub cross: Option<&'o C>,
    pub members: usize,
    pub target: &'o [Complex64],
    pub scatter: Option<(f64, &'o [f64])>,
}

impl<C: Comm> Ensemble<'_, '_, C> {
    /// Mean over all the members of the statistics `[b, len(plan)]` of this
    /// group's members (identical on every rank).
    pub fn mean(&self, values: &[Complex64]) -> Vec<Complex64> {
        let n = self.obj.plan.len();
        let mut sum = vec![0.0; 2 * n];
        for member in values.chunks(n) {
            for (i, v) in member.iter().enumerate() {
                sum[2 * i] += v.re;
                sum[2 * i + 1] += v.im;
            }
        }
        if let Some(cross) = self.cross {
            cross.allreduce_sum(&mut sum);
        }
        let b = self.members as f64;
        (0..n).map(|i| Complex64::new(sum[2 * i] / b, sum[2 * i + 1] / b)).collect()
    }

    /// Scatter of the members around `mean`, `D_i = sum_m |S_{m,i} - mean_i|^2 / (B - 1)`
    /// (identical on every rank; zero for one member).
    pub fn scatter(&self, values: &[Complex64], mean: &[Complex64]) -> Vec<f64> {
        let n = self.obj.plan.len();
        let mut d = vec![0.0; n];
        for member in values.chunks(n) {
            for (i, v) in member.iter().enumerate() {
                d[i] += (v - mean[i]).norm_sqr();
            }
        }
        if let Some(cross) = self.cross {
            cross.allreduce_sum(&mut d);
        }
        let dof = self.members.saturating_sub(1).max(1) as f64;
        d.iter_mut().for_each(|v| *v /= dof);
        d
    }

    /// Loss and owned slice of the gradient of this group's members.
    pub fn loss_and_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        self.obj.loss_and_grad_with(x, |values| {
            let n = self.obj.plan.len();
            let mean = self.mean(values);
            let b = self.members as f64;
            let mut loss = 0.0;
            let one: Vec<Complex64> = mean
                .iter()
                .zip(self.target)
                .zip(&self.obj.weights)
                .map(|((m, t), w)| {
                    let residual = m - t;
                    loss += w * residual.norm_sqr();
                    2.0 * w * residual / b
                })
                .collect();
            let mut cotangent = one.repeat(self.obj.batch);
            if let (Some((lambda, sigma2)), true) = (self.scatter, self.members > 1) {
                let d = self.scatter(values, &mean);
                let nf = n as f64;
                let coef: Vec<f64> = (0..n)
                    .map(|i| {
                        let r = d[i] / sigma2[i] - 1.0;
                        loss += lambda / nf * r * r;
                        4.0 * lambda / nf * r / sigma2[i] / (b - 1.0)
                    })
                    .collect();
                for (member, cot) in values.chunks(n).zip(cotangent.chunks_mut(n)) {
                    for i in 0..n {
                        cot[i] += coef[i] * (member[i] - mean[i]);
                    }
                }
            }
            (loss, cotangent)
        })
    }
}
