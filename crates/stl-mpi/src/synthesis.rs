//! Distributed synthesis: the map is the variable (STL-Dev
//! `variables="map"`), the loss is `sum w |S(x) - S_target|^2`, minimised
//! by the L-BFGS / strong Wolfe port of `stl_core::lbfgs`.  Every rank holds
//! the owned slice `[B, C, end - start]` of level 0.

use crate::comm::Comm;
use crate::decomposition::{build_partition_for, Decomposition};
use crate::halo::HaloExchange;
use num_complex::Complex64;
use std::time::Instant;
use stl_core::engine::{EngineConfig, LocalEngine, Partition};
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
        assert_eq!(target.len(), batch * plan.len());
        let dec = Decomposition::new(geo, cfg.levels(), comm.size());
        let part = build_partition_for(geo, cfg, &dec, comm.rank());
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
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        engine.forward(self.plan)
    }

    /// Statistics and, per coefficient, the spatial variance of the terms
    /// it averages ([`LocalEngine::forward_moments`]).
    pub fn statistics_moments(&self, x: &[f64]) -> (Vec<Complex64>, Vec<f64>) {
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        engine.forward_moments(self.plan)
    }

    /// Loss and owned slice of its gradient (`loss_and_grad_local`, map variable).
    pub fn loss_and_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let start = Instant::now();
        let mut engine =
            LocalEngine::distributed(self.geo, self.cfg, self.part.clone(), &self.halo, self.to_layout(x), self.batch, self.channels);
        let values = engine.forward(self.plan);
        self.tick("forward", start);
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
    mut report: impl FnMut(usize, f64, f64),
) -> Vec<f64> {
    let started = Instant::now();
    let mut history = Vec::new();
    let reduce = CommReduce(obj.comm);
    let mut f = |p: &[f64]| {
        let (loss, g) = obj.loss_and_grad(p);
        history.push(loss);
        report(history.len() - 1, loss, started.elapsed().as_secs_f64());
        (loss, g)
    };
    minimize(x, opts, &reduce, &mut f);
    history
}
