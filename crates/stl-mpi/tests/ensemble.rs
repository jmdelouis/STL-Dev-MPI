//! Ensemble loss (`stl-synth --ensemble`): the statistics of the members are
//! averaged before the comparison with the target.  Checked on the synthesis
//! fixture against the single-map objective and by finite differences.

use num_complex::Complex64;
use std::path::PathBuf;
use stl_mpi::check::Fixture;
use stl_mpi::synthesis::{Ensemble, Objective};
use stl_mpi::threads::run_ranks;
use stl_mpi::Comm;

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

/// Two identical members: same loss as the single map, gradient halved per member.
#[test]
fn identical_members_match_the_single_map() {
    let fx = fixture();
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let (target_map, initial) = (fx.f64("target_map"), fx.f64("initial_map"));
    for ranks in [1usize, 2] {
        run_ranks(ranks, |comm| {
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            let target = probe.statistics(&probe.slice(&target_map));
            let single = Objective::new(&geo, &cfg, &plan, &target, 1, 1, comm);
            let (loss1, g1) = single.loss_and_grad(&single.slice(&initial));
            let rows = target.repeat(2);
            let obj = Objective::new(&geo, &cfg, &plan, &rows, 2, 1, comm);
            let ens = Ensemble { obj: &obj, cross: None, members: 2, target: &target };
            let both = [initial.clone(), initial.clone()].concat();
            let (loss2, g2) = ens.loss_and_grad(&obj.slice(&both));
            assert!((loss2 - loss1).abs() <= 1e-13 * loss1, "P={ranks}: loss {loss2} vs {loss1}");
            let n = g1.len();
            for m in 0..2 {
                let err = (0..n).fold(0f64, |e, k| e.max((g2[m * n + k] - 0.5 * g1[k]).abs()));
                let scale = g1.iter().fold(0f64, |e, v| e.max(v.abs()));
                assert!(err <= 1e-12 * scale, "P={ranks}: member {m} gradient {err} vs scale {scale}");
            }
        });
    }
}

/// Gradient of the ensemble loss of two different members by central differences.
#[test]
fn gradient_matches_finite_differences() {
    let fx = fixture();
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let (target_map, initial) = (fx.f64("target_map"), fx.f64("initial_map"));
    let n0 = initial.len();
    let second: Vec<f64> = initial.iter().enumerate().map(|(p, v)| 0.7 * v + 0.3 * (0.37 * p as f64).sin()).collect();
    let both = [initial.clone(), second].concat();
    run_ranks(2, |comm| {
        let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
        let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
        let target = probe.statistics(&probe.slice(&target_map));
        let rows = target.repeat(2);
        let obj = Objective::new(&geo, &cfg, &plan, &rows, 2, 1, comm);
        let ens = Ensemble { obj: &obj, cross: None, members: 2, target: &target };
        let x = obj.slice(&both);
        let (_, g) = ens.loss_and_grad(&x);
        let len = x.len() / 2;
        // Pixels of both members, perturbed on rank 0 only (the loss is collective).
        for &k in &[0, 7, len / 2, len - 1, len + 3, 2 * len - 5] {
            let h = 1e-5;
            let mut xp = x.clone();
            let mut xm = x.clone();
            if comm.rank() == 0 {
                xp[k] += h;
                xm[k] -= h;
            }
            let (lp, _) = ens.loss_and_grad(&xp);
            let (lm, _) = ens.loss_and_grad(&xm);
            if comm.rank() == 0 {
                let fd = (lp - lm) / (2.0 * h);
                let rel = (fd - g[k]).abs() / g[k].abs().max(1e-8);
                assert!(rel <= 1e-5, "pixel {k} of {n0}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
            }
        }
    });
}
