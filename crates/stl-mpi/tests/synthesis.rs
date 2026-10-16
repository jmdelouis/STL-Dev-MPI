//! Milestone 5: the L-BFGS / strong Wolfe synthesis must follow STL-Dev's
//! `benchmark_synthesis.py --mode local --variables map` trajectory
//! (fixtures `fixtures/synthesis`, CLAUDE.md validation 5).
//!
//! `cargo test --release -p stl-mpi --test synthesis -- --ignored --nocapture`
//! runs the 20 iterations (minutes); the first evaluation is always checked.

use num_complex::Complex64;
use std::path::PathBuf;
use stl_mpi::check::Fixture;
use stl_mpi::synthesis::{run_lbfgs, Objective};
use stl_core::lbfgs::LbfgsOptions;
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

/// Relative agreement of two loss values at the printed precision (`{:.6e}`).
fn same_printed(a: f64, b: f64) -> bool {
    format!("{a:.6e}") == format!("{b:.6e}")
}

/// Target statistics and first loss/gradient on 1 and 2 ranks.
#[test]
fn first_evaluation() {
    let fx = fixture();
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let reference = fx.complex("target_values");
    let history = fx.f64("loss_history");
    for ranks in [1usize, 2] {
        let out = run_ranks(ranks, |comm| {
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            let target = probe.statistics(&probe.slice(&fx.f64("target_map")));
            let obj = Objective::new(&geo, &cfg, &plan, &target, 1, 1, comm);
            let (loss, _) = obj.loss_and_grad(&obj.slice(&fx.f64("initial_map")));
            (target, loss)
        });
        let (target, loss) = &out[0];
        let err = target.iter().zip(&reference).fold(0f64, |m, (a, b)| m.max((a - b).norm() / (1e-11 + 1e-10 * b.norm())));
        assert!(err <= 1.0, "P={ranks}: target statistics {err} x tolerance");
        let rel = (loss - history[0]).abs() / history[0];
        eprintln!("P={ranks}: first loss {loss:.15e} vs STL-Dev {:.15e} (rel {rel:.1e})", history[0]);
        assert!(rel <= 1e-12, "P={ranks}: first loss {loss} vs {}", history[0]);
    }
}

#[test]
#[ignore]
fn trajectory_matches_stl_dev() {
    let fx = fixture();
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let reference = fx.f64("loss_history");
    let ranks: usize = std::env::var("STL_RANKS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let niter = fx.int("niter") as usize;
    let opts = LbfgsOptions { max_iter: niter, history_size: fx.int("history_size") as usize, ..Default::default() };
    let histories = run_ranks(ranks, |comm| {
        let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
        let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
        let target = probe.statistics(&probe.slice(&fx.f64("target_map")));
        let obj = Objective::new(&geo, &cfg, &plan, &target, 1, 1, comm);
        let mut x = obj.slice(&fx.f64("initial_map"));
        run_lbfgs(&obj, &mut x, &opts, |i, loss, t| {
            if comm.rank() == 0 {
                let r = reference.get(i).copied().unwrap_or(f64::NAN);
                eprintln!("evaluation {:3}  rust {loss:.6e}  stl-dev {r:.6e}  rel {:.1e}  t = {t:6.1} s", i + 1, (loss - r).abs() / r);
            }
        })
    });
    let history = &histories[0];
    assert_eq!(history.len(), reference.len(), "number of evaluations");
    let printed = history.iter().zip(&reference).take_while(|(a, b)| same_printed(**a, **b)).count();
    eprintln!("P={ranks}: {printed} of {} evaluations identical at the printed precision", history.len());
    for (i, (a, b)) in history.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() / b <= 1e-6, "evaluation {}: {a:e} vs {b:e}", i + 1);
    }
}
