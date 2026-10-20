//! The synthesis on devices must follow STL-Dev's trajectory
//! (`benchmark_synthesis.py --mode local --variables map`, WebSky kappa at
//! nside 64, 20 L-BFGS iterations; CLAUDE.md validation 5), as the CPU
//! engine does (`stl-mpi/tests/synthesis.rs`).  `STL_RANKS` ranks (default 1).

use num_complex::Complex64;
use std::path::PathBuf;
use std::sync::Arc;
use stl_core::lbfgs::LbfgsOptions;
use stl_gpu::{Gpu, GpuEngine, GpuEvaluator, GpuOptions};
use stl_mpi::check::Fixture;
use stl_mpi::synthesis::{run_lbfgs, Objective};
use stl_mpi::threads::run_ranks;
use stl_mpi::Comm;

fn fixture() -> Fixture {
    let root = match std::env::var_os("STL_FIXTURES") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
    }
    .join("synthesis");
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
    Fixture::open(root.join(index[0]["name"].as_str().unwrap()))
}

fn same_printed(a: f64, b: f64) -> bool {
    format!("{a:.6e}") == format!("{b:.6e}")
}

#[test]
fn trajectory_matches_stl_dev() {
    if Gpu::count().map_or(true, |n| n == 0) {
        eprintln!("no device, test skipped");
        return;
    }
    let fx = fixture();
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let reference = fx.f64("loss_history");
    let reference_stats = fx.complex("target_values");
    let ranks: usize = std::env::var("STL_RANKS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let niter = fx.int("niter") as usize;
    let opts = LbfgsOptions { max_iter: niter, history_size: fx.int("history_size") as usize, ..Default::default() };
    let started = std::time::Instant::now();
    let out = run_ranks(ranks, |comm| {
        let gpu = Arc::new(Gpu::for_rank(comm.rank()).unwrap());
        let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
        let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
        let engine = GpuEngine::new(gpu, &geo, &cfg, probe.part.clone(), 1, 1, &GpuOptions::default());
        probe.set_evaluator(Box::new(GpuEvaluator::new(engine, &plan)));
        let target = probe.statistics(&probe.slice(&fx.f64("target_map")));
        let ev = probe.take_evaluator().unwrap();
        let mut obj = Objective::new(&geo, &cfg, &plan, &target, 1, 1, comm);
        obj.set_evaluator(ev);
        let mut x = obj.slice(&fx.f64("initial_map"));
        let history = run_lbfgs(&obj, &mut x, &opts, |i, loss, t| {
            if comm.rank() == 0 {
                let r = reference.get(i).copied().unwrap_or(f64::NAN);
                eprintln!("evaluation {:3}  gpu {loss:.6e}  stl-dev {r:.6e}  rel {:.1e}  t = {t:6.2} s", i + 1, (loss - r).abs() / r);
            }
        });
        drop(obj);
        (target, history)
    });
    let (target, history) = &out[0];
    let err = target.iter().zip(&reference_stats).fold(0f64, |m, (a, b)| m.max((a - b).norm() / (1e-11 + 1e-10 * b.norm())));
    assert!(err <= 1.0, "target statistics {err} x tolerance");
    assert_eq!(history.len(), reference.len(), "number of evaluations");
    let printed = history.iter().zip(&reference).take_while(|(a, b)| same_printed(**a, **b)).count();
    let worst = history.iter().zip(&reference).fold(0f64, |m, (a, b)| m.max((a - b).abs() / b));
    eprintln!(
        "P={ranks}: {printed} / {} loss values identical at the printed precision, max relative difference {worst:.1e}, {:.1} s",
        reference.len(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(printed, reference.len());
}
