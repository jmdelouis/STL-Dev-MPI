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

/// The objective on devices equals the CPU one for 1, 2 and 3 ranks, over
/// consecutive evaluations at different maps (state reused).
#[test]
fn objective_gradient_rank_counts() {
    use stl_core::engine::EngineConfig;
    use stl_core::local::LocalGeometry;
    use stl_core::plan::{build_plan, Family};
    use stl_core::stencil::Gauge;
    if Gpu::count().map_or(true, |n| n == 0) {
        return;
    }
    let nside: i64 = std::env::var("STL_TEST_NSIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let j = (63 - nside.leading_zeros() as usize).saturating_sub(1);
    let geo = LocalGeometry::new(nside, 5, 2, Gauge::Cosmo, 16384);
    let cfg = EngineConfig::new(j, false);
    let plan = build_plan(j, 2, &cfg.j_to_dg, 1, None, &Family::ALL);
    let n0 = geo.pixels(0);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let target_map: Vec<f64> = (0..n0).map(|_| next()).collect();
    let maps: Vec<Vec<f64>> = (0..3).map(|_| (0..n0).map(|_| next()).collect()).collect();
    let run = |p: usize, gpu: bool| -> Vec<(f64, Vec<f64>)> {
        let per_rank = run_ranks(p, |comm| {
            let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
            let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
            if gpu {
                let g = Arc::new(Gpu::for_rank(comm.rank()).unwrap());
                let e = GpuEngine::new(g, &geo, &cfg, probe.part.clone(), 1, 1, &GpuOptions::default());
                probe.set_evaluator(Box::new(GpuEvaluator::new(e, &plan)));
            }
            let target = probe.statistics(&probe.slice(&target_map));
            let ev = probe.take_evaluator();
            let mut obj = Objective::new(&geo, &cfg, &plan, &target, 1, 1, comm);
            if let Some(ev) = ev {
                obj.set_evaluator(ev);
            }
            let out: Vec<(f64, Vec<f64>, (i64, i64))> =
                maps.iter().map(|m| {
                    let (l, g) = obj.loss_and_grad(&obj.slice(m));
                    (l, g, obj.owned())
                }).collect();
            drop(obj);
            out
        });
        (0..maps.len())
            .map(|k| {
                let mut g = vec![0.0; n0];
                for r in &per_rank {
                    let (s, e) = r[k].2;
                    g[s as usize..e as usize].copy_from_slice(&r[k].1);
                }
                (per_rank[0][k].0, g)
            })
            .collect()
    };
    let cpu = run(1, false);
    for p in [1usize, 2, 3] {
        let dev = run(p, true);
        for (k, ((lc, gc), (lg, gg))) in cpu.iter().zip(&dev).enumerate() {
            let scale = gc.iter().fold(0f64, |m, v| m.max(v.abs()));
            let dg = gc.iter().zip(gg).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
            let dl = (lc - lg).abs() / lc;
            eprintln!("nside {nside} P={p} evaluation {k}: loss {dl:.1e}, gradient {dg:.1e} (relative to CPU)");
            assert!(dl <= 1e-10 && dg <= 1e-9, "P={p} evaluation {k}");
        }
    }
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
