//! Several ranks (threads, each driving a device; one device may serve
//! several ranks) against the fixtures and against one rank (CLAUDE.md
//! validation 4).  Under MPI: `STL_CHECK_GPU=1 tools/mpi_check.sh 2 3`.

use num_complex::Complex64;
use std::path::PathBuf;
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Family};
use stl_core::stencil::Gauge;
use stl_gpu::{Gpu, GpuOptions};
use stl_mpi::check::{assemble, compare, run_problem_gpu, Assembled, Fixture};
use stl_mpi::threads::run_ranks;

fn have_device() -> bool {
    match Gpu::count() {
        Ok(n) if n > 0 => true,
        r => {
            eprintln!("no device, test skipped: {r:?}");
            false
        }
    }
}

fn fixtures() -> Vec<Fixture> {
    let root = match std::env::var_os("STL_FIXTURES") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
    }
    .join("stats");
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
    index.as_array().unwrap().iter().map(|m| Fixture::open(root.join(m["name"].as_str().unwrap()))).collect()
}

fn run(fx: &Fixture, size: usize, unit: Option<i64>, opts: &GpuOptions) -> Assembled {
    let rows = (fx.int("batch") * fx.int("channels")) as usize;
    let (geo, cfg, plan) = (fx.geometry(), fx.config(), fx.plan());
    let (map0, gradient) = (fx.f64("map0"), fx.complex("gradient"));
    let (b, c) = (fx.int("batch") as usize, fx.int("channels") as usize);
    assemble(&run_ranks(size, |comm| run_problem_gpu(&geo, &cfg, &plan, &map0, &gradient, b, c, comm, unit, opts)), rows)
}

#[test]
fn ranks_agree_with_fixtures_and_each_other() {
    if !have_device() {
        return;
    }
    for fx in fixtures() {
        let mut reference: Option<Assembled> = None;
        for (size, block_rows) in [(1usize, 0usize), (2, 0), (4, 0), (4, 97)] {
            let opts = GpuOptions { block_rows, ..Default::default() };
            let a = run(&fx, size, None, &opts);
            let report = compare(&fx, &a, reference.as_ref());
            eprintln!(
                "{} P={size} blocks of {block_rows}: stats {:.1e} x tol, gradients {:.1e} x tol, vs P=1 {:?}",
                fx.name(),
                report.stats_ratio,
                report.gradient_ratio,
                report.other_diff
            );
            assert!(report.passed(), "{} with {size} ranks: {report:?}", fx.name());
            reference.get_or_insert(a);
        }
    }
}

#[test]
fn misaligned_ranges() {
    if !have_device() {
        return;
    }
    let fx = fixtures().into_iter().find(|f| f.name().contains("C2_B2_full_b64")).unwrap();
    let opts = GpuOptions::default();
    let reference = run(&fx, 1, None, &opts);
    for (size, unit) in [(3usize, 1i64), (5, 4), (7, 1)] {
        let a = run(&fx, size, Some(unit), &opts);
        let report = compare(&fx, &a, Some(&reference));
        eprintln!("{} P={size} unit={unit}: vs P=1 {:?}", fx.name(), report.other_diff);
        assert!(report.passed(), "P={size} unit={unit}: {report:?}");
    }
}

/// More ranks than cells at the coarsest level (nside 8, J = 3: 48 coarse
/// cells, 52 ranks), against the CPU engine.
#[test]
fn more_ranks_than_coarse_cells() {
    if !have_device() {
        return;
    }
    let geo = LocalGeometry::new(8, 3, 2, Gauge::Cosmo, 64);
    let cfg = EngineConfig::new(3, false);
    let plan = build_plan(3, 2, &cfg.j_to_dg, 1, None, &Family::ALL);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let map0: Vec<f64> = (0..geo.pixels(0)).map(|_| next()).collect();
    let gradient: Vec<Complex64> = plan.iter().map(|_| Complex64::new(next(), next())).collect();
    let mut engine = LocalEngine::new(&geo, &cfg, map0.clone(), 1, 1);
    let values = engine.forward(&plan);
    let mg = engine.reverse(&plan, &gradient);
    let (_, bands) = engine.analysis_vjp(mg.clone());
    let opts = GpuOptions { threads: 1, ..Default::default() };
    let results = run_ranks(52, |c| run_problem_gpu(&geo, &cfg, &plan, &map0, &gradient, 1, 1, c, None, &opts));
    let empty = results.iter().filter(|r| r.owned[2].0 == r.owned[2].1).count();
    let a = assemble(&results, 1);
    let diff = |x: &[f64], y: &[f64]| x.iter().zip(y).fold(0f64, |m, (a, b)| m.max((a - b).abs()));
    let dv = a.values.iter().zip(&values).fold(0f64, |m, (x, y)| m.max((x - y).norm()));
    let dg = (0..3).map(|d| diff(&a.maps_gradient[d], &mg[d]).max(diff(&a.bands[d], &bands[d]))).fold(0f64, f64::max);
    eprintln!("52 ranks, {empty} without coarse cells: statistics {dv:.1e}, gradients {dg:.1e} (vs CPU)");
    assert!(empty > 0);
    assert!(dv <= 1e-13 && dg <= 1e-13, "{dv:e} {dg:e}");
}
