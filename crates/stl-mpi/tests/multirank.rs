//! Milestone 4 without an MPI launcher: 1, 2 and 4 ranks as threads must
//! reproduce the STL-Dev fixtures and each other (CLAUDE.md, validation 4).

use std::path::PathBuf;
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_mpi::check::{assemble, compare, run_rank, Assembled, Fixture};
use stl_mpi::threads::run_ranks;

fn fixtures() -> Vec<Fixture> {
    let root = match std::env::var_os("STL_FIXTURES") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
    }
    .join("stats");
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
    index.as_array().unwrap().iter().map(|m| Fixture::open(root.join(m["name"].as_str().unwrap()))).collect()
}

fn rows(fx: &Fixture) -> usize {
    (fx.int("batch") * fx.int("channels")) as usize
}

/// One rank through the distributed path is the serial engine, bit for bit.
#[test]
fn one_rank_is_the_serial_engine() {
    let fx = fixtures().into_iter().find(|f| f.name().contains("_b64")).unwrap();
    let distributed = assemble(&run_ranks(1, |c| run_rank(&fx, c)), rows(&fx));
    let (geo, cfg): (_, EngineConfig) = (fx.geometry(), fx.config());
    let plan = fx.plan();
    let mut engine = LocalEngine::new(&geo, &cfg, fx.f64("map0"), fx.int("batch") as usize, fx.int("channels") as usize);
    let values = engine.forward(&plan);
    let mg = engine.reverse(&plan, &fx.complex("gradient"));
    assert_eq!(distributed.values, values);
    assert_eq!(distributed.maps_gradient, mg);
    let (_, bands) = engine.analysis_vjp(mg);
    assert_eq!(distributed.bands, bands);
}

/// Gathering the owned slices on rank 0 (what `stl-mpi-check` does)
/// assembles the same result as collecting them in-process.
#[test]
fn gather_to_root_assembles_the_same_result() {
    let fx = fixtures().into_iter().find(|f| f.name().contains("_b64")).unwrap();
    let rows = rows(&fx);
    let gathered = run_ranks(4, |c| {
        let res = run_rank(&fx, c);
        stl_mpi::check::gather_to_root(c, &res, rows).map(|all| assemble(&all, rows))
    });
    let direct = assemble(&run_ranks(4, |c| run_rank(&fx, c)), rows);
    let on_root = gathered[0].as_ref().unwrap();
    assert!(gathered[1..].iter().all(Option::is_none));
    assert_eq!(on_root.values, direct.values);
    assert_eq!(on_root.maps_gradient, direct.maps_gradient);
    assert_eq!(on_root.bands, direct.bands);
    let round = stl_mpi::check::from_flat(&stl_mpi::check::to_flat(&direct), &direct);
    assert_eq!(round.bands, direct.bands);
}

#[test]
fn ranks_agree_with_fixtures_and_each_other() {
    for fx in fixtures() {
        let mut reference: Option<Assembled> = None;
        for size in [1usize, 2, 4] {
            let results = run_ranks(size, |c| run_rank(&fx, c));
            let held: Vec<usize> = results.iter().map(|r| r.held[0]).collect();
            let halo: Vec<usize> = results.iter().map(|r| r.halo[0]).collect();
            let a = assemble(&results, rows(&fx));
            let report = compare(&fx, &a, reference.as_ref());
            eprintln!(
                "{} P={size}: stats {:.1e} x tol, gradients {:.1e} x tol, vs P=1 {:?}; level-0 cells held {held:?} (halo {halo:?})",
                fx.name(),
                report.stats_ratio,
                report.gradient_ratio,
                report.other_diff,
            );
            assert!(report.passed(), "{} with {size} ranks: {report:?}", fx.name());
            reference.get_or_insert(a);
        }
    }
}
