//! Distributed statistics and gradients of STL-Dev `stats` fixtures under
//! MPI, compared with the fixtures and, optionally, with a saved run.
//!
//!     mpirun -n 4 stl-mpi-check fixtures/stats/<case> [--save out.bin] [--reference ref.bin]
//!
//! Exit status 1 when a comparison fails (CLAUDE.md validation 2-4).

use stl_mpi::check::{assemble, compare, from_flat, gather_to_root, run_rank, to_flat, Fixture};
use stl_mpi::mpi_comm::MpiComm;
use stl_mpi::Comm;

fn main() {
    let universe = mpi::initialize().expect("MPI initialisation failed");
    let comm = MpiComm::new(universe.world());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let dir = args.first().filter(|a| !a.starts_with("--")).expect("usage: stl-mpi-check <case dir> [--save f] [--reference f]");
    let fx = Fixture::open(dir);
    let rows = (fx.int("batch") * fx.int("channels")) as usize;
    let start = std::time::Instant::now();
    let result = run_rank(&fx, &comm);
    let seconds = start.elapsed().as_secs_f64();
    let Some(all) = gather_to_root(&comm, &result, rows) else { return };
    let assembled = assemble(&all, rows);
    let reference = flag("--reference").map(|path| {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        from_flat(&v, &assembled)
    });
    let report = compare(&fx, &assembled, reference.as_ref());
    if let Some(path) = flag("--save") {
        let bytes: Vec<u8> = to_flat(&assembled).iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    }
    let held: Vec<usize> = all.iter().map(|r| r.held[0]).collect();
    println!(
        "{} P={}: {:.2} s, stats {:.1e} x tol, gradients {:.1e} x tol, vs reference {:?}, level-0 cells held {:?} of {}: {}",
        fx.name(),
        comm.size(),
        seconds,
        report.stats_ratio,
        report.gradient_ratio,
        report.other_diff,
        held,
        12 * fx.int("nside") * fx.int("nside"),
        if report.passed() { "ok" } else { "FAILED" },
    );
    if !report.passed() {
        std::process::exit(1);
    }
}
