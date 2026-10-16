//! Synthesis of a HEALPix map from ScatCov statistics (see `stl_cli::USAGE`).
//!
//!     stl-synth --target kap_nside64_nested.npy --niter 20 --out run
//!     mpirun -n 28 stl-synth --target ... --niter 20 --out run   (feature `mpi`)

use stl_cli::{run, Args};

/// glibc raises its mmap threshold after the first large free, so the
/// per-block buffers (tens of MB) then come from the heap and are never
/// returned: the resident peak grows to the sum of retained blocks.  A
/// fixed threshold (128 kB) keeps every large buffer in its own mapping,
/// released when dropped.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn fix_malloc_threshold() {
    extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
    }
    const M_MMAP_THRESHOLD: i32 = -3;
    if std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_none() {
        // SAFETY: mallopt only changes allocator parameters, before any thread starts.
        unsafe {
            mallopt(M_MMAP_THRESHOLD, 128 * 1024);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn fix_malloc_threshold() {}

fn main() {
    fix_malloc_threshold();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let outcome = launch(&args);
    match outcome {
        Ok(Some(report)) => {
            let short = serde_json::json!({
                "ranks": report["ranks"], "evaluations": report["evaluations"],
                "seconds": report["seconds"], "seconds_per_evaluation": report["seconds_per_evaluation"],
                "loss_first": report["loss_first"], "loss_last": report["loss_last"],
            });
            println!("{short}");
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("stl-synth: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "mpi")]
fn launch(args: &Args) -> Result<Option<serde_json::Value>, String> {
    let universe = mpi::initialize().ok_or("MPI initialisation failed")?;
    let comm = stl_mpi::mpi_comm::MpiComm::new(universe.world());
    run(args, &comm)
}

#[cfg(not(feature = "mpi"))]
fn launch(args: &Args) -> Result<Option<serde_json::Value>, String> {
    let results = stl_mpi::threads::run_ranks(args.ranks, |comm| run(args, comm));
    let mut root = None;
    for (rank, r) in results.into_iter().enumerate() {
        match r {
            Ok(v) if rank == 0 => root = v,
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(root)
}
