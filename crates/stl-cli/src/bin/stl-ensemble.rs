//! Ensemble synthesis with the objective of P. Campeti (see
//! `stl_cli::ensemble::cli::USAGE`).
//!
//!     stl-ensemble calibrate --references r000.npy ... r009.npy --nside 128 --out cal.json
//!     stl-ensemble run --calibration cal.json --niter 100 --gpu --out ens

use stl_cli::ensemble::cli::{run_calibrate, run_synthesis, Args};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    match launch(&args) {
        Ok(Some(report)) => {
            let short = serde_json::json!({
                "evaluations": report["evaluations"], "seconds": report["seconds"],
                "seconds_per_evaluation": report["seconds_per_evaluation"], "loss_final": report["loss_final"],
            });
            println!("{short}");
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("stl-ensemble: {e}");
            std::process::exit(1);
        }
    }
}

fn command<C: stl_mpi::Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    if args.command == "calibrate" {
        run_calibrate(args, comm)
    } else {
        run_synthesis(args, comm)
    }
}

#[cfg(feature = "mpi")]
fn launch(args: &Args) -> Result<Option<serde_json::Value>, String> {
    let universe = mpi::initialize().ok_or("MPI initialisation failed")?;
    let comm = stl_mpi::mpi_comm::MpiComm::new(universe.world());
    command(args, &comm)
}

#[cfg(not(feature = "mpi"))]
fn launch(args: &Args) -> Result<Option<serde_json::Value>, String> {
    let results = stl_mpi::threads::run_ranks(args.ranks, |comm| command(args, comm));
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
