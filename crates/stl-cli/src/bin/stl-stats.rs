//! ScatCov statistics of HEALPix maps (see `stl_cli::STATS_USAGE`).
//!
//!     stl-stats --out stats.json --norm r000.npy --gpu -- r00?_nside1024_nested.npy

use std::path::PathBuf;
use stl_cli::{statistics_of_maps, Args, STATS_USAGE};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(split) = argv.iter().position(|a| a == "--") else {
        eprintln!("{STATS_USAGE}");
        std::process::exit(2);
    };
    let maps: Vec<PathBuf> = argv[split + 1..].iter().map(PathBuf::from).collect();
    let (mut out, mut norm, mut engine) = (None, None, Vec::new());
    let mut it = argv[..split].iter();
    while let Some(k) = it.next() {
        match k.as_str() {
            "--out" => out = it.next().cloned(),
            "--norm" => norm = it.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("{STATS_USAGE}");
                return;
            }
            _ => engine.push(k.clone()),
        }
    }
    let (Some(out), Some(first)) = (out, maps.first()) else {
        eprintln!("--out and at least one map are required\n\n{STATS_USAGE}");
        std::process::exit(2);
    };
    engine.extend(["--target".to_string(), first.to_string_lossy().into_owned()]);
    let args = match Args::parse(&engine) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    match launch(&args, &maps, norm.as_deref()) {
        Ok(Some(report)) => {
            if let Err(e) = std::fs::write(&out, serde_json::to_string(&report).unwrap()) {
                eprintln!("stl-stats: {out}: {e}");
                std::process::exit(1);
            }
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("stl-stats: {e}");
            std::process::exit(1);
        }
    }
}

/// The report on rank 0, `None` on the other ranks.
#[cfg(feature = "mpi")]
fn launch(args: &Args, maps: &[PathBuf], norm: Option<&std::path::Path>) -> Result<Option<serde_json::Value>, String> {
    use stl_mpi::Comm;
    let universe = mpi::initialize().ok_or("MPI initialisation failed")?;
    let comm = stl_mpi::mpi_comm::MpiComm::new(universe.world());
    let report = statistics_of_maps(args, maps, norm, &comm)?;
    Ok((comm.rank() == 0).then_some(report))
}

#[cfg(not(feature = "mpi"))]
fn launch(args: &Args, maps: &[PathBuf], norm: Option<&std::path::Path>) -> Result<Option<serde_json::Value>, String> {
    let results = stl_mpi::threads::run_ranks(args.ranks, |comm| statistics_of_maps(args, maps, norm, comm));
    let mut root = None;
    for (rank, r) in results.into_iter().enumerate() {
        match r {
            Ok(v) if rank == 0 => root = Some(v),
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(root)
}
