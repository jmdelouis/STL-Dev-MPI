//! `stl-synth`: synthesis of a HEALPix map from the ScatCov statistics of a
//! target map, on any number of ranks.  Same objective, defaults and report
//! fields as STL-Dev's `benchmark_synthesis.py --mode local --variables map`.

use num_complex::Complex64;
use std::path::{Path, PathBuf};
use std::time::Instant;
use stl_core::engine::EngineConfig;
use stl_core::lbfgs::LbfgsOptions;
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Family};
use stl_core::stencil::Gauge;
use stl_mpi::io::{nside_of, read_map, write_healpix_fits, write_npy};
use stl_mpi::synthesis::{run_lbfgs, Objective};
use stl_mpi::Comm;

pub const USAGE: &str = "\
stl-synth --target MAP [options]

  --target MAP          NESTED full-sky map (.npy or HEALPix .fits), standardised
  --init MAP            initial map (NESTED, nside <= target); default white noise
  --init-units U        'target' (default: standardised like the target) or 'standardized'
  --init-noise X        noise std added to --init: 'auto' (default, RMS of the target's
                        missing small scales when the init nside is lower), a number, 0
  --seed N              white-noise seed (default 11)
  --J N                 scales (default log2(nside) - 1)
  --L N                 orientations (default 2)
  --kernel-size N       stencil (default 5)
  --fewer               has_fewer_convolutions definitions of S3/S4
  --block-pixels N      block size per level (default 16384)
  --niter N             L-BFGS iterations (default 50)
  --history-size N      L-BFGS history (default 10)
  --out PREFIX          writes PREFIX.json, PREFIX_synthesis.fits, PREFIX_synthesis_nested.npy,
                        PREFIX_target.fits (default synthesis_rust_nside<N>)
  --ranks N             thread ranks in this process (builds without MPI)
  --quiet               no line per evaluation
";

#[derive(Clone, Debug)]
pub struct Args {
    pub target: PathBuf,
    pub init: Option<PathBuf>,
    pub init_standardized: bool,
    pub init_noise: Option<f64>,
    pub seed: u64,
    pub j: Option<usize>,
    pub l: usize,
    pub kernel_size: usize,
    pub fewer: bool,
    pub block_pixels: usize,
    pub niter: usize,
    pub history_size: usize,
    pub out: Option<String>,
    pub ranks: usize,
    pub quiet: bool,
}

impl Args {
    pub fn parse(argv: &[String]) -> Result<Args, String> {
        let mut a = Args {
            target: PathBuf::new(),
            init: None,
            init_standardized: false,
            init_noise: None,
            seed: 11,
            j: None,
            l: 2,
            kernel_size: 5,
            fewer: false,
            block_pixels: 16384,
            niter: 50,
            history_size: 10,
            out: None,
            ranks: 1,
            quiet: false,
        };
        let mut it = argv.iter();
        let num = |v: Option<&String>, k: &str| -> Result<usize, String> {
            v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: not an integer"))
        };
        while let Some(k) = it.next() {
            match k.as_str() {
                "--target" => a.target = it.next().ok_or("--target needs a path")?.into(),
                "--init" => a.init = Some(it.next().ok_or("--init needs a path")?.into()),
                "--init-units" => {
                    a.init_standardized = match it.next().map(String::as_str) {
                        Some("target") => false,
                        Some("standardized") => true,
                        _ => return Err("--init-units is 'target' or 'standardized'".into()),
                    }
                }
                "--init-noise" => {
                    let v = it.next().ok_or("--init-noise needs a value")?;
                    a.init_noise = if v == "auto" { None } else { Some(v.parse().map_err(|_| "--init-noise: not a number")?) };
                }
                "--seed" => a.seed = num(it.next(), k)? as u64,
                "--J" => a.j = Some(num(it.next(), k)?),
                "--L" => a.l = num(it.next(), k)?,
                "--kernel-size" => a.kernel_size = num(it.next(), k)?,
                "--fewer" => a.fewer = true,
                "--block-pixels" => a.block_pixels = num(it.next(), k)?,
                "--niter" => a.niter = num(it.next(), k)?,
                "--history-size" => a.history_size = num(it.next(), k)?,
                "--out" => a.out = Some(it.next().ok_or("--out needs a prefix")?.clone()),
                "--ranks" => a.ranks = num(it.next(), k)?,
                "--quiet" => a.quiet = true,
                "-h" | "--help" => return Err(USAGE.into()),
                other => return Err(format!("unknown option {other}\n\n{USAGE}")),
            }
        }
        if a.target.as_os_str().is_empty() {
            return Err(format!("--target is required\n\n{USAGE}"));
        }
        Ok(a)
    }
}

/// Deterministic standard normals (xoshiro256** + Box-Muller).
pub struct Normal {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Normal {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0u64; 4];
        for v in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *v = x ^ (x >> 31);
        }
        Normal { s, spare: None }
    }

    fn uniform(&mut self) -> f64 {
        let s = &mut self.s;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        ((r >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn sample(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * a.sin());
        r * a.cos()
    }
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn upsample(v: &[f64], factor: usize) -> Vec<f64> {
    v.iter().flat_map(|&x| std::iter::repeat_n(x, factor)).collect()
}

fn downsample(v: &[f64], factor: usize) -> Vec<f64> {
    v.chunks(factor).map(|c| c.iter().sum::<f64>() / factor as f64).collect()
}

/// Full-sky map from the ranks' owned slices (on rank 0; `None` elsewhere).
fn gather_map<C: Comm>(comm: &C, local: &[f64]) -> Option<Vec<f64>> {
    let size = comm.size();
    let all = comm.alltoallv_f64((0..size).map(|p| if p == 0 { local.to_vec() } else { Vec::new() }).collect());
    (comm.rank() == 0).then(|| all.concat())
}

/// The whole command on the rank of `comm`; returns the JSON report on rank 0.
pub fn run<C: Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    let root = comm.rank() == 0;
    let raw = read_map(&args.target)?;
    let nside = nside_of(raw.len())?;
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(format!("{}: non-finite values", args.target.display()));
    }
    let (mean, std) = mean_std(&raw);
    let target_map: Vec<f64> = raw.iter().map(|v| (v - mean) / std).collect();
    let npix = target_map.len();
    let j = args.j.unwrap_or_else(|| (64 - (nside as u64).leading_zeros() as usize).saturating_sub(2).max(1));
    let geo = LocalGeometry::new(nside, args.kernel_size, args.l, Gauge::Cosmo, args.block_pixels);
    let cfg = EngineConfig::new(j, args.fewer);
    let plan = build_plan(j, args.l, &cfg.j_to_dg, 1, None, &Family::ALL);

    // Initial map (standardised units).
    let mut rng = Normal::new(args.seed);
    let mut init_info = serde_json::Value::Null;
    let initial: Vec<f64> = match &args.init {
        None => (0..npix).map(|_| rng.sample()).collect(),
        Some(path) => {
            let mut coarse = read_map(path)?;
            let nside_init = nside_of(coarse.len())?;
            if nside_init > nside || nside % nside_init != 0 {
                return Err(format!("{}: nside {nside_init} incompatible with {nside}", path.display()));
            }
            if !args.init_standardized {
                coarse.iter_mut().for_each(|v| *v = (*v - mean) / std);
            }
            let factor = (npix / coarse.len()).max(1);
            let base = upsample(&coarse, factor);
            let noise_std = match args.init_noise {
                Some(s) => s,
                None if factor > 1 => {
                    let back = upsample(&downsample(&target_map, factor), factor);
                    (target_map.iter().zip(&back).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / npix as f64).sqrt()
                }
                None => 0.0,
            };
            init_info = serde_json::json!({"file": path, "nside": nside_init, "noise_std": noise_std});
            base.iter().map(|b| if noise_std > 0.0 { b + noise_std * rng.sample() } else { *b }).collect()
        }
    };

    let started_setup = Instant::now();
    // Target statistics with the same distributed engine.
    let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
    let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
    let target_stats = probe.statistics(&probe.slice(&target_map));
    drop(probe);
    let obj = Objective::new(&geo, &cfg, &plan, &target_stats, 1, 1, comm);
    let setup_seconds = started_setup.elapsed().as_secs_f64();
    let mut x = obj.slice(&initial);

    let opts = LbfgsOptions { max_iter: args.niter, history_size: args.history_size, ..Default::default() };
    let started = Instant::now();
    let history = if args.niter == 0 {
        Vec::new()
    } else {
        run_lbfgs(&obj, &mut x, &opts, |i, loss, t| {
            if root && !args.quiet {
                println!("[rust-local P={}] evaluation {:4}  loss = {loss:.6e}  t = {t:7.1} s", comm.size(), i + 1);
            }
        })
    };
    let seconds = started.elapsed().as_secs_f64();
    let final_stats = obj.statistics(&x);
    let final_loss: f64 = final_stats.iter().zip(&target_stats).map(|(a, b)| (a - b).norm_sqr()).sum();
    let timers: serde_json::Map<String, serde_json::Value> =
        obj.timers.borrow().iter().map(|(k, v)| (k.to_string(), serde_json::json!((v * 10.0).round() / 10.0))).collect();
    let floor = obj.memory_floor_bytes(args.history_size);
    let floors: Vec<i64> = comm.allgather_i64(
        floor.as_object().unwrap().iter().filter_map(|(_, v)| v.as_i64()).sum(),
    );
    let result = gather_map(comm, &x);
    let Some(result) = result else { return Ok(None) };

    let prefix = args.out.clone().unwrap_or_else(|| format!("synthesis_rust_nside{nside}"));
    let file = |suffix: &str| PathBuf::from(format!("{prefix}{suffix}"));
    let unstd = |v: &[f64]| v.iter().map(|x| x * std + mean).collect::<Vec<_>>();
    write_healpix_fits(&file("_target.fits"), &unstd(&target_map), "TARGET", false)?;
    write_healpix_fits(&file("_synthesis.fits"), &unstd(&result), "SYNTHESIS", false)?;
    write_npy(&file("_synthesis_nested.npy"), &unstd(&result))?;
    write_npy(&file("_synthesis_standardized.npy"), &result)?;
    let report = serde_json::json!({
        "mode": "rust-local",
        "nside": nside,
        "target": {"file": args.target, "mean": mean, "std": std},
        "J": j, "L": args.l, "kernel_size": args.kernel_size, "fewer": args.fewer,
        "coefficients": plan.len(),
        "niter": args.niter,
        "evaluations": history.len(),
        "block_pixels": args.block_pixels,
        "variables": "map",
        "ranks": comm.size(),
        "setup_seconds": (setup_seconds * 100.0).round() / 100.0,
        "seconds": (seconds * 100.0).round() / 100.0,
        "seconds_per_evaluation": (seconds / history.len().max(1) as f64 * 1000.0).round() / 1000.0,
        "loss_first": history.first(),
        "loss_last": history.last(),
        "loss_final_map": final_loss,
        "loss_history": history,
        "fits": [file("_target.fits"), file("_synthesis.fits")],
        "init": init_info,
        "memory_floor_bytes": floor,
        "memory_floor_bytes_per_rank": floors,
        "timers_seconds": timers,
    });
    std::fs::write(file(".json"), serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    Ok(Some(report))
}

pub fn exists(p: &Path) -> bool {
    p.exists()
}
