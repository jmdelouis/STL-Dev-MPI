//! `stl-synth`: synthesis of a HEALPix map from the ScatCov statistics of a
//! target map, on any number of ranks.  Same objective, defaults and report
//! fields as STL-Dev's `benchmark_synthesis.py --mode local --variables map`,
//! except the orientations: L = 4 by default (pass `--L 2` to reproduce
//! STL-Dev's runs).

use num_complex::Complex64;
use std::path::{Path, PathBuf};
use std::time::Instant;
use stl_core::engine::EngineConfig;
use stl_core::lbfgs::LbfgsOptions;
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Family};
use stl_core::stencil::Gauge;
use stl_mpi::io::{map_len, nside_of, read_map_range, SlicedMap};
use stl_mpi::synthesis::{ergodic_weights, run_lbfgs, Objective};
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
  --L N                 orientations (default 4; L = 2 leaves anisotropic, blobby syntheses)
  --kernel-size N       stencil (default 5)
  --fewer               has_fewer_convolutions definitions of S3/S4
  --block-pixels N      block size per level (default 16384)
  --conv-chunk N        rebuild the convolution geometry by sub-blocks of N rows at every
                        application (less memory, more time; default: whole block)
  --niter N             L-BFGS iterations (default 50)
  --weights W           loss weights: 'none' (default, sum |S - S_target|^2) or 'ergodic'
                        (reduced chi^2: mean of |S - S_target|^2 / sigma^2, sigma^2 = spatial
                        variance of each coefficient's terms on the target / pixels of its level)
  --family-weights F    extra factors on the weights of whole families, e.g. 'S1=10,S2=10'
                        (families mean, var, S1, S2, S3, S4; default 1)
  --history-size N      L-BFGS history (default 10)
  --out PREFIX          writes PREFIX.json, PREFIX_synthesis.fits, PREFIX_synthesis_nested.npy,
                        PREFIX_target.fits (default synthesis_rust_nside<N>)
  --ranks N             thread ranks in this process (builds without MPI)
  --gpu                 one CUDA device per rank (feature gpu): device = local rank modulo the
                        visible devices, or STL_GPU_DEVICE
  --gpu-block-rows N    owned cells per device block (default: from the device memory)
  --gpu-cache C         convolution geometry kept on the device between sweeps: 'full', 'transposition'
                        (the weights are recomputed), 'none', or 'auto' (default: full if it fits,
                        else transposition); budget from the device memory or STL_GPU_CACHE_MB
  --quiet               no line per evaluation
";

/// `S1=10,S2=10` -> [(S1, 10), (S2, 10)]: factors on the weights of whole families.
fn parse_family_weights(v: &str) -> Result<Vec<(Family, f64)>, String> {
    v.split(',')
        .map(|item| {
            let (name, f) = item.split_once('=').ok_or(format!("--family-weights: '{item}' is not FAMILY=FACTOR"))?;
            let fam = Family::parse(name.trim()).ok_or(format!("--family-weights: unknown family '{name}'"))?;
            let f: f64 = f.trim().parse().map_err(|_| format!("--family-weights: '{f}' is not a number"))?;
            if !(f.is_finite() && f >= 0.0) {
                return Err(format!("--family-weights: factor {f} must be finite and >= 0"));
            }
            Ok((fam, f))
        })
        .collect()
}

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
    pub conv_chunk: Option<usize>,
    pub ergodic_weights: bool,
    pub family_weights: Vec<(Family, f64)>,
    pub niter: usize,
    pub history_size: usize,
    pub out: Option<String>,
    pub ranks: usize,
    pub quiet: bool,
    pub gpu: bool,
    pub gpu_block_rows: usize,
    pub gpu_cache: String,
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
            l: 4,
            kernel_size: 5,
            fewer: false,
            block_pixels: 16384,
            conv_chunk: None,
            ergodic_weights: false,
            family_weights: Vec::new(),
            niter: 50,
            history_size: 10,
            out: None,
            ranks: 1,
            quiet: false,
            gpu: false,
            gpu_block_rows: 0,
            gpu_cache: "auto".into(),
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
                "--conv-chunk" => a.conv_chunk = Some(num(it.next(), k)?),
                "--weights" => {
                    a.ergodic_weights = match it.next().map(String::as_str) {
                        Some("none") => false,
                        Some("ergodic") => true,
                        _ => return Err("--weights is 'none' or 'ergodic'".into()),
                    }
                }
                "--family-weights" => {
                    let v = it.next().ok_or("--family-weights needs a value")?;
                    a.family_weights = parse_family_weights(v)?;
                }
                "--niter" => a.niter = num(it.next(), k)?,
                "--history-size" => a.history_size = num(it.next(), k)?,
                "--out" => a.out = Some(it.next().ok_or("--out needs a prefix")?.clone()),
                "--ranks" => a.ranks = num(it.next(), k)?,
                "--quiet" => a.quiet = true,
                "--gpu" => a.gpu = true,
                "--gpu-block-rows" => a.gpu_block_rows = num(it.next(), k)?,
                "--gpu-cache" => {
                    a.gpu_cache = match it.next().map(String::as_str) {
                        Some(v @ ("auto" | "full" | "transposition" | "none")) => v.to_string(),
                        _ => return Err("--gpu-cache is 'auto', 'full', 'transposition' or 'none'".into()),
                    }
                }
                "-h" | "--help" => return Err(USAGE.into()),
                other => return Err(format!("unknown option {other}\n\n{USAGE}")),
            }
        }
        if a.target.as_os_str().is_empty() {
            return Err(format!("--target is required\n\n{USAGE}"));
        }
        if a.gpu && !cfg!(feature = "gpu") {
            return Err("--gpu: built without the feature gpu".into());
        }
        Ok(a)
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Standard normal value of pixel `id` for `seed` (counter-based: the
/// same map whatever the number of ranks and the decomposition).
pub fn pixel_normal(seed: u64, id: i64) -> f64 {
    let a = splitmix64(seed ^ splitmix64(id as u64));
    let b = splitmix64(a);
    let u1 = ((a >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((b >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Global sum of a local value (rank order).
fn global_sum<C: Comm>(comm: &C, local: f64) -> f64 {
    let mut v = [local];
    comm.allreduce_sum(&mut v);
    v[0]
}

/// A memory counter of this process from `/proc/self/status`, in bytes
/// (`VmRSS`: resident now, `VmHWM`: peak resident); 0 where unavailable.
pub fn proc_memory(key: &str) -> i64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<i64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

/// The whole command on the rank of `comm`; returns the JSON report on
/// rank 0.  No rank holds a full-sky map: inputs are read and outputs
/// written by owned slices of level 0.
pub fn run<C: Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    let root = comm.rank() == 0;
    let npix = map_len(&args.target)?;
    let nside = nside_of(npix)?;
    let j = args.j.unwrap_or_else(|| (64 - (nside as u64).leading_zeros() as usize).saturating_sub(2).max(1));
    let mut geo = LocalGeometry::new(nside, args.kernel_size, args.l, Gauge::Cosmo, args.block_pixels);
    geo.conv_chunk = args.conv_chunk;
    let cfg = EngineConfig::new(j, args.fewer);
    let plan = build_plan(j, args.l, &cfg.j_to_dg, 1, None, &Family::ALL);
    let started_setup = Instant::now();
    let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
    #[allow(unused_mut)]
    let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
    #[allow(unused_mut)]
    let mut gpu_info = serde_json::Value::Null;
    #[cfg(feature = "gpu")]
    if args.gpu {
        let (ev, info) = gpu::evaluator(args, &geo, &cfg, &plan, &probe.part, comm.rank())?;
        probe.set_evaluator(ev);
        gpu_info = info;
    }
    let (s, e) = probe.owned();
    let (s, e) = (s as usize, e as usize);

    // Target slice, standardised with global moments (two passes).
    let raw = read_map_range(&args.target, s, e)?;
    let bad = global_sum(comm, raw.iter().filter(|v| !v.is_finite()).count() as f64);
    if bad > 0.0 {
        return Err(format!("{}: {bad} non-finite values", args.target.display()));
    }
    let mean = global_sum(comm, raw.iter().sum()) / npix as f64;
    let std = (global_sum(comm, raw.iter().map(|x| (x - mean) * (x - mean)).sum()) / npix as f64).sqrt();
    let target: Vec<f64> = raw.iter().map(|v| (v - mean) / std).collect();
    drop(raw);

    // Initial slice (standardised units).
    let mut init_info = serde_json::Value::Null;
    let initial: Vec<f64> = match &args.init {
        None => (s..e).map(|p| pixel_normal(args.seed, p as i64)).collect(),
        Some(path) => {
            let n_init = map_len(path)?;
            let nside_init = nside_of(n_init)?;
            if nside_init > nside || nside % nside_init != 0 {
                return Err(format!("{}: nside {nside_init} incompatible with {nside}", path.display()));
            }
            let factor = npix / n_init;
            if s % factor != 0 || e % factor != 0 {
                return Err(format!("--init at nside {nside_init}: rank ranges are not aligned on its pixels (use fewer ranks)"));
            }
            let mut coarse = read_map_range(path, s / factor, e / factor)?;
            if !args.init_standardized {
                coarse.iter_mut().for_each(|v| *v = (*v - mean) / std);
            }
            let noise_std = match args.init_noise {
                Some(v) => v,
                None if factor > 1 => {
                    // RMS of the target's scales below the init resolution.
                    let local: f64 = target
                        .chunks(factor)
                        .map(|c| {
                            let m = c.iter().sum::<f64>() / factor as f64;
                            c.iter().map(|v| (v - m) * (v - m)).sum::<f64>()
                        })
                        .sum();
                    (global_sum(comm, local) / npix as f64).sqrt()
                }
                None => 0.0,
            };
            init_info = serde_json::json!({"file": path, "nside": nside_init, "noise_std": noise_std});
            (s..e)
                .map(|p| {
                    let b = coarse[p / factor - s / factor];
                    if noise_std > 0.0 { b + noise_std * pixel_normal(args.seed, p as i64) } else { b }
                })
                .collect()
        }
    };

    // Resident memory once the inputs are read.
    let rss_baseline = proc_memory("VmRSS:");
    // Target statistics with the same distributed engine, and the spatial
    // variance of every coefficient's terms for the ergodic weights.
    let (target_stats, target_var) = probe.statistics_moments(&target);
    let evaluator = probe.take_evaluator();
    let (dec, part) = (probe.dec.clone(), probe.part.clone());
    drop(probe);
    let mut obj = Objective::with_partition(&geo, &cfg, &plan, &target_stats, 1, 1, comm, dec, part);
    if let Some(ev) = evaluator {
        obj.set_evaluator(ev);
    }
    if args.ergodic_weights {
        obj.weights = ergodic_weights(&geo, &plan, &target_var);
    }
    for (w, c) in obj.weights.iter_mut().zip(&plan) {
        if let Some(&(_, f)) = args.family_weights.iter().find(|(fam, _)| *fam == c.family) {
            *w *= f;
        }
    }
    let setup_seconds = started_setup.elapsed().as_secs_f64();
    let mut x = initial;

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
    let final_loss: f64 =
        final_stats.iter().zip(&target_stats).zip(&obj.weights).map(|((a, b), w)| w * (a - b).norm_sqr()).sum();
    // Residual per family, in units of the ergodic sigma (|S - S_t| / sigma).
    let mut per_family = serde_json::Map::new();
    for fam in Family::ALL {
        let z: Vec<f64> = plan
            .iter()
            .enumerate()
            .filter(|(_, c)| c.family == fam)
            .map(|(i, c)| {
                let sigma = (target_var[i] / geo.pixels(c.native_dg) as f64).sqrt();
                (final_stats[i] - target_stats[i]).norm() / sigma.max(1e-300)
            })
            .collect();
        if !z.is_empty() {
            let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64).sqrt();
            per_family.insert(fam.name().into(), serde_json::json!({"count": z.len(), "rms_sigma": rms, "max_sigma": z.iter().cloned().fold(0.0, f64::max)}));
        }
    }
    let mut timers: serde_json::Map<String, serde_json::Value> =
        obj.timers.borrow().iter().map(|(k, v)| (k.to_string(), serde_json::json!((v * 10.0).round() / 10.0))).collect();
    // Device phases (whole run, target statistics included) and memory.
    let (device_timers, device_memory) = obj.evaluator_report();
    for (k, v) in device_timers {
        timers.insert(k.to_string(), serde_json::json!((v * 100.0).round() / 100.0));
    }
    let device_peaks = comm.allgather_i64(device_memory.map_or(0, |m| m.0 as i64));
    if let Some(info) = gpu_info.as_object_mut() {
        info.insert("memory_peak_bytes_per_rank".into(), serde_json::json!(device_peaks));
        info.insert("memory_total_bytes".into(), serde_json::json!(device_memory.map(|m| m.1)));
    }
    let floor = obj.memory_floor_bytes(args.history_size);
    let floors: Vec<i64> = comm.allgather_i64(
        floor.as_object().unwrap().iter().filter_map(|(_, v)| v.as_i64()).sum(),
    );
    let peaks = comm.allgather_i64(proc_memory("VmHWM:"));
    let baselines = comm.allgather_i64(rss_baseline);

    // Outputs written by slices: rank 0 sizes the files, every rank writes
    // its range (the collectives order creation before the writes).
    let prefix = args.out.clone().unwrap_or_else(|| format!("synthesis_rust_nside{nside}"));
    let file = |suffix: &str| PathBuf::from(format!("{prefix}{suffix}"));
    let unstd = |v: &[f64]| v.iter().map(|x| x * std + mean).collect::<Vec<_>>();
    let outputs = [
        (SlicedMap::new(&file("_target.fits"), false), "TARGET", unstd(&target)),
        (SlicedMap::new(&file("_synthesis.fits"), false), "SYNTHESIS", unstd(&x)),
        (SlicedMap::new(&file("_synthesis_nested.npy"), false), "", unstd(&x)),
        (SlicedMap::new(&file("_synthesis_standardized.npy"), false), "", x.clone()),
    ];
    let mut status = 0.0;
    if root {
        for (out, column, _) in &outputs {
            if let Err(e) = out.create(npix, column) {
                eprintln!("stl-synth: {e}");
                status = 1.0;
            }
        }
    }
    if global_sum(comm, status) > 0.0 {
        return Err("could not create the output files".into());
    }
    for (out, _, values) in &outputs {
        if let Err(e) = out.write(s, values) {
            eprintln!("stl-synth: rank {}: {e}", comm.rank());
            status = 1.0;
        }
    }
    if global_sum(comm, status) > 0.0 {
        return Err("could not write the output files".into());
    }
    if !root {
        return Ok(None);
    }
    let report = serde_json::json!({
        "mode": "rust-local",
        "nside": nside,
        "target": {"file": args.target, "mean": mean, "std": std},
        "J": j, "L": args.l, "kernel_size": args.kernel_size, "fewer": args.fewer,
        "coefficients": plan.len(),
        "niter": args.niter,
        "evaluations": history.len(),
        "block_pixels": args.block_pixels,
        "conv_chunk": args.conv_chunk,
        "variables": "map",
        "ranks": comm.size(),
        "backend": if args.gpu { "gpu" } else { "cpu" },
        "gpu": gpu_info,
        "setup_seconds": (setup_seconds * 100.0).round() / 100.0,
        "seconds": (seconds * 100.0).round() / 100.0,
        "seconds_per_evaluation": (seconds / history.len().max(1) as f64 * 1000.0).round() / 1000.0,
        "loss_first": history.first(),
        "loss_last": history.last(),
        "loss_final_map": final_loss,
        "weights": if args.ergodic_weights { "ergodic" } else { "none" },
        "family_weights": args.family_weights.iter().map(|(f, w)| (f.name().to_string(), serde_json::json!(w))).collect::<serde_json::Map<_, _>>(),
        "final_residual_in_ergodic_sigma": per_family,
        "loss_history": history,
        "fits": [file("_target.fits"), file("_synthesis.fits")],
        "init": init_info,
        "memory_floor_bytes": floor,
        "memory_floor_bytes_per_rank": floors,
        "memory_peak_bytes_per_rank": peaks,
        "rss_baseline_bytes_per_rank": baselines,
        "timers_seconds": timers,
    });
    std::fs::write(file(".json"), serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    Ok(Some(report))
}

pub fn exists(p: &Path) -> bool {
    p.exists()
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::Args;
    use std::sync::Arc;
    use stl_core::engine::{EngineConfig, Evaluator, Partition};
    use stl_core::local::LocalGeometry;
    use stl_core::plan::Coefficient;
    use stl_gpu::{GeometryCache, Gpu, GpuEngine, GpuEvaluator, GpuOptions};

    /// The device engine of the rank and a description for the report.
    pub fn evaluator<'a>(
        args: &Args,
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        plan: &'a [Coefficient],
        part: &Partition,
        rank: usize,
    ) -> Result<(Box<dyn Evaluator + 'a>, serde_json::Value), String> {
        let gpu = Arc::new(Gpu::for_rank(rank)?);
        let cache = GeometryCache::parse(&args.gpu_cache).ok_or("--gpu-cache: unknown policy")?;
        let mut opts = GpuOptions { block_rows: args.gpu_block_rows, cache, ..Default::default() };
        if let Some(mb) = std::env::var("STL_GPU_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            opts.cache_bytes = mb << 20;
        }
        if let Some(t) = std::env::var("STL_GPU_THREADS").ok().and_then(|v| v.parse().ok()) {
            opts.threads = t;
        }
        let engine = GpuEngine::new(gpu.clone(), geo, cfg, part.clone(), 1, 1, &opts);
        let info = serde_json::json!({
            "device": gpu.name,
            "devices_visible": Gpu::count()?,
            "device_ordinal": gpu.ordinal(),
            "blocks_per_level": engine.blocks_per_level(),
            "block_rows": engine.block_rows(),
            "geometry_cache": engine.geometry_cache().0.name(),
            "geometry_cache_budget_bytes": engine.geometry_cache().1,
        });
        Ok((Box::new(GpuEvaluator::new(engine, plan)), info))
    }
}
