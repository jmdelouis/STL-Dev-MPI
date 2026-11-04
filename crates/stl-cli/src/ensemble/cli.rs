//! `stl-ensemble calibrate | run`: the command of the ensemble objective.

use super::calibration::{calibrate, degrade, Calibration};
use super::objective::{EnsembleObjective, BLOCKS};
use crate::{pixel_normal, proc_memory};
use num_complex::Complex64;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use stl_core::engine::EngineConfig;
use stl_core::lbfgs::LbfgsOptions;
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::sht::Sht;
use stl_core::stencil::Gauge;
use stl_mpi::io::{read_map, write_npy};
use stl_mpi::synthesis::{run_lbfgs_with, Objective};
use stl_mpi::Comm;

/// Usage text of `stl-ensemble`, printed on `--help` and parse errors.
pub const USAGE: &str = "\
stl-ensemble calibrate --references MAP... --nside N --out CAL.json [options]
stl-ensemble run --calibration CAL.json --niter N --out PREFIX [options]

Ensemble synthesis with the objective of P. Campeti (STL_ensemble v2, variance
ratio): B members whose Gaussian maps g are the variable, kappa = f(g) - mean
through the pooled LUT of the references, and
  L = (L_GSC + L_PSC + L_PS + L_PDF + L_MF) / 5
on the SC statistics of g and of kappa (full plan, J = log2(nside) + 1, L = 4,
kernel 5), the 16 log C_l bands, the CIC PDF and the harmonic Minkowski
functionals of kappa, each on the mean over the members (against the mean of a
bootstrap draw of the references) and on their variance (against the variance
of the references).  See docs/campeti_merge.md.

calibrate:
  --references MAP...   reference maps (NESTED .npy / .fits, nside >= N), until the next option
  --nside N             nside of the synthesis (the maps are block-averaged: via 512 in f32)
  --bootstrap-ids I,J.. references whose mean is the target of the means (default: all of them
                        when there are as many as --ensemble, else a draw with replacement)
  --ensemble B          members (default 10)
  --seed N              seed of the bootstrap draw (default 20260914)
  --knots N             LUT knots (default 8192)
run:
  --calibration F       calibration JSON
  --niter N             L-BFGS iterations; --max-eval N evaluations (default 5 N / 4)
  --ensemble B          members (default 10, the calibration's draw size)
  --groups G            MPI groups of ranks sharing B / G members (default 1)
  --init MAP            Gaussian maps of the members ('{m}': one per member); default Gaussian
                        random fields with the mean C_l of the drawn references (--seed + m)
  --init-noise conditional
                        --init names physical maps of the members at a lower nside (PREFIX_m{m}_physical
                        of a coarser run): copied on the children, plus details drawn from the references'
                        per bin of the parent's value and local variation (docs/lagrangian_synthesis.md),
                        then Gaussianised with the calibration's LUT
  --init-upsample harmonic|copy
                        the coarse init of --init-noise conditional on the children: smooth (a_lm up to
                        2 nside_init, default; details around the same upsampling of the references'
                        parents) or copied (details around the parent means)
  --init-bins Q         quantile bins per feature of --init-noise conditional (default 8)
  --seed N              default 61022026
  --ps-variance-upper X upper band centre of the C_l variance term (default 383 at nside 128;
                        required otherwise)
  --history-size N      L-BFGS history (default 10)
  --checkpoint N        every N evaluations, the maps of the lowest loss so far: PREFIX_checkpoint_mMMM_*
                        and PREFIX_checkpoint.json (restart with --init PREFIX_checkpoint_m{m}_gaussian_nested.npy)
  --harmonic cpu|gpu    where the C_l, PDF and Minkowski terms run (default gpu with --gpu;
                        the device needs cuFFT, libcufft.so or STL_CUFFT)
  --out PREFIX          PREFIX.json, PREFIX_mMMM_gaussian_nested.npy, PREFIX_mMMM_physical_nested.npy
both:
  --block-pixels N      CPU engine block (default 16384)
  --ranks N             thread ranks (builds without MPI)
  --gpu, --gpu-conv F, --gpu-fields F, --gpu-cache C, --geometry-cache P, --gpu-block-rows N
                        as stl-synth
  --quiet
";

/// Options of `stl-ensemble` (see [`USAGE`]).
#[derive(Clone, Debug, Default)]
pub struct Args {
    /// The command: `calibrate` or `run`.
    pub command: String,
    /// `--references`: reference maps (NESTED `.npy` / `.fits`), up to the
    /// next option.
    pub references: Vec<PathBuf>,
    /// `--nside`: nside of the synthesis (`calibrate`; the references are
    /// block-averaged to it).
    pub nside: usize,
    /// `--bootstrap-ids`: references whose mean is the target of the means
    /// (default: all of them when there are as many as members, else a draw
    /// with replacement).
    pub bootstrap_ids: Option<Vec<usize>>,
    /// `--ensemble`: number of members `B` (default 10).
    pub ensemble: usize,
    /// `--groups`: MPI groups of ranks sharing `B / G` members (default 1).
    pub groups: usize,
    /// `--seed`: seed of the bootstrap draw (`calibrate`, default 20260914)
    /// or of the initial Gaussian fields (`run`, default 61022026, `+ m` per
    /// member).
    pub seed: Option<u64>,
    /// `--knots`: LUT knots (default 8192).
    pub knots: usize,
    /// `--calibration`: calibration JSON (`run`).
    pub calibration: Option<PathBuf>,
    /// `--niter`: L-BFGS iterations.
    pub niter: usize,
    /// `--max-eval`: L-BFGS evaluations (default `5 niter / 4`).
    pub max_eval: Option<usize>,
    /// `--init`: Gaussian maps g of the members (`{m}` replaced by the
    /// member); default Gaussian random fields of the calibration's C_l.
    pub init: Option<String>,
    /// `--init-noise conditional`: `--init` holds physical maps at a lower
    /// nside, refined by [`crate::conditional::Conditional`] on the references.
    pub init_conditional: bool,
    /// `--init-bins`: quantile bins per feature (default 8).
    pub init_bins: usize,
    /// `--init-upsample copy`: the coarse init copied on the children
    /// (default: harmonic upsampling).
    pub init_copy: bool,
    /// `--ps-variance-upper`: upper band centre of the C_l variance term
    /// (default 383 at nside 128, required otherwise).
    pub ps_variance_upper: Option<f64>,
    /// `--history-size`: L-BFGS history (default 10).
    pub history_size: usize,
    /// `--checkpoint`: evaluations between checkpoints of the lowest loss so
    /// far (0: none).
    pub checkpoint: usize,
    /// `--out`: calibration JSON (`calibrate`) or output prefix (`run`).
    pub out: Option<String>,
    /// `--block-pixels`: CPU engine block (default 16384).
    pub block_pixels: usize,
    /// `--ranks`: thread ranks in builds without MPI (default 1).
    pub ranks: usize,
    /// `--quiet`: no progress output.
    pub quiet: bool,
    /// `--gpu`: one CUDA device per rank (feature `gpu`).
    pub gpu: bool,
    /// `--gpu-conv` (`None`: `STL_GPU_CONV`, else dia).
    pub gpu_conv: Option<String>,
    /// `--gpu-fields` (`None`: `STL_GPU_FIELDS`, else auto).
    pub gpu_fields: Option<String>,
    /// `--gpu-cache`: convolution geometry kept on the device between
    /// sweeps (default `auto`).
    pub gpu_cache: String,
    /// `--geometry-cache`: where the convolution geometry is kept between
    /// evaluations (`None`: `--gpu-cache` without host copies).
    pub geometry_cache: Option<String>,
    /// `--gpu-block-rows`: owned cells per device block (0: from the device
    /// memory).
    pub gpu_block_rows: usize,
    /// `--harmonic`: `Some(true)` on the device.
    pub harmonic_gpu: Option<bool>,
}

impl Args {
    /// Parses `argv` (without the program name): the command, then the
    /// options with their defaults.  `Err` carries the usage or the message of
    /// a missing or invalid option.
    pub fn parse(argv: &[String]) -> Result<Args, String> {
        let mut a = Args {
            ensemble: 10,
            groups: 1,
            knots: 8192,
            history_size: 10,
            block_pixels: 16384,
            ranks: 1,
            gpu_cache: "auto".into(),
            init_bins: 8,
            ..Default::default()
        };
        let mut it = argv.iter().peekable();
        a.command = it.next().cloned().ok_or(USAGE)?;
        let num = |v: Option<&String>, k: &str| -> Result<usize, String> { v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: not an integer")) };
        while let Some(k) = it.next() {
            match k.as_str() {
                "--references" => {
                    while let Some(v) = it.peek() {
                        if v.starts_with("--") {
                            break;
                        }
                        a.references.push(PathBuf::from(it.next().unwrap()));
                    }
                }
                "--nside" => a.nside = num(it.next(), k)?,
                "--bootstrap-ids" => {
                    let v = it.next().ok_or("--bootstrap-ids needs a list")?;
                    a.bootstrap_ids = Some(v.split(',').map(|s| s.trim().parse().map_err(|_| "--bootstrap-ids: integers")).collect::<Result<_, _>>()?);
                }
                "--ensemble" => a.ensemble = num(it.next(), k)?,
                "--groups" => a.groups = num(it.next(), k)?,
                "--seed" => a.seed = Some(num(it.next(), k)? as u64),
                "--knots" => a.knots = num(it.next(), k)?,
                "--calibration" => a.calibration = Some(it.next().ok_or("--calibration needs a path")?.into()),
                "--niter" => a.niter = num(it.next(), k)?,
                "--max-eval" => a.max_eval = Some(num(it.next(), k)?),
                "--init" => a.init = Some(it.next().ok_or("--init needs a path")?.clone()),
                "--init-noise" => match it.next().map(|v| v.as_str()) {
                    Some("conditional") => a.init_conditional = true,
                    _ => return Err("--init-noise: conditional".into()),
                },
                "--init-bins" => a.init_bins = num(it.next(), k)?,
                "--init-upsample" => match it.next().map(|v| v.as_str()) {
                    Some("copy") => a.init_copy = true,
                    Some("harmonic") => a.init_copy = false,
                    _ => return Err("--init-upsample: harmonic or copy".into()),
                },
                "--ps-variance-upper" => a.ps_variance_upper = Some(it.next().ok_or("--ps-variance-upper needs a value")?.parse().map_err(|_| "--ps-variance-upper: a number")?),
                "--history-size" => a.history_size = num(it.next(), k)?,
                "--checkpoint" => a.checkpoint = num(it.next(), k)?,
                "--out" => a.out = Some(it.next().ok_or("--out needs a value")?.clone()),
                "--block-pixels" => a.block_pixels = num(it.next(), k)?,
                "--ranks" => a.ranks = num(it.next(), k)?,
                "--quiet" => a.quiet = true,
                "--gpu" => a.gpu = true,
                "--gpu-conv" => a.gpu_conv = Some(it.next().ok_or("--gpu-conv needs a value")?.clone()),
                "--gpu-fields" => a.gpu_fields = Some(it.next().ok_or("--gpu-fields needs a value")?.clone()),
                "--gpu-cache" => a.gpu_cache = it.next().ok_or("--gpu-cache needs a value")?.clone(),
                "--geometry-cache" => a.geometry_cache = Some(it.next().ok_or("--geometry-cache needs a value")?.clone()),
                "--gpu-block-rows" => a.gpu_block_rows = num(it.next(), k)?,
                "--harmonic" => {
                    a.harmonic_gpu = Some(match it.next().map(|s| s.as_str()) {
                        Some("gpu") => true,
                        Some("cpu") => false,
                        _ => return Err("--harmonic: cpu or gpu".into()),
                    })
                }
                "-h" | "--help" => return Err(USAGE.into()),
                other => return Err(format!("unknown option {other}\n\n{USAGE}")),
            }
        }
        match a.command.as_str() {
            "calibrate" if a.references.len() < 2 || a.nside == 0 || a.out.is_none() => Err("calibrate needs --references (two or more), --nside and --out".into()),
            "run" if a.calibration.is_none() || a.out.is_none() => Err("run needs --calibration and --out".into()),
            "run" if a.init_conditional && a.init.is_none() => Err("--init-noise conditional needs --init".into()),
            "calibrate" | "run" => {
                if a.gpu && !cfg!(feature = "gpu") {
                    return Err("--gpu: built without the feature gpu".into());
                }
                if a.harmonic_gpu == Some(true) && !a.gpu {
                    return Err("--harmonic gpu needs --gpu".into());
                }
                Ok(a)
            }
            _ => Err(USAGE.into()),
        }
    }
}

/// The plan of the method: complete, one channel, J = log2(nside) + 1, L = 4, kernel 5.
pub fn setup(nside: usize, scales: usize, block_pixels: usize) -> (LocalGeometry, EngineConfig, Vec<Coefficient>) {
    let geo = LocalGeometry::new(nside as i64, 5, 4, Gauge::Cosmo, block_pixels);
    let cfg = EngineConfig::new(scales, false);
    let plan = build_plan(scales, 4, &cfg.j_to_dg, 1, None, &Family::ALL);
    (geo, cfg, plan)
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(feature = "gpu")]
mod device {
    use super::Args;
    use std::sync::Arc;
    use stl_core::engine::{EngineConfig, Partition};
    use stl_core::local::LocalGeometry;
    use stl_core::plan::Coefficient;
    use stl_gpu::{ConvForm, FieldStore, GeometryCache, Gpu, GpuEngine, GpuEvaluator, GpuOptions};

    /// The device engine of the rank for `batch` maps, as an evaluator of
    /// `plan`, its description and the device.
    pub fn evaluator<'a>(
        args: &Args,
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        plan: &[Coefficient],
        part: &Partition,
        rank: usize,
        batch: usize,
    ) -> Result<(GpuEvaluator<'a>, serde_json::Value, Arc<Gpu>), String> {
        let gpu = Arc::new(Gpu::for_rank(rank)?);
        let (cache, host) = match &args.geometry_cache {
            Some(p) => GeometryCache::policy(p).ok_or("--geometry-cache: 'auto', 'device', 'host' or 'off'")?,
            None => (GeometryCache::parse(&args.gpu_cache).ok_or("--gpu-cache: unknown policy")?, false),
        };
        let host_cache_bytes = if host { stl_gpu::host_cache_budget() } else { 0 };
        let host_calibrate = args.geometry_cache.as_deref() == Some("auto");
        let mut opts = GpuOptions { block_rows: args.gpu_block_rows, cache, host_cache_bytes, host_calibrate, ..Default::default() };
        if let Some(c) = &args.gpu_conv {
            opts.conv = ConvForm::parse(c).ok_or("--gpu-conv: unknown form")?;
        }
        if let Some(f) = &args.gpu_fields {
            opts.fields = FieldStore::parse(f).ok_or("--gpu-fields: unknown policy")?;
        }
        if let Some(mb) = std::env::var("STL_GPU_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            opts.cache_bytes = mb << 20;
        }
        if let Some(t) = std::env::var("STL_GPU_THREADS").ok().and_then(|v| v.parse().ok()) {
            opts.threads = t;
        }
        let engine = GpuEngine::new(gpu.clone(), geo, cfg, part.clone(), batch, 1, &opts);
        let info = serde_json::json!({
            "device": gpu.name, "device_ordinal": gpu.ordinal(), "batch": batch,
            "conv_form": engine.conv_form().name(), "fields": engine.field_store().name(),
            "geometry_cache": engine.geometry_cache().0.name(), "block_rows": engine.block_rows(),
        });
        Ok((GpuEvaluator::new(engine, plan), info, gpu))
    }
}

/// `calibrate`: the targets from the reference maps, written by rank 0.
pub fn run_calibrate<C: Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    let started = Instant::now();
    let nside = args.nside;
    let scales = nside.ilog2() as usize + 1;
    let (geo, cfg, plan) = setup(nside, scales, args.block_pixels);
    let mut physical = Vec::new();
    for p in &args.references {
        let x = read_map(p).map_err(|e| format!("{}: {e}", p.display()))?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(format!("{}: non-finite values", p.display()));
        }
        physical.push(degrade(&x, nside)?);
    }
    let r = physical.len();
    let ids = match &args.bootstrap_ids {
        Some(v) => v.clone(),
        None if r == args.ensemble => (0..r).collect(),
        None => {
            let seed = args.seed.unwrap_or(20260914);
            (0..args.ensemble).map(|k| (splitmix(seed ^ splitmix(k as u64)) % r as u64) as usize).collect()
        }
    };
    let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
    #[allow(unused_mut)]
    let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
    #[cfg(feature = "gpu")]
    if args.gpu {
        let (ev, _, _) = device::evaluator(args, &geo, &cfg, &plan, &probe.part, comm.rank(), 1)?;
        probe.set_evaluator(Box::new(ev));
    }
    let names = args.references.iter().map(|p| p.display().to_string()).collect();
    let cal = calibrate(&physical, names, ids, &plan, scales, args.knots, |x| probe.statistics_moments(&probe.slice(x)))?;
    if comm.rank() == 0 {
        let out = args.out.as_ref().unwrap();
        std::fs::write(out, serde_json::to_string(&cal.to_json()).map_err(|e| e.to_string())?).map_err(|e| format!("{out}: {e}"))?;
        if !args.quiet {
            println!("calibration of {r} references at nside {nside} in {:.1} s: {out}", started.elapsed().as_secs_f64());
        }
    }
    Ok(None)
}

/// `run`: the synthesis; the report on rank 0.
pub fn run_synthesis<C: Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    let started_setup = Instant::now();
    let path = args.calibration.as_ref().unwrap();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let cal = Calibration::from_json(&serde_json::from_str(&text).map_err(|e| e.to_string())?)?;
    let nside = cal.nside;
    let cutoff = match (args.ps_variance_upper, nside) {
        (Some(x), _) => x,
        (None, 128) => 383.0,
        _ => return Err("--ps-variance-upper: required at this nside (383 is the source's value at 128)".into()),
    };
    let (members, groups) = (args.ensemble, args.groups);
    if members % groups != 0 || comm.size() % groups != 0 {
        return Err(format!("--groups {groups} must divide --ensemble {members} and the number of ranks {}", comm.size()));
    }
    let per_group = comm.size() / groups;
    let (group_id, group_rank) = (comm.rank() / per_group, comm.rank() % per_group);
    let (group, cross) = if groups > 1 {
        let g = comm.split(group_id, group_rank).ok_or("--groups needs the MPI backend")?;
        let c = comm.split(group_rank, group_id).ok_or("--groups needs the MPI backend")?;
        (Some(g), Some(c))
    } else {
        (None, None)
    };
    let gcomm: &C = group.as_ref().unwrap_or(comm);
    let local = members / groups;
    let first = group_id * local;
    let (geo, cfg, plan) = setup(nside, cal.scales, args.block_pixels);
    let dummy = vec![Complex64::new(0.0, 0.0); local * plan.len()];
    #[allow(unused_mut)]
    let mut og = Objective::new(&geo, &cfg, &plan, &dummy, local, 1, gcomm);
    #[allow(unused_mut)]
    let mut ok = Objective::with_partition(&geo, &cfg, &plan, &dummy, local, 1, gcomm, og.dec.clone(), og.part.clone());
    #[allow(unused_mut)]
    let mut gpu_info = serde_json::Value::Null;
    // One device engine for this group's members, shared by the SC of g and of kappa.
    #[cfg(feature = "gpu")]
    let (_device, dev_gpu) = if args.gpu {
        let (ev, info, gpu) = device::evaluator(args, &geo, &cfg, &plan, &og.part, comm.rank(), local)?;
        og.set_evaluator(Box::new(ev.share(&plan, local)));
        ok.set_evaluator(Box::new(ev.share(&plan, local)));
        gpu_info = info;
        (Some(ev), Some(gpu))
    } else {
        (None, None)
    };
    #[allow(unused_mut)]
    let mut obj = EnsembleObjective::new(&og, &ok, cross.as_ref(), &cal, &plan, members, first, cutoff)?;
    #[cfg(feature = "gpu")]
    if let (Some(gpu), true) = (dev_gpu, args.harmonic_gpu.unwrap_or(args.gpu)) {
        obj.use_device_terms(gpu)?;
    }
    let (s, e) = og.owned();
    let (s, e) = (s as usize, e as usize);
    let npix = 12 * nside * nside;
    // Initial Gaussian maps of this group's members.
    let seed = args.seed.unwrap_or(61022026);
    let mut x = Vec::with_capacity(local * (e - s));
    match &args.init {
        Some(p) if args.init_conditional => {
            // The references at this nside, the parents' features and their details.
            let refs = cal.references.iter().map(|r| read_map(&PathBuf::from(r)).map_err(|e| format!("{r}: {e}")).and_then(|x| degrade(&x, nside))).collect::<Result<Vec<_>, _>>()?;
            let slices: Vec<&[f64]> = refs.iter().map(|r| &r[s..e]).collect();
            // The physical outputs have their monopole removed: the references' is put back.
            let monopole = refs.iter().map(|r| r.iter().sum::<f64>() / npix as f64).sum::<f64>() / refs.len() as f64;
            let member = |m: usize| -> Result<Vec<f64>, String> {
                let f = PathBuf::from(p.replace("{m}", &format!("{m:03}")));
                let mut v = read_map(&f).map_err(|e| format!("{}: {e}", f.display()))?;
                v.iter_mut().for_each(|k| *k += monopole);
                let factor = npix / v.len().max(1);
                if factor < 4 || factor * v.len() != npix || !factor.is_power_of_two() {
                    return Err(format!("{}: --init-noise conditional needs a map at a lower nside than {nside}", f.display()));
                }
                Ok(v)
            };
            let factor = npix / member(first)?.len();
            let nside_c = (nside as f64 / (factor as f64).sqrt()).round() as usize;
            let up = |v: &[f64]| crate::conditional::upsample(v, nside_c, nside);
            let bases: Vec<Vec<f64>> = if args.init_copy { Vec::new() } else { refs.iter().map(|r| up(&degrade(r, nside_c).unwrap())[s..e].to_vec()).collect() };
            let base_slices: Vec<&[f64]> = bases.iter().map(|b| b.as_slice()).collect();
            let cond = crate::conditional::Conditional::pooled(gcomm, &slices, (!args.init_copy).then_some(&base_slices[..]), s, factor, nside_c as i64, args.init_bins);
            for m in first..first + local {
                let v = member(m)?;
                let base = (!args.init_copy).then(|| up(&v)[s..e].to_vec());
                x.extend(cond.draw_on(&v, base.as_deref(), s, e, seed + m as u64).into_iter().map(|k| cal.lut.gaussianize(k)));
            }
        }
        Some(p) => {
            for m in first..first + local {
                let f = PathBuf::from(p.replace("{m}", &format!("{m:03}")));
                let v = read_map(&f).map_err(|e| format!("{}: {e}", f.display()))?;
                if v.len() != npix {
                    return Err(format!("{}: not at nside {nside}", f.display()));
                }
                x.extend_from_slice(&v[s..e]);
            }
        }
        None => {
            let sht = Sht::new(nside, 3 * nside - 1);
            let cl: Vec<f64> = cal.targets.initialization_cl.iter().enumerate().map(|(l, &c)| if l == 0 { 0.0 } else { c.max(0.0) }).collect();
            for m in first..first + local {
                let alm: Vec<Complex64> = sht
                    .lm()
                    .iter()
                    .enumerate()
                    .map(|(k, &(l, mm))| {
                        let (a, b) = (pixel_normal(seed + m as u64, 2 * k as i64), pixel_normal(seed + m as u64, 2 * k as i64 + 1));
                        if mm == 0 {
                            Complex64::new(a * cl[l].sqrt(), 0.0)
                        } else {
                            Complex64::new(a, b) * (0.5 * cl[l]).sqrt()
                        }
                    })
                    .collect();
                let g = sht.synthesis(&alm);
                x.extend(g[s..e].iter().map(|v| v + cal.targets.initialization_reference_mean));
            }
        }
    }
    let setup_seconds = started_setup.elapsed().as_secs_f64();
    let opts = LbfgsOptions { max_iter: args.niter, max_eval: args.max_eval, history_size: args.history_size, ..Default::default() };
    let root = comm.rank() == 0;
    let blocks_history = std::cell::RefCell::new(Vec::new());
    let eval_seconds = std::cell::RefCell::new(Vec::new());
    let prefix = args.out.as_ref().unwrap();
    // Whole Gaussian and physical maps of this group's members from the owned
    // slices, written by the group's rank 0 (through a temporary name).
    let write_members = |x: &[f64], prefix: &str| -> Result<(), String> {
        let g = obj.whole(x);
        let (kappa, _) = obj.physical(&g);
        if gcomm.rank() == 0 {
            for (m, (gr, kr)) in g.chunks(npix).zip(kappa.chunks(npix)).enumerate() {
                let id = first + m;
                for (kind, map) in [("gaussian", gr), ("physical", kr)] {
                    let path = format!("{prefix}_m{id:03}_{kind}_nested.npy");
                    write_npy(&PathBuf::from(format!("{path}.tmp")), map).map_err(|e| e.to_string())?;
                    std::fs::rename(format!("{path}.tmp"), &path).map_err(|e| format!("{path}: {e}"))?;
                }
            }
        }
        Ok(())
    };
    // Lowest loss so far: (loss, owned slices, evaluation); the evaluation last written.
    let mut best = (f64::INFINITY, Vec::new(), 0usize);
    let mut written = 0usize;
    let started = Instant::now();
    let f = |p: &[f64]| {
        let t = Instant::now();
        let out = obj.loss_and_grad(p);
        eval_seconds.borrow_mut().push(t.elapsed().as_secs_f64());
        blocks_history.borrow_mut().push(obj.last.borrow().map(|b| b.loss));
        if args.checkpoint > 0 {
            let n = eval_seconds.borrow().len();
            if out.0 < best.0 {
                best.1.clear();
                best.1.extend_from_slice(p);
                (best.0, best.2) = (out.0, n);
            }
            if n % args.checkpoint == 0 && best.2 != written {
                let ck = format!("{prefix}_checkpoint");
                match write_members(&best.1, &ck) {
                    Ok(()) if root => {
                        let state = json!({"evaluation": best.2, "loss": best.0, "blocks": blocks_history.borrow()[best.2 - 1],
                            "evaluations_done": n, "seconds": started.elapsed().as_secs_f64(),
                            "restart": format!("--init {ck}_m{{m}}_gaussian_nested.npy")});
                        if let Err(e) = std::fs::write(format!("{ck}.json"), serde_json::to_string_pretty(&state).unwrap()) {
                            eprintln!("stl-ensemble: {ck}.json: {e}");
                        }
                    }
                    Ok(()) => {}
                    Err(e) => eprintln!("stl-ensemble: checkpoint: {e}"),
                }
                written = best.2;
            }
        }
        out
    };
    let print = |i: usize, loss: f64, t: f64| {
        if root && !args.quiet {
            let b = obj.last.borrow();
            println!(
                "[stl-ensemble] evaluation {:4}  loss = {loss:.10e}  blocks = [{}]  t = {t:8.1} s",
                i + 1,
                b.iter().map(|v| format!("{:.6e}", v.loss)).collect::<Vec<_>>().join(", ")
            );
        }
    };
    let history = if args.niter > 0 { run_lbfgs_with(comm, &mut x, &opts, f, print) } else { Vec::new() };
    let seconds = started.elapsed().as_secs_f64();
    // Final maps of this group's members (whole maps on every rank of the group).
    let (final_loss, _) = obj.loss_and_grad(&x);
    let final_blocks = *obj.last.borrow();
    write_members(&x, prefix)?;
    let peak = comm.allreduce_max(proc_memory("VmHWM:") as f64);
    #[allow(unused_mut)]
    let mut device_memory = serde_json::Value::Null;
    if let (_, Some((peak_dev, total))) = og.evaluator_report() {
        device_memory = json!({"peak_bytes": comm.allreduce_max(peak_dev as f64), "total_bytes": total});
    }
    if !root {
        return Ok(None);
    }
    let evals = eval_seconds.borrow();
    let per_eval = if evals.len() > 1 { evals[1..].iter().sum::<f64>() / (evals.len() - 1) as f64 } else { evals.first().copied().unwrap_or(0.0) };
    let report = json!({
        "method": "P. Campeti, STL_ensemble v2 (variance ratio), five blocks, ported to Rust",
        "nside": nside, "scales": cal.scales, "orientations": 4, "kernel": 5, "members": members, "groups": groups,
        "ranks": comm.size(), "calibration": path, "references": cal.references, "bootstrap_ids": cal.bootstrap_ids,
        "ps_variance_upper": cutoff, "niter": args.niter, "max_eval": args.max_eval, "history_size": args.history_size,
        "checkpoint": args.checkpoint, "init": args.init, "init_noise": if args.init_conditional { json!({"conditional": true, "bins": args.init_bins, "upsample": if args.init_copy { "copy" } else { "harmonic" }}) } else { serde_json::Value::Null }, "seed": seed, "gpu": gpu_info, "harmonic": if obj.device_terms() { "gpu" } else { "cpu" },
        "evaluations": history.len(), "seconds": seconds, "setup_seconds": setup_seconds,
        "seconds_per_evaluation": per_eval, "first_evaluation_seconds": evals.first(),
        "loss_history": &history, "blocks": BLOCKS, "blocks_history": &*blocks_history.borrow(),
        "loss_final": final_loss, "blocks_final": final_blocks.iter().map(|b| json!({"loss": b.loss, "mean": b.mean, "variance": b.variance})).collect::<Vec<_>>(),
        "timers_seconds": &*obj.timers.borrow(), "peak_rss_bytes_max_rank": peak, "device_memory": device_memory,
        "lbfgs_threads": std::env::var("STL_THREADS").ok(),
    });
    std::fs::write(format!("{prefix}.json"), serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    Ok(Some(report))
}
