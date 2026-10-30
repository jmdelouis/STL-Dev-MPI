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
use stl_mpi::synthesis::{ergodic_weights, run_lbfgs_with, Ensemble, Objective};
use stl_mpi::Comm;

#[cfg(feature = "gpu")]
mod device_terms;
mod kappa;
mod kurtosis;
mod level_power;
mod marginal;
pub mod minkowski;
use kappa::Transform;
use kurtosis::Kurtosis;
use level_power::LevelPower;
use marginal::Marginal;
use minkowski::{Minkowski, SigmaFile};

pub const USAGE: &str = "\
stl-synth --target MAP [options]

  --target MAP          NESTED full-sky map (.npy or HEALPix .fits), standardised
  --init MAP            initial map (NESTED, nside <= target); default white noise; with
                        --ensemble, '{m}' in MAP gives one init per member (3-digit index,
                        e.g. prev_m{m}_synthesis_nested.npy)
  --init-units U        'target' (default: standardised like the target) or 'standardized'
  --init-noise X        noise std added to --init: 'auto' (default, RMS of the target's
                        missing small scales when the init nside is lower), a number, 0
  --seed N              white-noise seed (default 11)
  --J N                 scales (default log2(nside) - 1)
  --L N                 orientations (default 4; L = 2 leaves anisotropic, blobby syntheses)
  --kernel-size N       stencil (default 5)
  --fewer               has_fewer_convolutions definitions of S3/S4
  --block-pixels N      block size per level (default 16384)
  --overlap N           overlapping patches: levels < N evaluated with redundant halos (no
                        exchange during the sweeps), levels >= N on rank 0; for a few ranks
  --conv-chunk N        rebuild the convolution geometry by sub-blocks of N rows at every
                        application (less memory, more time; default: whole block)
  --niter N             L-BFGS iterations (default 50)
  --weights W           loss weights: 'none' (default, sum |S - S_target|^2) or 'ergodic'
                        (reduced chi^2: mean of |S - S_target|^2 / sigma^2, sigma^2 = spatial
                        variance of each coefficient's terms on the target / pixels of its level)
  --family-weights F    extra factors on the weights of whole families, e.g. 'S1=10,S2=10'
                        (families mean, var, S1, S2, S3, S4; default 1)
  --ensemble B          synthesise B maps whose MEAN statistics match the target (loss on the
                        mean over the members; member m starts from seed + m); default 1
  --groups G            split the ranks into G groups of consecutive ranks, each sharing the
                        sky of B / G members (MPI; G divides B and the number of ranks); default 1
  --scatter-weight X    with --ensemble B > 1, adds X / n sum_i (D_i / sigma_i^2 - 1)^2: the
                        scatter D_i of the members around their mean held at the ergodic
                        variance sigma_i^2 of a map (second moment of the ensemble); default 0
  --level-power-weight X adds X / B sum_m sum_n (V_n / V_n(target) - 1)^2, V_n the variance of the
                        details of level n of the 4-children pyramid of the map (finest level down
                        to nside 8): holds the power of every level; default 0
  --kappa-table KNOTS   adds the statistics of kappa = f(g), f the monotone interpolation of the
                        knots (g, kappa) of KNOTS.npy (tools/gaussianize.py knots): the term
                        X sum_i w_i |mean over members of S_i(f(g)) - S_i(f(g_target))|^2, w_i the
                        ergodic weights of f(g_target); g and kappa standardised by the target
  --kappa-weight X      factor X of that term (default 0.1; 0 drops it)
  --kappa-families F    families of that term (default 'mean,var,S1,S2': the power and the l1 norm
                        of every scale and orientation)
  --kappa-s3-weight X   adds the S3 of kappa as a separate reduced chi^2 of factor X (ergodic weights
                        normalised over its own coefficients); default 0
  --kappa-s3-nside A-B  restricts it to the S3 (j1 <= j3) whose scale j3 is at the levels of nside A
                        to B (e.g. 64-128); default every level
  --kurtosis-weight X   adds X / B sum_m sum_n (K_n / K_n(target) - 1)^2, K_n = <d_n^4> / <d_n^2>^2 the
                        kurtosis of the details of level n of the 4-children pyramid of kappa = f(g)
                        (of the map without --kappa-table); default 0
  --kurtosis-nside A-B  levels of that term by nside (e.g. 64-128); default every level of
                        --level-power-weight (finest to nside 8)
  --minkowski-weight X  adds X / B sum_m mean_{n,k,i} ((V_k(t_i) - V_k(t_i; target)) / sigma_k,i)^2, the
                        Minkowski functionals V0, V1, V2 of level n of the 4-children pyramid of
                        kappa = f(g) (of the map without --kappa-table) at the thresholds
                        t_i = mu_n + nu_i sigma_n of the target (nu = -2 ... 5 by 0.2), soft (logistic of
                        width 0.2 sigma_n) with least-squares derivatives on the 3x3 cells; default 0
  --minkowski-nside A-B levels of that term by nside; default 64 to nside / 2 (at the map's own nside
                        the functionals of kappa are those of g: the level power holds them)
  --minkowski-standardise standardise every level map by its own mean and std first (thresholds nu_i
                        for every map, as mf_levels.py): the functionals then measure the shape only
  --minkowski-sigma F   JSON of stl-minkowski --sigma-out: std of the functionals over realizations,
                        with its estimator (thresholds, width, --minkowski-standardise or not);
                        default 1 % of the target's maximum
  --marginal-weight X   adds X / B sum_m (1/N) sum_i (x_m,(i) - t_(i))^2, the squared 1-D Wasserstein
                        distance between the sorted values of every member and of the target: holds
                        the pixel histogram; default 0
  --freeze-nside M      hold the means of the map over the pixels of nside M at those of the
                        initial map: only the scales finer than M move (coarse-to-fine steps)
  --history-size N      L-BFGS history (default 10)
  --out PREFIX          writes PREFIX.json, PREFIX_synthesis.fits, PREFIX_synthesis_nested.npy,
                        PREFIX_target.fits (default synthesis_rust_nside<N>); with --ensemble B > 1,
                        PREFIX_mMMM_synthesis_nested.npy and _standardized.npy per member
  --ranks N             thread ranks in this process (builds without MPI)
  --gpu                 one CUDA device per rank (feature gpu): device = local rank modulo the
                        visible devices, or STL_GPU_DEVICE
  --gpu-block-rows N    owned cells per device block (default: from the device memory)
  --gpu-cache C         convolution geometry kept on the device between sweeps: 'full', 'transposition'
                        (the weights are recomputed), 'none', or 'auto' (default: full if it fits,
                        else transposition); budget from the device memory or STL_GPU_CACHE_MB
  --gpu-conv F          form of the convolution geometry: 'dia' (masked diagonals,
                        docs/dia_convolution.md) or 'stencils' (bound stencils and their
                        transposition); default STL_GPU_CONV, else dia
  --gpu-fields F        wavelet fields z_n, |z_n| of every level 'keep' (from the forward sweep to
                        the reverse one) or 'recompute' (per block in the reverse sweep, as the CPU
                        engine; ~3 L doubles less per pixel and map), or 'auto': keep when it takes
                        at most half of the free device memory; default STL_GPU_FIELDS, else auto
  --gpu-terms T         where the objective's terms run with --gpu: 'device' (default: the members'
                        maps go to the device once per evaluation; level power, marginal, kappa
                        transform, kurtosis and the gradient sums run there; one rank per group,
                        no Minkowski term or frozen scales, else 'host') or 'host'
  --quiet               no line per evaluation
";

/// `64-128` (or `64`) -> (64, 128): levels given by their nside.
fn parse_nside_range(k: &str, v: Option<&String>) -> Result<(usize, usize), String> {
    let v = v.ok_or(format!("{k} needs a value"))?;
    let (a, b) = v.split_once('-').unwrap_or((v, v));
    let p = |t: &str| t.trim().parse::<usize>().ok().filter(|n| n.is_power_of_two()).ok_or(format!("{k}: '{t}' is not a power of 2"));
    let (a, b) = (p(a)?, p(b)?);
    Ok((a.min(b), a.max(b)))
}

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
    pub overlap: Option<usize>,
    pub ergodic_weights: bool,
    pub family_weights: Vec<(Family, f64)>,
    pub ensemble: usize,
    pub groups: usize,
    pub scatter_weight: f64,
    pub freeze_nside: Option<usize>,
    pub level_power_weight: f64,
    pub kappa_table: Option<PathBuf>,
    pub kappa_weight: f64,
    pub kappa_families: Vec<Family>,
    pub kappa_s3_weight: f64,
    pub kappa_s3_nside: Option<(usize, usize)>,
    pub kurtosis_weight: f64,
    pub kurtosis_nside: Option<(usize, usize)>,
    pub minkowski_weight: f64,
    pub minkowski_nside: Option<(usize, usize)>,
    pub minkowski_sigma: Option<PathBuf>,
    pub minkowski_standardise: bool,
    pub marginal_weight: f64,
    pub niter: usize,
    pub history_size: usize,
    pub out: Option<String>,
    pub ranks: usize,
    pub quiet: bool,
    pub gpu: bool,
    pub gpu_block_rows: usize,
    pub gpu_cache: String,
    /// `--gpu-conv` (`None`: `STL_GPU_CONV`, else dia).
    pub gpu_conv: Option<String>,
    /// `--gpu-fields` (`None`: `STL_GPU_FIELDS`, else auto).
    pub gpu_fields: Option<String>,
    /// `--gpu-terms host`: the objective's terms on the host.
    pub gpu_terms_host: bool,
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
            overlap: None,
            ergodic_weights: false,
            family_weights: Vec::new(),
            ensemble: 1,
            groups: 1,
            scatter_weight: 0.0,
            freeze_nside: None,
            level_power_weight: 0.0,
            kappa_table: None,
            kappa_weight: 0.1,
            kappa_families: vec![Family::Mean, Family::Var, Family::S1, Family::S2],
            kappa_s3_weight: 0.0,
            kappa_s3_nside: None,
            kurtosis_weight: 0.0,
            kurtosis_nside: None,
            minkowski_weight: 0.0,
            minkowski_nside: None,
            minkowski_sigma: None,
            minkowski_standardise: false,
            marginal_weight: 0.0,
            niter: 50,
            history_size: 10,
            out: None,
            ranks: 1,
            quiet: false,
            gpu: false,
            gpu_block_rows: 0,
            gpu_cache: "auto".into(),
            gpu_conv: None,
            gpu_fields: None,
            gpu_terms_host: false,
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
                "--overlap" => a.overlap = Some(num(it.next(), k)?),
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
                "--ensemble" => a.ensemble = num(it.next(), k)?.max(1),
                "--groups" => a.groups = num(it.next(), k)?.max(1),
                "--scatter-weight" => {
                    let v: f64 = it.next().ok_or("--scatter-weight needs a value")?.parse().map_err(|_| "--scatter-weight: not a number")?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("--scatter-weight must be finite and >= 0".into());
                    }
                    a.scatter_weight = v;
                }
                "--level-power-weight" => {
                    let v: f64 = it.next().ok_or("--level-power-weight needs a value")?.parse().map_err(|_| "--level-power-weight: not a number")?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("--level-power-weight must be finite and >= 0".into());
                    }
                    a.level_power_weight = v;
                }
                "--kappa-table" => a.kappa_table = Some(it.next().ok_or("--kappa-table needs a path")?.into()),
                "--kappa-weight" => {
                    let v: f64 = it.next().ok_or("--kappa-weight needs a value")?.parse().map_err(|_| "--kappa-weight: not a number")?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("--kappa-weight must be finite and >= 0".into());
                    }
                    a.kappa_weight = v;
                }
                "--kappa-families" => {
                    let v = it.next().ok_or("--kappa-families needs a value")?;
                    a.kappa_families = v
                        .split(',')
                        .map(|f| Family::parse(f.trim()).ok_or(format!("--kappa-families: unknown family '{f}'")))
                        .collect::<Result<_, _>>()?;
                }
                "--kappa-s3-weight" | "--kurtosis-weight" | "--minkowski-weight" => {
                    let v: f64 = it.next().ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: not a number"))?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err(format!("{k} must be finite and >= 0"));
                    }
                    match k.as_str() {
                        "--kappa-s3-weight" => a.kappa_s3_weight = v,
                        "--kurtosis-weight" => a.kurtosis_weight = v,
                        _ => a.minkowski_weight = v,
                    }
                }
                "--kappa-s3-nside" => a.kappa_s3_nside = Some(parse_nside_range(k, it.next())?),
                "--kurtosis-nside" => a.kurtosis_nside = Some(parse_nside_range(k, it.next())?),
                "--minkowski-nside" => a.minkowski_nside = Some(parse_nside_range(k, it.next())?),
                "--minkowski-sigma" => a.minkowski_sigma = Some(it.next().ok_or("--minkowski-sigma needs a path")?.into()),
                "--minkowski-standardise" => a.minkowski_standardise = true,
                "--marginal-weight" => {
                    let v: f64 = it.next().ok_or("--marginal-weight needs a value")?.parse().map_err(|_| "--marginal-weight: not a number")?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("--marginal-weight must be finite and >= 0".into());
                    }
                    a.marginal_weight = v;
                }
                "--freeze-nside" => a.freeze_nside = Some(num(it.next(), k)?),
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
                "--gpu-conv" => {
                    a.gpu_conv = match it.next().map(String::as_str) {
                        Some(v @ ("stencils" | "dia")) => Some(v.to_string()),
                        _ => return Err("--gpu-conv is 'stencils' or 'dia'".into()),
                    }
                }
                "--gpu-fields" => {
                    a.gpu_fields = match it.next().map(String::as_str) {
                        Some(v @ ("auto" | "keep" | "recompute")) => Some(v.to_string()),
                        _ => return Err("--gpu-fields is 'auto', 'keep' or 'recompute'".into()),
                    }
                }
                "--gpu-terms" => {
                    a.gpu_terms_host = match it.next().map(String::as_str) {
                        Some("device") => false,
                        Some("host") => true,
                        _ => return Err("--gpu-terms is 'device' or 'host'".into()),
                    }
                }
                "-h" | "--help" => return Err(USAGE.into()),
                other => return Err(format!("unknown option {other}\n\n{USAGE}")),
            }
        }
        if a.target.as_os_str().is_empty() {
            return Err(format!("--target is required\n\n{USAGE}"));
        }
        if a.kappa_s3_weight > 0.0 && a.kappa_weight > 0.0 && a.kappa_families.contains(&Family::S3) {
            return Err("--kappa-s3-weight: S3 is already in --kappa-families".into());
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
///
/// With `--ensemble B --groups G`, the ranks form G groups of consecutive
/// ranks; each group shares the sky of B / G members (the batch of its
/// objective) and the loss is on the mean statistics of the B members.
pub fn run<C: Comm>(args: &Args, comm: &C) -> Result<Option<serde_json::Value>, String> {
    let root = comm.rank() == 0;
    let (members, groups) = (args.ensemble, args.groups);
    if members % groups != 0 || comm.size() % groups != 0 {
        return Err(format!("--groups {groups} must divide --ensemble {members} and the number of ranks {}", comm.size()));
    }
    let per_group = comm.size() / groups;
    let (group_id, group_rank) = (comm.rank() / per_group, comm.rank() % per_group);
    // `group`: the ranks sharing the sky of this group's members; `cross`:
    // the ranks of the same position in every group (sums over the groups).
    let (group, cross) = if groups > 1 {
        let g = comm.split(group_id, group_rank).ok_or("--groups needs the MPI backend")?;
        let c = comm.split(group_rank, group_id).ok_or("--groups needs the MPI backend")?;
        (Some(g), Some(c))
    } else {
        (None, None)
    };
    let gcomm: &C = group.as_ref().unwrap_or(comm);
    let local_members = members / groups;
    let first_member = group_id * local_members;

    let npix = map_len(&args.target)?;
    let nside = nside_of(npix)?;
    let j = args.j.unwrap_or_else(|| (64 - (nside as u64).leading_zeros() as usize).saturating_sub(2).max(1));
    let mut geo = LocalGeometry::new(nside, args.kernel_size, args.l, Gauge::Cosmo, args.block_pixels);
    geo.conv_chunk = args.conv_chunk;
    let mut cfg = EngineConfig::new(j, args.fewer);
    if let Some(coarse) = args.overlap {
        if args.gpu {
            return Err("--overlap runs on the CPU engine only (the device engine needs the standard partition)".into());
        }
        if coarse < 1 || coarse > cfg.levels() {
            return Err(format!("--overlap {coarse}: the first coarse level must be in 1..={}", cfg.levels()));
        }
        cfg.overlap = Some(coarse);
    }
    let plan = build_plan(j, args.l, &cfg.j_to_dg, 1, None, &Family::ALL);
    let started_setup = Instant::now();
    let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
    #[allow(unused_mut)]
    let mut probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, gcomm);
    #[allow(unused_mut)]
    let mut gpu_info = serde_json::Value::Null;
    // One device engine per rank, sized for this group's members and shared
    // by every objective (target statistics, synthesis, kappa term): one
    // topology, geometry cache and set of level fields on the device.
    // The objective's terms on the device too (`--gpu-terms device`), when
    // they apply; the device keeps room for their buffers.
    #[cfg(feature = "gpu")]
    let device_terms = args.gpu
        && !args.gpu_terms_host
        && device_terms::applies(gcomm.size(), args.minkowski_weight > 0.0, args.freeze_nside.is_some());
    #[cfg(feature = "gpu")]
    let device = if args.gpu {
        let (s, e) = probe.owned();
        let reserve = if device_terms { device_terms::reserve_bytes(local_members, (e - s) as usize) } else { 0 };
        let (ev, info) = gpu::evaluator(args, &geo, &cfg, &plan, &probe.part, comm.rank(), local_members, reserve)?;
        probe.set_evaluator(Box::new(ev.share(&plan, 1)));
        gpu_info = info;
        Some(ev)
    } else {
        None
    };
    let (s, e) = probe.owned();
    let (s, e) = (s as usize, e as usize);

    // Target slice, standardised with global moments (two passes).
    let raw = read_map_range(&args.target, s, e)?;
    let bad = global_sum(gcomm, raw.iter().filter(|v| !v.is_finite()).count() as f64);
    if bad > 0.0 {
        return Err(format!("{}: {bad} non-finite values", args.target.display()));
    }
    let mean = global_sum(gcomm, raw.iter().sum()) / npix as f64;
    let std = (global_sum(gcomm, raw.iter().map(|x| (x - mean) * (x - mean)).sum()) / npix as f64).sqrt();
    let target: Vec<f64> = raw.iter().map(|v| (v - mean) / std).collect();
    drop(raw);

    // Initial slices of this group's members (standardised units); member m
    // draws its noise with seed + m (member 0 is the single-map synthesis).
    let mut init_info = serde_json::Value::Null;
    let mut initial: Vec<f64> = Vec::with_capacity(local_members * (e - s));
    match &args.init {
        None => {
            for m in first_member..first_member + local_members {
                initial.extend((s..e).map(|p| pixel_normal(args.seed + m as u64, p as i64)));
            }
        }
        Some(path) => {
            // A path holding "{m}" is one init per member: "{m}" becomes the
            // member index on three digits (PREFIX_mMMM_... of an ensemble).
            let per_member = path.to_string_lossy().contains("{m}");
            let member_path = |m: usize| PathBuf::from(path.to_string_lossy().replace("{m}", &format!("{m:03}")));
            let first = member_path(first_member);
            let n_init = map_len(&first)?;
            let nside_init = nside_of(n_init)?;
            if nside_init > nside || nside % nside_init != 0 {
                return Err(format!("{}: nside {nside_init} incompatible with {nside}", first.display()));
            }
            let factor = npix / n_init;
            if s % factor != 0 || e % factor != 0 {
                return Err(format!("--init at nside {nside_init}: rank ranges are not aligned on its pixels (use fewer ranks)"));
            }
            let mut coarse: Vec<Vec<f64>> = Vec::with_capacity(local_members);
            for m in first_member..first_member + local_members {
                let p = member_path(m);
                if map_len(&p)? != n_init {
                    return Err(format!("{}: not the nside of {}", p.display(), first.display()));
                }
                let mut c = read_map_range(&p, s / factor, e / factor)?;
                if !args.init_standardized {
                    c.iter_mut().for_each(|v| *v = (*v - mean) / std);
                }
                coarse.push(c);
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
                    (global_sum(gcomm, local) / npix as f64).sqrt()
                }
                None => 0.0,
            };
            if members > 1 && !per_member && noise_std <= 0.0 {
                return Err("--ensemble from one --init needs a noise (--init-noise > 0 or auto below the target nside), or one init per member ({m} in the path): identical members stay identical".into());
            }
            init_info = serde_json::json!({"file": path, "per_member": per_member, "nside": nside_init, "noise_std": noise_std});
            for (k, m) in (first_member..first_member + local_members).enumerate() {
                initial.extend((s..e).map(|p| {
                    let b = coarse[k][p / factor - s / factor];
                    if noise_std > 0.0 { b + noise_std * pixel_normal(args.seed + m as u64, p as i64) } else { b }
                }));
            }
        }
    }

    // Resident memory once the inputs are read.
    let rss_baseline = proc_memory("VmRSS:");
    // Target statistics with the same distributed engine, and the spatial
    // variance of every coefficient's terms for the ergodic weights.
    let (target_stats, target_var) = probe.statistics_moments(&target);
    let (dec, part) = (probe.dec.clone(), probe.part.clone());
    drop(probe);
    let target_rows = target_stats.repeat(local_members);
    #[allow(unused_mut)]
    let mut obj = Objective::with_partition(&geo, &cfg, &plan, &target_rows, local_members, 1, gcomm, dec, part);
    #[cfg(feature = "gpu")]
    if let Some(dev) = &device {
        obj.set_evaluator(Box::new(dev.share(&plan, local_members)));
    }
    if args.ergodic_weights {
        obj.weights = ergodic_weights(&geo, &plan, &target_var);
        // Reduced chi^2 of the mean of B members (variance sigma^2 / B).
        obj.weights.iter_mut().for_each(|w| *w *= members as f64);
    }
    for (w, c) in obj.weights.iter_mut().zip(&plan) {
        if let Some(&(_, f)) = args.family_weights.iter().find(|(fam, _)| *fam == c.family) {
            *w *= f;
        }
    }
    // Ergodic variance of every coefficient for one map.
    let sigma2: Vec<f64> =
        plan.iter().enumerate().map(|(i, c)| (target_var[i] / geo.pixels(c.native_dg) as f64).max(1e-300)).collect();
    let scatter = (args.scatter_weight > 0.0 && members > 1).then_some((args.scatter_weight, &sigma2[..]));
    let ens = Ensemble { obj: &obj, cross: cross.as_ref(), members, target: &target_stats, scatter };
    let level_power = (args.level_power_weight > 0.0).then(|| {
        let levels = LevelPower::levels(nside as usize, s, e, gcomm);
        LevelPower::new(args.level_power_weight, &target, levels, npix, members, gcomm)
    });
    // Statistics of kappa = f(g) (--kappa-table): own plan (the families of
    // --kappa-families, then the S3 of --kappa-s3-weight at the levels of
    // --kappa-s3-nside), target statistics (of the transformed target) and
    // ergodic weights, loss on the mean of the members.
    let at_levels = |range: Option<(usize, usize)>, dg: usize| range.is_none_or(|(a, b)| (a..=b).contains(&(nside as usize >> dg)));
    let mut kplan = if args.kappa_weight > 0.0 { build_plan(j, args.l, &cfg.j_to_dg, 1, None, &args.kappa_families) } else { Vec::new() };
    let kbase = kplan.len();
    if args.kappa_s3_weight > 0.0 {
        let s3 = build_plan(j, args.l, &cfg.j_to_dg, 1, None, &[Family::S3]);
        kplan.extend(s3.into_iter().filter(|c| at_levels(args.kappa_s3_nside, c.native_dg)));
        if kplan.len() == kbase {
            return Err("--kappa-s3-nside: no S3 at these levels".into());
        }
    }
    let kdummy = vec![Complex64::new(0.0, 0.0); kplan.len()];
    let kappa_tf = match &args.kappa_table {
        Some(p) => Some(Transform::new(&read_map_range(p, 0, map_len(p)?)?, &target, mean, std, npix, gcomm)?),
        None => None,
    };
    let ktarget = kappa_tf.as_ref().map(|tf| tf.apply(&target).0);
    let kappa_stats = !kplan.is_empty();
    let (kstats, kvar) = match &ktarget {
        Some(kt) if kappa_stats => {
            #[allow(unused_mut)]
            let mut probe = Objective::with_partition(&geo, &cfg, &kplan, &kdummy, 1, 1, gcomm, obj.dec.clone(), obj.part.clone());
            #[cfg(feature = "gpu")]
            if let Some(dev) = &device {
                probe.set_evaluator(Box::new(dev.share(&kplan, 1)));
            }
            probe.statistics_moments(kt)
        }
        _ => (Vec::new(), Vec::new()),
    };
    let krows = kstats.repeat(local_members);
    let kobj = match &kappa_tf {
        Some(_) if kappa_stats => {
            let mut k = Objective::with_partition(&geo, &cfg, &kplan, &krows, local_members, 1, gcomm, obj.dec.clone(), obj.part.clone());
            #[cfg(feature = "gpu")]
            if let Some(dev) = &device {
                k.set_evaluator(Box::new(dev.share(&kplan, local_members)));
            }
            // Two reduced chi^2: the families, and the S3 of --kappa-s3-weight.
            let (base, s3) = (ergodic_weights(&geo, &kplan[..kbase], &kvar[..kbase]), ergodic_weights(&geo, &kplan[kbase..], &kvar[kbase..]));
            let scaled = |w: Vec<f64>, f: f64| w.into_iter().map(move |v| v * members as f64 * f);
            k.weights = scaled(base, args.kappa_weight).chain(scaled(s3, args.kappa_s3_weight)).collect();
            Some(k)
        }
        _ => None,
    };
    let kens = kobj.as_ref().map(|k| Ensemble { obj: k, cross: cross.as_ref(), members, target: &kstats, scatter: None });
    // Kurtosis per level (--kurtosis-weight), of kappa = f(g) with --kappa-table;
    // the kurtosis of every level is reported whenever one of them is given.
    let pyramid_levels = LevelPower::levels(nside as usize, s, e, gcomm);
    let kurtosis_levels: Vec<usize> = (0..pyramid_levels).filter(|&n| at_levels(args.kurtosis_nside, n)).collect();
    if args.kurtosis_weight > 0.0 && kurtosis_levels.is_empty() {
        return Err("--kurtosis-nside: no level of the pyramid in this range".into());
    }
    let kurtosis_source: &[f64] = ktarget.as_deref().unwrap_or(&target);
    let kurtosis = (args.kurtosis_weight > 0.0)
        .then(|| Kurtosis::new(args.kurtosis_weight, kurtosis_source, kurtosis_levels.clone(), npix, members, gcomm));
    let kurtosis_all = (args.kurtosis_weight > 0.0 || kappa_tf.is_some())
        .then(|| Kurtosis::new(0.0, kurtosis_source, (0..pyramid_levels).collect(), npix, members, gcomm));
    // Minkowski functionals per level (--minkowski-weight), of the same field
    // as the kurtosis; levels of means aligned on the ranks (0 ..= pyramid_levels).
    let minkowski = if args.minkowski_weight > 0.0 {
        let half = nside as usize / 2;
        let levels: Vec<usize> = (0..=pyramid_levels)
            .filter(|&n| match args.minkowski_nside {
                Some((a, b)) => (a..=b).contains(&(nside as usize >> n)),
                None => n >= 1 && (64..=half).contains(&(nside as usize >> n)),
            })
            .collect();
        if levels.is_empty() {
            return Err("--minkowski-nside: no level of the pyramid in this range".into());
        }
        let file = match &args.minkowski_sigma {
            Some(p) => Some(SigmaFile::read(p)?),
            None => None,
        };
        let path = args.minkowski_sigma.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
        // The sigma file's estimator wins.
        let standardise = file.as_ref().map_or(args.minkowski_standardise, |f| f.standardise);
        let sigma = file.as_ref().map(|f| (f, path.as_str()));
        Some(Minkowski::new(args.minkowski_weight, kurtosis_source, s, npix, members, &levels, sigma, standardise, gcomm)?)
    } else {
        None
    };
    let marginal = (args.marginal_weight > 0.0).then(|| Marginal::new(args.marginal_weight, &target, s, npix, members, gcomm));
    let setup_seconds = started_setup.elapsed().as_secs_f64();
    let mut x = initial;

    // --freeze-nside M: the means of the map over the pixels of nside M (groups
    // of `freeze` consecutive NESTED pixels) stay those of the initial map; the
    // gradient is projected on the zero-mean details, so L-BFGS (directions
    // built from projected gradients and steps) only moves the scales finer than M.
    let freeze = match args.freeze_nside {
        None => None,
        Some(m) => {
            let n = nside as usize;
            if m == 0 || m > n || n % m != 0 || !m.is_power_of_two() {
                return Err(format!("--freeze-nside {m}: a power of 2 dividing the target nside {nside}"));
            }
            let group = (n / m) * (n / m);
            if s % group != 0 || e % group != 0 {
                return Err(format!("--freeze-nside {m}: rank ranges are not aligned on its pixels (use fewer ranks)"));
            }
            Some(group)
        }
    };
    let project = |g: &mut [f64]| {
        if let Some(group) = freeze {
            for c in g.chunks_mut(group) {
                let mean = c.iter().sum::<f64>() / group as f64;
                c.iter_mut().for_each(|v| *v -= mean);
            }
        }
    };

    let opts = LbfgsOptions { max_iter: args.niter, history_size: args.history_size, ..Default::default() };
    let calls_before = obj.exchange_calls();
    let started = Instant::now();
    let print = |i: usize, loss: f64, t: f64| {
        if root && !args.quiet {
            println!("[rust-local P={}] evaluation {:4}  loss = {loss:.6e}  t = {t:7.1} s", comm.size(), i + 1);
        }
    };
    // Wall time of every part of an evaluation (host_*: the terms with their
    // gradient accumulation, the objectives with their device sweeps).
    let host_timers: std::cell::RefCell<std::collections::BTreeMap<&'static str, f64>> = Default::default();
    let tick = |name: &'static str, since: Instant| *host_timers.borrow_mut().entry(name).or_default() += since.elapsed().as_secs_f64();
    // The whole objective on the device (--gpu-terms device): its own
    // evaluators on the shared engine.
    #[allow(unused_mut)]
    let mut device_history: Option<Vec<f64>> = None;
    #[cfg(feature = "gpu")]
    if let (Some(dev), true, true) = (&device, device_terms, args.niter > 0) {
        let kappa = kappa_tf.as_ref().filter(|_| kens.is_some() || kurtosis.is_some()).map(|tf| device_terms::KappaPart {
            transform: tf,
            ensemble: kens.as_ref(),
            evaluator: kens.as_ref().map(|_| dev.share(&kplan, local_members)),
        });
        let mut dobj = device_terms::DeviceObjective::new(
            &obj,
            (members > 1).then_some(&ens),
            dev.share(&plan, local_members),
            kappa,
            level_power.as_ref(),
            marginal.as_ref(),
            kurtosis.as_ref(),
            cross.as_ref(),
        );
        let f = |p: &[f64]| {
            let t = Instant::now();
            let out = dobj.loss_and_grad(p, &tick);
            tick("host_evaluations", t);
            out
        };
        device_history = Some(run_lbfgs_with(comm, &mut x, &opts, f, print));
    }
    let history = if let Some(h) = device_history {
        h
    } else if args.niter == 0 {
        Vec::new()
    } else if members == 1 && level_power.is_none() && kens.is_none() && marginal.is_none() && kurtosis.is_none() && minkowski.is_none() && freeze.is_none() {
        let f = |p: &[f64]| {
            let t = Instant::now();
            let out = obj.loss_and_grad(p);
            tick("host_evaluations", t);
            out
        };
        run_lbfgs_with(comm, &mut x, &opts, f, print)
    } else {
        let loss_and_grad = |p: &[f64]| {
            let t_eval = Instant::now();
            let t = Instant::now();
            let (mut loss, mut g) = if members == 1 { obj.loss_and_grad(p) } else { ens.loss_and_grad(p) };
            tick("host_synthesis_objective", t);
            if let Some(lp) = &level_power {
                let t = Instant::now();
                let (l2, g2) = lp.loss_and_grad(p, e - s, gcomm, cross.as_ref());
                loss += l2;
                g.iter_mut().zip(&g2).for_each(|(a, b)| *a += b);
                tick("host_level_power", t);
            }
            if let Some(mg) = &marginal {
                let t = Instant::now();
                let (lm, gm) = mg.loss_and_grad(p, e - s, s, gcomm, cross.as_ref());
                loss += lm;
                g.iter_mut().zip(&gm).for_each(|(a, b)| *a += b);
                tick("host_marginal", t);
            }
            // Terms on kappa = f(g): cotangent on kappa, chained by f'(g).
            let t = Instant::now();
            let kd = match &kappa_tf {
                Some(tf) if kens.is_some() || kurtosis.is_some() || minkowski.is_some() => Some(tf.apply(p)),
                _ => None,
            };
            tick("host_kappa_transform", t);
            if let (Some((k, dk)), Some(ke)) = (&kd, &kens) {
                let t = Instant::now();
                let (lk, gk) = ke.loss_and_grad(k);
                loss += lk;
                g.iter_mut().zip(gk.iter().zip(dk)).for_each(|(a, (b, d))| *a += b * d);
                tick("host_kappa_objective", t);
            }
            if let Some(ku) = &kurtosis {
                let t = Instant::now();
                let (l4, g4) = ku.loss_and_grad(kd.as_ref().map_or(p, |(k, _)| k.as_slice()), e - s, gcomm, cross.as_ref());
                loss += l4;
                match &kd {
                    Some((_, dk)) => g.iter_mut().zip(g4.iter().zip(dk)).for_each(|(a, (b, d))| *a += b * d),
                    None => g.iter_mut().zip(&g4).for_each(|(a, b)| *a += b),
                }
                tick("host_kurtosis", t);
            }
            if let Some(mk) = &minkowski {
                let t = Instant::now();
                let (l5, g5) = mk.loss_and_grad(kd.as_ref().map_or(p, |(k, _)| k.as_slice()), e - s, gcomm, cross.as_ref());
                loss += l5;
                match &kd {
                    Some((_, dk)) => g.iter_mut().zip(g5.iter().zip(dk)).for_each(|(a, (b, d))| *a += b * d),
                    None => g.iter_mut().zip(&g5).for_each(|(a, b)| *a += b),
                }
                tick("host_minkowski", t);
            }
            project(&mut g);
            drop(kd);
            tick("host_evaluations", t_eval);
            (loss, g)
        };
        run_lbfgs_with(comm, &mut x, &opts, loss_and_grad, print)
    };
    let seconds = started.elapsed().as_secs_f64();
    // L-BFGS: what the evaluations leave of the run (two-loop recursion,
    // line search, copies of the variable).
    let lbfgs_seconds = seconds - host_timers.borrow().get("host_evaluations").copied().unwrap_or(0.0);
    host_timers.borrow_mut().insert("host_lbfgs", lbfgs_seconds);
    let calls_after = obj.exchange_calls();
    let per_eval = |a: u64, b: u64| ((b - a) as f64 / history.len().max(1) as f64 * 10.0).round() / 10.0;
    let mpi_calls = serde_json::json!({
        "neighbour_exchanges_per_evaluation": per_eval(calls_before.0, calls_after.0),
        "engine_allreduces_per_evaluation": per_eval(calls_before.1, calls_after.1),
        "halo_mode": std::env::var("STL_MPI_HALO").unwrap_or_else(|_| "point-to-point".into()),
        "overlap_first_coarse_level": args.overlap,
    });
    // Final statistics of this group's members and their mean over all members.
    let final_members = obj.statistics(&x);
    let final_mean = ens.mean(&final_members);
    let loss_mean: f64 =
        final_mean.iter().zip(&target_stats).zip(&obj.weights).map(|((a, b), w)| w * (a - b).norm_sqr()).sum();
    let n = plan.len();
    // Scatter of the members over its ergodic expectation, per coefficient.
    let scatter_ratio: Vec<f64> = ens.scatter(&final_members, &final_mean).iter().zip(&sigma2).map(|(d, s)| d / s).collect();
    let loss_scatter = if scatter.is_some() {
        args.scatter_weight / n as f64 * scatter_ratio.iter().map(|r| (r - 1.0) * (r - 1.0)).sum::<f64>()
    } else {
        0.0
    };
    let (loss_levels, level_ratios) = match &level_power {
        Some(lp) => (lp.loss_and_grad(&x, e - s, gcomm, cross.as_ref()).0, lp.ratios(&x, e - s, gcomm, cross.as_ref())),
        None => (0.0, Vec::new()),
    };
    // Kappa term: final value and residual of the mean per family (sigma / sqrt(B)).
    let kfinal = kappa_tf.as_ref().map(|tf| tf.apply(&x).0);
    let (loss_kappa, kappa_report) = match (&kappa_tf, &kens, &kfinal) {
        (Some(tf), Some(ke), Some(kx)) => {
            let kmean = ke.mean(&ke.obj.statistics(kx));
            let kloss: f64 = kmean.iter().zip(&kstats).zip(&ke.obj.weights).map(|((a, b), w)| w * (a - b).norm_sqr()).sum();
            let mut fams = serde_json::Map::new();
            for fam in &Family::ALL {
                let z: Vec<f64> = (0..kplan.len())
                    .filter(|&i| kplan[i].family == *fam)
                    .map(|i| {
                        let sigma = (kvar[i] / geo.pixels(kplan[i].native_dg) as f64).max(1e-300).sqrt();
                        (kmean[i] - kstats[i]).norm() / sigma * (members as f64).sqrt()
                    })
                    .collect();
                if !z.is_empty() {
                    let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64).sqrt();
                    fams.insert(fam.name().into(), serde_json::json!({"count": z.len(), "rms_sigma": rms, "max_sigma": z.iter().cloned().fold(0.0, f64::max)}));
                }
            }
            let (km, ks) = tf.kappa_moments();
            let report = serde_json::json!({
                "table": args.kappa_table,
                "weight": args.kappa_weight,
                "families": if kbase > 0 { args.kappa_families.iter().map(|f| f.name()).collect::<Vec<_>>() } else { Vec::new() },
                "s3_weight": args.kappa_s3_weight,
                "s3_levels_nside": args.kappa_s3_nside.map(|(a, b)| [a, b]),
                "s3_coefficients": kplan.len() - kbase,
                "coefficients": kplan.len(),
                "kappa_mean": km,
                "kappa_std": ks,
                "loss_final_term": kloss,
                "final_residual_in_ergodic_sigma": fams,
            });
            (kloss, report)
        }
        _ => (0.0, serde_json::Value::Null),
    };
    let loss_marginal = marginal.as_ref().map_or(0.0, |mg| mg.loss_and_grad(&x, e - s, s, gcomm, cross.as_ref()).0);
    // Kurtosis of the details (of kappa with --kappa-table): term and every level.
    let kurtosis_x: &[f64] = kfinal.as_deref().unwrap_or(&x);
    let loss_kurtosis = kurtosis.as_ref().map_or(0.0, |ku| ku.loss_and_grad(kurtosis_x, e - s, gcomm, cross.as_ref()).0);
    let kurtosis_report = match &kurtosis_all {
        Some(ka) => {
            let excess = |v: &[f64]| v.iter().map(|k| k - 3.0).collect::<Vec<_>>();
            serde_json::json!({
                "weight": args.kurtosis_weight,
                "of": if kappa_tf.is_some() { "kappa" } else { "map" },
                "levels_nside": kurtosis.as_ref().map(|ku| ku.levels.iter().map(|n| nside >> n).collect::<Vec<_>>()),
                "loss_final_term": loss_kurtosis,
                "all_levels_nside": (0..pyramid_levels).map(|n| nside >> n).collect::<Vec<_>>(),
                "target_excess": excess(&ka.target),
                "final_excess_mean_over_members": excess(&ka.values(kurtosis_x, e - s, gcomm, cross.as_ref())),
            })
        }
        None => serde_json::Value::Null,
    };
    let (loss_minkowski, minkowski_report) = match &minkowski {
        Some(mk) => {
            let l = mk.loss_and_grad(kurtosis_x, e - s, gcomm, cross.as_ref()).0;
            let report = serde_json::json!({
                "weight": args.minkowski_weight,
                "of": if kappa_tf.is_some() { "kappa" } else { "map" },
                "levels_nside": mk.levels_nside(),
                "nu": mk.nu,
                "h_over_level_std": mk.h,
                "eps_over_mean_squared_gradient": mk.eps,
                "standardised_levels": mk.standardise,
                "sigma": mk.sigma_source,
                "loss_final_term": l,
                "final_z_over_members": mk.report(kurtosis_x, e - s, first_member, gcomm, cross.as_ref()),
            });
            (l, report)
        }
        None => (0.0, serde_json::Value::Null),
    };
    let final_loss = loss_mean + loss_scatter + loss_levels + loss_kappa + loss_marginal + loss_kurtosis + loss_minkowski;
    let sigma: Vec<f64> = sigma2.iter().map(|v| v.sqrt()).collect();
    // Residual of the mean per family, in units of its ergodic sigma
    // (sigma / sqrt(B): sigma itself for one map).
    let rescale = (members as f64).sqrt();
    let mut per_family = serde_json::Map::new();
    // Per member: residual to the target and scatter around the mean, in units of sigma
    // (sums of squares per family, added over the groups).
    let mut member_sq = vec![0.0; 2 * Family::ALL.len()];
    for (f, fam) in Family::ALL.iter().enumerate() {
        let idx: Vec<usize> = (0..n).filter(|&i| plan[i].family == *fam).collect();
        if idx.is_empty() {
            continue;
        }
        let z: Vec<f64> = idx.iter().map(|&i| (final_mean[i] - target_stats[i]).norm() / sigma[i] * rescale).collect();
        let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64).sqrt();
        per_family.insert(fam.name().into(), serde_json::json!({"count": z.len(), "rms_sigma": rms, "max_sigma": z.iter().cloned().fold(0.0, f64::max)}));
        for member in final_members.chunks(n) {
            for &i in &idx {
                member_sq[2 * f] += ((member[i] - target_stats[i]).norm() / sigma[i]).powi(2);
                member_sq[2 * f + 1] += ((member[i] - final_mean[i]).norm() / sigma[i]).powi(2);
            }
        }
    }
    if let Some(c) = &cross {
        c.allreduce_sum(&mut member_sq);
    }
    let mut members_report = serde_json::Map::new();
    for (f, fam) in Family::ALL.iter().enumerate() {
        let count = plan.iter().filter(|c| c.family == *fam).count();
        if count > 0 {
            let k = (count * members) as f64;
            // Mean over the family of D_i / sigma_i^2 (1 for an ergodic ensemble).
            let ratio: f64 = (0..n).filter(|&i| plan[i].family == *fam).map(|i| scatter_ratio[i]).sum::<f64>() / count as f64;
            members_report.insert(
                fam.name().into(),
                serde_json::json!({
                    "residual_rms_sigma": (member_sq[2 * f] / k).sqrt(),
                    "scatter_rms_sigma": (member_sq[2 * f + 1] / k).sqrt(),
                    "scatter_over_ergodic_variance": ratio,
                }),
            );
        }
    }
    let mut timers: serde_json::Map<String, serde_json::Value> =
        obj.timers.borrow().iter().map(|(k, v)| (k.to_string(), serde_json::json!((v * 10.0).round() / 10.0))).collect();
    // Device phases (whole run: target statistics and kappa term included,
    // one engine) and memory.
    let (device_timers, device_memory) = obj.evaluator_report();
    for (k, v) in device_timers {
        timers.insert(k.to_string(), serde_json::json!((v * 100.0).round() / 100.0));
    }
    for (k, v) in host_timers.borrow().iter() {
        timers.insert(k.to_string(), serde_json::json!((v * 100.0).round() / 100.0));
    }
    let device_peaks = comm.allgather_i64(device_memory.map_or(0, |m| m.0 as i64));
    #[allow(unused_mut)]
    let mut cache_used = 0i64;
    #[cfg(feature = "gpu")]
    if let Some(dev) = &device {
        cache_used = dev.engine().geometry_cache().2 as i64;
    }
    let cache_used = comm.allgather_i64(cache_used);
    if let Some(info) = gpu_info.as_object_mut() {
        info.insert("memory_peak_bytes_per_rank".into(), serde_json::json!(device_peaks));
        info.insert("memory_total_bytes".into(), serde_json::json!(device_memory.map(|m| m.1)));
        info.insert("geometry_cache_used_bytes_per_rank".into(), serde_json::json!(cache_used));
    }
    let floor = obj.memory_floor_bytes(args.history_size);
    let floors: Vec<i64> = comm.allgather_i64(
        floor.as_object().unwrap().iter().filter_map(|(_, v)| v.as_i64()).sum(),
    );
    let peaks = comm.allgather_i64(proc_memory("VmHWM:"));
    let baselines = comm.allgather_i64(rss_baseline);

    // Outputs written by slices: rank 0 sizes the files, every rank writes
    // its range (the collectives order creation before the writes).  One map:
    // PREFIX_synthesis*; an ensemble: PREFIX_mMMM_synthesis_{nested,standardized}.npy.
    let prefix = args.out.clone().unwrap_or_else(|| format!("synthesis_rust_nside{nside}"));
    let file = |suffix: &str| PathBuf::from(format!("{prefix}{suffix}"));
    let member_file = |m: usize, suffix: &str| {
        if members == 1 { file(suffix) } else { PathBuf::from(format!("{prefix}_m{m:03}{suffix}")) }
    };
    let unstd = |v: &[f64]| v.iter().map(|x| x * std + mean).collect::<Vec<_>>();
    // (file, FITS column, written by this rank: values)
    let mut outputs: Vec<(SlicedMap, &str, Option<Vec<f64>>)> =
        vec![(SlicedMap::new(&file("_target.fits"), false), "TARGET", (group_id == 0).then(|| unstd(&target)))];
    let len = e - s;
    for m in 0..members {
        let mine = (first_member..first_member + local_members).contains(&m);
        let slice = || &x[(m - first_member) * len..(m - first_member + 1) * len];
        if members == 1 {
            outputs.push((SlicedMap::new(&file("_synthesis.fits"), false), "SYNTHESIS", mine.then(|| unstd(slice()))));
        }
        outputs.push((SlicedMap::new(&member_file(m, "_synthesis_nested.npy"), false), "", mine.then(|| unstd(slice()))));
        outputs.push((SlicedMap::new(&member_file(m, "_synthesis_standardized.npy"), false), "", mine.then(|| slice().to_vec())));
    }
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
        if let Some(values) = values {
            if let Err(e) = out.write(s, values) {
                eprintln!("stl-synth: rank {}: {e}", comm.rank());
                status = 1.0;
            }
        }
    }
    if global_sum(comm, status) > 0.0 {
        return Err("could not write the output files".into());
    }
    if !root {
        return Ok(None);
    }
    let synthesis_files: Vec<PathBuf> = (0..members).map(|m| member_file(m, "_synthesis_nested.npy")).collect();
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
        "ensemble": {
            "members": members,
            "groups": groups,
            "ranks_per_group": per_group,
            "loss": "sum_i w_i |mean over members of S_i - S_target,i|^2 + scatter_weight / n sum_i (D_i / sigma_i^2 - 1)^2",
            "scatter_weight": args.scatter_weight,
            "loss_final_mean_term": loss_mean,
            "loss_final_scatter_term": loss_scatter,
            "member_files": synthesis_files,
            "per_member_in_ergodic_sigma": members_report,
        },
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
        "freeze_nside": args.freeze_nside,
        "level_power": {
            "weight": args.level_power_weight,
            "levels_nside": (0..level_ratios.len()).map(|n| nside >> n).collect::<Vec<_>>(),
            "variance_over_target": level_ratios,
            "loss_final_term": loss_levels,
        },
        "kappa": kappa_report,
        "kurtosis": kurtosis_report,
        "minkowski": minkowski_report,
        "marginal": {"weight": args.marginal_weight, "loss_final_term": loss_marginal},
        "memory_floor_bytes": floor,
        "memory_floor_bytes_per_rank": floors,
        "memory_peak_bytes_per_rank": peaks,
        "rss_baseline_bytes_per_rank": baselines,
        "timers_seconds": timers,
        "mpi_calls": mpi_calls,
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
    use stl_core::engine::{EngineConfig, Partition};
    use stl_core::local::LocalGeometry;
    use stl_core::plan::Coefficient;
    use stl_gpu::{ConvForm, FieldStore, GeometryCache, Gpu, GpuEngine, GpuEvaluator, GpuOptions};

    /// The device engine of the rank, sized for `batch` maps (the largest
    /// batch it evaluates), as the evaluator of `plan` at that batch (other
    /// plans and batches: `GpuEvaluator::share`), and a description for
    /// the report.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluator<'a>(
        args: &Args,
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        plan: &[Coefficient],
        part: &Partition,
        rank: usize,
        batch: usize,
        reserve_bytes: usize,
    ) -> Result<(GpuEvaluator<'a>, serde_json::Value), String> {
        let gpu = Arc::new(Gpu::for_rank(rank)?);
        let cache = GeometryCache::parse(&args.gpu_cache).ok_or("--gpu-cache: unknown policy")?;
        let mut opts = GpuOptions { block_rows: args.gpu_block_rows, cache, reserve_bytes, ..Default::default() };
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
            "device": gpu.name,
            "devices_visible": Gpu::count()?,
            "device_ordinal": gpu.ordinal(),
            "blocks_per_level": engine.blocks_per_level(),
            "block_rows": engine.block_rows(),
            "geometry_cache": engine.geometry_cache().0.name(),
            "conv_form": engine.conv_form().name(),
            "fields": engine.field_store().name(),
            "field_bytes_estimate": engine.memory_parts().0,
            "decimation_rows_bytes": engine.memory_parts().1,
            "geometry_cache_budget_bytes": engine.geometry_cache().1,
        });
        Ok((GpuEvaluator::new(engine, plan), info))
    }
}
