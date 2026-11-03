//! Synthesis of planar images from the statistics of target images
//! (`stl_cli::planar`, `docs/planar_2d.md`).
//!
//!     stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --niter 100 --out run

use std::path::{Path, PathBuf};
use std::time::Instant;
use stl_cli::planar::{nyquist_filter, standardize, white_noise, Config, Mode, PlanarObjective};
use stl_cli::proc_memory;
use stl_core::lbfgs::{minimize, LbfgsOptions, Local};
use stl_mpi::io::{read_npy_shape, write_npy_shape};

const USAGE: &str = "\
stl-synth2d: synthesis of periodic planar images (L-BFGS on the images)

  --target PATH       target images, .npy [Bt, H, W] or [Bt, C, H, W]
  --ntarget N         first N target images (default all)
  --crop S            top-left S x S of every image (tests)
  --nbatch B          images synthesised (default 10)
  --mode M            ensemble (default): ScatCov + power spectrum + Minkowski,
                      batch mean and variance against the targets';
                      stldev: the loss of STL-Dev's synthesize_from_maps
  --J N               scales (default log2(min(H, W)) - 2)
  --tile T            tile side of the block-wise engine (default 64)
  --fewer             has_fewer_convolutions definitions of S3/S4
  --bins N            power-spectrum bins (default 2^(log2(min) - 4); 0: none)
  --w-sc X --w-ps X --w-mf X   ensemble block weights (default 1)
  --niter N           L-BFGS iterations (default 100), --history N (default 50)
  --warm N            first N iterations on the stldev loss (ensemble mode:
                      a well-conditioned start), then --niter on the ensemble
  --init PATH         initial images (standardised units), .npy [B, (C,) H, W]
  --seed N            seed of the initial white noise (default 26)
  --bench K           only time K evaluations of the loss and gradient
  --trace K           every K evaluations, the batch-mean feature residuals
                      (mean_b f - mean_t f) and image 0: PREFIX_trace.npy
                      [S, n, 2], PREFIX_trace_maps.npy [S, C, H, W]
  --gpu               the running batch on a CUDA device (feature gpu;
                      STL_GPU_DEVICE, default 0): ScatCov, spectra, Minkowski;
                      --tile as on the CPU (a tile larger than the image:
                      whole levels resident)
  --out PREFIX        outputs: PREFIX_maps.npy [B, C, H, W] (target units),
                      PREFIX_features_{target,init,final}.npy [*, n, 2],
                      PREFIX_mf_{target,init,final}.npy, PREFIX.json
Threads: STL_THREADS (default all available).";

#[derive(Debug)]
struct Args {
    target: PathBuf,
    ntarget: Option<usize>,
    crop: Option<usize>,
    nbatch: usize,
    mode: Mode,
    j: Option<usize>,
    tile: usize,
    fewer: bool,
    bins: Option<usize>,
    weights: [f64; 3],
    niter: usize,
    warm: usize,
    history: usize,
    init: Option<PathBuf>,
    seed: u64,
    bench: usize,
    trace: usize,
    gpu: bool,
    out: Option<String>,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        target: PathBuf::new(),
        ntarget: None,
        crop: None,
        nbatch: 10,
        mode: Mode::Ensemble,
        j: None,
        tile: 64,
        fewer: false,
        bins: None,
        weights: [1.0; 3],
        niter: 100,
        warm: 0,
        history: 50,
        init: None,
        seed: 26,
        bench: 0,
        trace: 0,
        gpu: false,
        out: None,
    };
    let mut it = argv.iter();
    fn num<T: std::str::FromStr>(v: Option<&String>, k: &str) -> Result<T, String> {
        v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: bad value"))
    }
    while let Some(k) = it.next() {
        match k.as_str() {
            "--target" => a.target = PathBuf::from(it.next().ok_or("--target needs a path")?),
            "--ntarget" => a.ntarget = Some(num(it.next(), k)?),
            "--crop" => a.crop = Some(num(it.next(), k)?),
            "--nbatch" => a.nbatch = num(it.next(), k)?,
            "--mode" => {
                a.mode = match it.next().map(String::as_str) {
                    Some("ensemble") => Mode::Ensemble,
                    Some("stldev") => Mode::StlDev,
                    _ => return Err("--mode is ensemble or stldev".into()),
                }
            }
            "--J" => a.j = Some(num(it.next(), k)?),
            "--tile" => a.tile = num(it.next(), k)?,
            "--fewer" => a.fewer = true,
            "--bins" => a.bins = Some(num(it.next(), k)?),
            "--w-sc" => a.weights[0] = num(it.next(), k)?,
            "--w-ps" => a.weights[1] = num(it.next(), k)?,
            "--w-mf" => a.weights[2] = num(it.next(), k)?,
            "--niter" => a.niter = num(it.next(), k)?,
            "--history" => a.history = num(it.next(), k)?,
            "--warm" => a.warm = num(it.next(), k)?,
            "--init" => a.init = Some(PathBuf::from(it.next().ok_or("--init needs a path")?)),
            "--seed" => a.seed = num(it.next(), k)?,
            "--bench" => a.bench = num(it.next(), k)?,
            "--trace" => a.trace = num(it.next(), k)?,
            "--gpu" => a.gpu = true,
            "--out" => a.out = Some(it.next().ok_or("--out needs a prefix")?.clone()),
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.target.as_os_str().is_empty() {
        return Err(format!("--target is required\n{USAGE}"));
    }
    Ok(a)
}

/// Images `[B, C, H, W]` of a `.npy` of 2, 3 or 4 dimensions.
fn read_images(path: &Path) -> Result<(usize, usize, usize, usize, Vec<f64>), String> {
    let (shape, v) = read_npy_shape(path)?;
    let (b, c, h, w) = match shape[..] {
        [h, w] => (1, 1, h, w),
        [b, h, w] => (b, 1, h, w),
        [b, c, h, w] => (b, c, h, w),
        _ => return Err(format!("{}: shape {shape:?} is not [B, (C,) H, W]", path.display())),
    };
    Ok((b, c, h, w, v))
}

/// The first `count` images, cropped to their top-left `s x s`.
fn select(v: &[f64], (c, h, w): (usize, usize, usize), count: usize, s: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count * c * s * s);
    for p in 0..count * c {
        for r in 0..s {
            out.extend_from_slice(&v[(p * h + r) * w..][..s]);
        }
    }
    out
}

fn complex_rows(x: &[num_complex::Complex64]) -> Vec<f64> {
    x.iter().flat_map(|v| [v.re, v.im]).collect()
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    let (bt, c, h0, w0, data) = read_images(&a.target)?;
    let bt_use = a.ntarget.unwrap_or(bt).min(bt);
    let (h, w) = a.crop.map_or((h0, w0), |s| (s.min(h0), s.min(w0)));
    if h != w && a.crop.is_some() {
        return Err("--crop needs square images".into());
    }
    let target = if (h, w) == (h0, w0) { data[..bt_use * c * h * w].to_vec() } else { select(&data, (c, h0, w0), bt_use, h) };
    let target = nyquist_filter(&target, h, w);
    let (t, mean, std) = standardize(&target, c, h * w);
    let mut cfg = Config::new(h, w, c, a.mode);
    cfg.fewer = a.fewer;
    cfg.tile = a.tile;
    cfg.weights = a.weights;
    if let Some(j) = a.j {
        cfg.j = j;
    }
    if let Some(n) = a.bins {
        cfg.n_bins = n;
    }
    let t0 = Instant::now();
    #[allow(unused_mut)]
    let mut obj = PlanarObjective::new(cfg.clone(), &t)?;
    if a.gpu {
        attach_gpu(&mut obj, a.nbatch)?;
    }
    let setup = t0.elapsed().as_secs_f64();
    let n = c * h * w;
    let mut u = match &a.init {
        Some(p) => {
            let (b, ci, hi, wi, v) = read_images(p)?;
            if (ci, hi, wi) != (c, h, w) || b < a.nbatch {
                return Err(format!("{}: [{b}, {ci}, {hi}, {wi}] does not hold {} images of [{c}, {h}, {w}]", p.display(), a.nbatch));
            }
            v[..a.nbatch * n].to_vec()
        }
        None => nyquist_filter(&white_noise(a.nbatch * n, a.seed), h, w),
    };
    let mut report = serde_json::json!({
        "target": a.target.display().to_string(), "ntarget": bt_use, "nbatch": a.nbatch,
        "shape": [c, h, w], "J": cfg.j, "L": 4, "tile": cfg.tile, "n_bins": cfg.n_bins, "fewer": cfg.fewer,
        "mode": format!("{:?}", cfg.mode), "weights": cfg.weights, "threads": stl_cli::ensemble::fields::host_threads(),
        "coefficients": obj.feature_count(), "minkowski_values": obj.mf_count(),
        "target_mean": mean, "target_std": std, "setup_seconds": setup,
    });
    if a.bench > 0 {
        let mut times = Vec::new();
        let mut loss = 0.0;
        for _ in 0..a.bench {
            let t = Instant::now();
            loss = obj.loss_and_grad(&u).0;
            times.push(t.elapsed().as_secs_f64());
        }
        report["bench_seconds"] = serde_json::json!(times);
        report["loss"] = serde_json::json!(loss);
        report["blocks"] = blocks(&obj);
        report["peak_rss_bytes"] = serde_json::json!(proc_memory("VmHWM:"));
        report["peak_device_bytes"] = serde_json::json!(obj.device_peak_bytes());
        return Ok(report);
    }
    let (f_init, m_init) = obj.batch_features(&u);
    let mut history = Vec::new();
    let mut times = Vec::new();
    // Feature residuals and image 0 every `--trace` evaluations (an extra forward each).
    let nf = obj.feature_count();
    let target_mean: Vec<num_complex::Complex64> = (0..nf).map(|i| (0..bt_use).map(|k| obj.target[k * nf + i]).sum::<num_complex::Complex64>() / bt_use as f64).collect();
    let mut trace: Vec<f64> = Vec::new();
    let mut trace_maps: Vec<f64> = Vec::new();
    let mut trace_evals: Vec<usize> = Vec::new();
    let t1 = Instant::now();
    let mut lbfgs = |obj: &PlanarObjective, niter: usize, u: &mut Vec<f64>, history: &mut Vec<f64>, warm_offset: usize| {
        let opts = LbfgsOptions { lr: 1.0, max_iter: niter, max_eval: None, tolerance_grad: 1e-12, tolerance_change: 1e-15, history_size: a.history };
        let mut f = |x: &[f64]| {
            let t = Instant::now();
            let r = obj.loss_and_grad(x);
            times.push(t.elapsed().as_secs_f64());
            history.push(r.0);
            if a.trace > 0 && warm_offset != usize::MAX && (history.len() - 1) % a.trace == 0 {
                let (f, _) = obj.batch_features(x);
                let b = f.len() / nf;
                for i in 0..nf {
                    let d = (0..b).map(|k| f[k * nf + i]).sum::<num_complex::Complex64>() / b as f64 - target_mean[i];
                    trace.extend([d.re, d.im]);
                }
                trace_maps.extend_from_slice(&x[..n]);
                trace_evals.push(history.len() - 1 + warm_offset);
            }
            if history.len() % 10 == 1 {
                eprintln!("eval {:4}  loss {:.6e}  {:.3} s", history.len(), r.0, times.last().unwrap());
            }
            r
        };
        minimize(u, &opts, &Local, &mut f)
    };
    let mut warm_history = Vec::new();
    let mut evals = 0;
    if a.warm > 0 && cfg.mode != Mode::StlDev {
        let mut wc = cfg.clone();
        wc.mode = Mode::StlDev;
        let warm = PlanarObjective::new(wc, &t)?;
        evals += lbfgs(&warm, a.warm, &mut u, &mut warm_history, usize::MAX);
        report["warm_loss_history"] = serde_json::json!(warm_history);
    }
    evals += lbfgs(&obj, a.niter, &mut u, &mut history, evals);
    drop(lbfgs);
    let seconds = t1.elapsed().as_secs_f64();
    let (f_final, m_final) = obj.batch_features(&u);
    report["evaluations"] = serde_json::json!(evals);
    report["seconds"] = serde_json::json!(seconds);
    report["seconds_per_evaluation"] = serde_json::json!(seconds / evals as f64);
    report["loss_history"] = serde_json::json!(history);
    report["loss_first"] = serde_json::json!(history.first());
    report["loss_last"] = serde_json::json!(history.last());
    report["timers_seconds"] = serde_json::json!(times);
    report["blocks"] = blocks(&obj);
    report["peak_rss_bytes"] = serde_json::json!(proc_memory("VmHWM:"));
    report["peak_device_bytes"] = serde_json::json!(obj.device_peak_bytes());
    report["families"] = serde_json::json!(obj.plan().iter().map(|q| q.family.name()).chain((obj.plan().len()..obj.feature_count()).map(|_| "PS")).collect::<Vec<_>>());
    if let Some(p) = &a.out {
        let nf = obj.feature_count();
        let (mean, std) = (&mean, &std);
        let maps: Vec<f64> = u.chunks(h * w).enumerate().flat_map(|(k, v)| v.iter().map(move |x| x * std[k % c] + mean[k % c])).collect();
        write_npy_shape(Path::new(&format!("{p}_maps.npy")), &maps, &[a.nbatch, c, h, w])?;
        write_npy_shape(Path::new(&format!("{p}_target.npy")), &target, &[bt_use, c, h, w])?;
        for (name, f, rows) in [("target", &obj.target, bt_use), ("init", &f_init, a.nbatch), ("final", &f_final, a.nbatch)] {
            write_npy_shape(Path::new(&format!("{p}_features_{name}.npy")), &complex_rows(f), &[rows, nf, 2])?;
        }
        let nm = obj.mf_count();
        if nm > 0 {
            for (name, m, rows) in [("target", &obj.target_mf, bt_use), ("init", &m_init, a.nbatch), ("final", &m_final, a.nbatch)] {
                write_npy_shape(Path::new(&format!("{p}_mf_{name}.npy")), m, &[rows, nm])?;
            }
        }
        if a.trace > 0 {
            let s = trace_evals.len();
            write_npy_shape(Path::new(&format!("{p}_trace.npy")), &trace, &[s, nf, 2])?;
            write_npy_shape(Path::new(&format!("{p}_trace_maps.npy")), &trace_maps, &[s, c, h, w])?;
            report["trace_evaluations"] = serde_json::json!(trace_evals);
            report["plan"] = serde_json::json!(obj.plan().iter().map(|q| serde_json::json!({"family": q.family.name(), "j1": q.j1, "j2": q.j2, "j3": q.j3, "l1": q.l1, "l2": q.l2, "l3": q.l3})).collect::<Vec<_>>());
        }
        std::fs::write(format!("{p}.json"), serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    }
    Ok(report)
}

#[cfg(feature = "gpu")]
fn attach_gpu(obj: &mut PlanarObjective, batch: usize) -> Result<(), String> {
    let dev = std::env::var("STL_GPU_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    obj.use_gpu(std::sync::Arc::new(stl_gpu::Gpu::new(dev)?), batch)
}

#[cfg(not(feature = "gpu"))]
fn attach_gpu(_obj: &mut PlanarObjective, _batch: usize) -> Result<(), String> {
    Err("--gpu needs the gpu feature (cargo build --features gpu)".into())
}

fn blocks(obj: &PlanarObjective) -> serde_json::Value {
    serde_json::Value::Object(obj.last.lock().unwrap().iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(r) => {
            let keys = ["evaluations", "seconds_per_evaluation", "loss_first", "loss_last", "bench_seconds", "loss", "peak_rss_bytes", "peak_device_bytes"];
            let short: serde_json::Map<String, serde_json::Value> = keys.iter().filter(|k| !r[**k].is_null()).map(|k| (k.to_string(), r[*k].clone())).collect();
            println!("{}", serde_json::Value::Object(short));
        }
        Err(e) => {
            eprintln!("stl-synth2d: {e}");
            std::process::exit(1);
        }
    }
}
