//! Synthesis of planar images from the statistics of target images
//! (`stl_cli::planar`, `docs/planar_2d.md`).
//!
//! ```text
//! stl-synth2d --target Turb_6.npy --ntarget 10 --nbatch 10 --niter 100 --out run
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;
use stl_cli::planar::{nyquist_filter, standardize, white_noise, Config, Mode, PlanarObjective};
use stl_cli::proc_memory;
use stl_core::lbfgs::{minimize, minimize_preconditioned, LbfgsOptions, Local, Preconditioner};
use stl_core::planar::cascade::{resample, ConditionalDetail, SplitMix};
use stl_core::planar::transport::{bicubic, gradient, Smoother};
use stl_core::planar::warp::{reproject, Flow, Warp, WarpState};
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
  --lagrangian F      optimise a displacement of the initial images instead of
                      their pixels (docs/lagrangian_synthesis.md): F = stream
                      (incompressible), potential (compressive) or both
  --disp X            rms displacement in pixels of unit variables (default 1)
  --disp-scale S      Gaussian cutoff of the displacement in pixels (default 4)
  --disp-slope B      displacement spectrum ~ |k|^-B (default 1)
  --residual S        plus an Eulerian residual S * rho (default 0: none)
  --anchor K          re-anchor every K iterations: x0 <- x, variables <- 0,
                      new L-BFGS history (default 0: never)
  --reproject         rank-match x0 onto the targets' histogram at every anchor
  --transport         the --niter iterations are transport steps on the current images
                      (scheme P): x <- x(p + tau u), u = -K(g grad x), bicubic (Keys),
                      Armijo backtracking on tau; after --c2f, at the last level only
  --transport-precond S2
                      pixel L-BFGS with H0 = S2 I + grad x K grad x^T, frozen and
                      refreshed every --transport-refresh iterations (default 20)
  --transport-levels N, --transport-slope B
                      K = sum_l 2^(B l) 4^l (D^T)^l D^l over N pyramid levels (defaults 6, 0)
  --transport-step PX first tau: largest displacement of PX pixels (default 0.5);
                      tau is capped at 2 pixels of displacement
  --c2f LEVELS        coarse-to-fine start (docs/lagrangian_synthesis.md 6.4):
                      square sides, e.g. 32,64,128 (the image side last is
                      implied); pixel L-BFGS at each coarse level (--warm at
                      the first one), then upsampling plus detail drawn
                      conditionally on the coarse value and gradient bins;
                      with --out, PREFIX_c2f_<n>_maps.npy per coarse level
  --c2f-iter LIST     L-BFGS iterations per coarse level (default 300,200,...)
  --c2f-detail D      conditional (default) or white
  --c2f-bins Q        quantile bins per feature (default 8)
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
    c2f: Vec<usize>,
    c2f_iter: Vec<usize>,
    c2f_white: bool,
    c2f_bins: usize,
    lagrangian: Option<Flow>,
    disp: [f64; 3],
    residual: f64,
    anchor: usize,
    reproject: bool,
    transport: bool,
    transport_precond: Option<f64>,
    transport_refresh: usize,
    transport_levels: usize,
    transport_slope: f64,
    transport_step: f64,
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
        c2f: Vec::new(),
        c2f_iter: Vec::new(),
        c2f_white: false,
        c2f_bins: 8,
        lagrangian: None,
        disp: [1.0, 4.0, 1.0],
        residual: 0.0,
        anchor: 0,
        reproject: false,
        transport: false,
        transport_precond: None,
        transport_refresh: 20,
        transport_levels: 6,
        transport_slope: 0.0,
        transport_step: 0.5,
        gpu: false,
        out: None,
    };
    let mut it = argv.iter();
    fn num<T: std::str::FromStr>(v: Option<&String>, k: &str) -> Result<T, String> {
        v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: bad value"))
    }
    fn list(v: Option<&String>, k: &str) -> Result<Vec<usize>, String> {
        v.ok_or(format!("{k} needs a value"))?.split(',').map(|x| x.trim().parse().map_err(|_| format!("{k}: bad value"))).collect()
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
            "--c2f" => a.c2f = list(it.next(), k)?,
            "--c2f-iter" => a.c2f_iter = list(it.next(), k)?,
            "--c2f-detail" => {
                a.c2f_white = match it.next().map(String::as_str) {
                    Some("conditional") => false,
                    Some("white") => true,
                    _ => return Err("--c2f-detail is conditional or white".into()),
                }
            }
            "--c2f-bins" => a.c2f_bins = num(it.next(), k)?,
            "--lagrangian" => {
                a.lagrangian = Some(match it.next().map(String::as_str) {
                    Some("stream") => Flow::Stream,
                    Some("potential") => Flow::Potential,
                    Some("both") => Flow::Both,
                    _ => return Err("--lagrangian is stream, potential or both".into()),
                })
            }
            "--disp" => a.disp[0] = num(it.next(), k)?,
            "--disp-scale" => a.disp[1] = num(it.next(), k)?,
            "--disp-slope" => a.disp[2] = num(it.next(), k)?,
            "--residual" => a.residual = num(it.next(), k)?,
            "--anchor" => a.anchor = num(it.next(), k)?,
            "--reproject" => a.reproject = true,
            "--transport" => a.transport = true,
            "--transport-precond" => a.transport_precond = Some(num::<f64>(it.next(), k)?),
            "--transport-refresh" => a.transport_refresh = num(it.next(), k)?,
            "--transport-levels" => a.transport_levels = num(it.next(), k)?,
            "--transport-slope" => a.transport_slope = num::<f64>(it.next(), k)?,
            "--transport-step" => a.transport_step = num::<f64>(it.next(), k)?,
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

/// The Lagrangian variable `z = [theta (planes x warp.len()), rho (planes x npix)]`
/// of the images `x = x0 o (id + u(theta)) + residual * rho`, plane by plane.
struct Lag<'a> {
    warp: &'a Warp,
    x0: Vec<f64>,
    residual: f64,
    planes: usize,
}

impl Lag<'_> {
    fn len(&self) -> usize {
        let np = self.warp.h * self.warp.w;
        self.planes * (self.warp.len() + if self.residual > 0.0 { np } else { 0 })
    }

    fn apply(&self, z: &[f64]) -> (Vec<f64>, Vec<WarpState>) {
        let (np, nv) = (self.warp.h * self.warp.w, self.warp.len());
        let mut x = Vec::with_capacity(self.planes * np);
        let mut st = Vec::with_capacity(self.planes);
        for p in 0..self.planes {
            let (xp, s) = self.warp.forward(&self.x0[p * np..][..np], &z[p * nv..][..nv]);
            x.extend(xp);
            st.push(s);
        }
        if self.residual > 0.0 {
            let rho = &z[self.planes * nv..];
            x.iter_mut().zip(rho).for_each(|(a, r)| *a += self.residual * r);
        }
        (x, st)
    }

    fn pull(&self, st: &[WarpState], g: &[f64]) -> Vec<f64> {
        let np = self.warp.h * self.warp.w;
        let mut gz: Vec<f64> = (0..self.planes).flat_map(|p| self.warp.vjp(&st[p], &g[p * np..][..np])).collect();
        if self.residual > 0.0 {
            gz.extend(g.iter().map(|v| self.residual * v));
        }
        gz
    }
}

fn complex_rows(x: &[num_complex::Complex64]) -> Vec<f64> {
    x.iter().flat_map(|v| [v.re, v.im]).collect()
}

/// Planes `[P, n, n]` of `x` resampled to `m x m`.
fn resample_planes(x: &[f64], n: usize, m: usize) -> Vec<f64> {
    x.chunks(n * n).flat_map(|p| resample(p, n, m)).collect()
}

/// Planes of channel `ch` among the first `rows` images of `v` `[B, C, n, n]`.
fn channel_planes(v: &[f64], rows: usize, c: usize, ch: usize, np: usize) -> Vec<f64> {
    (0..rows).flat_map(|k| v[(k * c + ch) * np..][..np].to_vec()).collect()
}

/// `--c2f`: syntheses of the coarse levels and the initial images of the
/// finest one, in its standardised units (`mean`, `std` of `run`).
/// `target`: the Nyquist-filtered targets `[Bt, C, H, H]` in target units.
fn c2f_init(a: &Args, cfg: &Config, target: &[f64], (bt, c, h): (usize, usize, usize), (mean, std): (&[f64], &[f64])) -> Result<(Vec<f64>, serde_json::Value), String> {
    let mut levels = a.c2f.clone();
    levels.push(h);
    if levels.windows(2).any(|w| w[1] != 2 * w[0]) {
        return Err("--c2f: every level must be twice the previous one, up to the image side".into());
    }
    let tl: Vec<Vec<f64>> = levels.iter().map(|&n| if n == h { target.to_vec() } else { resample_planes(target, h, n) }).collect();
    let mut prev: Vec<f64> = Vec::new(); // synthesis of the previous level, target units
    let mut per_level = Vec::new();
    let mut equivalent = 0.0;
    for (i, &n) in levels.iter().enumerate() {
        let np = n * n;
        // Standardisation of this level, as run() does for the image side.
        let (tn, mn, sn) = if n == h { (Vec::new(), mean.to_vec(), std.to_vec()) } else { standardize(&nyquist_filter(&tl[i], n, n), c, np) };
        let init = if i == 0 {
            nyquist_filter(&white_noise(a.nbatch * c * np, a.seed), n, n)
        } else {
            let m = levels[i - 1];
            let up = resample_planes(&prev, m, n);
            let tu = resample_planes(&tl[i - 1], m, n);
            let mut rng = SplitMix(a.seed ^ (0x5eed_0000 + n as u64));
            let mut x = up.clone();
            for ch in 0..c {
                let cu = channel_planes(&tu, bt, c, ch, np);
                let dt: Vec<f64> = channel_planes(&tl[i], bt, c, ch, np).iter().zip(&cu).map(|(a, b)| a - b).collect();
                let s_up = channel_planes(&up, a.nbatch, c, ch, np);
                let d: Vec<f64> = if a.c2f_white {
                    let mu = dt.iter().sum::<f64>() / dt.len() as f64;
                    let sd = (dt.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / dt.len() as f64).sqrt();
                    (0..s_up.len()).map(|_| sd * rng.normal()).collect()
                } else {
                    ConditionalDetail::new(&cu, &dt, n, a.c2f_bins).draw(&s_up, &mut rng)
                };
                for k in 0..a.nbatch {
                    x[(k * c + ch) * np..][..np].iter_mut().zip(&d[k * np..][..np]).for_each(|(v, e)| *v += e);
                }
            }
            x.chunks(np).enumerate().flat_map(|(k, p)| p.iter().map(|v| (v - mn[k % c]) / sn[k % c]).collect::<Vec<_>>()).collect()
        };
        if n == h {
            let report = serde_json::json!({"levels": levels, "detail": if a.c2f_white { "white" } else { "conditional" }, "bins": a.c2f_bins,
                "per_level": per_level, "equivalent_evaluations": equivalent});
            return Ok((init, report));
        }
        let mut lc = Config::new(n, n, c, cfg.mode);
        lc.fewer = cfg.fewer;
        lc.tile = cfg.tile.min(n);
        lc.weights = cfg.weights;
        #[allow(unused_mut)]
        let mut obj = PlanarObjective::new(lc.clone(), &tn)?;
        if a.gpu {
            attach_gpu(&mut obj, a.nbatch)?;
        }
        let t0 = Instant::now();
        let mut u = init;
        let mut evals = 0;
        let mut hist = Vec::new();
        let opts = |k: usize| LbfgsOptions { lr: 1.0, max_iter: k, max_eval: None, tolerance_grad: 1e-12, tolerance_change: 1e-15, history_size: a.history };
        if i == 0 && a.warm > 0 && cfg.mode != Mode::StlDev {
            let mut wc = lc.clone();
            wc.mode = Mode::StlDev;
            let warm = PlanarObjective::new(wc, &tn)?;
            evals += minimize(&mut u, &opts(a.warm), &Local, &mut |x: &[f64]| warm.loss_and_grad(x));
        }
        let iters = a.c2f_iter.get(i).copied().unwrap_or(if i == 0 { 300 } else { 200 });
        evals += minimize(&mut u, &opts(iters), &Local, &mut |x: &[f64]| {
            let r = obj.loss_and_grad(x);
            hist.push(r.0);
            r
        });
        let eq = evals as f64 * (n as f64 / h as f64).powi(2);
        equivalent += eq;
        eprintln!("c2f level {n}: {evals} evaluations, loss {:.4e} -> {:.4e}", hist[0], hist[hist.len() - 1]);
        per_level.push(serde_json::json!({"n": n, "evaluations": evals, "seconds": t0.elapsed().as_secs_f64(),
            "loss_first": hist[0], "loss_last": hist[hist.len() - 1], "equivalent_evaluations": eq}));
        prev = u.chunks(np).enumerate().flat_map(|(k, p)| p.iter().map(|v| v * sn[k % c] + mn[k % c]).collect::<Vec<_>>()).collect();
        if let Some(p) = &a.out {
            write_npy_shape(Path::new(&format!("{p}_c2f_{n}_maps.npy")), &prev, &[a.nbatch, c, n, n])?;
        }
    }
    unreachable!()
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
    let mut c2f_report = serde_json::Value::Null;
    let t_c2f = Instant::now();
    if !a.c2f.is_empty() {
        if h != w {
            return Err("--c2f needs square images".into());
        }
        let (init, r) = c2f_init(a, &cfg, &target, (bt_use, c, h), (&mean, &std))?;
        u = init;
        c2f_report = r;
        c2f_report["seconds"] = serde_json::json!(t_c2f.elapsed().as_secs_f64());
    }
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
    let mut lbfgs = |obj: &PlanarObjective, niter: usize, u: &mut Vec<f64>, history: &mut Vec<f64>, warm_offset: usize, lag: Option<&Lag>, pre: Option<&mut dyn Preconditioner>| {
        let opts = LbfgsOptions { lr: 1.0, max_iter: niter, max_eval: None, tolerance_grad: 1e-12, tolerance_change: 1e-15, history_size: a.history };
        let mut f = |z: &[f64]| {
            let t = Instant::now();
            let (xl, st) = match lag {
                Some(l) => {
                    let (x, st) = l.apply(z);
                    (Some(x), st)
                }
                None => (None, Vec::new()),
            };
            let x = xl.as_deref().unwrap_or(z);
            let mut r = obj.loss_and_grad(x);
            if let Some(l) = lag {
                r.1 = l.pull(&st, &r.1);
            }
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
        minimize_preconditioned(u, &opts, &Local, &mut f, pre)
    };
    let mut warm_history = Vec::new();
    let mut transport_times: Vec<f64> = Vec::new();
    let mut evals = 0;
    if a.warm > 0 && cfg.mode != Mode::StlDev && a.c2f.is_empty() {
        let mut wc = cfg.clone();
        wc.mode = Mode::StlDev;
        let warm = PlanarObjective::new(wc, &t)?;
        evals += lbfgs(&warm, a.warm, &mut u, &mut warm_history, usize::MAX, None, None);
        report["warm_loss_history"] = serde_json::json!(warm_history);
    }
    if let Some(flow) = a.lagrangian {
        let [rms, scale, slope] = a.disp;
        let warp = Warp::new(h, w, flow, scale, slope, rms);
        // Sorted standardised target values per channel, for --reproject.
        let pools: Vec<Vec<f64>> = (0..c)
            .map(|ch| {
                let mut v: Vec<f64> = (0..bt_use).flat_map(|k| t[(k * c + ch) * h * w..][..h * w].iter().copied()).collect();
                v.sort_by(f64::total_cmp);
                v
            })
            .collect();
        let (mut done, mut cycles) = (0, 0);
        while done < a.niter {
            let k = if a.anchor > 0 { a.anchor.min(a.niter - done) } else { a.niter - done };
            if a.reproject {
                u.chunks_mut(h * w).enumerate().for_each(|(p, x)| reproject(x, &pools[p % c]));
            }
            let lag = Lag { warp: &warp, x0: u.clone(), residual: a.residual, planes: a.nbatch * c };
            let mut z = vec![0.0; lag.len()];
            let offset = evals - history.len();
            evals += lbfgs(&obj, k, &mut z, &mut history, offset, Some(&lag), None);
            u = lag.apply(&z).0;
            done += k;
            cycles += 1;
        }
        report["lagrangian"] = serde_json::json!({"flow": format!("{flow:?}"), "disp": rms, "disp_scale": scale, "disp_slope": slope,
            "residual": a.residual, "anchor": a.anchor, "reproject": a.reproject, "cycles": cycles});
    } else if let Some(s2) = a.transport_precond {
        let k = Smoother::new(h, w, a.transport_levels, a.transport_slope);
        let (mut done, mut refreshes) = (0, 0);
        while done < a.niter {
            let it = a.transport_refresh.min(a.niter - done);
            let mut pre = TransportMetric::new(&k, &u, s2);
            let offset = evals - history.len();
            evals += lbfgs(&obj, it, &mut u, &mut history, offset, None, Some(&mut pre));
            done += it;
            refreshes += 1;
        }
        report["transport_precond"] = serde_json::json!({"sigma2": s2, "refresh": a.transport_refresh, "refreshes": refreshes, "weights": k.weights});
    } else if a.transport {
        let k = Smoother::new(h, w, a.transport_levels, a.transport_slope);
        let r = transport(&obj, &k, &mut u, a.niter, a.transport_step, &mut history, &mut transport_times);
        evals += history.len();
        report["transport"] = serde_json::json!({"weights": k.weights, "first_step_pixels": a.transport_step, "steps": r.0, "taus": r.1, "max_displacements": r.2});
    } else {
        evals += lbfgs(&obj, a.niter, &mut u, &mut history, evals, None, None);
    }
    drop(lbfgs);
    times.extend(transport_times);
    let seconds = t1.elapsed().as_secs_f64();
    let (f_final, m_final) = obj.batch_features(&u);
    report["evaluations"] = serde_json::json!(evals);
    if !c2f_report.is_null() {
        c2f_report["total_equivalent_evaluations"] = serde_json::json!(c2f_report["equivalent_evaluations"].as_f64().unwrap() + evals as f64);
        report["c2f"] = c2f_report;
    }
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

/// `P = s2 I + grad x K grad x^T` per plane, `grad x` frozen at the images
/// it was built from: the metric of the transport step, linearised.
struct TransportMetric<'a> {
    k: &'a Smoother,
    grads: Vec<(Vec<f64>, Vec<f64>)>,
    s2: f64,
}

impl<'a> TransportMetric<'a> {
    fn new(k: &'a Smoother, u: &[f64], s2: f64) -> Self {
        TransportMetric { k, grads: u.chunks(k.h * k.w).map(|x| gradient(x, k.h, k.w)).collect(), s2 }
    }
}

impl Preconditioner for TransportMetric<'_> {
    fn apply(&self, g: &[f64]) -> Vec<f64> {
        let n = self.k.h * self.k.w;
        let mut out = Vec::with_capacity(g.len());
        for (gp, (gr, gc)) in g.chunks(n).zip(&self.grads) {
            let kr = self.k.apply(&gp.iter().zip(gr).map(|(a, b)| a * b).collect::<Vec<_>>());
            let kc = self.k.apply(&gp.iter().zip(gc).map(|(a, b)| a * b).collect::<Vec<_>>());
            out.extend((0..n).map(|p| self.s2 * gp[p] + gr[p] * kr[p] + gc[p] * kc[p]));
        }
        out
    }
}

/// Velocity `-K(g grad x)` of every plane, and the slope `<g, grad x . v>`.
fn velocity(k: &Smoother, u: &[f64], g: &[f64]) -> (Vec<(Vec<f64>, Vec<f64>)>, f64) {
    let n = k.h * k.w;
    let mut slope = 0.0;
    let v = u
        .chunks(n)
        .zip(g.chunks(n))
        .map(|(x, gp)| {
            let (gr, gc) = gradient(x, k.h, k.w);
            let vr: Vec<f64> = k.apply(&gp.iter().zip(&gr).map(|(a, b)| a * b).collect::<Vec<_>>()).into_iter().map(|v| -v).collect();
            let vc: Vec<f64> = k.apply(&gp.iter().zip(&gc).map(|(a, b)| a * b).collect::<Vec<_>>()).into_iter().map(|v| -v).collect();
            slope += (0..n).map(|p| gp[p] * (gr[p] * vr[p] + gc[p] * vc[p])).sum::<f64>();
            (vr, vc)
        })
        .collect();
    (v, slope)
}

/// Scheme P: `niter` transport steps `x <- x(p + tau v)`, `v = -K(g grad x)`,
/// Armijo backtracking on `tau` (doubled after an accepted step, capped at a
/// displacement of 2 pixels).  Every loss evaluation is pushed on
/// `history`.  Returns the accepted steps, their `tau` and largest
/// displacements.
fn transport(obj: &PlanarObjective, k: &Smoother, u: &mut Vec<f64>, niter: usize, first_px: f64, history: &mut Vec<f64>, times: &mut Vec<f64>) -> (usize, Vec<f64>, Vec<f64>) {
    let n = k.h * k.w;
    let mut eval = |x: &[f64], history: &mut Vec<f64>| {
        let t = Instant::now();
        let r = obj.loss_and_grad(x);
        times.push(t.elapsed().as_secs_f64());
        history.push(r.0);
        if history.len() % 10 == 1 {
            eprintln!("eval {:4}  loss {:.6e}  {:.3} s", history.len(), r.0, times.last().unwrap());
        }
        r
    };
    let (mut loss, mut g) = eval(u, history);
    let (mut taus, mut disps) = (Vec::new(), Vec::new());
    let mut tau = f64::NAN;
    let mut steps = 0;
    while history.len() < niter {
        let (v, slope) = velocity(k, u, &g);
        if slope >= 0.0 {
            break;
        }
        let vmax = v.iter().flat_map(|(r, c)| r.iter().chain(c)).fold(0.0f64, |m, x| m.max(x.abs()));
        if vmax == 0.0 {
            break;
        }
        tau = if tau.is_nan() { first_px / vmax } else { (2.0 * tau).min(2.0 / vmax) };
        let mut accepted = false;
        while history.len() < niter {
            let x: Vec<f64> = u
                .chunks(n)
                .zip(&v)
                .flat_map(|(x, (vr, vc))| {
                    let (sr, sc): (Vec<f64>, Vec<f64>) = (vr.iter().map(|a| tau * a).collect(), vc.iter().map(|a| tau * a).collect());
                    bicubic(x, k.h, k.w, &sr, &sc)
                })
                .collect();
            let (l, gn) = eval(&x, history);
            if l <= loss + 1e-4 * tau * slope {
                (*u, loss, g) = (x, l, gn);
                accepted = true;
                break;
            }
            tau *= 0.5;
        }
        if !accepted {
            break;
        }
        taus.push(tau);
        disps.push(tau * vmax);
        steps += 1;
    }
    (steps, taus, disps)
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
