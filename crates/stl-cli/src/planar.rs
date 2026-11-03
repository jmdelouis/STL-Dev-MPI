//! Planar (2D image) synthesis, `stl-synth2d`: a batch of `B` periodic
//! images is the variable, its statistics are matched to those of `Bt`
//! target images.  The images are in the target's standardised units (the
//! target batch minus its mean, over its standard deviation, per channel).
//!
//! Two objectives:
//!
//! - [`Mode::StlDev`], the loss of STL-Dev's `Synthesis.optimize_from_maps`
//!   (`synthesize_from_maps`, `mean_field=True`): the statistics of every
//!   image are normalised by the target's (`var / var_ref`, `S_k` by the
//!   square roots of the target's `S2` diagonal, the spectra by those of the
//!   target's auto-spectra), S3 and S4 weighted by 3.5 and 3.5^2, averaged
//!   over the batch, and `L = sum_i |mean_b f_i(u_b) - mean_t f_i(t)|^2`;
//! - [`Mode::Ensemble`], the ensemble objective of the HEALPix runs
//!   (`stl-ensemble`, P. Campeti) on the plane: the same normalised ScatCov
//!   features, the auto power spectra in Gaussian log rings and the soft
//!   Minkowski functionals (V0, V1, V2 at 65 thresholds, spectral
//!   derivatives), each block comparing the batch mean with the target mean
//!   and the batch variance with the target variance
//!   (`ensemble::losses::{sc_loss, power_loss_masked, mf_loss}`):
//!   `L = w_sc L_SC + w_ps L_PS + w_mf L_MF`.
//!
//! The images are evaluated in parallel on the host threads (one engine
//! per image, its tiles on the remaining threads).

use crate::ensemble::fields::{chunked, host_threads, Minkowski, MinkowskiState};
use crate::ensemble::losses::{mf_loss, moments, power_loss_masked, sc_loss, torch_median, Block};
use num_complex::Complex64;
use std::sync::Mutex;
use std::time::Instant;
use stl_core::plan::{Coefficient, Family};
use stl_core::planar::spectrum::{Fft2, GaussianRings};
use stl_core::planar::{PlanarEngine, Pyramid};

type C64 = Complex64;

/// Loss definition of the planar synthesis (module header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// STL-Dev's `Synthesis.optimize_from_maps` loss on the mean features.
    StlDev,
    /// Ensemble objective: SC, power-spectrum and Minkowski blocks on the
    /// batch mean and variance.
    Ensemble,
}

/// Image size, scales and loss weights of a [`PlanarObjective`].
#[derive(Clone, Debug)]
pub struct Config {
    /// Image height in pixels.
    pub h: usize,
    /// Image width in pixels.
    pub w: usize,
    /// Channels per image.
    pub channels: usize,
    /// Number of scales `J`.
    pub j: usize,
    /// `has_fewer_convolutions` definitions of S3/S4.
    pub fewer: bool,
    /// Tile side in pixels of every level (clipped to the level).
    pub tile: usize,
    /// Bins of the power spectrum (0: no spectrum).
    pub n_bins: usize,
    /// Loss definition.
    pub mode: Mode,
    /// Ad-hoc weights of S3 and S4 (STL-Dev: 3.5 and 3.5^2).
    pub s3_weight: f64,
    /// Ad-hoc weight of S4 (STL-Dev: 3.5^2).
    pub s4_weight: f64,
    /// Ensemble block weights: SC, power spectrum, Minkowski.
    pub weights: [f64; 3],
}

impl Config {
    /// STL-Dev's defaults for `h x w` images: `J = log2(min) - 2`,
    /// `n_bins = 2^(log2(min) - 4)`.
    pub fn new(h: usize, w: usize, channels: usize, mode: Mode) -> Self {
        let lg = (h.min(w) as f64).log2().floor() as usize;
        Config {
            h,
            w,
            channels,
            j: lg.saturating_sub(2).max(1),
            fewer: false,
            tile: 64,
            n_bins: 1 << lg.saturating_sub(4),
            mode,
            s3_weight: 3.5,
            s4_weight: 3.5 * 3.5,
            weights: [1.0, 1.0, 1.0],
        }
    }
}

/// `x - mean` over `std` per channel, both over the whole batch
/// (`wavelet_op.standardize(mean_field=True)`): `(images, mean, std)`.
pub fn standardize(x: &[f64], channels: usize, npix: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let b = x.len() / (channels * npix);
    let mean: Vec<f64> = (0..channels).map(|c| (0..b).map(|i| x[(i * channels + c) * npix..][..npix].iter().sum::<f64>() / npix as f64).sum::<f64>() / b as f64).collect();
    let mut out = x.to_vec();
    for (k, v) in out.chunks_mut(npix).enumerate() {
        v.iter_mut().for_each(|a| *a -= mean[k % channels]);
    }
    let std: Vec<f64> = (0..channels).map(|c| ((0..b).map(|i| out[(i * channels + c) * npix..][..npix].iter().map(|v| v * v).sum::<f64>() / npix as f64).sum::<f64>() / b as f64).sqrt()).collect();
    for (k, v) in out.chunks_mut(npix).enumerate() {
        v.iter_mut().for_each(|a| *a /= std[k % channels]);
    }
    (out, mean, std)
}

/// Features of one image and what their cotangent needs.
struct Eval {
    sc: Vec<C64>,
    pyr: Pyramid,
    ps: Vec<C64>,
    xs: Vec<Vec<C64>>,
    /// Minkowski values per channel, and their states (host evaluation).
    mf: Vec<Vec<f64>>,
    mfs: Vec<MinkowskiState>,
}

/// The device of the running batch: ScatCov and Fourier terms.
#[cfg(feature = "gpu")]
pub struct Device {
    /// ScatCov engine of the batch on the device.
    pub sc: stl_gpu::planar::GpuPlanar,
    /// Fourier terms of the objective on the device.
    pub terms: stl_gpu::planar::GpuPlanarTerms,
}

#[cfg(feature = "gpu")]
struct DeviceState {
    pyr: stl_gpu::planar::GpuPyramid,
    terms: stl_gpu::planar::TermsState,
}

#[cfg(not(feature = "gpu"))]
struct DeviceState;

/// Planar synthesis objective: the normalised features of the `Bt` target
/// images and the loss and gradient of a batch of `B` images for the
/// [`Mode`] of its [`Config`].
pub struct PlanarObjective {
    /// Image size, scales and loss definition.
    pub cfg: Config,
    /// ScatCov engine of one image (tiles on host threads).
    pub engine: PlanarEngine,
    rings: Option<GaussianRings>,
    fft: Fft2,
    mk: Minkowski,
    /// Feature `f = raw / div * weight` of every SC coefficient and spectrum value.
    div: Vec<f64>,
    weight: Vec<f64>,
    ps_div: Vec<f64>,
    /// Target features `[Bt, n]` (SC then spectra) and Minkowski `[Bt, C * 195]`.
    pub target: Vec<C64>,
    /// Target Minkowski values `[Bt, C * 195]` ([`Mode::Ensemble`] only, else empty).
    pub target_mf: Vec<f64>,
    /// Number `Bt` of target images.
    pub target_count: usize,
    /// Per-block values and timers of the last evaluation.
    pub last: Mutex<Vec<(&'static str, f64)>>,
    /// The running batch on a CUDA device ([`PlanarObjective::use_gpu`]).
    #[cfg(feature = "gpu")]
    pub device: Option<Mutex<Device>>,
}

impl PlanarObjective {
    /// Objective for the standardised target images `t` `[Bt, C, H, W]`.
    pub fn new(cfg: Config, t: &[f64]) -> Result<Self, String> {
        let npix = cfg.h * cfg.w;
        let bt = t.len() / (cfg.channels * npix);
        if bt * cfg.channels * npix != t.len() || bt == 0 {
            return Err("target size is not a whole number of images".into());
        }
        if cfg.mode == Mode::Ensemble && bt < 2 {
            return Err("the ensemble objective needs at least two target images".into());
        }
        let engine = PlanarEngine::new(cfg.h, cfg.w, cfg.channels, cfg.j, cfg.fewer, cfg.tile);
        let rings = (cfg.n_bins > 0).then(|| GaussianRings::new(cfg.h, cfg.w, cfg.n_bins, cfg.j, 0));
        let mut obj = PlanarObjective {
            fft: Fft2::new(cfg.h, cfg.w),
            mk: Minkowski::default(),
            div: Vec::new(),
            weight: Vec::new(),
            ps_div: Vec::new(),
            target: Vec::new(),
            target_mf: Vec::new(),
            target_count: bt,
            last: Mutex::new(Vec::new()),
            #[cfg(feature = "gpu")]
            device: None,
            cfg,
            engine,
            rings,
        };
        let evals = obj.evaluate(t, false);
        obj.normalisation(&evals);
        obj.target = evals.iter().flat_map(|e| obj.features(e)).collect();
        if obj.cfg.mode == Mode::Ensemble {
            obj.target_mf = evals.iter().flat_map(|e| e.mf.concat()).collect();
        }
        Ok(obj)
    }

    /// Coefficient plan of the ScatCov engine (order of the SC features).
    pub fn plan(&self) -> &[Coefficient] {
        self.engine.plan()
    }

    /// Features per image: SC coefficients then spectra.
    pub fn feature_count(&self) -> usize {
        self.plan().len() + self.ps_div.len()
    }

    /// Minkowski values per image.
    pub fn mf_count(&self) -> usize {
        if self.cfg.mode == Mode::Ensemble {
            self.cfg.channels * self.mk.count()
        } else {
            0
        }
    }

    /// Engine threads per image when `images` run side by side.
    fn tile_threads(&self, images: usize) -> usize {
        (host_threads() / images.max(1)).max(1)
    }

    /// Statistics of a batch of `b` images on the device, if one is
    /// attached for that batch size: ScatCov `[b, len(plan)]`, spectra
    /// `[b, pairs x bins]`, Minkowski values `[b, C x 3K]` and the state.
    #[cfg(feature = "gpu")]
    fn device_forward(&self, x: &[f64], b: usize) -> Option<(Vec<C64>, Vec<C64>, Vec<f64>, DeviceState)> {
        let dev = self.device.as_ref()?.lock().unwrap();
        if dev.sc.batch != b {
            return None;
        }
        let xd = dev.sc.gpu.upload(x);
        let (sc, pyr) = dev.sc.forward(&xd);
        let (ps, mf, terms) = dev.terms.forward(&xd);
        Some((sc, ps, mf, DeviceState { pyr, terms }))
    }

    #[cfg(not(feature = "gpu"))]
    fn device_forward(&self, _x: &[f64], _b: usize) -> Option<(Vec<C64>, Vec<C64>, Vec<f64>, DeviceState)> {
        None
    }

    /// Map cotangents `[B, C, H, W]` on the device from those of the
    /// ScatCov, the spectra and the Minkowski values.
    #[cfg(feature = "gpu")]
    fn device_reverse(&self, st: &DeviceState, csc: &[C64], cps: &[C64], cmf: &[f64]) -> Vec<f64> {
        let dev = self.device.as_ref().unwrap().lock().unwrap();
        let g = dev.sc.reverse(&st.pyr, csc);
        dev.terms.vjp(&st.terms, cps, cmf, &g);
        dev.sc.gpu.download(&g)
    }

    #[cfg(not(feature = "gpu"))]
    fn device_reverse(&self, _st: &DeviceState, _csc: &[C64], _cps: &[C64], _cmf: &[f64]) -> Vec<f64> {
        unreachable!("no device without the gpu feature")
    }

    /// Runs the running batches of `batch` images on `gpu`: ScatCov,
    /// spectra and Minkowski functionals.
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self, gpu: std::sync::Arc<stl_gpu::Gpu>, batch: usize) -> Result<(), String> {
        let mk = (self.cfg.mode == Mode::Ensemble).then_some((&self.mk.nu[..], self.mk.eps));
        let terms = stl_gpu::planar::GpuPlanarTerms::new(gpu.clone(), self.cfg.h, self.cfg.w, self.cfg.channels, batch, self.rings.as_ref(), mk)?;
        let sc = stl_gpu::planar::GpuPlanar::new(gpu, self.engine.clone(), batch);
        self.device = Some(Mutex::new(Device { sc, terms }));
        Ok(())
    }

    /// Largest device memory in use seen so far (0 without a device).
    pub fn device_peak_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        if let Some(d) = &self.device {
            return d.lock().unwrap().sc.peak_used.load(std::sync::atomic::Ordering::Relaxed);
        }
        0
    }

    fn evaluate(&self, x: &[f64], keep: bool) -> Vec<Eval> {
        self.evaluate_with(x, keep).0
    }

    fn evaluate_with(&self, x: &[f64], keep: bool) -> (Vec<Eval>, Option<DeviceState>) {
        let (c, npix) = (self.cfg.channels, self.cfg.h * self.cfg.w);
        let b = x.len() / (c * npix);
        let device = if keep { self.device_forward(x, b) } else { None };
        let engine = self.engine.clone().with_threads(self.tile_threads(b));
        let mf = self.cfg.mode == Mode::Ensemble;
        let np = self.plan().len();
        let none = || Pyramid { maps: Vec::new(), m: Vec::new() };
        if let Some((sc, ps, mfv, st)) = device {
            let nps = self.rings.as_ref().map_or(0, |r| r.count(c));
            let k3 = self.mk.count();
            let evals = (0..b)
                .map(|i| Eval {
                    sc: sc[i * np..][..np].to_vec(),
                    pyr: none(),
                    ps: ps[i * nps..][..nps].to_vec(),
                    xs: Vec::new(),
                    mf: if mf { (0..c).map(|ch| mfv[(i * c + ch) * k3..][..k3].to_vec()).collect() } else { Vec::new() },
                    mfs: Vec::new(),
                })
                .collect();
            return (evals, Some(st));
        }
        let evals = chunked(b, 1, |i, _| {
            let img = &x[i * c * npix..][..c * npix];
            let (sc, pyr) = engine.forward(img);
            let (ps, xs) = self.rings.as_ref().map_or((Vec::new(), Vec::new()), |r| r.forward(img, c));
            let (mfv, mfs): (Vec<Vec<f64>>, Vec<MinkowskiState>) = if mf { (0..c).map(|ch| self.mk.forward(&self.fft, &img[ch * npix..][..npix])).unzip() } else { (Vec::new(), Vec::new()) };
            let pyr = if keep { pyr } else { none() };
            Eval { sc, pyr, ps, xs: if keep { xs } else { Vec::new() }, mf: mfv, mfs: if keep { mfs } else { Vec::new() } }
        });
        (evals, None)
    }

    /// References of the target batch: `var_ref`, `sqrt(mean_b S2)` of
    /// the diagonal, `sqrt(mean_b PS)` of the auto-spectra.
    fn normalisation(&mut self, evals: &[Eval]) {
        let plan = self.engine.plan().to_vec();
        let bt = evals.len() as f64;
        let mean_of = |i: usize| evals.iter().map(|e| e.sc[i]).sum::<C64>() / bt;
        let find = |f: Family, c: usize, j: usize, l: usize| {
            plan.iter().position(|q| q.family == f && q.c1 == c && q.c2 == c && (f == Family::Var || (q.j1 == Some(j) && q.l1 == Some(l)))).unwrap()
        };
        let var_ref = |c: usize| mean_of(find(Family::Var, c, 0, 0)).re;
        let r = |c: usize, j: Option<usize>, l: Option<usize>| mean_of(find(Family::S2, c, j.unwrap(), l.unwrap())).sqrt().re;
        let (w3, w4) = match self.cfg.mode {
            Mode::StlDev => (self.cfg.s3_weight, self.cfg.s4_weight),
            Mode::Ensemble => (1.0, 1.0),
        };
        self.div = plan
            .iter()
            .map(|q| match q.family {
                Family::Mean => 1.0,
                Family::Var => var_ref(q.c1),
                Family::S1 => (r(q.c1, q.j1, q.l1) * r(q.c2, q.j1, q.l1)).sqrt(),
                Family::S2 => r(q.c1, q.j1, q.l1) * r(q.c2, q.j1, q.l1),
                Family::S3 => r(q.c1, q.j1, q.l1) * r(q.c2, q.j3, q.l3),
                Family::S4 => r(q.c1, q.j1, q.l1) * r(q.c2, q.j2, q.l2),
            })
            .collect();
        self.weight = plan
            .iter()
            .map(|q| match q.family {
                Family::S3 => w3,
                Family::S4 => w4,
                _ => 1.0,
            })
            .collect();
        if let Some(rings) = &self.rings {
            let (c, nb) = (self.cfg.channels, rings.n_bins);
            let p = |ch: usize, k: usize| {
                let start = (0..ch).map(|a| c - a).sum::<usize>() * nb;
                (evals.iter().map(|e| e.ps[start + k]).sum::<C64>() / bt).sqrt().re
            };
            self.ps_div = (0..c).flat_map(|c1| (c1..c).flat_map(move |c2| (0..nb).map(move |k| (c1, c2, k)))).map(|(c1, c2, k)| p(c1, k) * p(c2, k)).collect();
        }
    }

    fn features(&self, e: &Eval) -> Vec<C64> {
        let sc = e.sc.iter().zip(&self.div).zip(&self.weight).map(|((v, d), w)| v / d * w);
        let ps = e.ps.iter().zip(&self.ps_div).map(|(v, d)| v / d);
        sc.chain(ps).collect()
    }

    /// Features `[B, n]` and Minkowski values `[B, C * 195]` of a batch.
    pub fn batch_features(&self, x: &[f64]) -> (Vec<C64>, Vec<f64>) {
        let evals = self.evaluate(x, false);
        (evals.iter().flat_map(|e| self.features(e)).collect(), evals.iter().flat_map(|e| e.mf.concat()).collect())
    }

    /// Loss and gradient for the batch `x` `[B, C, H, W]`.
    pub fn loss_and_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let t0 = Instant::now();
        let (evals, device) = self.evaluate_with(x, true);
        let t1 = Instant::now();
        let b = evals.len();
        let n = self.feature_count();
        let feats: Vec<C64> = evals.iter().flat_map(|e| self.features(e)).collect();
        let mut blocks: Vec<(&'static str, f64)> = Vec::new();
        let (loss, cot_f, cot_mf) = match self.cfg.mode {
            Mode::StlDev => {
                let bt = self.target_count as f64;
                let mut loss = 0.0;
                let mut cot = vec![C64::default(); b * n];
                for i in 0..n {
                    let mu = (0..b).map(|k| feats[k * n + i]).sum::<C64>() / b as f64;
                    let t = (0..self.target_count).map(|k| self.target[k * n + i]).sum::<C64>() / bt;
                    let r = mu - t;
                    loss += r.norm_sqr();
                    for k in 0..b {
                        cot[k * n + i] = 2.0 * r / b as f64;
                    }
                }
                (loss, cot, Vec::new())
            }
            Mode::Ensemble => self.ensemble_loss(&feats, &evals, &mut blocks),
        };
        let t2 = Instant::now();
        let (c, npix) = (self.cfg.channels, self.cfg.h * self.cfg.w);
        let ns = self.plan().len();
        let engine = self.engine.clone().with_threads(self.tile_threads(b));
        let nm = self.mk.count();
        let csc_all: Vec<C64> = (0..b).flat_map(|k| cot_f[k * n..][..ns].iter().zip(&self.div).zip(&self.weight).map(|((g, d), w)| g * w / d)).collect();
        if let Some(st) = &device {
            let cps: Vec<C64> = (0..b).flat_map(|k| cot_f[k * n + ns..][..n - ns].iter().zip(&self.ps_div).map(|(g, d)| g / d)).collect();
            let cmf = if cot_mf.is_empty() { vec![0.0; b * c * nm] } else { cot_mf.clone() };
            let g = self.device_reverse(st, &csc_all, &cps, &cmf);
            let t3 = Instant::now();
            blocks.extend([("forward", (t1 - t0).as_secs_f64()), ("losses", (t2 - t1).as_secs_f64()), ("reverse", (t3 - t2).as_secs_f64())]);
            *self.last.lock().unwrap() = blocks;
            return (loss, g);
        }
        let grads = chunked(b, 1, |k, _| {
            let e = &evals[k];
            let cf = &cot_f[k * n..][..n];
            let mut g = engine.reverse(&e.pyr, &csc_all[k * ns..][..ns]);
            if let Some(rings) = &self.rings {
                let cps: Vec<C64> = cf[ns..].iter().zip(&self.ps_div).map(|(g, d)| g / d).collect();
                g.iter_mut().zip(rings.vjp(&e.xs, &cps)).for_each(|(a, v)| *a += v);
            }
            for (ch, state) in e.mfs.iter().enumerate() {
                let gm = self.mk.vjp(&self.fft, state, &cot_mf[(k * c + ch) * nm..][..nm]);
                g[ch * npix..][..npix].iter_mut().zip(gm).for_each(|(a, v)| *a += v);
            }
            g
        });
        let t3 = Instant::now();
        blocks.extend([
            ("forward", (t1 - t0).as_secs_f64()),
            ("losses", (t2 - t1).as_secs_f64()),
            ("reverse", (t3 - t2).as_secs_f64()),
        ]);
        *self.last.lock().unwrap() = blocks;
        (loss, grads.concat())
    }

    fn ensemble_loss(&self, feats: &[C64], evals: &[Eval], blocks: &mut Vec<(&'static str, f64)>) -> (f64, Vec<C64>, Vec<f64>) {
        let b = evals.len();
        let n = self.feature_count();
        let ns = self.plan().len();
        let bt = self.target_count;
        let [wsc, wps, wmf] = self.cfg.weights;
        // SC block on the normalised coefficients.
        let take = |x: &[C64], rows: usize, r: std::ops::Range<usize>| -> Vec<C64> { (0..rows).flat_map(|k| x[k * n..][r.clone()].iter().copied()).collect() };
        let xs = take(feats, b, 0..ns);
        let ts = take(&self.target, bt, 0..ns);
        let (tm, tv) = moments(&ts, ns);
        let floor = 1e-12 * torch_median(&tv).max(f64::MIN_POSITIVE);
        let tv: Vec<f64> = tv.iter().map(|v| v.max(floor)).collect();
        let sigma: Vec<f64> = tv.iter().map(|v| v.sqrt()).collect();
        let (sc, csc) = sc_loss(&xs, &tm, &tv, &sigma, self.plan(), false).expect("SC block");
        let mut cot = vec![C64::default(); b * n];
        for k in 0..b {
            for i in 0..ns {
                cot[k * n + i] = wsc * csc[k * ns + i];
            }
        }
        let mut loss = wsc * sc.loss;
        push_block(blocks, "sc", sc);
        // Power block on the auto-spectra.
        if let Some(rings) = &self.rings {
            let (c, nb) = (self.cfg.channels, rings.n_bins);
            let auto: Vec<usize> = (0..c).flat_map(|c1| (c1..c).map(move |c2| (c1, c2))).enumerate().filter(|(_, (a, b))| a == b).flat_map(|(p, _)| (0..nb).map(move |k| ns + p * nb + k)).collect();
            let re = |x: &[C64], rows: usize| -> Vec<f64> { (0..rows).flat_map(|k| auto.iter().map(move |&i| x[k * n + i].re)).collect() };
            let tp = re(&self.target, bt);
            let (tm, tv) = moments(&tp.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>(), auto.len());
            let tm: Vec<f64> = tm.iter().map(|v| v.re).collect();
            let mask = vec![true; auto.len()];
            let (ps, cps) = power_loss_masked(&re(feats, b), &tm, &tv, &mask).expect("power block");
            for k in 0..b {
                for (q, &i) in auto.iter().enumerate() {
                    cot[k * n + i] = C64::new(wps * cps[k * auto.len() + q], 0.0);
                }
            }
            loss += wps * ps.loss;
            push_block(blocks, "power", ps);
        }
        // Minkowski block.
        let nm = self.mf_count();
        let xm: Vec<f64> = evals.iter().flat_map(|e| e.mf.concat()).collect();
        let (tm, tv) = moments(&self.target_mf.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>(), nm);
        let tm: Vec<f64> = tm.iter().map(|v| v.re).collect();
        let (mf, cmf) = mf_loss(&xm, &tm, &tv);
        loss += wmf * mf.loss;
        push_block(blocks, "mf", mf);
        (loss, cot, cmf.iter().map(|v| wmf * v).collect())
    }
}

fn push_block(blocks: &mut Vec<(&'static str, f64)>, name: &'static str, b: Block) {
    let (m, v): (&'static str, &'static str) = match name {
        "sc" => ("sc_mean", "sc_variance"),
        "power" => ("power_mean", "power_variance"),
        _ => ("mf_mean", "mf_variance"),
    };
    blocks.extend([(name, b.loss), (m, b.mean), (v, b.variance)]);
}

/// Gaussian white noise (SplitMix64 and Box-Muller), reproducible from `seed`.
pub fn white_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u = 1.0 - next();
            let v = next();
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        })
        .collect()
}

/// [`Fft2::nyquist_filter`] of every `h x w` plane of `x`.
pub fn nyquist_filter(x: &[f64], h: usize, w: usize) -> Vec<f64> {
    let fft = Fft2::new(h, w);
    x.chunks(h * w).flat_map(|p| fft.nyquist_filter(p)).collect()
}
