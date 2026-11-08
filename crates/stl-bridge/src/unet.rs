//! Native U-Net handles (`stlp_unet_*`, `stlp_train_*`): the spherical
//! U-Net of `stl_core::nn` (host) or `stl_gpu::nn` (device), its inference
//! and parameter gradients, and its training with the scattering losses of
//! Healpix-STL-IA (`stlia/train_denoise.py`, `--real --bias frozen`): the
//! network, the statistics engines, the observations and the noise draws
//! stay on one device; only coefficient vectors (values, variances,
//! cotangents) and the losses go through the host.
//!
//! Loss terms of a field (observations `d`, half missions `d1`, `d2`, the
//! 857 GHz map `T`; `u = f(d)`; frozen biases `B` estimated at a fixed `u`
//! with noise draws), each a reduced chi^2 with ergodic weights
//! (`stlia.stl_torch.DenoiseLoss`): `sum_c w_c |model_c - target_c|^2`,
//! `w_c = N_c / V_c / n` (`V_c` the spatial variance of the target's terms,
//! `N_c` its native pixels, `n` the number of coefficients; `V_c = 0`: the
//! largest weight):
//!
//! - `data`: `Phi(u) + B_data` vs `Phi(d)`;
//! - `l1`: `Phi(u, u) + B_l1` vs `Phi(d1, d2)`;
//! - `l2`: `Phi(u, u) + B_l2` vs `Phi(d, u)` (the target moves with `u`:
//!   gradient through both, weights from the variances of `(d, u)`);
//! - `l3`: `Phi(T, u) + B_l3` vs `Phi(T, d)`;
//! - `cl` (with `stlp_train_set_cl`): the binned pseudo-spectra of `u`
//!   under apodised masks against the half-mission cross spectrum
//!   (`stl_port.denoise.SpectrumLoss`, see `ClTerm`), no noise bias; on the
//!   device with the GPU backend, in the same network forward and backward
//!   as the other terms;
//!
//! `B_data = <Phi(u + n)> - Phi(u)`, `B_l1 = <Phi(u + n1, u + n2)> - Phi(u,
//! u)`, `B_l2 = <Phi(u + n, u)> - Phi(u, u)`, `B_l3 = <Phi(T, u + n)> -
//! Phi(T, u)`, the means over draws `(n, n1, n2)` of one simulation with a
//! sign, set to zero on the coefficients without noise bias (`Stats::exact`;
//! `bias_all` estimates every one).
//!
//! Joint pair (`maps` = 2, `stlp_train_set_joint`): one network maps an
//! input pair (E, B; or Q, U) to an output pair `u`; a linear front end `F`
//! (identity, `conj-eth^2` from (Q, U) to `(chi_E, chi_B)`, or the harmonic
//! E and B maps of (Q, U)) gives the two
//! fields of the terms, `F u`, whose observations and noise pools are set
//! as two fields (already in the domain of `F`).  A joint step runs the
//! network once, sums the terms of both fields (times a factor), and sends
//! the cotangent `[2, npix]` back through `F^T` and the network once.  The
//! frozen biases of both fields are estimated together from the same draws
//! (the same simulations and signs for every term and both fields).

use super::*;
use std::collections::HashMap;
use stl_core::lbfgs::{minimize, LbfgsOptions, Local};
use stl_core::nn::{std_unbiased, Adam, ParamSpec, UNet, UNetConfig};

type C64 = Complex64;

/// The two statistics engines: one channel, or the cross terms of two.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Auto,
    Cross,
}

/// A row of a batch of maps: `a[off..] + s b[off..]`.
type Row<'a, M> = (&'a M, usize, Option<(&'a M, usize, f64)>);

/// What the trainer needs from a backend.
trait Backend {
    type Map;
    type Tape;
    fn npix(&self) -> usize;
    /// Maps of the network's input and output (1 or 2).
    fn maps(&self) -> usize;
    fn upload(&self, x: &[f64]) -> Self::Map;
    fn download(&self, m: &Self::Map) -> Vec<f64>;
    /// The maps `rows` one after the other.
    fn combine(&self, rows: &[Row<Self::Map>]) -> Self::Map;
    /// `dst += s src[off..off + npix]`.
    fn add_scaled(&self, dst: &mut Self::Map, src: &Self::Map, off: usize, s: f64);
    fn zeros(&self, n: usize) -> Self::Map;
    /// `u = d - scale r(d / scale)` of one input `[maps, npix]`, one scale per map.
    fn net_forward(&self, d: &Self::Map, scales: &[f64]) -> (Self::Map, Self::Tape);
    fn net_backward(&mut self, tape: &Self::Tape, scales: &[f64], du: &Self::Map);
    /// Statistics of the batch `x` (`[batch, channels, npix]`), kept for [`stats_grad`](Self::stats_grad).
    fn stats(&mut self, kind: Kind, batch: usize, x: Self::Map, moments: bool) -> (Vec<C64>, Vec<f64>);
    /// Map cotangent of the last [`stats`](Self::stats) of this kind and batch.
    fn stats_grad(&mut self, kind: Kind, batch: usize, cot: &[C64]) -> Self::Map;
    fn params(&self) -> Vec<f32>;
    fn set_params(&mut self, w: &[f32]);
    fn grads(&self) -> Vec<f32>;
    fn zero_grad(&mut self);
    fn adam(&mut self, lr: f64);
    fn reset_adam(&mut self);
    /// Conditioning maps `[cond][npix]`.
    fn set_cond(&mut self, cond: &[Vec<f32>]);
    /// Masks of the statistics (drops the engines).
    fn set_masks(&mut self, masks: Option<Masks>);
    /// Native pixels `[K, ncoeff]` of the averages of `kind`.
    fn stats_pixels(&self, kind: Kind) -> Vec<f64>;
    /// Coefficients without noise bias of a term ([`Stats::exact`]).
    fn exact(&self, term: &str) -> Vec<bool>;
    /// Geometry of the statistics engines.
    fn geometry(&self) -> &LocalGeometry;
    /// Builds (`Some(iter)`) or drops the harmonic E/B transform of [`eb`](Self::eb).
    fn set_eb(&mut self, iter: Option<usize>);
    /// The harmonic E and B maps of the `(Q, U)` pairs `x` `[b, 2, npix]`, or
    /// with `adjoint` its transpose (`stl_core::sht::EbTransform`; on the
    /// device, `stl_gpu::GpuEb`).
    fn eb(&self, x: &Self::Map, adjoint: bool) -> Self::Map;
    /// `dst += s src` over the whole buffer.
    /// Waits for the device (the phase timers, `stl_core::prof`).
    fn sync(&self);
    /// Cumulative phase seconds of the GPU statistics engines (`engine.<kind>.<phase>`).
    fn engine_timers(&self) -> Vec<(String, f64)>;
    fn axpy(&self, dst: &mut Self::Map, src: &Self::Map, s: f64);
    /// Builds (`Some`) or drops the pseudo-spectra of the cl term: anafast
    /// (`iter = 0`) up to `lmax` of the maps times each of the weights
    /// `masks` `[K][npix]`, spin 0 or (`spin2`) the `(E, B)` of `(Q, U)` pairs.
    fn set_cl(&mut self, cfg: Option<(usize, &[Vec<f64>], bool)>);
    /// The `a_lm` (on the host) of the rows of `x` starting at `offs` (spin
    /// 2: the pairs `(Q, U)` at `off`, `off + npix`) times each mask: `[row][K]`
    /// (spin 2: `[row][K][E, B]`).
    fn cl_alms(&mut self, x: &Self::Map, offs: &[usize]) -> Vec<Vec<C64>>;
    /// `C_l` `[row][K][comp][lmax + 1]` (comp: TT, or EE and BB) of the rows
    /// as [`cl_alms`](Self::cl_alms), kept for [`cl_backward`](Self::cl_backward).
    fn cl_forward(&mut self, x: &Self::Map, offs: &[usize]) -> Vec<f64>;
    /// Map cotangent `[rows (spin 2: x 2), npix]` of `sum(cot C_l)` after the
    /// last [`cl_forward`](Self::cl_forward).
    fn cl_backward(&mut self, cot: &[f64]) -> Self::Map;
}

/// Host pseudo-spectra of the cl term ([`Backend::set_cl`]).
struct ClHost {
    sht: Sht,
    masks: Vec<Vec<f64>>,
    spin2: bool,
    rows: usize,
    alms: Vec<Vec<C64>>,
}

impl ClHost {
    fn alms(&self, x: &[f64], offs: &[usize]) -> Vec<Vec<C64>> {
        let n = self.sht.npix;
        let masked = |o: usize, w: &[f64]| -> Vec<f64> { x[o..o + n].iter().zip(w).map(|(a, b)| a * b).collect() };
        let mut out = Vec::new();
        for &o in offs {
            for w in &self.masks {
                if self.spin2 {
                    let (q, u) = (masked(o, w), masked(o + n, w));
                    let (e, b) = self.sht.analysis_spin(2, &[(&q, &u)]).pop().unwrap();
                    out.push(e);
                    out.push(b);
                } else {
                    out.push(self.sht.analysis(&masked(o, w)));
                }
            }
        }
        out
    }

    fn backward(&self, cot: &[f64]) -> Vec<f64> {
        let (n, nl, k) = (self.sht.npix, self.sht.lmax + 1, self.masks.len());
        let ca: Vec<Vec<C64>> = self.alms.iter().zip(cot.chunks(nl)).map(|(a, d)| self.sht.power_vjp(a, d)).collect();
        let parts = if self.spin2 { 2 } else { 1 };
        let mut out = vec![0.0; self.rows * parts * n];
        for (i, w) in (0..self.rows * k).map(|i| (i, &self.masks[i % k])) {
            let r = i / k;
            let maps = if self.spin2 {
                let (q, u) = self.sht.analysis_spin_vjp(2, &[(&ca[2 * i], &ca[2 * i + 1])]).pop().unwrap();
                vec![q, u]
            } else {
                vec![self.sht.analysis_vjp(&ca[i])]
            };
            for (c, g) in maps.iter().enumerate() {
                let o = &mut out[(r * parts + c) * n..][..n];
                o.iter_mut().zip(g.iter().zip(w)).for_each(|(o, (v, m))| *o += v * m);
            }
        }
        out
    }
}

/// Device pseudo-spectra of the cl term (spin 0 through `eb.sht`).
#[cfg(feature = "gpu")]
struct ClDev {
    eb: stl_gpu::GpuEb,
    masks: Vec<stl_gpu::CudaSlice<f64>>,
    spin2: bool,
    rows: usize,
    alms: Vec<stl_gpu::CudaSlice<f64>>,
}

/// Spin-0 maps per device transform batch (bounds its Legendre sums).
#[cfg(feature = "gpu")]
const CL_CHUNK: usize = 16;

/// Engine geometry shared by the backends (boxed: the engines borrow it).
struct Stats {
    geo: Box<LocalGeometry>,
    cfg: Box<EngineConfig>,
    plans: [Vec<Coefficient>; 2],
    masks: Option<Masks>,
}

impl Stats {
    fn new(nside: usize, j: usize, block_pixels: usize) -> Self {
        let geo = Box::new(LocalGeometry::new(nside as i64, 5, 4, Gauge::Cosmo, block_pixels));
        let cfg = Box::new(EngineConfig::new(j, false));
        let plans = [plan_of(j, geo.l, &cfg.j_to_dg, false), plan_of(j, geo.l, &cfg.j_to_dg, true)];
        Stats { geo, cfg, plans, masks: None }
    }

    fn plan(&self, kind: Kind) -> &[Coefficient] {
        &self.plans[kind as usize]
    }

    /// `(&'static geometry, &'static config)`: the boxes outlive the engines
    /// (declared after them in the backends).
    unsafe fn refs(&self) -> (&'static LocalGeometry, &'static EngineConfig) {
        (&*(&*self.geo as *const LocalGeometry), &*(&*self.cfg as *const EngineConfig))
    }

    /// `[K, ncoeff]`: the coefficients of `term` without noise bias, whose
    /// noise average is the noise-free statistic (the noise enters them only
    /// linearly; `unbiased()` of Healpix-STL-IA's `train_denoise.py`): the
    /// mean of `data`; S2 of `l1`, `l2`, `l3`; S3 with the modulus of `u`
    /// (`c1 = 1`) for `l2`, with the modulus of `T` (`c1 = 0`) for `l3`.
    fn exact(&self, term: &str) -> Vec<bool> {
        let kind = if term == "data" { Kind::Auto } else { Kind::Cross };
        let plan = self.plan(kind);
        let one: Vec<bool> = plan
            .iter()
            .map(|c| {
                let f = c.family.name();
                match term {
                    "data" => f == "mean",
                    "l2" => f == "S2" || (f == "S3" && c.c1 == 1),
                    "l3" => f == "S2" || (f == "S3" && c.c1 == 0),
                    _ => f == "S2",
                }
            })
            .collect();
        let k = self.masks.as_ref().map_or(1, Masks::len);
        (0..k).flat_map(|_| one.iter().cloned()).collect()
    }

    /// Native pixels `[K, ncoeff]` of the averages of `kind`.
    fn pixels(&self, kind: Kind) -> Vec<f64> {
        let plan = self.plan(kind);
        match &self.masks {
            None => plan.iter().map(|c| self.geo.pixels(c.native_dg) as f64).collect(),
            Some(m) => (0..m.len()).flat_map(|k| plan.iter().map(move |c| m.counts[k][c.native_dg] as f64)).collect(),
        }
    }
}

// -- host backend ----------------------------------------------------------------

struct CpuBack {
    engines: HashMap<(Kind, usize), LocalEngine<'static>>,
    net: UNet,
    w: Vec<f32>,
    g: Vec<f32>,
    adam: Adam,
    cond: Vec<Vec<f32>>,
    stats: Stats,
    eb: Option<stl_core::sht::EbTransform>,
    cl: Option<ClHost>,
}

/// `nside` of `npix = 12 nside^2`.
fn nside_of(npix: usize) -> usize {
    ((npix / 12) as f64).sqrt().round() as usize
}

impl Backend for CpuBack {
    type Map = Vec<f64>;
    type Tape = stl_core::nn::Tape;

    fn npix(&self) -> usize {
        self.net.pixels(0)
    }

    fn maps(&self) -> usize {
        self.net.cfg.maps
    }

    fn upload(&self, x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }

    fn download(&self, m: &Vec<f64>) -> Vec<f64> {
        m.clone()
    }

    fn combine(&self, rows: &[Row<Vec<f64>>]) -> Vec<f64> {
        let n = self.npix();
        let mut out = Vec::with_capacity(rows.len() * n);
        for &(a, ao, b) in rows {
            match b {
                Some((b, bo, s)) => out.extend(a[ao..ao + n].iter().zip(&b[bo..bo + n]).map(|(x, y)| x + s * y)),
                None => out.extend_from_slice(&a[ao..ao + n]),
            }
        }
        out
    }

    fn add_scaled(&self, dst: &mut Vec<f64>, src: &Vec<f64>, off: usize, s: f64) {
        let n = self.npix();
        dst.iter_mut().zip(&src[off..off + n]).for_each(|(a, b)| *a += s * b);
    }

    fn zeros(&self, n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    fn net_forward(&self, d: &Vec<f64>, scales: &[f64]) -> (Vec<f64>, Self::Tape) {
        let n = self.npix();
        assert_eq!(scales.len(), self.maps(), "one scale per map");
        let x: Vec<f32> = d.iter().enumerate().map(|(i, &v)| (v / scales[i / n]) as f32).collect();
        let (r, t) = self.net.forward(&self.w, &x, &self.cond);
        (d.iter().zip(&r).enumerate().map(|(i, (&a, &b))| a - scales[i / n] * b as f64).collect(), t)
    }

    fn net_backward(&mut self, tape: &Self::Tape, scales: &[f64], du: &Vec<f64>) {
        let n = self.npix();
        let dr: Vec<f32> = du.iter().enumerate().map(|(i, &v)| (-scales[i / n] * v) as f32).collect();
        self.net.backward(&self.w, tape, &dr, &mut self.g);
    }

    fn stats(&mut self, kind: Kind, batch: usize, x: Vec<f64>, moments: bool) -> (Vec<C64>, Vec<f64>) {
        let (geo, cfg) = unsafe { self.stats.refs() };
        let masks = self.stats.masks.clone();
        let channels = if kind == Kind::Cross { 2 } else { 1 };
        let e = self.engines.entry((kind, batch)).or_insert_with(|| {
            let mut e = LocalEngine::new(geo, cfg, vec![0.0; batch * channels * geo.pixels(0)], batch, channels);
            e.set_masks(masks);
            e
        });
        e.reload(&x);
        let plan = &self.stats.plans[kind as usize];
        if moments {
            e.forward_moments(plan)
        } else {
            (e.forward(plan), Vec::new())
        }
    }

    fn stats_grad(&mut self, kind: Kind, batch: usize, cot: &[C64]) -> Vec<f64> {
        let e = self.engines.get_mut(&(kind, batch)).expect("statistics before their gradient");
        let mg = e.reverse(&self.stats.plans[kind as usize], cot);
        e.map_cotangent(mg)
    }

    fn params(&self) -> Vec<f32> {
        self.w.clone()
    }

    fn set_params(&mut self, w: &[f32]) {
        self.w.copy_from_slice(w);
    }

    fn grads(&self) -> Vec<f32> {
        self.g.clone()
    }

    fn zero_grad(&mut self) {
        self.g.fill(0.0);
    }

    fn adam(&mut self, lr: f64) {
        self.adam.step(&mut self.w, &self.g, lr);
    }

    fn reset_adam(&mut self) {
        self.adam = Adam::new(self.net.size);
    }

    fn set_cond(&mut self, cond: &[Vec<f32>]) {
        self.cond = self.net.cond_levels(cond);
    }

    fn set_masks(&mut self, masks: Option<Masks>) {
        self.engines.clear();
        self.stats.masks = masks;
    }

    fn stats_pixels(&self, kind: Kind) -> Vec<f64> {
        self.stats.pixels(kind)
    }

    fn exact(&self, term: &str) -> Vec<bool> {
        self.stats.exact(term)
    }

    fn geometry(&self) -> &LocalGeometry {
        &self.stats.geo
    }

    fn set_eb(&mut self, iter: Option<usize>) {
        let ns = nside_of(self.npix());
        self.eb = iter.map(|k| stl_core::sht::EbTransform::new(ns, 3 * ns - 1, k));
    }

    fn eb(&self, x: &Vec<f64>, adjoint: bool) -> Vec<f64> {
        let t = self.eb.as_ref().expect("no E/B transform (front eb)");
        if adjoint {
            t.adjoint(x)
        } else {
            t.apply(x)
        }
    }

    fn engine_timers(&self) -> Vec<(String, f64)> {
        Vec::new()
    }

    fn sync(&self) {}

    fn axpy(&self, dst: &mut Vec<f64>, src: &Vec<f64>, s: f64) {
        assert_eq!(dst.len(), src.len());
        dst.iter_mut().zip(src).for_each(|(a, b)| *a += s * b);
    }

    fn set_cl(&mut self, cfg: Option<(usize, &[Vec<f64>], bool)>) {
        let ns = nside_of(self.npix());
        self.cl = cfg.map(|(lmax, masks, spin2)| ClHost { sht: Sht::new(ns, lmax), masks: masks.to_vec(), spin2, rows: 0, alms: Vec::new() });
    }

    fn cl_alms(&mut self, x: &Vec<f64>, offs: &[usize]) -> Vec<Vec<C64>> {
        self.cl.as_ref().expect("no cl spectra (stlp_train_set_cl)").alms(x, offs)
    }

    fn cl_forward(&mut self, x: &Vec<f64>, offs: &[usize]) -> Vec<f64> {
        let c = self.cl.as_mut().expect("no cl spectra (stlp_train_set_cl)");
        c.alms = c.alms(x, offs);
        c.rows = offs.len();
        c.alms.iter().flat_map(|a| c.sht.power(a)).collect()
    }

    fn cl_backward(&mut self, cot: &[f64]) -> Vec<f64> {
        self.cl.as_ref().expect("cl forward before its gradient").backward(cot)
    }
}

// -- device backend --------------------------------------------------------------

#[cfg(feature = "gpu")]
struct GpuBack {
    evals: HashMap<(Kind, usize), stl_gpu::GpuEvaluator<'static>>,
    net: stl_gpu::nn::GpuUNet,
    cond: Vec<stl_gpu::CudaSlice<f32>>,
    gpu: Arc<Gpu>,
    /// Options of the statistics engines (`stlp_new`'s defaults).
    opts: GpuOptions,
    stats: Stats,
    eb: Option<stl_gpu::GpuEb>,
    cl: Option<ClDev>,
}

#[cfg(feature = "gpu")]
impl GpuBack {
    fn eval(&mut self, kind: Kind, batch: usize) -> &mut stl_gpu::GpuEvaluator<'static> {
        if !self.evals.contains_key(&(kind, batch)) {
            let base = self.evals.iter().find(|((k, _), _)| *k == kind).map(|(_, e)| e.share(self.stats.plan(kind), batch));
            let ev = base.unwrap_or_else(|| {
                let (geo, cfg) = unsafe { self.stats.refs() };
                let channels = if kind == Kind::Cross { 2 } else { 1 };
                let opts = self.opts.clone();
                let mut e = GpuEngine::new(self.gpu.clone(), geo, cfg, Partition::serial(geo, cfg.levels()), batch, channels, &opts);
                e.set_masks(self.stats.masks.as_ref());
                stl_gpu::GpuEvaluator::new(e, self.stats.plan(kind))
            });
            self.evals.insert((kind, batch), ev);
        }
        self.evals.get_mut(&(kind, batch)).unwrap()
    }

    /// The device `a_lm` of the rows of `x` at `offs` times each mask (see
    /// [`Backend::cl_alms`]), packed.
    fn cl_dev_alms(&self, x: &stl_gpu::CudaSlice<f64>, offs: &[usize]) -> Vec<stl_gpu::CudaSlice<f64>> {
        let c = self.cl.as_ref().expect("no cl spectra (stlp_train_set_cl)");
        let (g, n) = (&self.gpu, self.npix());
        let masked = |o: usize, w: &stl_gpu::CudaSlice<f64>| {
            let out = g.zeros::<f64>(n);
            g.launch1("px_mul", n, &[g.at(x, o), g.a(w), n.into(), g.a(&out)]);
            out
        };
        if c.spin2 {
            let pairs: Vec<stl_gpu::harmonic::Pair> = offs.iter().flat_map(|&o| c.masks.iter().map(move |w| (masked(o, w), masked(o + n, w)))).collect();
            return c.eb.analysis_pairs(g, &pairs).into_iter().flat_map(|(e, b)| [e, b]).collect();
        }
        let maps: Vec<_> = offs.iter().flat_map(|&o| c.masks.iter().map(move |w| masked(o, w))).collect();
        maps.chunks(CL_CHUNK).flat_map(|p| c.eb.sht.analysis_many(g, &p.iter().collect::<Vec<_>>())).collect()
    }
}

#[cfg(feature = "gpu")]
impl Backend for GpuBack {
    type Map = stl_gpu::CudaSlice<f64>;
    type Tape = stl_gpu::nn::Tape;

    fn npix(&self) -> usize {
        self.net.net.pixels(0)
    }

    fn maps(&self) -> usize {
        self.net.net.cfg.maps
    }

    fn upload(&self, x: &[f64]) -> Self::Map {
        self.gpu.upload(x)
    }

    fn download(&self, m: &Self::Map) -> Vec<f64> {
        self.gpu.download(m)
    }

    fn combine(&self, rows: &[Row<Self::Map>]) -> Self::Map {
        let g = &self.gpu;
        let n = self.npix();
        let out = g.zeros::<f64>(rows.len() * n);
        for (r, &(a, ao, b)) in rows.iter().enumerate() {
            let (bp, s) = b.map_or((stl_gpu::device::NULL, 0.0), |(b, bo, s)| (g.at(b, bo), s));
            g.launch1("nn_axpy", n, &[g.at(&out, r * n), g.at(a, ao), bp, s.into(), n.into()]);
        }
        out
    }

    fn add_scaled(&self, dst: &mut Self::Map, src: &Self::Map, off: usize, s: f64) {
        let g = &self.gpu;
        let n = self.npix();
        g.launch1("nn_axpy", n, &[g.a(dst), g.a(dst), g.at(src, off), s.into(), n.into()]);
    }

    fn zeros(&self, n: usize) -> Self::Map {
        self.gpu.zeros(n)
    }

    fn net_forward(&self, d: &Self::Map, scales: &[f64]) -> (Self::Map, Self::Tape) {
        self.net.forward(d, scales, &self.cond)
    }

    fn net_backward(&mut self, tape: &Self::Tape, scales: &[f64], du: &Self::Map) {
        self.net.backward(tape, scales, du);
    }

    fn stats(&mut self, kind: Kind, batch: usize, x: Self::Map, moments: bool) -> (Vec<C64>, Vec<f64>) {
        self.eval(kind, batch).forward_device(&Serial, x, moments)
    }

    fn stats_grad(&mut self, kind: Kind, batch: usize, cot: &[C64]) -> Self::Map {
        let ev = self.evals.get_mut(&(kind, batch)).expect("statistics before their gradient");
        ev.gradient_device(&Serial, cot)
    }

    fn params(&self) -> Vec<f32> {
        self.net.params()
    }

    fn set_params(&mut self, w: &[f32]) {
        self.net.set_params(w);
    }

    fn grads(&self) -> Vec<f32> {
        self.net.grads()
    }

    fn zero_grad(&mut self) {
        self.net.zero_grad();
    }

    fn adam(&mut self, lr: f64) {
        self.net.adam_step(lr);
    }

    fn reset_adam(&mut self) {
        self.net.reset_adam();
    }

    fn set_cond(&mut self, cond: &[Vec<f32>]) {
        let nc = self.net.net.cfg.cond;
        assert_eq!(cond.len(), nc, "conditioning maps");
        if nc == 0 {
            self.cond.clear();
            return;
        }
        let n = self.npix();
        let mut pm = vec![0f32; n * nc];
        for (k, c) in cond.iter().enumerate() {
            for i in 0..n {
                pm[i * nc + k] = c[i];
            }
        }
        self.cond = self.net.cond_levels(&self.gpu.upload(&pm));
    }

    fn set_masks(&mut self, masks: Option<Masks>) {
        self.evals.clear();
        self.stats.masks = masks;
    }

    fn stats_pixels(&self, kind: Kind) -> Vec<f64> {
        self.stats.pixels(kind)
    }

    fn exact(&self, term: &str) -> Vec<bool> {
        self.stats.exact(term)
    }

    fn geometry(&self) -> &LocalGeometry {
        &self.stats.geo
    }

    fn set_eb(&mut self, iter: Option<usize>) {
        let ns = nside_of(self.npix());
        self.eb = iter.map(|k| stl_gpu::GpuEb::new(&self.gpu, ns, 3 * ns - 1, k).expect("device E/B transform"));
    }

    fn eb(&self, x: &Self::Map, adjoint: bool) -> Self::Map {
        let t = self.eb.as_ref().expect("no E/B transform (front eb)");
        if adjoint {
            t.adjoint(&self.gpu, x)
        } else {
            t.apply(&self.gpu, x)
        }
    }

    fn engine_timers(&self) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for kind in [Kind::Auto, Kind::Cross] {
            if let Some((_, ev)) = self.evals.iter().find(|((k, _), _)| *k == kind) {
                let name = if kind == Kind::Auto { "auto" } else { "cross" };
                out.extend(ev.engine().timers.iter().map(|(p, s)| (format!("engine.{name}.{p}"), *s)));
            }
        }
        out
    }

    fn sync(&self) {
        self.gpu.sync();
    }

    fn axpy(&self, dst: &mut Self::Map, src: &Self::Map, s: f64) {
        assert_eq!(dst.len(), src.len());
        let g = &self.gpu;
        g.launch1("nn_axpy", dst.len(), &[g.a(dst), g.a(dst), g.a(src), s.into(), dst.len().into()]);
    }

    fn set_cl(&mut self, cfg: Option<(usize, &[Vec<f64>], bool)>) {
        let ns = nside_of(self.npix());
        self.cl = cfg.map(|(lmax, masks, spin2)| ClDev {
            eb: stl_gpu::GpuEb::new(&self.gpu, ns, lmax, 0).expect("device cl transforms"),
            masks: masks.iter().map(|w| self.gpu.upload(w)).collect(),
            spin2,
            rows: 0,
            alms: Vec::new(),
        });
    }

    fn cl_alms(&mut self, x: &Self::Map, offs: &[usize]) -> Vec<Vec<C64>> {
        self.cl_dev_alms(x, offs).iter().map(|a| self.gpu.download(a).chunks(2).map(|v| C64::new(v[0], v[1])).collect()).collect()
    }

    fn cl_forward(&mut self, x: &Self::Map, offs: &[usize]) -> Vec<f64> {
        let alms = self.cl_dev_alms(x, offs);
        let c = self.cl.as_mut().unwrap();
        let cl: Vec<f64> = alms.iter().flat_map(|a| c.eb.sht.power(&self.gpu, a)).collect();
        c.alms = alms;
        c.rows = offs.len();
        cl
    }

    fn cl_backward(&mut self, cot: &[f64]) -> Self::Map {
        let c = self.cl.as_ref().expect("cl forward before its gradient");
        let (g, n) = (&self.gpu, self.npix());
        let (nl, k) = (c.eb.sht.lmax + 1, c.masks.len());
        assert_eq!(cot.len(), c.alms.len() * nl, "cl cotangent");
        // Omega power_vjp / mult: the map cotangent is its (spin-2) synthesis.
        let coef: Vec<_> = c.alms.iter().zip(cot.chunks(nl)).map(|(a, d)| c.eb.sht.power_vjp_coefficients(g, a, d)).collect();
        let maps: Vec<stl_gpu::CudaSlice<f64>> = if c.spin2 {
            let mut it = coef.into_iter();
            let pairs: Vec<stl_gpu::harmonic::Pair> = (0..c.alms.len() / 2).map(|_| (it.next().unwrap(), it.next().unwrap())).collect();
            c.eb.synthesis_pairs(g, &pairs).into_iter().flat_map(|(q, u)| [q, u]).collect()
        } else {
            coef.chunks(CL_CHUNK).flat_map(|p| c.eb.sht.synthesis_many(g, &p.iter().collect::<Vec<_>>())).collect()
        };
        let parts = if c.spin2 { 2 } else { 1 };
        let out = g.zeros::<f64>(c.rows * parts * n);
        // The masks are summed in mask order.
        for (i, m) in maps.iter().enumerate() {
            let (rk, part) = (i / parts, i % parts);
            let (r, kk) = (rk / k, rk % k);
            g.launch1("px_mul_add", n, &[g.at(&out, (r * parts + part) * n), g.a(m), g.a(&c.masks[kk]), n.into()]);
        }
        out
    }
}

// -- trainer ---------------------------------------------------------------------

/// Field id of the joint pair (`stlp_train_set_joint`) in the `stlp_train_*` calls.
const JOINT: u32 = u32::MAX;

/// Loss weights of the terms.
#[derive(Clone, Copy, Debug, Default)]
struct Weights {
    data: f64,
    l1: f64,
    l2: f64,
    l3: f64,
    /// The cl term (with [`ClTerm`] set; else the driver computes it).
    cl: f64,
}

impl Weights {
    fn cross(&self) -> bool {
        self.l1 != 0.0 || self.l2 != 0.0 || self.l3 != 0.0
    }

    fn scaled(&self, s: f64) -> Self {
        Weights { data: s * self.data, l1: s * self.l1, l2: s * self.l2, l3: s * self.l3, cl: s * self.cl }
    }
}

/// Frozen noise biases of a field.
#[derive(Clone, Default)]
struct Bias {
    data: Vec<C64>,
    l1: Vec<C64>,
    l2: Vec<C64>,
    l3: Vec<C64>,
}

/// Targets of a field that do not depend on `u`: values and variances.
struct Targets {
    data: Option<(Vec<C64>, Vec<f64>)>,
    l1: Option<(Vec<C64>, Vec<f64>)>,
    l3: Option<(Vec<C64>, Vec<f64>)>,
}

/// One field: observations, noise pools `[S, npix]`, scale, targets, biases.
struct Field<M> {
    d: M,
    d1: Option<M>,
    d2: Option<M>,
    noise: [Option<M>; 3],
    sims: usize,
    scale: f64,
    targets: Option<Targets>,
    bias: Option<Bias>,
}

/// The joint network (`maps` = 2): its input `[2, npix]` and scales, the
/// fields of the terms of its two outputs, and the linear front end from
/// the output to those fields (`None`: the outputs are the fields).
struct Joint<M> {
    d: M,
    scales: [f64; 2],
    fields: [u32; 2],
    front: Option<Front>,
    /// Factor on the terms of every field (0.5: their mean over the pair).
    weight: f64,
}

/// Linear front end of the joint pair, from the network output `[2, npix]`
/// to the fields of its terms (on `[b, 2, npix]` maps).
enum Front {
    /// `chi_E + i chi_B = conj-eth^2 (Q + iU)` at level 0, the scalar
    /// pyramid of `NativeEngine(spin=2)` (masks eroded as there; host).
    Eth(SpinPyramid),
    /// The harmonic E and B maps of `(Q, U)`, `stl_port.denoise.eb_maps`
    /// (healpy `map2alm(pol=True, iter)` then `alm2map`, `lmax = 3 nside -
    /// 1`), by the backend ([`Backend::eb`]: on the device with the GPU
    /// backend); full sky, the masks are not eroded.
    Eb,
}

/// The cl term (`stlp_train_set_cl`): `stl_port.denoise.SpectrumLoss` over
/// `BinnedSpectra`, the binned pseudo-spectra `C_b(m x) = sum_l B_bl C_l(m
/// x) / mean(m^2)` (anafast, `iter = 0`, under each apodised mask `m`; `B`
/// the mean over the multipoles of each bin) and the reduced chi^2 `mean
/// (C_b(m u) - C_b(m d1, m d2))^2 / V_b` against the half-mission cross
/// spectrum, `V_b = (C_b(d1) C_b(d2) + C_b(d1, d2)^2) / modes_b` (`modes_b =
/// f_sky sum (2l + 1)`, `f_sky = mean(m^2)^2 / mean(m^4)`).  Spin 0: one
/// unit per field of the terms (the fields after the front end), the
/// values summed over the fields of `f`; spin 2: one unit for the joint
/// pair, the EE and BB spectra of the network output `(Q, U)` (before the
/// front end).  Not scaled by the joint factor (the driver's sum).
struct ClTerm {
    /// Host transforms of the layout and cross spectra of the targets.
    sht: Sht,
    k: usize,
    spin2: bool,
    edges: Vec<usize>,
    /// `mean(m^2)` per mask.
    w2: Vec<f64>,
    /// `[K][nb]`.
    modes: Vec<f64>,
    /// Half missions `(d1, d2)` of the joint pair, `[2, npix]` each (spin 2).
    hm: Option<(Vec<f64>, Vec<f64>)>,
    /// `(target, variance)` `[K][comp][nb]` per unit (a field, or [`JOINT`]).
    targets: HashMap<u32, (Vec<f64>, Vec<f64>)>,
}

impl ClTerm {
    fn comps(&self) -> usize {
        if self.spin2 {
            2
        } else {
            1
        }
    }

    fn nb(&self) -> usize {
        self.edges.len() - 1
    }

    /// `[rows][K][comp][nb]` of the `C_l` `[rows][K][comp][lmax + 1]`.
    fn bin(&self, cl: &[f64]) -> Vec<f64> {
        let (nl, c) = (self.sht.lmax + 1, self.comps());
        cl.chunks(nl)
            .enumerate()
            .flat_map(|(i, row)| {
                let w2 = self.w2[(i / c) % self.k];
                self.edges.windows(2).map(move |e| row[e[0]..e[1]].iter().sum::<f64>() / (e[1] - e[0]) as f64 / w2)
            })
            .collect()
    }

    /// `C_l` cotangents of the binned cotangents (transpose of [`bin`](Self::bin)).
    fn bin_t(&self, cot: &[f64]) -> Vec<f64> {
        let (nl, c, nb) = (self.sht.lmax + 1, self.comps(), self.nb());
        let mut out = vec![0.0; cot.len() / nb * nl];
        for (i, (row, o)) in cot.chunks(nb).zip(out.chunks_mut(nl)).enumerate() {
            let w2 = self.w2[(i / c) % self.k];
            for (e, &g) in self.edges.windows(2).zip(row) {
                o[e[0]..e[1]].iter_mut().for_each(|x| *x = g / (e[1] - e[0]) as f64 / w2);
            }
        }
        out
    }

    /// Target and variance of a unit from the `a_lm` `[2][K]` (spin 2:
    /// `[2][K][E, B]`) of its half missions.
    fn target(&self, alms: &[Vec<C64>]) -> (Vec<f64>, Vec<f64>) {
        let h = alms.len() / 2;
        let (a1, a2) = (&alms[..h], &alms[h..]);
        let spectra = |x: &[Vec<C64>], y: &[Vec<C64>]| -> Vec<f64> { self.bin(&x.iter().zip(y).flat_map(|(p, q)| self.sht.cross_power(p, q)).collect::<Vec<_>>()) };
        let (c12, c11, c22) = (spectra(a1, a2), spectra(a1, a1), spectra(a2, a2));
        let (nb, c) = (self.nb(), self.comps());
        let var = (0..c12.len()).map(|i| ((c11[i] * c22[i] + c12[i] * c12[i]) / self.modes[(i / (c * nb)) * nb + i % nb]).max(1e-300)).collect();
        (c12, var)
    }

    /// `mean(r^2 / V)` of the binned spectra `cb` of a unit, and the binned
    /// cotangent `2 r / V / n` times `s`.
    fn chi2(&self, unit: u32, cb: &[f64], s: f64) -> (f64, Vec<f64>) {
        let (t, v) = &self.targets[&unit];
        let n = t.len() as f64;
        let mut loss = 0.0;
        let cot = cb
            .iter()
            .zip(t.iter().zip(v))
            .map(|(m, (a, b))| {
                let r = m - a;
                loss += r * r / b;
                s * 2.0 * r / b / n
            })
            .collect();
        (loss / n, cot)
    }
}

struct Trainer<B: Backend> {
    back: B,
    weights: Weights,
    t857: Option<B::Map>,
    fields: HashMap<u32, Field<B::Map>>,
    joint: Option<Joint<B::Map>>,
    /// Draws per statistics call of the bias estimates (`bias_batch` maps,
    /// or `3 bias_batch` pairs).
    bias_batch: usize,
    /// Estimate the bias of every coefficient (else zero where there is no
    /// noise bias, [`Stats::exact`]).
    bias_all: bool,
    cl: Option<ClTerm>,
}

/// `(w_c)` of `DenoiseLoss.weights`: `N_c / V_c / n`, the largest weight
/// where `V_c = 0`.
fn chi2_weights(var: &[f64], pixels: &[f64]) -> Vec<f64> {
    let n = var.len() as f64;
    let mut w: Vec<f64> = var.iter().zip(pixels).map(|(&v, &p)| if v > 0.0 { 1.0 / (v / p).max(1e-300) } else { 0.0 }).collect();
    let wmax = w.iter().cloned().fold(0.0, f64::max);
    for (x, &v) in w.iter_mut().zip(var) {
        if v <= 0.0 {
            *x = wmax;
        }
    }
    w.iter_mut().for_each(|x| *x /= n);
    w
}

/// `sum w |m - t|^2` and its cotangent with respect to `m`, `2 w (m - t)`.
fn chi2(m: &[C64], t: &[C64], w: &[f64]) -> (f64, Vec<C64>) {
    let mut loss = 0.0;
    let cot = m
        .iter()
        .zip(t)
        .zip(w)
        .map(|((a, b), &wc)| {
            let r = a - b;
            loss += wc * r.norm_sqr();
            r * (2.0 * wc)
        })
        .collect();
    (loss, cot)
}

fn add(a: &[C64], b: &[C64]) -> Vec<C64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `total = sum_t w_t terms[t]` of terms `[total, data, l1, l2, l3, cl]`.
fn total(w: &Weights, terms: &mut [f64; 6]) {
    terms[0] = w.data * terms[1] + w.l1 * terms[2] + w.l2 * terms[3] + w.l3 * terms[4] + w.cl * terms[5];
}

impl<B: Backend> Trainer<B> {
    fn npix(&self) -> usize {
        self.back.npix()
    }

    fn field(&self, f: u32) -> &Field<B::Map> {
        self.fields.get(&f).unwrap_or_else(|| panic!("field {f} not set"))
    }

    fn joint(&self) -> &Joint<B::Map> {
        self.joint.as_ref().expect("no joint pair (stlp_train_set_joint)")
    }

    /// The fields of the terms of `f` (`f` itself, or the joint pair) and
    /// the factor on their terms.
    fn parts(&self, f: u32) -> (Vec<u32>, f64) {
        if f == JOINT {
            let j = self.joint();
            (j.fields.to_vec(), j.weight)
        } else {
            (vec![f], 1.0)
        }
    }

    /// Simulations in the noise pools of every field of `f`.
    fn sims(&self, f: u32) -> usize {
        self.parts(f).0.iter().map(|&g| self.field(g).sims).min().unwrap()
    }

    /// The network output `u` of `f` (`[npix]`, or `[2, npix]` for the
    /// joint pair), its tape and the scales.
    fn output(&self, f: u32) -> (B::Map, B::Tape, Vec<f64>) {
        let (d, scales) = if f == JOINT {
            let j = self.joint();
            (&j.d, j.scales.to_vec())
        } else {
            assert_eq!(self.back.maps(), 1, "a network of 2 maps trains on the joint pair");
            let fd = self.field(f);
            (&fd.d, vec![fd.scale])
        };
        let (u, tape) = self.back.net_forward(d, &scales);
        (u, tape, scales)
    }

    /// The front end of `f` applied to the output `u` (`None`: identity).
    fn front(&self, f: u32, u: &B::Map) -> Option<B::Map> {
        let s = (f == JOINT).then(|| self.joint().front.as_ref()).flatten()?;
        Some(self.apply_front(s, u, false))
    }

    /// The front end `s` (or its transpose) on `[b, 2, npix]` maps.
    fn apply_front(&self, s: &Front, x: &B::Map, adjoint: bool) -> B::Map {
        match s {
            Front::Eth(p) => {
                let h = self.back.download(x);
                self.back.upload(&if adjoint { p.eth[0].adjoint(&h, p.threads) } else { p.eth[0].apply(&h, p.threads) })
            }
            Front::Eb => self.back.eb(x, adjoint),
        }
    }

    /// Transpose of [`front`](Self::front).
    fn front_adjoint(&self, f: u32, g: B::Map) -> B::Map {
        match (f == JOINT).then(|| self.joint().front.as_ref()).flatten() {
            Some(s) => self.apply_front(s, &g, true),
            None => g,
        }
    }

    /// The map cotangents of the fields, one after the other.
    fn stack(&self, mut parts: Vec<B::Map>) -> B::Map {
        if parts.len() == 1 {
            return parts.pop().unwrap();
        }
        let rows: Vec<Row<B::Map>> = parts.iter().map(|m| (m, 0, None)).collect();
        self.back.combine(&rows)
    }

    /// The targets of field `f` that do not depend on `u` (computed once).
    fn targets(&mut self, f: u32) {
        if self.field(f).targets.is_some() {
            return;
        }
        let w = self.weights;
        let fd = self.field(f);
        let data = (w.data != 0.0).then(|| self.back.combine(&[(&fd.d, 0, None)]));
        let data = data.map(|x| self.back.stats(Kind::Auto, 1, x, true));
        let fd = self.field(f);
        let cross = w.cross().then(|| {
            let t = self.t857.as_ref().expect("the l terms need the 857 map (stlp_train_set_common)");
            let (d1, d2) = (fd.d1.as_ref().expect("half mission d1"), fd.d2.as_ref().expect("half mission d2"));
            self.back.combine(&[(d1, 0, None), (d2, 0, None), (t, 0, None), (&fd.d, 0, None)])
        });
        let (l1, l3) = match cross {
            Some(x) => {
                let (v, var) = self.back.stats(Kind::Cross, 2, x, true);
                let h = v.len() / 2;
                (Some((v[..h].to_vec(), var[..h].to_vec())), Some((v[h..].to_vec(), var[h..].to_vec())))
            }
            None => (None, None),
        };
        self.fields.get_mut(&f).unwrap().targets = Some(Targets { data, l1, l3 });
    }

    /// Frozen biases of `f` (a field, or both fields of the joint pair) at
    /// the current network output, from the draws `(sim, sign)`: the same
    /// draws for every term and every field.
    fn estimate_bias(&mut self, f: u32, draws: &[(usize, f64)]) {
        assert!(!draws.is_empty(), "no draws");
        let sims = self.sims(f);
        for (s, _) in draws {
            assert!(*s < sims, "simulation {s} of {sims}");
        }
        let mut t0 = self.tstart();
        let (u, _, _) = self.output(f);
        let y = self.front(f, &u);
        self.tmark(&mut t0, "bias.unet_forward_and_front");
        let n = self.npix();
        for (k, g) in self.parts(f).0.into_iter().enumerate() {
            self.field_bias(g, y.as_ref().unwrap_or(&u), k * n, draws);
        }
        self.tmark(&mut t0, "bias.stats");
    }

    /// Frozen biases of field `f` at the output `u[uo..uo + npix]`.
    fn field_bias(&mut self, f: u32, u: &B::Map, uo: usize, draws: &[(usize, f64)]) {
        let n = self.npix();
        let w = self.weights;
        let m = draws.len() as f64;
        let fd = self.fields.get(&f).unwrap();
        let mut bias = Bias::default();
        let bb = self.bias_batch.max(1);
        if w.data != 0.0 {
            let noise = fd.noise[0].as_ref().expect("noise pool");
            let mut acc: Vec<C64> = Vec::new();
            for chunk in draws.chunks(bb) {
                let rows: Vec<Row<B::Map>> = chunk.iter().map(|&(s, sg)| (u, uo, Some((noise, s * n, sg)))).collect();
                let x = self.back.combine(&rows);
                let (v, _) = self.back.stats(Kind::Auto, chunk.len(), x, false);
                let per = v.len() / chunk.len();
                if acc.is_empty() {
                    acc = vec![C64::default(); per];
                }
                for row in v.chunks(per) {
                    acc.iter_mut().zip(row).for_each(|(a, b)| *a += b);
                }
            }
            let x = self.back.combine(&[(u, uo, None)]);
            let (phi, _) = self.back.stats(Kind::Auto, 1, x, false);
            bias.data = acc.iter().zip(&phi).map(|(a, p)| a / m - p).collect();
        }
        if w.cross() {
            let fd = self.fields.get(&f).unwrap();
            let t = self.t857.as_ref().expect("857 map");
            let [n0, n1, n2] = [&fd.noise[0], &fd.noise[1], &fd.noise[2]].map(|p| p.as_ref().expect("full and half-mission noise pools"));
            let x = self.back.combine(&[(u, uo, None), (u, uo, None), (t, 0, None), (u, uo, None)]);
            let (phi, _) = self.back.stats(Kind::Cross, 2, x, false);
            let h = phi.len() / 2;
            let (uu, tu) = (&phi[..h], &phi[h..]);
            let mut acc = [vec![C64::default(); h], vec![C64::default(); h], vec![C64::default(); h]];
            for chunk in draws.chunks(bb) {
                let mut rows: Vec<Row<B::Map>> = Vec::new();
                for &(s, sg) in chunk {
                    rows.push((u, uo, Some((n1, s * n, sg))));
                    rows.push((u, uo, Some((n2, s * n, sg))));
                }
                for &(s, sg) in chunk {
                    rows.push((u, uo, Some((n0, s * n, sg))));
                    rows.push((u, uo, None));
                }
                for &(s, sg) in chunk {
                    rows.push((t, 0, None));
                    rows.push((u, uo, Some((n0, s * n, sg))));
                }
                let k = chunk.len();
                let x = self.back.combine(&rows);
                let (v, _) = self.back.stats(Kind::Cross, 3 * k, x, false);
                for (i, row) in v.chunks(h).enumerate() {
                    acc[i / k].iter_mut().zip(row).for_each(|(a, b)| *a += b);
                }
            }
            bias.l1 = acc[0].iter().zip(uu).map(|(a, p)| a / m - p).collect();
            bias.l2 = acc[1].iter().zip(uu).map(|(a, p)| a / m - p).collect();
            bias.l3 = acc[2].iter().zip(tu).map(|(a, p)| a / m - p).collect();
        }
        if !self.bias_all {
            // No noise bias there: the estimate would only add scatter.
            for (term, b) in [("data", &mut bias.data), ("l1", &mut bias.l1), ("l2", &mut bias.l2), ("l3", &mut bias.l3)] {
                if !b.is_empty() {
                    for (v, e) in b.iter_mut().zip(self.back.exact(term)) {
                        if e {
                            *v = C64::default();
                        }
                    }
                }
            }
        }
        self.fields.get_mut(&f).unwrap().bias = Some(bias);
    }

    /// The loss of `f` (a field, or the sum over the fields of the joint
    /// pair): terms `[total, data, l1, l2, l3, cl]`; `live = Some((data draws,
    /// hm draws))` for the live-bias loss (else the frozen biases); with
    /// `backward`, the parameter cotangents of the total are added.
    fn loss_of(&mut self, f: u32, live: Option<(&[(usize, f64)], &[(usize, f64)])>, backward: bool) -> [f64; 6] {
        let mut t0 = self.tstart();
        let (u, tape, scales) = self.output(f);
        self.tmark(&mut t0, "step.unet_forward");
        let y = self.front(f, &u);
        self.tmark(&mut t0, "step.front");
        let n = self.npix();
        let (fields, wt) = self.parts(f);
        let mut terms = [0.0; 6];
        let mut du = Vec::new();
        for (k, g) in fields.into_iter().enumerate() {
            let x = y.as_ref().unwrap_or(&u);
            let (t, d) = match live {
                Some((data, hm)) => self.field_live(g, x, k * n, data, hm, wt, backward),
                None => self.field_loss(g, x, k * n, wt, backward),
            };
            terms.iter_mut().zip(&t).for_each(|(a, b)| *a += b);
            du.extend(d);
        }
        self.tmark(&mut t0, if live.is_some() { "step.stats_live" } else { "step.stats_frozen" });
        // The cl term: on the fields (spin 0) or on the output (spin 2).
        let spin2 = self.cl.as_ref().is_some_and(|c| c.spin2);
        let cl = (self.cl.is_some() && self.weights.cl != 0.0).then(|| self.cl_term(f, if spin2 { &u } else { y.as_ref().unwrap_or(&u) }, backward));
        if let Some((v, _)) = &cl {
            terms[5] = *v;
            terms[0] += self.weights.cl * v;
        }
        self.tmark(&mut t0, "step.cl");
        if backward {
            let mut du = self.stack(du);
            if let Some((_, Some(g))) = &cl {
                if !spin2 {
                    self.back.axpy(&mut du, g, 1.0);
                }
            }
            self.tmark(&mut t0, "step.cotangent_sum");
            let mut du = self.front_adjoint(f, du);
            if let Some((_, Some(g))) = &cl {
                if spin2 {
                    self.back.axpy(&mut du, g, 1.0);
                }
            }
            self.tmark(&mut t0, "step.front_adjoint");
            self.back.net_backward(&tape, &scales, &du);
            self.tmark(&mut t0, "step.unet_backward");
        }
        terms
    }

    /// A phase timer start (`stl_core::prof`; `None` when off).
    fn tstart(&self) -> Option<std::time::Instant> {
        stl_core::prof::start(|| self.back.sync())
    }

    fn tmark(&self, t0: &mut Option<std::time::Instant>, name: &str) {
        stl_core::prof::mark(t0, name, || self.back.sync());
    }

    /// The units of the cl term of `f` (see [`ClTerm`]).
    fn cl_units(&self, f: u32) -> Vec<u32> {
        if self.cl.as_ref().unwrap().spin2 {
            assert_eq!(f, JOINT, "the spin-2 cl term is a term of the joint pair");
            vec![JOINT]
        } else {
            self.parts(f).0
        }
    }

    /// Targets of the cl units of `f` (computed once from the half missions).
    fn cl_targets(&mut self, f: u32) {
        let n = self.npix();
        for g in self.cl_units(f) {
            if self.cl.as_ref().unwrap().targets.contains_key(&g) {
                continue;
            }
            let (x, offs) = if g == JOINT {
                let (h1, h2) = self.cl.as_ref().unwrap().hm.clone().expect("spin-2 cl: the half missions of the pair (stlp_train_set_cl)");
                (self.back.upload(&[h1, h2].concat()), vec![0, 2 * n])
            } else {
                let fd = self.field(g);
                let (d1, d2) = (fd.d1.as_ref().expect("cl: half mission d1"), fd.d2.as_ref().expect("cl: half mission d2"));
                (self.back.combine(&[(d1, 0, None), (d2, 0, None)]), vec![0, n])
            };
            let alms = self.back.cl_alms(&x, &offs);
            let c = self.cl.as_mut().unwrap();
            let t = c.target(&alms);
            c.targets.insert(g, t);
        }
    }

    /// The cl term of `f` at `x` (the fields `[units, npix]`, or the output
    /// pair with spin 2) and with `backward` its map cotangent at `x` (times
    /// the weight of the term).
    fn cl_term(&mut self, f: u32, x: &B::Map, backward: bool) -> (f64, Option<B::Map>) {
        self.cl_targets(f);
        let n = self.npix();
        let units = self.cl_units(f);
        let spin2 = self.cl.as_ref().unwrap().spin2;
        let offs: Vec<usize> = (0..units.len()).map(|k| k * n * if spin2 { 2 } else { 1 }).collect();
        let raw = self.back.cl_forward(x, &offs);
        let c = self.cl.as_ref().unwrap();
        let cb = c.bin(&raw);
        let per = cb.len() / units.len();
        let mut loss = 0.0;
        let mut cot = Vec::with_capacity(cb.len());
        for (g, part) in units.iter().zip(cb.chunks(per)) {
            let (l, d) = c.chi2(*g, part, self.weights.cl);
            loss += l;
            cot.extend(d);
        }
        let g = backward.then(|| {
            let dcl = c.bin_t(&cot);
            self.back.cl_backward(&dcl)
        });
        (loss, g)
    }

    /// The live-bias terms of field `f` at the output `u[uo..uo + npix]`
    /// (`train_denoise.py --bias live`), times `wt`, and with `backward`
    /// the map cotangent of their total: the models are means over the draws
    /// with the gradient through every one, `<Phi(u + n_i)>` (draws `data`),
    /// `<Phi(u + n1_i, u + n2_i)>`, `<Phi(u + n_i, u)>`, `<Phi(T, u + n_i)>`
    /// (draws `hm`, full and half missions of one simulation); targets and
    /// weights as the frozen loss.
    #[allow(clippy::too_many_arguments)]
    fn field_live(&mut self, f: u32, u: &B::Map, uo: usize, data: &[(usize, f64)], hm: &[(usize, f64)], wt: f64, backward: bool) -> ([f64; 6], Option<B::Map>) {
        self.targets(f);
        let mut tp = self.tstart();
        let w = self.weights.scaled(wt);
        let n = self.npix();
        let mut du = self.back.zeros(n);
        let mut terms = [0.0; 6];
        let targets = self.field(f).targets.as_ref().unwrap();
        let (data_t, l1_t, l3_t) = (targets.data.clone(), targets.l1.clone(), targets.l3.clone());
        let mean = |v: &[C64], k: usize, at: usize, h: usize| -> Vec<C64> {
            let mut m = vec![C64::default(); h];
            for i in 0..k {
                m.iter_mut().zip(&v[(at + i) * h..(at + i + 1) * h]).for_each(|(a, b)| *a += b);
            }
            m.iter_mut().for_each(|a| *a /= k as f64);
            m
        };
        if w.data != 0.0 {
            assert!(!data.is_empty(), "live data term: no draws");
            let fd = self.fields.get(&f).unwrap();
            let noise = fd.noise[0].as_ref().expect("noise pool");
            let rows: Vec<Row<B::Map>> = data.iter().map(|&(s, sg)| (u, uo, Some((noise, s * n, sg)))).collect();
            let x = self.back.combine(&rows);
            let k = data.len();
            let (v, _) = self.back.stats(Kind::Auto, k, x, false);
            self.tmark(&mut tp, "stats.live.data.forward");
            let h = v.len() / k;
            let (t, tv) = data_t.unwrap();
            let wc = chi2_weights(&tv, &self.back.stats_pixels(Kind::Auto));
            let (l, cot) = chi2(&mean(&v, k, 0, h), &t, &wc);
            terms[1] = l;
            if backward {
                let one: Vec<C64> = cot.iter().map(|c| c * (w.data / k as f64)).collect();
                let all: Vec<C64> = (0..k).flat_map(|_| one.iter().cloned()).collect();
                let g = self.back.stats_grad(Kind::Auto, k, &all);
            self.tmark(&mut tp, "stats.live.data.vjp");
                for i in 0..k {
                    self.back.add_scaled(&mut du, &g, i * n, 1.0);
                }
            }
        }
        if w.cross() {
            assert!(!hm.is_empty(), "live l terms: no draws");
            let k = hm.len();
            let fd = self.fields.get(&f).unwrap();
            let t857 = self.t857.as_ref().expect("857 map");
            let [n0, n1, n2] = [&fd.noise[0], &fd.noise[1], &fd.noise[2]].map(|p| p.as_ref().expect("full and half-mission noise pools"));
            let mut rows: Vec<Row<B::Map>> = vec![(&fd.d, 0, None), (u, uo, None)];
            for &(s, sg) in hm {
                rows.push((u, uo, Some((n1, s * n, sg))));
                rows.push((u, uo, Some((n2, s * n, sg))));
            }
            for &(s, sg) in hm {
                rows.push((u, uo, Some((n0, s * n, sg))));
                rows.push((u, uo, None));
            }
            for &(s, sg) in hm {
                rows.push((t857, 0, None));
                rows.push((u, uo, Some((n0, s * n, sg))));
            }
            let x = self.back.combine(&rows);
            let batch = 1 + 3 * k;
            let (v, var) = self.back.stats(Kind::Cross, batch, x, true);
            self.tmark(&mut tp, "stats.live.cross.forward");
            let h = v.len() / batch;
            let pix = self.back.stats_pixels(Kind::Cross);
            let (t1, v1) = l1_t.unwrap();
            let (t3, v3) = l3_t.unwrap();
            let du_t = &v[..h];
            let (a1, c1) = chi2(&mean(&v, k, 1, h), &t1, &chi2_weights(&v1, &pix));
            let (a2, c2) = chi2(&mean(&v, k, 1 + k, h), du_t, &chi2_weights(&var[..h], &pix));
            let (a3, c3) = chi2(&mean(&v, k, 1 + 2 * k, h), &t3, &chi2_weights(&v3, &pix));
            terms[2] = a1;
            terms[3] = a2;
            terms[4] = a3;
            if backward {
                let kf = k as f64;
                let mut cot: Vec<C64> = c2.iter().map(|c| -c * w.l2).collect();
                for (c, s) in [(&c1, w.l1), (&c2, w.l2), (&c3, w.l3)] {
                    for _ in 0..k {
                        cot.extend(c.iter().map(|x| x * (s / kf)));
                    }
                }
                let g = self.back.stats_grad(Kind::Cross, batch, &cot);
            self.tmark(&mut tp, "stats.live.cross.vjp");
                // [pair, channel, npix]: u is channel 1 of (d, u), both channels of the l1
                // and l2 pairs, channel 1 of the l3 pairs.
                self.back.add_scaled(&mut du, &g, n, 1.0);
                for pair in 1..1 + 2 * k {
                    self.back.add_scaled(&mut du, &g, 2 * pair * n, 1.0);
                    self.back.add_scaled(&mut du, &g, (2 * pair + 1) * n, 1.0);
                }
                for pair in 1 + 2 * k..batch {
                    self.back.add_scaled(&mut du, &g, (2 * pair + 1) * n, 1.0);
                }
            }
        }
        terms[1..].iter_mut().for_each(|t| *t *= wt);
        total(&self.weights, &mut terms);
        (terms, backward.then_some(du))
    }

    /// The frozen-bias terms of field `f` at the output `u[uo..uo + npix]`,
    /// times `wt`, and with `backward` the map cotangent of their total.
    fn field_loss(&mut self, f: u32, u: &B::Map, uo: usize, wt: f64, backward: bool) -> ([f64; 6], Option<B::Map>) {
        self.targets(f);
        let mut tp = self.tstart();
        let w = self.weights.scaled(wt);
        let n = self.npix();
        let mut du = self.back.zeros(n);
        let mut terms = [0.0; 6];
        let fd = self.fields.get(&f).unwrap();
        let bias = fd.bias.clone().expect("frozen biases (stlp_train_bias) before the loss");
        let targets = fd.targets.as_ref().unwrap();
        let data_t = targets.data.clone();
        let (l1_t, l3_t) = (targets.l1.clone(), targets.l3.clone());
        if w.data != 0.0 {
            let (t, tv) = data_t.unwrap();
            let x = self.back.combine(&[(u, uo, None)]);
            let (v, _) = self.back.stats(Kind::Auto, 1, x, false);
            self.tmark(&mut tp, "stats.frozen.data.forward");
            let wc = chi2_weights(&tv, &self.back.stats_pixels(Kind::Auto));
            let (l, cot) = chi2(&add(&v, &bias.data), &t, &wc);
            terms[1] = l;
            if backward {
                let cot: Vec<C64> = cot.iter().map(|c| c * w.data).collect();
                let g = self.back.stats_grad(Kind::Auto, 1, &cot);
            self.tmark(&mut tp, "stats.frozen.data.vjp");
                self.back.add_scaled(&mut du, &g, 0, 1.0);
            }
        }
        if w.cross() {
            let fd = self.fields.get(&f).unwrap();
            let t857 = self.t857.as_ref().unwrap();
            let x = self.back.combine(&[(&fd.d, 0, None), (u, uo, None), (u, uo, None), (u, uo, None), (t857, 0, None), (u, uo, None)]);
            let (v, var) = self.back.stats(Kind::Cross, 3, x, true);
            self.tmark(&mut tp, "stats.frozen.cross.forward");
            let h = v.len() / 3;
            let pix = self.back.stats_pixels(Kind::Cross);
            let (du_t, uu, tu) = (&v[..h], &v[h..2 * h], &v[2 * h..]);
            let (t1, v1) = l1_t.unwrap();
            let (t3, v3) = l3_t.unwrap();
            let (a1, c1) = chi2(&add(uu, &bias.l1), &t1, &chi2_weights(&v1, &pix));
            let (a2, c2) = chi2(&add(uu, &bias.l2), du_t, &chi2_weights(&var[..h], &pix));
            let (a3, c3) = chi2(&add(tu, &bias.l3), &t3, &chi2_weights(&v3, &pix));
            terms[2] = a1;
            terms[3] = a2;
            terms[4] = a3;
            if backward {
                let mut cot = Vec::with_capacity(3 * h);
                cot.extend(c2.iter().map(|c| -c * w.l2));
                cot.extend(c1.iter().zip(&c2).map(|(a, b)| a * w.l1 + b * w.l2));
                cot.extend(c3.iter().map(|c| c * w.l3));
                let g = self.back.stats_grad(Kind::Cross, 3, &cot);
            self.tmark(&mut tp, "stats.frozen.cross.vjp");
                // [pair, channel, npix]: u is channel 1 of (d, u), both of (u, u), channel 1 of (T, u).
                for (pair, ch) in [(0, 1), (1, 0), (1, 1), (2, 1)] {
                    self.back.add_scaled(&mut du, &g, (2 * pair + ch) * n, 1.0);
                }
            }
        }
        terms[1..].iter_mut().for_each(|t| *t *= wt);
        total(&self.weights, &mut terms);
        (terms, backward.then_some(du))
    }

    /// The frozen-bias loss of `f` (see [`loss_of`](Self::loss_of)).
    fn loss(&mut self, f: u32, backward: bool) -> [f64; 6] {
        self.loss_of(f, None, backward)
    }

    /// The live-bias loss of `f` (see [`loss_of`](Self::loss_of)).
    fn loss_live(&mut self, f: u32, data: &[(usize, f64)], hm: &[(usize, f64)], backward: bool) -> [f64; 6] {
        self.loss_of(f, Some((data, hm)), backward)
    }
}


// -- C ABI -------------------------------------------------------------------------

enum AnyTrainer {
    Cpu(Trainer<CpuBack>),
    #[cfg(feature = "gpu")]
    Gpu(Trainer<GpuBack>),
}

/// `$body` with `$t` the trainer of either backend.
macro_rules! with {
    ($h:expr, $t:ident => $body:expr) => {
        match &mut $h.inner {
            AnyTrainer::Cpu($t) => $body,
            #[cfg(feature = "gpu")]
            AnyTrainer::Gpu($t) => $body,
        }
    };
}

/// A native U-Net and, for training, its statistics engines, observations
/// and noise pools (see the module documentation).
pub struct UNetHandle {
    inner: AnyTrainer,
    specs: Vec<ParamSpec>,
    size: usize,
    npix: usize,
    cond: usize,
    backend: &'static str,
    config: Value,
    /// Levels of the statistics engines (for the masks).
    levels: usize,
    /// Maps of the network.
    maps: usize,
    /// Indicators `[K][npix]` of the statistics masks (empty: the whole sky).
    mask_weights: Vec<Vec<f64>>,
}

/// The masks of the statistics from the stored indicators; with the
/// spin-2 front end, eroded as `NativeEngine(spin=2)`'s (`stlp_set_masks`):
/// the pixels whose `chi` and decimations only read pixels of the mask.
/// Drops the targets and biases.
fn install_masks(h: &mut UNetHandle) {
    let nside = h.config["nside"].as_u64().unwrap() as i64;
    let levels = h.levels;
    let w = h.mask_weights.clone();
    with!(h, t => {
        let masks = (!w.is_empty()).then(|| match t.joint.as_ref().and_then(|j| j.front.as_ref()) {
            Some(Front::Eth(s)) => Masks::from_levels(w.iter().map(|m| keep_nonempty(scalar_pyramid_valid(t.back.geometry(), levels, s.valid_rows(m).swap_remove(0)))).collect()),
            _ => Masks::from_weights(nside, levels, w),
        });
        t.back.set_masks(masks);
        t.fields.values_mut().for_each(|f| {
            f.targets = None;
            f.bias = None;
        });
    });
}

/// New U-Net handle from a JSON configuration: network `nside`, `channels`
/// (list, one per level), `ksz` (3), `gauge` (`phi`), `slope` (0.2), `cond`
/// (0), `maps` (1, or 2: a pair of maps in and out, `UNet(maps=2)`), `scale`
/// (null: the standard deviation of the input), `seed` (0, of
/// torch's initialisation scheme); `backend` (`cpu` | `gpu`), `device` (0);
/// training: `j` (scales, `log2(nside) - 1`), `loss` (`{"data", "l1", "l2",
/// "l3"}` weights, 0 by default), `bias_batch` (5, draws per statistics
/// call of a bias estimate: 5 maps, 15 pairs, as `train_denoise.py` with
/// `B = 4`), `bias_all` (false: the frozen biases are zero on the
/// coefficients without noise bias, as `train_denoise.py` without
/// `--bias-all`), `block_pixels` (4096); GPU engines: `cache`
/// (`transposition`), `cache_bytes` (1 GiB), `threads` (1), `conv`,
/// `fields` (as `stlp_new`).  Null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_new(config: *const c_char) -> *mut UNetHandle {
    guard(std::ptr::null_mut(), || {
        let v = parse(config);
        let nside = integer(&v, "nside", 64);
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let channels: Vec<usize> = v["channels"].as_array().expect("channels: one count per level").iter().map(|c| c.as_u64().expect("channel count") as usize).collect();
        let gauge = Gauge::parse(v["gauge"].as_str().unwrap_or("phi")).expect("gauge: phi or cosmo");
        let cfg = UNetConfig {
            nside,
            channels,
            ksz: integer(&v, "ksz", 3),
            gauge,
            slope: real(&v, "slope", 0.2) as f32,
            cond: integer(&v, "cond", 0),
            maps: integer(&v, "maps", 1),
            scale: v["scale"].as_f64(),
        };
        let net = UNet::new(cfg);
        let w = net.init(integer(&v, "seed", 0) as u64);
        let specs = net.params.clone();
        let (size, npix, cond, maps) = (net.size, net.pixels(0), net.cfg.cond, net.cfg.maps);
        let j = integer(&v, "j", (nside.ilog2() as usize).saturating_sub(1).max(1));
        assert!(j > 0 && j <= nside.ilog2() as usize + 1, "invalid number of scales");
        let stats = Stats::new(nside, j, integer(&v, "block_pixels", 4096));
        let levels = stats.cfg.levels();
        let l = &v["loss"];
        let weights = Weights { data: real(l, "data", 0.0), l1: real(l, "l1", 0.0), l2: real(l, "l2", 0.0), l3: real(l, "l3", 0.0), cl: real(l, "cl", 0.0) };
        let bias_batch = integer(&v, "bias_batch", 5);
        let bias_all = v["bias_all"].as_bool().unwrap_or(false);
        let backend = v["backend"].as_str().unwrap_or("cpu");
        let inner = match backend {
            "cpu" => AnyTrainer::Cpu(Trainer {
                back: CpuBack { engines: HashMap::new(), adam: Adam::new(size), g: vec![0.0; size], w, cond: vec![Vec::new(); net.stencils.len()], net, stats, eb: None, cl: None },
                weights,
                t857: None,
                fields: HashMap::new(),
                joint: None,
                bias_batch,
                bias_all,
                cl: None,
            }),
            #[cfg(feature = "gpu")]
            "gpu" => {
                let gpu = Arc::new(Gpu::new(integer(&v, "device", 0)).expect("CUDA initialisation"));
                let dev = stl_gpu::nn::GpuUNet::new(gpu.clone(), net, &w).expect("device network");
                let mut opts = GpuOptions {
                    block_rows: stats.geo.block_pixels,
                    cache: GeometryCache::parse(v["cache"].as_str().unwrap_or("transposition")).expect("geometry cache mode"),
                    cache_bytes: integer(&v, "cache_bytes", 1 << 30),
                    threads: integer(&v, "threads", 1),
                    ..GpuOptions::default()
                };
                if let Some(s) = v["conv"].as_str() {
                    opts.conv = ConvForm::parse(s).expect("conv: dia or stencils");
                }
                if let Some(s) = v["fields"].as_str() {
                    opts.fields = FieldStore::parse(s).expect("fields: auto, keep or recompute");
                }
                AnyTrainer::Gpu(Trainer { back: GpuBack { evals: HashMap::new(), net: dev, cond: Vec::new(), gpu, opts, stats, eb: None, cl: None }, weights, t857: None, fields: HashMap::new(), joint: None, bias_batch, bias_all, cl: None })
            }
            #[cfg(not(feature = "gpu"))]
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        let backend = if backend == "gpu" { "gpu" } else { "cpu" };
        Box::into_raw(Box::new(UNetHandle { inner, specs, size, npix, cond, backend, config: v, levels, maps, mask_weights: Vec::new() }))
    })
}

/// Frees a U-Net handle (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_free(p: *mut UNetHandle) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// JSON: version, configuration, backend, parameter layout (`params`: name,
/// shape, offset, len, in `state_dict` order), sizes, fields set.  Free
/// with `stlp_string_free`.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_info(p: *mut UNetHandle) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &mut *p;
        let params: Vec<Value> = h.specs.iter().map(|s| json!({"name": s.name, "shape": s.shape, "offset": s.offset, "len": s.len})).collect();
        let (fields, ncoeff): (Vec<u32>, [usize; 2]) = with!(h, t => {
            let mut f: Vec<u32> = t.fields.keys().cloned().collect();
            f.sort();
            (f, [t.back.stats_pixels(Kind::Auto).len(), t.back.stats_pixels(Kind::Cross).len()])
        });
        let info = json!({"version": VERSION, "upstream_revision": env!("STL_BRIDGE_REVISION"), "config": h.config,
            "backend": h.backend, "size": h.size, "npix": h.npix, "cond": h.cond, "maps": h.maps, "params": params, "fields": fields,
            "coefficients": {"auto": ncoeff[0], "cross": ncoeff[1]}, "dtype": "float32 network, float64 maps and statistics"});
        CString::new(info.to_string()).unwrap().into_raw()
    })
}

/// Parameters to `out` (`size` floats).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_get_params(p: *mut UNetHandle, out: *mut f32) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let w = with!(h, t => t.back.params());
        std::slice::from_raw_parts_mut(out, h.size).copy_from_slice(&w);
        0
    })
}

/// Replaces the parameters (`n = size` floats).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_set_params(p: *mut UNetHandle, w: *const f32, n: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(n, h.size, "parameter count");
        let w = std::slice::from_raw_parts(w, n);
        assert!(w.iter().all(|x| x.is_finite()), "non-finite parameters");
        with!(h, t => t.back.set_params(w));
        0
    })
}

/// Accumulated parameter cotangents to `out` (`size` floats).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_get_grads(p: *mut UNetHandle, out: *mut f32) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let g = with!(h, t => t.back.grads());
        std::slice::from_raw_parts_mut(out, h.size).copy_from_slice(&g);
        0
    })
}

/// Zeroes the parameter cotangents.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_zero_grad(p: *mut UNetHandle) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        with!(h, t => t.back.zero_grad());
        0
    })
}

/// One Adam step of learning rate `lr` from the accumulated cotangents
/// (torch's `Adam`); `lr < 0`: resets the Adam state instead.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_adam(p: *mut UNetHandle, lr: f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        if lr < 0.0 {
            with!(h, t => t.back.reset_adam());
        } else {
            with!(h, t => t.back.adam(lr));
        }
        0
    })
}

/// Conditioning maps (`n = cond * 12 nside^2` floats, map after map, NESTED).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_set_cond(p: *mut UNetHandle, cond: *const f32, n: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(n, h.cond * h.npix, "conditioning maps");
        let c: Vec<Vec<f32>> = if n > 0 { std::slice::from_raw_parts(cond, n).chunks(h.npix).map(<[f32]>::to_vec).collect() } else { Vec::new() };
        with!(h, t => t.back.set_cond(&c));
        0
    })
}

/// Standard deviation (unbiased) of map `j` of every input of `d` `[b,
/// maps, n]`.
fn map_std(d: &[f64], maps: usize, n: usize, j: usize) -> f64 {
    let x: Vec<f64> = d.chunks(maps * n).flat_map(|r| r[j * n..(j + 1) * n].iter().cloned()).collect();
    std_unbiased(&x)
}

/// `u = d - scale r(d / scale)` of the batch `d` (`b` inputs of `maps`
/// NESTED maps) to `out`; `scales` (`maps` values, or null: the configured
/// scale if any, else the standard deviation, unbiased, of each map over
/// the batch).  With `cot` (non-null, shaped as `d`), the parameter
/// cotangents of `sum(cot u)` are added.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_apply(p: *mut UNetHandle, d: *const f64, b: usize, scales: *const f64, cot: *const f64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let (n, m) = (h.npix, h.maps);
        let len = m * n;
        let d = std::slice::from_raw_parts(d, b * len);
        let fixed = h.config["scale"].as_f64();
        let scales: Vec<f64> = if scales.is_null() {
            (0..m).map(|j| fixed.unwrap_or_else(|| map_std(d, m, n, j))).collect()
        } else {
            std::slice::from_raw_parts(scales, m).to_vec()
        };
        assert!(scales.iter().all(|s| *s > 0.0 && s.is_finite()), "scales must be positive");
        let cot = (!cot.is_null()).then(|| std::slice::from_raw_parts(cot, b * len));
        let o = std::slice::from_raw_parts_mut(out, b * len);
        with!(h, t => {
            for i in 0..b {
                let di = t.back.upload(&d[i * len..(i + 1) * len]);
                let (u, tape) = t.back.net_forward(&di, &scales);
                if let Some(c) = cot {
                    let ci = t.back.upload(&c[i * len..(i + 1) * len]);
                    t.back.net_backward(&tape, &scales, &ci);
                }
                o[i * len..(i + 1) * len].copy_from_slice(&t.back.download(&u));
            }
        });
        0
    })
}

/// Statistics averaged over the `k` masks `weights` (`[k, 12 nside^2]`
/// indicators, as `stlp_set_masks`; eroded with the spin-2 front end of
/// the joint pair), or the whole sky (`k = 0`); drops the targets and
/// biases.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_set_masks(p: *mut UNetHandle, weights: *const f64, k: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        h.mask_weights = if k > 0 { std::slice::from_raw_parts(weights, k * n).chunks(n).map(<[f64]>::to_vec).collect() } else { Vec::new() };
        install_masks(h);
        0
    })
}

/// The 857 GHz map `t` (`12 nside^2`, NESTED) of the `l` terms.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_set_common(p: *mut UNetHandle, t857: *const f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        let x = std::slice::from_raw_parts(t857, n);
        with!(h, t => {
            t.t857 = Some(t.back.upload(x));
            t.fields.values_mut().for_each(|f| f.targets = None);
        });
        0
    })
}

/// Field `field`: the observation `d`, the half missions `d1`, `d2` and the
/// noise pools `n0`, `n1`, `n2` (`[sims, 12 nside^2]`: full mission and
/// half missions of the same simulations; `d1`, `d2`, `n1`, `n2` may be
/// null without the `l` terms), uploaded once; `scale <= 0`: the standard
/// deviation of `d`.  0, or -1 on failure.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stlp_train_set_field(
    p: *mut UNetHandle,
    field: u32,
    d: *const f64,
    d1: *const f64,
    d2: *const f64,
    n0: *const f64,
    n1: *const f64,
    n2: *const f64,
    sims: usize,
    scale: f64,
) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        let map = |q: *const f64, len: usize| (!q.is_null()).then(|| std::slice::from_raw_parts(q, len));
        let dv = map(d, n).expect("observation d");
        let scale = if scale > 0.0 { scale } else { h.config["scale"].as_f64().unwrap_or_else(|| std_unbiased(dv)) };
        let parts = [map(d1, n), map(d2, n)];
        let pools = [map(n0, sims * n), map(n1, sims * n), map(n2, sims * n)];
        for x in parts.iter().chain(pools.iter()).flatten().chain(std::iter::once(&dv)) {
            assert!(x.iter().all(|v| v.is_finite()), "non-finite observation or noise");
        }
        with!(h, t => {
            let up = |x: Option<&[f64]>| x.map(|x| t.back.upload(x));
            let fd = Field {
                d: t.back.upload(dv),
                d1: up(parts[0]),
                d2: up(parts[1]),
                noise: [up(pools[0]), up(pools[1]), up(pools[2])],
                sims,
                scale,
                targets: None,
                bias: None,
            };
            t.fields.insert(field, fd);
            if let Some(c) = t.cl.as_mut() {
                c.targets.remove(&field);
            }
        });
        0
    })
}

/// The joint pair of a network of 2 maps (see the module documentation):
/// the network input `d` `[2, 12 nside^2]`, its `scales` (2 values, or
/// null: the configured scale if any, else the standard deviation of each
/// map), the fields `f0`, `f1` of the terms of the two outputs (set by
/// `stlp_train_set_field` in the domain of the front end, with the
/// simulations of the noise pools in the same rows), the front end `front`
/// (`"none"` or null: identity; `"eth"`: `conj-eth^2` of `(Q, U)` at level
/// 0 with a `spin_kernel` stencil (5), the masks then eroded as
/// `NativeEngine(spin=2)`'s; `"eb"` or `"eb:K"`: the harmonic E and B maps
/// of `(Q, U)`, healpy `map2alm(pol=True, iter=K)` (K = 3) then `alm2map`,
/// `lmax = 3 nside - 1`, as `stl_port.denoise.eb_maps`, masks unchanged)
/// and the factor `weight` on the terms of each
/// field.  The `stlp_train_*` calls on field `0xFFFFFFFF` then train the
/// pair.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_set_joint(p: *mut UNetHandle, d: *const f64, scales: *const f64, f0: u32, f1: u32, front: *const c_char, weight: f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(h.maps, 2, "the joint pair needs a network of 2 maps");
        assert!(f0 != f1 && f0 != JOINT && f1 != JOINT, "two distinct fields");
        assert!(weight > 0.0 && weight.is_finite(), "positive weight");
        let n = h.npix;
        let dv = std::slice::from_raw_parts(d, 2 * n);
        assert!(dv.iter().all(|v| v.is_finite()), "non-finite input");
        let fixed = h.config["scale"].as_f64();
        let sc: Vec<f64> = if scales.is_null() { (0..2).map(|j| fixed.unwrap_or_else(|| map_std(dv, 2, n, j))).collect() } else { std::slice::from_raw_parts(scales, 2).to_vec() };
        assert!(sc.iter().all(|s| *s > 0.0 && s.is_finite()), "scales must be positive");
        let kind = if front.is_null() { "none".to_string() } else { CStr::from_ptr(front).to_str().expect("front: UTF-8").to_string() };
        let nside = h.config["nside"].as_u64().unwrap() as i64;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let (spin, eb) = match kind.as_str() {
            "none" => (None, None),
            "eth" => (Some(Front::Eth(SpinPyramid::new(nside, 1, integer(&h.config, "spin_kernel", 5), threads))), None),
            k if k == "eb" || k.starts_with("eb:") => {
                let iter: usize = k.strip_prefix("eb:").map_or(3, |v| v.parse().expect("front eb:K, K the iterations"));
                (Some(Front::Eb), Some(iter))
            }
            other => panic!("front must be none, eth, eb or eb:K, not {other:?}"),
        };
        with!(h, t => {
            t.back.set_eb(eb);
            t.joint = Some(Joint { d: t.back.upload(dv), scales: [sc[0], sc[1]], fields: [f0, f1], front: spin, weight });
        });
        install_masks(h);
        0
    })
}

/// The cl term of the loss (see `ClTerm`, weight `loss.cl`): JSON `config`
/// `lmax` (`2 nside - 1`), `edges` (the multipole bin edges, `[lmin,
/// lmin + width, ..., lmax + 1]`), `spin` (0: the scalar spectra of the
/// fields of the terms, targets from their half missions; 2: the EE and BB
/// spectra of the joint pair's output `(Q, U)`, targets from the half
/// missions `d1`, `d2` `[2, 12 nside^2]` given here); the `k` apodised mask
/// weights `masks` `[k, 12 nside^2]`; `k = 0` drops the term (the driver
/// may then compute it).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_set_cl(p: *mut UNetHandle, config: *const c_char, masks: *const f64, k: usize, d1: *const f64, d2: *const f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        let nside = h.config["nside"].as_u64().unwrap() as usize;
        if k == 0 {
            with!(h, t => {
                t.cl = None;
                t.back.set_cl(None);
            });
            return 0;
        }
        let v = parse(config);
        let lmax = integer(&v, "lmax", 2 * nside - 1);
        assert!(lmax < 4 * nside, "lmax must be below 4 nside");
        let edges: Vec<usize> = v["edges"].as_array().expect("edges: the bin edges").iter().map(|e| e.as_u64().expect("bin edge") as usize).collect();
        assert!(edges.len() >= 2 && edges.windows(2).all(|e| e[0] < e[1]) && *edges.last().unwrap() <= lmax + 1, "edges: increasing, at most lmax + 1");
        let spin = integer(&v, "spin", 0);
        assert!(spin == 0 || spin == 2, "spin 0 or 2");
        let w: Vec<Vec<f64>> = std::slice::from_raw_parts(masks, k * n).chunks(n).map(<[f64]>::to_vec).collect();
        assert!(w.iter().flatten().all(|x| x.is_finite()), "mask weights must be finite");
        let hm = (spin == 2).then(|| {
            assert!(!d1.is_null() && !d2.is_null(), "spin-2 cl: the half missions d1, d2 of the pair");
            (std::slice::from_raw_parts(d1, 2 * n).to_vec(), std::slice::from_raw_parts(d2, 2 * n).to_vec())
        });
        let w2: Vec<f64> = w.iter().map(|m| m.iter().map(|x| x * x).sum::<f64>() / n as f64).collect();
        let w4: Vec<f64> = w.iter().map(|m| m.iter().map(|x| x.powi(4)).sum::<f64>() / n as f64).collect();
        let modes: Vec<f64> = (0..k)
            .flat_map(|i| {
                let fsky = w2[i] * w2[i] / w4[i];
                edges.windows(2).map(move |e| fsky * (e[0]..e[1]).map(|l| (2 * l + 1) as f64).sum::<f64>()).collect::<Vec<_>>()
            })
            .collect();
        with!(h, t => {
            t.back.set_cl(Some((lmax, &w, spin == 2)));
            t.cl = Some(ClTerm { sht: Sht::new(nside, lmax), k, spin2: spin == 2, edges: edges.clone(), w2: w2.clone(), modes: modes.clone(), hm: hm.clone(), targets: HashMap::new() });
        });
        0
    })
}

/// The front end of the joint pair (or its transpose with `adjoint`) on
/// `b` pairs `x` `[b, 2, 12 nside^2]` to `out` (same shape): to bring
/// observations and noise pools to the domain of the fields.  0, or -1 on
/// failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_front(p: *mut UNetHandle, x: *const f64, b: usize, adjoint: bool, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let len = b * 2 * h.npix;
        let x = std::slice::from_raw_parts(x, len);
        let o = std::slice::from_raw_parts_mut(out, len);
        with!(h, t => {
            match t.joint.as_ref().and_then(|j| j.front.as_ref()) {
                Some(s) => o.copy_from_slice(&t.back.download(&t.apply_front(s, &t.back.upload(x), adjoint))),
                None => o.copy_from_slice(x),
            }
        });
        0
    })
}

/// Frozen biases of `field` (`0xFFFFFFFF`: both fields of the joint pair,
/// from the same draws) at the current network output, from the `m` draws
/// `(sims[i], signs[i])` of its noise pools.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_bias(p: *mut UNetHandle, field: u32, sims: *const u64, signs: *const f64, m: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let s = std::slice::from_raw_parts(sims, m);
        let g = std::slice::from_raw_parts(signs, m);
        let draws: Vec<(usize, f64)> = s.iter().zip(g).map(|(&a, &b)| (a as usize, b)).collect();
        with!(h, t => t.estimate_bias(field, &draws));
        0
    })
}

/// Loss terms `[total, data, l1, l2, l3, cl]` of `field` (`0xFFFFFFFF`: the
/// joint pair, the terms summed over its fields) to `out`; with
/// `backward`, the parameter cotangents of the total are added.  0, or -1
/// on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_loss(p: *mut UNetHandle, field: u32, backward: bool, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let terms = with!(h, t => t.loss(field, backward));
        std::slice::from_raw_parts_mut(out, 6).copy_from_slice(&terms);
        0
    })
}

/// Live-bias loss terms of `field` (`[total, data, l1, l2, l3, cl]` to `out`):
/// the `kd` draws `(dsims, dsigns)` of the data term, the `kh` draws
/// `(hsims, hsigns)` of the `l` terms; with `backward` the parameter
/// cotangents are added, and with `lr > 0` an Adam step follows (the
/// cotangents zeroed first).  0, or -1 on failure.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stlp_train_live(
    p: *mut UNetHandle,
    field: u32,
    dsims: *const u64,
    dsigns: *const f64,
    kd: usize,
    hsims: *const u64,
    hsigns: *const f64,
    kh: usize,
    backward: bool,
    lr: f64,
    out: *mut f64,
) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let draws = |s: *const u64, g: *const f64, k: usize| -> Vec<(usize, f64)> {
            if k == 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(s, k).iter().zip(std::slice::from_raw_parts(g, k)).map(|(&a, &b)| (a as usize, b)).collect()
        };
        let (dd, hd) = (draws(dsims, dsigns, kd), draws(hsims, hsigns, kh));
        let terms = with!(h, t => {
            let sims = t.sims(field);
            for &(s, _) in dd.iter().chain(&hd) {
                assert!(s < sims, "simulation {s} out of the pool");
            }
            let mut t0 = t.tstart();
            if lr > 0.0 {
                t.back.zero_grad();
            }
            t.tmark(&mut t0, "step.zero_grad");
            let terms = t.loss_live(field, &dd, &hd, backward || lr > 0.0);
            assert!(terms.iter().all(|v| v.is_finite()), "non-finite loss");
            let mut t0 = t.tstart();
            if lr > 0.0 {
                t.back.adam(lr);
            }
            t.tmark(&mut t0, "step.adam");
            terms
        });
        std::slice::from_raw_parts_mut(out, 6).copy_from_slice(&terms);
        0
    })
}

/// One Adam step on `field`: zeroes the cotangents, loss and its gradient,
/// update with learning rate `lr`; the terms (before the update) to `out`
/// (`[total, data, l1, l2, l3, cl]`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_step(p: *mut UNetHandle, field: u32, lr: f64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let terms = with!(h, t => {
            let mut t0 = t.tstart();
            t.back.zero_grad();
            t.tmark(&mut t0, "step.zero_grad");
            let terms = t.loss(field, true);
            assert!(terms.iter().all(|v| v.is_finite()), "non-finite loss");
            let mut t0 = t.tstart();
            t.back.adam(lr);
            t.tmark(&mut t0, "step.adam");
            terms
        });
        std::slice::from_raw_parts_mut(out, 6).copy_from_slice(&terms);
        0
    })
}

/// L-BFGS with strong Wolfe line search (`stl_core::lbfgs`) on the
/// parameters for the sum of the losses of the `nf` fields `fields` (frozen
/// biases: a deterministic objective); options as `stlp_lbfgs`.  Returns a
/// JSON report (`evaluations`, `loss_history`, `terms_history`; free with
/// `stlp_string_free`), or null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_lbfgs(p: *mut UNetHandle, fields: *const u32, nf: usize, options: *const c_char) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &mut *p;
        let fields = std::slice::from_raw_parts(fields, nf).to_vec();
        let v = parse(options);
        let opts = LbfgsOptions {
            lr: real(&v, "lr", 1.0),
            max_iter: integer(&v, "max_iter", 20),
            max_eval: v["max_eval"].as_u64().map(|n| n as usize),
            tolerance_grad: real(&v, "tolerance_grad", 1e-12),
            tolerance_change: real(&v, "tolerance_change", 1e-15),
            history_size: integer(&v, "history_size", 10),
        };
        let (evals, history, terms_history) = with!(h, t => {
            let mut x: Vec<f64> = t.back.params().iter().map(|&a| a as f64).collect();
            let mut history = Vec::new();
            let mut terms_history = Vec::new();
            let evals = {
            let mut f = |q: &[f64]| {
                let w: Vec<f32> = q.iter().map(|&a| a as f32).collect();
                t.back.set_params(&w);
                t.back.zero_grad();
                let mut total = [0.0; 6];
                for &fi in &fields {
                    let terms = t.loss(fi, true);
                    total.iter_mut().zip(&terms).for_each(|(a, b)| *a += b);
                }
                let g: Vec<f64> = t.back.grads().iter().map(|&a| a as f64).collect();
                assert!(total[0].is_finite() && g.iter().all(|a| a.is_finite()), "non-finite objective or gradient");
                history.push(total[0]);
                terms_history.push(total.to_vec());
                (total[0], g)
            };
            minimize(&mut x, &opts, &Local, &mut f)
            };
            let w: Vec<f32> = x.iter().map(|&a| a as f32).collect();
            t.back.set_params(&w);
            (evals, history, terms_history)
        });
        let report = json!({"evaluations": evals, "loss_history": history, "terms_history": terms_history,
            "terms": ["total", "data", "l1", "l2", "l3", "cl"], "optimizer": "native STL L-BFGS (strong Wolfe)"});
        CString::new(report.to_string()).unwrap().into_raw()
    })
}

/// The phase timers (`STL_PROFILE=1`, `stl_core::prof`) as JSON `{name:
/// [seconds, count]}`, then reset; `{}` when profiling is off.  Free with
/// `stlp_string_free`; null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_profile(p: *mut UNetHandle) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &mut *p;
        with!(h, t => t.back.sync());
        let mut tab: serde_json::Map<String, Value> = stl_core::prof::take().into_iter().map(|(k, (s, c))| (k, json!([s, c]))).collect();
        // The engines' own phase timers: cumulative since their creation (count -1).
        if stl_core::prof::enabled() {
            for (k, s) in with!(h, t => t.back.engine_timers()) {
                tab.insert(k, json!([s, -1]));
            }
        }
        CString::new(Value::Object(tab).to_string()).unwrap().into_raw()
    })
}
