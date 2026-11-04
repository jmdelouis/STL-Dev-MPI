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
//!
//! `B_data = <Phi(u + n)> - Phi(u)`, `B_l1 = <Phi(u + n1, u + n2)> - Phi(u,
//! u)`, `B_l2 = <Phi(u + n, u)> - Phi(u, u)`, `B_l3 = <Phi(T, u + n)> -
//! Phi(T, u)`, the means over draws `(n, n1, n2)` of one simulation with a
//! sign.

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
    fn upload(&self, x: &[f64]) -> Self::Map;
    fn download(&self, m: &Self::Map) -> Vec<f64>;
    /// The maps `rows` one after the other.
    fn combine(&self, rows: &[Row<Self::Map>]) -> Self::Map;
    /// `dst += s src[off..off + npix]`.
    fn add_scaled(&self, dst: &mut Self::Map, src: &Self::Map, off: usize, s: f64);
    fn zeros(&self, n: usize) -> Self::Map;
    fn net_forward(&self, d: &Self::Map, scale: f64) -> (Self::Map, Self::Tape);
    fn net_backward(&mut self, tape: &Self::Tape, scale: f64, du: &Self::Map);
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
}

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
}

impl Backend for CpuBack {
    type Map = Vec<f64>;
    type Tape = stl_core::nn::Tape;

    fn npix(&self) -> usize {
        self.net.pixels(0)
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

    fn net_forward(&self, d: &Vec<f64>, scale: f64) -> (Vec<f64>, Self::Tape) {
        let x: Vec<f32> = d.iter().map(|&v| (v / scale) as f32).collect();
        let (r, t) = self.net.forward(&self.w, &x, &self.cond);
        (d.iter().zip(&r).map(|(&a, &b)| a - scale * b as f64).collect(), t)
    }

    fn net_backward(&mut self, tape: &Self::Tape, scale: f64, du: &Vec<f64>) {
        let dr: Vec<f32> = du.iter().map(|&v| (-scale * v) as f32).collect();
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
}

#[cfg(feature = "gpu")]
impl Backend for GpuBack {
    type Map = stl_gpu::CudaSlice<f64>;
    type Tape = stl_gpu::nn::Tape;

    fn npix(&self) -> usize {
        self.net.net.pixels(0)
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

    fn net_forward(&self, d: &Self::Map, scale: f64) -> (Self::Map, Self::Tape) {
        self.net.forward(d, scale, &self.cond)
    }

    fn net_backward(&mut self, tape: &Self::Tape, scale: f64, du: &Self::Map) {
        self.net.backward(tape, scale, du);
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
}

// -- trainer ---------------------------------------------------------------------

/// Loss weights of the terms.
#[derive(Clone, Copy, Debug, Default)]
struct Weights {
    data: f64,
    l1: f64,
    l2: f64,
    l3: f64,
}

impl Weights {
    fn cross(&self) -> bool {
        self.l1 != 0.0 || self.l2 != 0.0 || self.l3 != 0.0
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

struct Trainer<B: Backend> {
    back: B,
    weights: Weights,
    t857: Option<B::Map>,
    fields: HashMap<u32, Field<B::Map>>,
    /// Draws per statistics call of the bias estimates (`bias_batch` maps,
    /// or `3 bias_batch` pairs).
    bias_batch: usize,
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

impl<B: Backend> Trainer<B> {
    fn npix(&self) -> usize {
        self.back.npix()
    }

    fn field(&self, f: u32) -> &Field<B::Map> {
        self.fields.get(&f).unwrap_or_else(|| panic!("field {f} not set"))
    }

    /// The targets of field `f` that do not depend on `u` (computed once).
    fn targets(&mut self, f: u32) {
        if self.field(f).targets.is_some() {
            return;
        }
        let w = self.weights;
        let n = self.npix();
        let fd = self.field(f);
        let data = (w.data != 0.0).then(|| {
            let x = self.back.combine(&[(&fd.d, 0, None)]);
            x
        });
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
        let _ = n;
        self.fields.get_mut(&f).unwrap().targets = Some(Targets { data, l1, l3 });
    }

    /// Frozen biases of field `f` at the current `u`, from the draws
    /// `(sim, sign)`.
    fn estimate_bias(&mut self, f: u32, draws: &[(usize, f64)]) {
        let n = self.npix();
        let w = self.weights;
        let m = draws.len() as f64;
        assert!(!draws.is_empty(), "no draws");
        let (u, _) = {
            let fd = self.field(f);
            self.back.net_forward(&fd.d, fd.scale)
        };
        let fd = self.fields.get(&f).unwrap();
        for (s, _) in draws {
            assert!(*s < fd.sims, "simulation {s} of {}", fd.sims);
        }
        let mut bias = Bias::default();
        let bb = self.bias_batch.max(1);
        if w.data != 0.0 {
            let noise = fd.noise[0].as_ref().expect("noise pool");
            let mut acc: Vec<C64> = Vec::new();
            for chunk in draws.chunks(bb) {
                let rows: Vec<Row<B::Map>> = chunk.iter().map(|&(s, sg)| (&u, 0, Some((noise, s * n, sg)))).collect();
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
            let x = self.back.combine(&[(&u, 0, None)]);
            let (phi, _) = self.back.stats(Kind::Auto, 1, x, false);
            bias.data = acc.iter().zip(&phi).map(|(a, p)| a / m - p).collect();
        }
        if w.cross() {
            let fd = self.fields.get(&f).unwrap();
            let t = self.t857.as_ref().expect("857 map");
            let [n0, n1, n2] = [&fd.noise[0], &fd.noise[1], &fd.noise[2]].map(|p| p.as_ref().expect("full and half-mission noise pools"));
            let x = self.back.combine(&[(&u, 0, None), (&u, 0, None), (t, 0, None), (&u, 0, None)]);
            let (phi, _) = self.back.stats(Kind::Cross, 2, x, false);
            let h = phi.len() / 2;
            let (uu, tu) = (&phi[..h], &phi[h..]);
            let mut acc = [vec![C64::default(); h], vec![C64::default(); h], vec![C64::default(); h]];
            for chunk in draws.chunks(bb) {
                let mut rows: Vec<Row<B::Map>> = Vec::new();
                for &(s, sg) in chunk {
                    rows.push((&u, 0, Some((n1, s * n, sg))));
                    rows.push((&u, 0, Some((n2, s * n, sg))));
                }
                for &(s, sg) in chunk {
                    rows.push((&u, 0, Some((n0, s * n, sg))));
                    rows.push((&u, 0, None));
                }
                for &(s, sg) in chunk {
                    rows.push((t, 0, None));
                    rows.push((&u, 0, Some((n0, s * n, sg))));
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
        self.fields.get_mut(&f).unwrap().bias = Some(bias);
    }

    /// The frozen-bias loss of field `f`: terms `[total, data, l1, l2, l3]`;
    /// with `backward`, the parameter cotangents of the total are added.
    fn loss(&mut self, f: u32, backward: bool) -> [f64; 5] {
        self.targets(f);
        let w = self.weights;
        let n = self.npix();
        let (u, tape) = {
            let fd = self.field(f);
            self.back.net_forward(&fd.d, fd.scale)
        };
        let scale = self.field(f).scale;
        let mut du = self.back.zeros(n);
        let mut terms = [0.0; 5];
        let fd = self.fields.get(&f).unwrap();
        let bias = fd.bias.clone().expect("frozen biases (stlp_train_bias) before the loss");
        let targets = fd.targets.as_ref().unwrap();
        let data_t = targets.data.clone();
        let (l1_t, l3_t) = (targets.l1.clone(), targets.l3.clone());
        if w.data != 0.0 {
            let (t, tv) = data_t.unwrap();
            let x = self.back.combine(&[(&u, 0, None)]);
            let (v, _) = self.back.stats(Kind::Auto, 1, x, false);
            let wc = chi2_weights(&tv, &self.back.stats_pixels(Kind::Auto));
            let (l, cot) = chi2(&add(&v, &bias.data), &t, &wc);
            terms[1] = l;
            if backward {
                let cot: Vec<C64> = cot.iter().map(|c| c * w.data).collect();
                let g = self.back.stats_grad(Kind::Auto, 1, &cot);
                self.back.add_scaled(&mut du, &g, 0, 1.0);
            }
        }
        if w.cross() {
            let fd = self.fields.get(&f).unwrap();
            let t857 = self.t857.as_ref().unwrap();
            let x = self.back.combine(&[(&fd.d, 0, None), (&u, 0, None), (&u, 0, None), (&u, 0, None), (t857, 0, None), (&u, 0, None)]);
            let (v, var) = self.back.stats(Kind::Cross, 3, x, true);
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
                // [pair, channel, npix]: u is channel 1 of (d, u), both of (u, u), channel 1 of (T, u).
                for (pair, ch) in [(0, 1), (1, 0), (1, 1), (2, 1)] {
                    self.back.add_scaled(&mut du, &g, (2 * pair + ch) * n, 1.0);
                }
            }
        }
        terms[0] = w.data * terms[1] + w.l1 * terms[2] + w.l2 * terms[3] + w.l3 * terms[4];
        if backward {
            self.back.net_backward(&tape, scale, &du);
        }
        terms
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
}

/// New U-Net handle from a JSON configuration: network `nside`, `channels`
/// (list, one per level), `ksz` (3), `gauge` (`phi`), `slope` (0.2), `cond`
/// (0), `scale` (null: the standard deviation of the input), `seed` (0, of
/// torch's initialisation scheme); `backend` (`cpu` | `gpu`), `device` (0);
/// training: `j` (scales, `log2(nside) - 1`), `loss` (`{"data", "l1", "l2",
/// "l3"}` weights, 0 by default), `bias_batch` (5, draws per statistics
/// call of a bias estimate: 5 maps, 15 pairs, as `train_denoise.py` with
/// `B = 4`), `block_pixels` (4096); GPU engines: `cache`
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
            scale: v["scale"].as_f64(),
        };
        let net = UNet::new(cfg);
        let w = net.init(integer(&v, "seed", 0) as u64);
        let specs = net.params.clone();
        let (size, npix, cond) = (net.size, net.pixels(0), net.cfg.cond);
        let j = integer(&v, "j", (nside.ilog2() as usize).saturating_sub(1).max(1));
        assert!(j > 0 && j <= nside.ilog2() as usize + 1, "invalid number of scales");
        let stats = Stats::new(nside, j, integer(&v, "block_pixels", 4096));
        let levels = stats.cfg.levels();
        let l = &v["loss"];
        let weights = Weights { data: real(l, "data", 0.0), l1: real(l, "l1", 0.0), l2: real(l, "l2", 0.0), l3: real(l, "l3", 0.0) };
        let bias_batch = integer(&v, "bias_batch", 5);
        let backend = v["backend"].as_str().unwrap_or("cpu");
        let inner = match backend {
            "cpu" => AnyTrainer::Cpu(Trainer {
                back: CpuBack { engines: HashMap::new(), adam: Adam::new(size), g: vec![0.0; size], w, cond: vec![Vec::new(); net.stencils.len()], net, stats },
                weights,
                t857: None,
                fields: HashMap::new(),
                bias_batch,
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
                AnyTrainer::Gpu(Trainer { back: GpuBack { evals: HashMap::new(), net: dev, cond: Vec::new(), gpu, opts, stats }, weights, t857: None, fields: HashMap::new(), bias_batch })
            }
            #[cfg(not(feature = "gpu"))]
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        let backend = if backend == "gpu" { "gpu" } else { "cpu" };
        Box::into_raw(Box::new(UNetHandle { inner, specs, size, npix, cond, backend, config: v, levels }))
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
            "backend": h.backend, "size": h.size, "npix": h.npix, "cond": h.cond, "params": params, "fields": fields,
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

/// `u = d - scale r(d / scale)` of the batch `d` (`b` NESTED maps) to `out`;
/// `scale <= 0`: the standard deviation of the batch (unbiased; the
/// configured scale if any).  With `cot` (non-null, `b` maps), the
/// parameter cotangents of `sum(cot u)` are added.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_unet_apply(p: *mut UNetHandle, d: *const f64, b: usize, scale: f64, cot: *const f64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        let d = std::slice::from_raw_parts(d, b * n);
        let fixed = h.config["scale"].as_f64();
        let scale = if scale > 0.0 { scale } else { fixed.unwrap_or_else(|| std_unbiased(d)) };
        let cot = (!cot.is_null()).then(|| std::slice::from_raw_parts(cot, b * n));
        let o = std::slice::from_raw_parts_mut(out, b * n);
        with!(h, t => {
            for i in 0..b {
                let di = t.back.upload(&d[i * n..(i + 1) * n]);
                let (u, tape) = t.back.net_forward(&di, scale);
                if let Some(c) = cot {
                    let ci = t.back.upload(&c[i * n..(i + 1) * n]);
                    t.back.net_backward(&tape, scale, &ci);
                }
                o[i * n..(i + 1) * n].copy_from_slice(&t.back.download(&u));
            }
        });
        0
    })
}

/// Statistics averaged over the `k` masks `weights` (`[k, 12 nside^2]`
/// indicators, as `stlp_set_masks`), or the whole sky (`k = 0`); drops the
/// targets and biases.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_set_masks(p: *mut UNetHandle, weights: *const f64, k: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let n = h.npix;
        let nside = h.config["nside"].as_u64().unwrap() as i64;
        let levels = h.levels;
        with!(h, t => {
            let masks = (k > 0).then(|| {
                let w = std::slice::from_raw_parts(weights, k * n);
                Masks::from_weights(nside, levels, w.chunks(n).map(<[f64]>::to_vec).collect())
            });
            t.back.set_masks(masks);
            t.fields.values_mut().for_each(|f| {
                f.targets = None;
                f.bias = None;
            });
        });
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
        });
        0
    })
}

/// Frozen biases of `field` at the current network output, from the `m`
/// draws `(sims[i], signs[i])` of its noise pools.  0, or -1 on failure.
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

/// Loss terms `[total, data, l1, l2, l3]` of `field` to `out`; with
/// `backward`, the parameter cotangents of the total are added.  0, or -1
/// on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_loss(p: *mut UNetHandle, field: u32, backward: bool, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let terms = with!(h, t => t.loss(field, backward));
        std::slice::from_raw_parts_mut(out, 5).copy_from_slice(&terms);
        0
    })
}

/// One Adam step on `field`: zeroes the cotangents, loss and its gradient,
/// update with learning rate `lr`; the terms (before the update) to `out`
/// (`[total, data, l1, l2, l3]`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_train_step(p: *mut UNetHandle, field: u32, lr: f64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let terms = with!(h, t => {
            t.back.zero_grad();
            let terms = t.loss(field, true);
            assert!(terms.iter().all(|v| v.is_finite()), "non-finite loss");
            t.back.adam(lr);
            terms
        });
        std::slice::from_raw_parts_mut(out, 5).copy_from_slice(&terms);
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
                let mut total = [0.0; 5];
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
            "terms": ["total", "data", "l1", "l2", "l3"], "optimizer": "native STL L-BFGS (strong Wolfe)"});
        CString::new(report.to_string()).unwrap().into_raw()
    })
}
