//! C ABI over the native engines and L-BFGS, for the Python ensemble driver
//! of P. Campeti (`python/stl_ensemble`, `stl_port/native.py`, `ctypes`).
//!
//! One handle owns one engine (CPU [`stl_core::engine::LocalEngine`] or
//! `stl_gpu::GpuEngine`) whose batch, geometry and plan are fixed at
//! creation: the complete plan (`Family::ALL`, one channel,
//! `fewer = false`, `dg(j) = j`, gauge cosmo), or with `cross` the cross
//! terms of two channels (`c1 != c2`: S1, S2, S3, S4).
//! Maps are `f64` NESTED `[batch, 12 nside^2]` (`[batch, 2, 12 nside^2]`
//! with `cross`); complex values and cotangents
//! are interleaved `(re, im)` pairs, and the gradient is that of
//! `Re sum(conj(cotangent) * values)`.  Every forward state carries an
//! identity (handle, trial generation, member, objective); `stlp_vjp` and
//! `stlp_restore` refuse a stale or foreign one.  One caller thread per
//! handle.  Panics are caught at the boundary: the call returns its failure
//! value and `stlp_last_error` the message.
//!
//! Masks (`stlp_set_masks`, [`stl_core::mask`]): the coefficients are then
//! averaged over each of `K` masks of their native level, the fields being
//! those of the full sky; values, variances and cotangents become
//! `[batch, K, len(plan)]`.
//!
//! A spectrum handle (`stlp_spec_*`) gives the pseudo-C_l of masked
//! full-sky maps (anafast, `iter = 0`) and their VJP.
//!
//! A patch handle (`stlp_patch_*`) evaluates the same statistics on disc
//! patches ([`stl_core::disc`]): a batch of discs with their own centres,
//! each map holding only its disc's pixels.

#![warn(missing_docs)]

use num_complex::Complex64;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::lbfgs::{minimize, LbfgsOptions, Local};
use stl_core::local::LocalGeometry;
use stl_core::mask::Masks;
use stl_core::disc::{pixel_vector, unit, DiscGeometry, DiscModel, DiscState, DiscSupport};
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::sht::Sht;
use stl_core::stencil::Gauge;
#[cfg(feature = "gpu")]
use stl_core::engine::{Partition, Serial};
#[cfg(feature = "gpu")]
use stl_gpu::{disc::{GpuDiscEngine, Patches}, ConvForm, FieldStore, GeometryCache, Gpu, GpuEngine, GpuOptions, GpuSht, HostForwardState};

/// Version of the C ABI and its JSON (`stlp_info`).
pub const VERSION: &str = "2026-11-06-v6";

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(e) => {
            let s = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "native panic".into());
            ERROR.with(|m| *m.borrow_mut() = CString::new(s.replace('\0', " ")).unwrap());
            fallback
        }
    }
}

unsafe fn parse(p: *const c_char) -> Value {
    assert!(!p.is_null(), "null JSON input");
    serde_json::from_str(CStr::from_ptr(p).to_str().expect("UTF-8 JSON")).expect("valid JSON")
}

fn integer(v: &Value, key: &str, default: usize) -> usize {
    v[key].as_u64().map_or(default, |n| n as usize)
}

fn real(v: &Value, key: &str, default: f64) -> f64 {
    v[key].as_f64().unwrap_or(default)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    owner: u64,
    generation: u64,
    member: u64,
    objective: u64,
}

enum Backend {
    Cpu(LocalEngine<'static>),
    #[cfg(feature = "gpu")]
    Gpu(GpuEngine<'static>),
}

/// An engine and what it borrows.  The engine borrows the two boxes, whose
/// heap contents never move; the fields drop in declaration order, the
/// engine first.
pub struct Handle {
    backend: Backend,
    _geo: Box<LocalGeometry>,
    _cfg: Box<EngineConfig>,
    plan: Vec<Coefficient>,
    id: u64,
    nside: usize,
    batch: usize,
    channels: usize,
    npix: usize,
    current: Option<Identity>,
    /// Masks of the averages (none: the whole sky).
    masks: Option<Masks>,
}

enum Fields {
    Cpu { maps: Vec<Vec<f64>>, m: BTreeMap<(usize, usize), Vec<f64>> },
    #[cfg(feature = "gpu")]
    Gpu(HostForwardState),
}

/// Host copy of a forward state (`stlp_checkpoint`).
pub struct Snapshot {
    identity: Identity,
    fields: Fields,
}

impl Handle {
    fn identity(&self, generation: u64, member: u64, objective: u64) -> Identity {
        Identity { owner: self.id, generation, member, objective }
    }
}

/// The complete one-channel plan, or the cross terms of two channels.
fn plan_of(j: usize, l: usize, j_to_dg: &[usize], cross: bool) -> Vec<Coefficient> {
    if cross {
        build_plan(j, l, j_to_dg, 2, None, &Family::ALL).into_iter().filter(|c| c.c1 != c.c2).collect()
    } else {
        build_plan(j, l, j_to_dg, 1, None, &Family::ALL)
    }
}

fn plan_json(plan: &[Coefficient]) -> Vec<Value> {
    plan.iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"index": i, "family": c.family.name(), "native_dg": c.native_dg, "c1": c.c1, "c2": c.c2,
                   "j1": c.j1, "j2": c.j2, "j3": c.j3, "l1": c.l1, "l2": c.l2, "l3": c.l3})
        })
        .collect()
}

/// Message of the last failure caught on the calling thread (empty if
/// none).  The pointer is owned by the library and valid until the next
/// failure on that thread; do not free it.
#[no_mangle]
pub extern "C" fn stlp_last_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

/// Frees a string returned by `stlp_info`.
#[no_mangle]
pub unsafe extern "C" fn stlp_string_free(p: *mut c_char) {
    if !p.is_null() {
        drop(CString::from_raw(p));
    }
}

/// New handle from a JSON configuration: `nside`, `j` (scales), `batch`,
/// `kernel` (5), `orientations` (4), `block_pixels` (4096), `backend`
/// (`cpu` | `gpu`); GPU only: `device` (0), `cache` (`transposition`),
/// `cache_bytes` (1 GiB), `threads` (1), `conv` (`dia` | `stencils`) and
/// `fields` (`auto` | `keep` | `recompute`), else `STL_GPU_CONV`,
/// `STL_GPU_FIELDS` or their defaults; `cross` (false): two channels and
/// the plan of their cross terms.  Null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_new(config: *const c_char) -> *mut Handle {
    guard(std::ptr::null_mut(), || {
        let v = parse(config);
        let nside = integer(&v, "nside", 128);
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let j = integer(&v, "j", 7);
        assert!(j > 0 && j <= nside.ilog2() as usize + 1, "invalid number of scales");
        let batch = integer(&v, "batch", 1);
        assert!(batch > 0, "batch must be positive");
        let block_pixels = integer(&v, "block_pixels", 4096);
        let geo = Box::new(LocalGeometry::new(
            nside as i64,
            integer(&v, "kernel", 5),
            integer(&v, "orientations", 4),
            Gauge::Cosmo,
            block_pixels,
        ));
        let cfg = Box::new(EngineConfig::new(j, false));
        let cross = v["cross"].as_bool().unwrap_or(false);
        let channels = if cross { 2 } else { 1 };
        let plan = plan_of(j, geo.l, &cfg.j_to_dg, cross);
        let npix = 12 * nside * nside;
        // Safety: the boxes outlive the engine (see `Handle`).
        let g: &'static LocalGeometry = &*(&*geo as *const LocalGeometry);
        let c: &'static EngineConfig = &*(&*cfg as *const EngineConfig);
        let backend = match v["backend"].as_str().unwrap_or("cpu") {
            "cpu" => Backend::Cpu(LocalEngine::new(g, c, vec![0.0; batch * channels * npix], batch, channels)),
            #[cfg(feature = "gpu")]
            "gpu" => {
                let gpu = std::sync::Arc::new(Gpu::new(integer(&v, "device", 0)).expect("CUDA initialisation"));
                let mut opts = GpuOptions {
                    block_rows: block_pixels,
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
                Backend::Gpu(GpuEngine::new(gpu, g, c, Partition::serial(g, c.levels()), batch, channels, &opts))
            }
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Box::into_raw(Box::new(Handle { backend, _geo: geo, _cfg: cfg, plan, id, nside, batch, channels, npix, current: None, masks: None }))
    })
}

/// Frees a handle returned by `stlp_new` and its engine (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn stlp_free(p: *mut Handle) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// JSON: versions, sizes, the plan and, on a GPU, device memory, timers,
/// geometry cache, convolution form and field store.  Free with
/// `stlp_string_free`.
#[no_mangle]
pub unsafe extern "C" fn stlp_info(p: *mut Handle) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &*p;
        #[allow(unused_mut)]
        let mut info = json!({"version": VERSION, "upstream_revision": env!("STL_BRIDGE_REVISION"),
            "nside": h.nside, "batch": h.batch, "channels": h.channels, "cross": h.channels == 2,
            "dtype": "float64/complex128", "fewer": false, "plan": plan_json(&h.plan),
            "masks": h.masks.as_ref().map_or(0, Masks::len)});
        #[cfg(feature = "gpu")]
        if let Backend::Gpu(e) = &h.backend {
            let (peak, total) = e.device_memory();
            let (mode, budget, used) = e.geometry_cache();
            let (fields, rows) = e.memory_parts();
            info["device_memory"] = json!({"peak_phase_bytes": peak, "total_bytes": total, "fields_bytes": fields, "decimation_rows_bytes": rows});
            info["timers"] = json!(e.timers);
            info["cache"] = json!({"mode": mode.name(), "budget": budget, "used": used});
            info["conv"] = json!(e.conv_form().name());
            info["fields"] = json!(e.field_store().name());
        }
        CString::new(info.to_string()).unwrap().into_raw()
    })
}

/// Averages over the `k` masks `weights` (`[k, 12 nside^2]`, nonzero: in
/// the mask; level `n + 1` holds the parents of level `n`) from the next
/// forward sweep on, or over the whole sky (`k = 0`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_set_masks(p: *mut Handle, weights: *const f64, k: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        h.current = None;
        let masks = (k > 0).then(|| {
            let w = std::slice::from_raw_parts(weights, k * h.npix);
            Masks::from_weights(h.nside as i64, h._cfg.levels(), w.chunks(h.npix).map(<[f64]>::to_vec).collect())
        });
        match &mut h.backend {
            Backend::Cpu(e) => e.set_masks(masks.clone()),
            #[cfg(feature = "gpu")]
            Backend::Gpu(e) => e.set_masks(masks.as_ref()),
        }
        h.masks = masks;
        0
    })
}

/// Number of pixels each coefficient averages to `out` (`[K, len(plan)]`,
/// `K = 1` without masks): the pixels of the mask at the coefficient's
/// native level.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_native_pixels(p: *mut Handle, out: *mut usize) -> i32 {
    guard(-1, || {
        let h = &*p;
        let k_count = h.masks.as_ref().map_or(1, Masks::len);
        let out = std::slice::from_raw_parts_mut(out, k_count * h.plan.len());
        for (k, row) in out.chunks_mut(h.plan.len()).enumerate() {
            for (o, c) in row.iter_mut().zip(&h.plan) {
                *o = h.masks.as_ref().map_or_else(|| h._geo.pixels(c.native_dg), |m| m.counts[k][c.native_dg]);
            }
        }
        0
    })
}

/// Forward sweep of the maps `x` (`n = batch * channels * 12 nside^2`): values to `out`
/// (`2 batch K len(plan)`, `K = 1` without masks), and with `moments` the
/// spatial variances of the terms to `var` (`batch K len(plan)`).  0, or -1
/// on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_forward(
    p: *mut Handle,
    x: *const f64,
    n: usize,
    generation: u64,
    member: u64,
    objective: u64,
    moments: bool,
    out: *mut f64,
    var: *mut f64,
) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(n, h.batch * h.channels * h.npix, "input shape mismatch");
        h.current = None;
        let x = std::slice::from_raw_parts(x, n);
        let (values, variance) = match &mut h.backend {
            Backend::Cpu(e) => {
                e.reload(x);
                if moments {
                    e.forward_moments(&h.plan)
                } else {
                    (e.forward(&h.plan), vec![])
                }
            }
            #[cfg(feature = "gpu")]
            Backend::Gpu(e) => {
                e.load(&Serial, x);
                e.forward(&Serial, &h.plan, moments)
            }
        };
        let o = std::slice::from_raw_parts_mut(out, 2 * values.len());
        for (o, z) in o.chunks_mut(2).zip(&values) {
            o[0] = z.re;
            o[1] = z.im;
        }
        if moments {
            std::slice::from_raw_parts_mut(var, variance.len()).copy_from_slice(&variance);
        }
        h.current = Some(h.identity(generation, member, objective));
        0
    })
}

/// Map cotangent `out` (`batch * channels * 12 nside^2`) of the cotangents `cot`
/// (`2 n`, `n = batch K len(plan)`) after the forward sweep of the same
/// identity.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_vjp(
    p: *mut Handle,
    cot: *const f64,
    n: usize,
    generation: u64,
    member: u64,
    objective: u64,
    out: *mut f64,
) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(h.current, Some(h.identity(generation, member, objective)), "stale, wrong-member or wrong-objective forward state");
        assert_eq!(n, h.batch * h.masks.as_ref().map_or(1, Masks::len) * h.plan.len(), "cotangent shape mismatch");
        let c: Vec<Complex64> = std::slice::from_raw_parts(cot, 2 * n).chunks(2).map(|v| Complex64::new(v[0], v[1])).collect();
        let grad = match &mut h.backend {
            Backend::Cpu(e) => {
                let mg = e.reverse(&h.plan, &c);
                e.map_cotangent(mg)
            }
            #[cfg(feature = "gpu")]
            Backend::Gpu(e) => {
                let mg = e.reverse(&Serial, &h.plan, &c);
                let g = e.analysis_vjp(&Serial, mg);
                e.gpu.download(&g)
            }
        };
        std::slice::from_raw_parts_mut(out, h.batch * h.channels * h.npix).copy_from_slice(&grad);
        0
    })
}

/// Host copy of the current forward state (null on failure); free with
/// `stlp_checkpoint_free`.
#[no_mangle]
pub unsafe extern "C" fn stlp_checkpoint(p: *mut Handle) -> *mut Snapshot {
    guard(std::ptr::null_mut(), || {
        let h = &*p;
        let identity = h.current.clone().expect("checkpoint requires a forward state");
        let fields = match &h.backend {
            Backend::Cpu(e) => Fields::Cpu { maps: e.maps.clone(), m: e.m.clone() },
            #[cfg(feature = "gpu")]
            Backend::Gpu(e) => Fields::Gpu(e.checkpoint_forward()),
        };
        Box::into_raw(Box::new(Snapshot { identity, fields }))
    })
}

/// Frees a checkpoint returned by `stlp_checkpoint` (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn stlp_checkpoint_free(p: *mut Snapshot) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// Bytes of host memory of a checkpoint (0 for null).
#[no_mangle]
pub unsafe extern "C" fn stlp_checkpoint_bytes(s: *const Snapshot) -> usize {
    if s.is_null() {
        return 0;
    }
    match &(*s).fields {
        Fields::Cpu { maps, m } => 8 * maps.iter().chain(m.values()).map(Vec::len).sum::<usize>(),
        #[cfg(feature = "gpu")]
        Fields::Gpu(f) => f.bytes(),
    }
}

/// Restores a checkpoint of this handle with the given identity; a
/// `stlp_vjp` may follow.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_restore(p: *mut Handle, s: *const Snapshot, generation: u64, member: u64, objective: u64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let s = &*s;
        assert_eq!(s.identity, h.identity(generation, member, objective), "stale, wrong-member, wrong-objective or foreign checkpoint");
        match (&mut h.backend, &s.fields) {
            (Backend::Cpu(e), Fields::Cpu { maps, m }) => {
                assert_eq!(e.maps.len(), maps.len(), "checkpoint levels");
                for (a, b) in e.maps.iter_mut().zip(maps) {
                    a.copy_from_slice(b);
                }
                assert!(m.keys().eq(e.m.keys()), "checkpoint stored moduli");
                for (k, v) in m {
                    e.m.get_mut(k).unwrap().copy_from_slice(v);
                }
            }
            #[cfg(feature = "gpu")]
            (Backend::Gpu(e), Fields::Gpu(f)) => e.restore_forward(f),
            #[allow(unreachable_patterns)]
            _ => panic!("checkpoint of another backend"),
        }
        h.current = Some(s.identity.clone());
        0
    })
}

/// `loss = callback(x, n, grad, context)`: fills `grad` (`n`) at `x`.
pub type Callback = unsafe extern "C" fn(*const f64, usize, *mut f64, *mut c_void) -> f64;

/// L-BFGS with strong Wolfe line search (`stl_core::lbfgs::minimize`) of the
/// callback from `x` (`n`, updated in place).  Options (JSON): `lr` (1),
/// `max_iter` (20), `max_eval` (`max_iter * 5 / 4`), `tolerance_grad`
/// (1e-12), `tolerance_change` (1e-15), `history_size` (10).  The vector
/// operations run on `STL_THREADS` host threads.  Returns the number of
/// evaluations, or -1 on failure (a non-finite loss or gradient included).
#[no_mangle]
pub unsafe extern "C" fn stlp_lbfgs(x: *mut f64, n: usize, options: *const c_char, callback: Callback, context: *mut c_void) -> i64 {
    guard(-1, || {
        let v = parse(options);
        let opts = LbfgsOptions {
            lr: real(&v, "lr", 1.0),
            max_iter: integer(&v, "max_iter", 20),
            max_eval: v["max_eval"].as_u64().map(|n| n as usize),
            tolerance_grad: real(&v, "tolerance_grad", 1e-12),
            tolerance_change: real(&v, "tolerance_change", 1e-15),
            history_size: integer(&v, "history_size", 10),
        };
        assert!(opts.history_size > 0 && opts.lr > 0.0 && opts.lr.is_finite(), "invalid L-BFGS options");
        let mut f = |p: &[f64]| {
            let mut g = vec![0.0; p.len()];
            let loss = callback(p.as_ptr(), p.len(), g.as_mut_ptr(), context);
            assert!(loss.is_finite() && g.iter().all(|x| x.is_finite()), "non-finite objective or gradient (or Python callback exception)");
            (loss, g)
        };
        minimize(std::slice::from_raw_parts_mut(x, n), &opts, &Local, &mut f) as i64
    })
}

// -- disc patches -----------------------------------------------------------------

type DiscKey = [u64; 5];

/// Disc-patch statistics ([`stl_core::disc`]): the plan of `stlp_new`
/// (one channel, or the cross terms of two).  CPU: the patches of a batch
/// evaluated on `threads` host threads, the geometry of the last `cache`
/// discs kept (key: centre, radius, core).  GPU (`stl_gpu::disc`): the
/// supports and decimations of the last `cache` discs kept on the host
/// (built on `threads` threads), their convolution geometry on the device
/// within `cache_bytes`.
pub struct PatchHandle {
    geo: LocalGeometry,
    cfg: EngineConfig,
    plan: Vec<Coefficient>,
    channels: usize,
    threads: usize,
    capacity: usize,
    cache: HashMap<DiscKey, Disc>,
    order: VecDeque<DiscKey>,
    hits: u64,
    misses: u64,
    selected: Vec<(DiscKey, Disc)>,
    states: Vec<DiscState>,
    current: Option<u64>,
    #[cfg(feature = "gpu")]
    device: Option<GpuDiscEngine>,
}

/// A cached disc: the CPU operators, or the support of the device engine.
#[derive(Clone)]
enum Disc {
    Cpu(Arc<DiscGeometry>),
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    Gpu(Arc<DiscSupport>),
}

impl Disc {
    fn ids(&self) -> &[i64] {
        match self {
            Disc::Cpu(g) => &g.ids[0],
            Disc::Gpu(s) => &s.ids[0],
        }
    }

    fn core_pixels(&self, n: usize) -> usize {
        match self {
            Disc::Cpu(g) => g.core_pixels(n),
            Disc::Gpu(s) => s.core_pixels(n),
        }
    }

    fn cpu(&self) -> &Arc<DiscGeometry> {
        match self {
            Disc::Cpu(g) => g,
            Disc::Gpu(_) => unreachable!("device disc on the CPU path"),
        }
    }
}

/// `f(item)` for every item on `threads` scoped threads (contiguous
/// chunks), results in item order.
fn parallel<T: Send, R: Send>(items: Vec<T>, threads: usize, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let t = threads.clamp(1, n.max(1));
    if t == 1 {
        return items.into_iter().map(f).collect();
    }
    let chunk = n.div_ceil(t);
    let mut parts: Vec<Vec<T>> = Vec::new();
    let mut it = items.into_iter().peekable();
    while it.peek().is_some() {
        parts.push(it.by_ref().take(chunk).collect());
    }
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = parts.into_iter().map(|p| s.spawn(move || p.into_iter().map(f).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    })
}

impl PatchHandle {
    fn model(&self) -> DiscModel<'_> {
        DiscModel { geo: &self.geo, cfg: &self.cfg, channels: self.channels }
    }

    fn sizes(&self) -> Vec<usize> {
        self.selected.iter().map(|(_, g)| g.ids().len()).collect()
    }

    fn on_device(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.device.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }
}

/// New patch handle from a JSON configuration: `nside`, `j`, `kernel` (5),
/// `orientations` (4), `cross` (false), `threads` (0: `STL_THREADS`, else
/// every CPU), `cache` (256 discs), `backend` (`cpu` | `gpu`); GPU only:
/// `device` (0), `reserve_bytes` (2 GiB: disc geometries are evicted
/// before a new one is built until the device keeps this much free beyond
/// it, the free memory read at that time), `cache_bytes` (optional cap of
/// the cached geometries).  Null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_new(config: *const c_char) -> *mut PatchHandle {
    guard(std::ptr::null_mut(), || {
        let v = parse(config);
        let nside = integer(&v, "nside", 128);
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let j = integer(&v, "j", 7);
        assert!(j > 0 && j <= nside.ilog2() as usize + 1, "invalid number of scales");
        let geo = LocalGeometry::new(nside as i64, integer(&v, "kernel", 5), integer(&v, "orientations", 4), Gauge::Cosmo, 4096);
        let cfg = EngineConfig::new(j, false);
        let cross = v["cross"].as_bool().unwrap_or(false);
        let plan = plan_of(j, geo.l, &cfg.j_to_dg, cross);
        let threads = match integer(&v, "threads", 0) {
            0 => std::env::var("STL_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
            t => t,
        };
        let channels = if cross { 2 } else { 1 };
        #[cfg(feature = "gpu")]
        let device = match v["backend"].as_str().unwrap_or("cpu") {
            "gpu" => {
                let gpu = Arc::new(Gpu::new(integer(&v, "device", 0)).expect("CUDA initialisation"));
                let budget = v["cache_bytes"].as_u64().map(|b| b as usize);
                let reserve = integer(&v, "reserve_bytes", 2 << 30);
                Some(GpuDiscEngine::new(gpu, &geo, &cfg, channels, budget, reserve))
            }
            "cpu" => None,
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        #[cfg(not(feature = "gpu"))]
        match v["backend"].as_str().unwrap_or("cpu") {
            "cpu" => {}
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        }
        Box::into_raw(Box::new(PatchHandle {
            geo,
            cfg,
            plan,
            channels,
            threads,
            capacity: integer(&v, "cache", 256),
            cache: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
            selected: Vec::new(),
            states: Vec::new(),
            current: None,
            #[cfg(feature = "gpu")]
            device,
        }))
    })
}

/// Frees a patch handle (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_free(p: *mut PatchHandle) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// JSON: version, sizes, plan, threads, geometry cache and the sizes of
/// the selected discs.  Free with `stlp_string_free`.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_info(p: *mut PatchHandle) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &*p;
        let info = json!({"version": VERSION, "upstream_revision": env!("STL_BRIDGE_REVISION"),
            "nside": h.geo.nside0, "j": h.cfg.j_count, "levels": h.cfg.levels(), "channels": h.channels, "cross": h.channels == 2,
            "dtype": "float64/complex128", "fewer": false, "threads": h.threads, "plan": plan_json(&h.plan),
            "cache": {"capacity": h.capacity, "entries": h.cache.len(), "hits": h.hits, "misses": h.misses},
            "backend": if h.on_device() { "gpu" } else { "cpu" }, "selected": h.sizes()});
        #[cfg(feature = "gpu")]
        let info = {
            let mut info = info;
            if let Some(e) = &h.device {
                let (entries, used, budget, reserve) = e.cache_info();
                let (free, total) = e.gpu().memory();
                info["device"] = json!({"name": e.gpu().name, "ordinal": e.gpu().ordinal(), "free": free, "total": total,
                    "cache": {"entries": entries, "bytes": used, "budget": budget, "reserve": reserve, "hits": e.hits, "misses": e.misses},
                    "timers": e.timers()});
            }
            info
        };
        CString::new(info.to_string()).unwrap().into_raw()
    })
}

/// Unit vectors `out` (`3 n`) of the NESTED pixel centres `ids` (`n`) at
/// `nside`.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_pixel_vectors(nside: usize, ids: *const i64, n: usize, out: *mut f64) -> i32 {
    guard(-1, || {
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let ids = std::slice::from_raw_parts(ids, n);
        let out = std::slice::from_raw_parts_mut(out, 3 * n);
        for (o, &id) in out.chunks_mut(3).zip(ids) {
            assert!(id >= 0 && (id as usize) < 12 * nside * nside, "pixel index out of range");
            o.copy_from_slice(&pixel_vector(id, nside as i64));
        }
        0
    })
}

/// Selects a batch of `b` discs: centres `centres` (`3 b`, any nonzero
/// vectors), angular radius `radius` (radians; `>= pi`: the full sky),
/// statistics averaged on the core of radius `core` when `0 < core <
/// radius`, else on the whole disc.  Writes the number of level-0 pixels
/// of every disc to `counts` (`b`) and drops the forward state.  0, or -1
/// on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_select(p: *mut PatchHandle, centres: *const f64, b: usize, radius: f64, core: f64, counts: *mut usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        h.current = None;
        h.states.clear();
        h.selected.clear();
        assert!(b > 0, "empty batch");
        assert!(radius.is_finite() && radius > 0.0, "disc radius must be positive");
        let core = (core.is_finite() && core > 0.0 && core < radius).then_some(core);
        let centres: Vec<[f64; 3]> = std::slice::from_raw_parts(centres, 3 * b).chunks(3).map(|c| unit([c[0], c[1], c[2]])).collect();
        let keys: Vec<DiscKey> = centres
            .iter()
            .map(|c| [c[0].to_bits(), c[1].to_bits(), c[2].to_bits(), radius.to_bits(), core.unwrap_or(-1.0).to_bits()])
            .collect();
        let mut missing: Vec<(DiscKey, [f64; 3])> = Vec::new();
        for (k, c) in keys.iter().zip(&centres) {
            if h.cache.contains_key(k) || missing.iter().any(|(m, _)| m == k) {
                h.hits += 1;
            } else {
                h.misses += 1;
                missing.push((*k, *c));
            }
        }
        let levels = h.cfg.levels();
        let geo = &h.geo;
        let device = h.on_device();
        let built = parallel(missing, h.threads, |(k, c)| {
            let s = DiscSupport::new(geo, levels, c, radius, core);
            (k, if device { Disc::Gpu(Arc::new(s)) } else { Disc::Cpu(Arc::new(DiscGeometry::bind(geo, s))) })
        });
        let fresh: HashMap<DiscKey, Disc> = built.into_iter().collect();
        h.selected = keys.iter().map(|k| (*k, fresh.get(k).or_else(|| h.cache.get(k)).unwrap().clone())).collect();
        for (k, g) in fresh {
            if h.capacity > 0 {
                h.cache.insert(k, g);
                h.order.push_back(k);
            }
        }
        while h.cache.len() > h.capacity {
            let old = h.order.pop_front().unwrap();
            h.cache.remove(&old);
        }
        std::slice::from_raw_parts_mut(counts, b).copy_from_slice(&h.sizes());
        0
    })
}

/// Sorted NESTED pixels (level 0) of the selected disc `i` to `out`
/// (`counts[i]` entries).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_ids(p: *mut PatchHandle, i: usize, out: *mut i64) -> i32 {
    guard(-1, || {
        let h = &*p;
        let ids = h.selected.get(i).expect("disc index out of range").1.ids();
        std::slice::from_raw_parts_mut(out, ids.len()).copy_from_slice(ids);
        0
    })
}

/// Number of pixels averaged at every level of every selected disc to
/// `out` (`b levels`: a coefficient of native level `n` of disc `i` is a
/// mean over `out[i levels + n]` pixels).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_core_pixels(p: *mut PatchHandle, out: *mut usize) -> i32 {
    guard(-1, || {
        let h = &*p;
        let levels = h.cfg.levels();
        let out = std::slice::from_raw_parts_mut(out, h.selected.len() * levels);
        for (o, (_, g)) in out.chunks_mut(levels).zip(&h.selected) {
            for (n, v) in o.iter_mut().enumerate() {
                *v = g.core_pixels(n);
            }
        }
        0
    })
}

/// Statistics of the selected discs: `x` holds, disc after disc, the
/// patch `[channels, counts[i]]` (`n` values in all); values to `out`
/// (`2 b len(plan)`), with `moments` the spatial variances over the
/// averaging pixels to `var` (`b len(plan)`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_forward(p: *mut PatchHandle, x: *const f64, n: usize, generation: u64, moments: bool, out: *mut f64, var: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        h.current = None;
        h.states.clear();
        assert!(!h.selected.is_empty(), "no disc selected");
        let sizes: Vec<usize> = h.sizes().iter().map(|s| s * h.channels).collect();
        assert_eq!(n, sizes.iter().sum::<usize>(), "input shape mismatch");
        let x = std::slice::from_raw_parts(x, n);
        #[cfg(feature = "gpu")]
        if h.on_device() {
            device_forward(h, Patches::Host(x), generation, moments, out, var);
            return 0;
        }
        let mut offset = 0;
        let jobs: Vec<(Arc<DiscGeometry>, &[f64])> = h
            .selected
            .iter()
            .zip(&sizes)
            .map(|((_, g), &s)| {
                let part = &x[offset..offset + s];
                offset += s;
                (g.cpu().clone(), part)
            })
            .collect();
        let (model, plan) = (h.model(), &h.plan);
        let results = parallel(jobs, h.threads, |(g, xi)| model.forward(&g, xi, plan, moments));
        let nc = plan.len();
        let o = std::slice::from_raw_parts_mut(out, 2 * nc * results.len());
        let mut states = Vec::with_capacity(results.len());
        for (i, (values, variance, state)) in results.into_iter().enumerate() {
            for (k, z) in values.iter().enumerate() {
                o[2 * (i * nc + k)] = z.re;
                o[2 * (i * nc + k) + 1] = z.im;
            }
            if moments {
                std::slice::from_raw_parts_mut(var.add(i * nc), nc).copy_from_slice(&variance);
            }
            states.push(state);
        }
        h.states = states;
        h.current = Some(generation);
        0
    })
}

/// Patch cotangents `out` (the layout of the input of the forward) of the
/// cotangents `cot` (`2 n`, `n = b len(plan)`) after the forward of the
/// same generation.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_vjp(p: *mut PatchHandle, cot: *const f64, n: usize, generation: u64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(h.current, Some(generation), "stale forward state");
        let nc = h.plan.len();
        assert_eq!(n, h.selected.len() * nc, "cotangent shape mismatch");
        let c: Vec<Complex64> = std::slice::from_raw_parts(cot, 2 * n).chunks(2).map(|v| Complex64::new(v[0], v[1])).collect();
        #[cfg(feature = "gpu")]
        if let Some(e) = h.device.as_mut() {
            let g = e.reverse(&h.plan, &c);
            std::slice::from_raw_parts_mut(out, g.len()).copy_from_slice(&g);
            return 0;
        }
        let h = &*h;
        let (model, plan) = (h.model(), &h.plan);
        let grads = parallel((0..h.selected.len()).collect(), h.threads, |i: usize| {
            model.reverse(h.selected[i].1.cpu(), &h.states[i], plan, &c[i * nc..(i + 1) * nc])
        });
        let total: usize = grads.iter().map(Vec::len).sum();
        let o = std::slice::from_raw_parts_mut(out, total);
        let mut offset = 0;
        for g in grads {
            o[offset..offset + g.len()].copy_from_slice(&g);
            offset += g.len();
        }
        0
    })
}

/// The device forward of the selected discs (GPU patch handle).
#[cfg(feature = "gpu")]
unsafe fn device_forward(h: &mut PatchHandle, x: Patches, generation: u64, moments: bool, out: *mut f64, var: *mut f64) {
    let e = h.device.as_mut().expect("not a GPU patch handle");
    let discs: Vec<(DiscKey, &DiscSupport)> = h
        .selected
        .iter()
        .map(|(k, d)| match d {
            Disc::Gpu(s) => (*k, &**s),
            Disc::Cpu(_) => unreachable!("CPU disc on the device path"),
        })
        .collect();
    let (values, variance) = e.forward_from(&discs, x, &h.plan, moments);
    let o = std::slice::from_raw_parts_mut(out, 2 * values.len());
    for (k, z) in values.iter().enumerate() {
        o[2 * k] = z.re;
        o[2 * k + 1] = z.im;
    }
    if moments {
        std::slice::from_raw_parts_mut(var, variance.len()).copy_from_slice(&variance);
    }
    h.current = Some(generation);
}

/// [`stlp_patch_forward`] reading the patches at the device address `x`
/// (`f64`, `n` values, the layout of `stlp_patch_forward`) of the handle's
/// device, e.g. a contiguous CUDA tensor whose stream the caller has
/// synchronised.  GPU patch handles only.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_forward_device(p: *mut PatchHandle, x: u64, n: usize, generation: u64, moments: bool, out: *mut f64, var: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        h.current = None;
        h.states.clear();
        assert!(!h.selected.is_empty(), "no disc selected");
        assert_eq!(n, h.channels * h.sizes().iter().sum::<usize>(), "input shape mismatch");
        assert!(x != 0, "null device address");
        #[cfg(feature = "gpu")]
        if h.on_device() {
            device_forward(h, Patches::Device(x), generation, moments, out, var);
            return 0;
        }
        let _ = (generation, moments, out, var);
        panic!("device input needs a GPU patch handle (backend gpu)")
    })
}

/// [`stlp_patch_vjp`] writing the patch cotangents to the device address
/// `out` (`f64`, the layout of the forward input) of the handle's device;
/// the device work is finished on return.  GPU patch handles only.  0, or
/// -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_patch_vjp_device(p: *mut PatchHandle, cot: *const f64, n: usize, generation: u64, out: u64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        assert_eq!(h.current, Some(generation), "stale forward state");
        assert_eq!(n, h.selected.len() * h.plan.len(), "cotangent shape mismatch");
        assert!(out != 0, "null device address");
        let c: Vec<Complex64> = std::slice::from_raw_parts(cot, 2 * n).chunks(2).map(|v| Complex64::new(v[0], v[1])).collect();
        #[cfg(feature = "gpu")]
        if let Some(e) = h.device.as_mut() {
            e.reverse_device(&h.plan, &c, out);
            return 0;
        }
        let _ = c;
        panic!("device output needs a GPU patch handle (backend gpu)")
    })
}

// -- pseudo-spectra ---------------------------------------------------------------

/// Pseudo-power spectra (anafast, `iter = 0`, no ring weights) of full-sky
/// NESTED maps, optionally multiplied by `K` mask weights before the
/// analysis ([`stl_core::sht::Sht`]; on a GPU the transforms of
/// `stl_gpu::GpuSht`, the spectra and their cotangents on the host).  Maps
/// `[batch, channels, 12 nside^2]`; one channel: the auto spectra, two
/// (`cross`): the cross spectra `Re(a_lm conj(b_lm))` only.  Spectra
/// `[batch, K, lmax + 1]` (`K = 1` without masks).
pub struct SpectrumHandle {
    sht: Sht,
    #[cfg(feature = "gpu")]
    device: Option<(GpuSht, Arc<Gpu>)>,
    id: u64,
    batch: usize,
    channels: usize,
    /// Mask weights `[K][npix]` (empty: the whole sky).
    masks: Vec<Vec<f64>>,
    /// `a_lm` of the last forward, `[batch][K][channels]`.
    alms: Vec<Vec<Complex64>>,
    current: Option<Identity>,
}

/// Maps per transform batch on the device (bounds its Legendre sums).
#[cfg(feature = "gpu")]
const SPEC_CHUNK: usize = 16;

impl SpectrumHandle {
    fn averages(&self) -> usize {
        self.masks.len().max(1)
    }

    fn identity(&self, generation: u64, member: u64, objective: u64) -> Identity {
        Identity { owner: self.id, generation, member, objective }
    }

    /// `a_lm` of every map, in order.
    fn analysis(&self, maps: &[Vec<f64>]) -> Vec<Vec<Complex64>> {
        #[cfg(feature = "gpu")]
        if let Some((s, gpu)) = &self.device {
            let mut out = Vec::with_capacity(maps.len());
            for part in maps.chunks(SPEC_CHUNK) {
                let up: Vec<_> = part.iter().map(|x| gpu.upload(x)).collect();
                for a in s.analysis_many(gpu, &up.iter().collect::<Vec<_>>()) {
                    out.push(gpu.download(&a).chunks(2).map(|v| Complex64::new(v[0], v[1])).collect());
                }
            }
            return out;
        }
        maps.iter().map(|x| self.sht.analysis(x)).collect()
    }

    /// Map cotangents of the analysis from the `a_lm` cotangents, in order.
    fn analysis_vjp(&self, cots: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
        #[cfg(feature = "gpu")]
        if let Some((s, gpu)) = &self.device {
            // `Omega Y(c / mult)`: the scaling on the host, the synthesis on the device.
            let area = self.sht.area();
            let lm = self.sht.lm();
            let mut out = Vec::with_capacity(cots.len());
            for part in cots.chunks(SPEC_CHUNK) {
                let up: Vec<_> = part
                    .iter()
                    .map(|c| {
                        let v: Vec<f64> = lm
                            .iter()
                            .zip(c)
                            .flat_map(|(&(_, m), z)| {
                                let z = z * if m == 0 { area } else { area / 2.0 };
                                [z.re, z.im]
                            })
                            .collect();
                        gpu.upload(&v)
                    })
                    .collect();
                for g in s.synthesis_many(gpu, &up.iter().collect::<Vec<_>>()) {
                    out.push(gpu.download(&g));
                }
            }
            return out;
        }
        cots.iter().map(|c| self.sht.analysis_vjp(c)).collect()
    }
}

/// New spectrum handle from a JSON configuration: `nside`, `lmax`
/// (`3 nside - 1`), `batch` (1), `cross` (false), `backend` (`cpu` |
/// `gpu`), `device` (0).  The CPU transforms run on `STL_THREADS` host
/// threads.  Null on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_new(config: *const c_char) -> *mut SpectrumHandle {
    guard(std::ptr::null_mut(), || {
        let v = parse(config);
        let nside = integer(&v, "nside", 128);
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let lmax = integer(&v, "lmax", 3 * nside - 1);
        assert!(lmax <= 4 * nside, "lmax must be at most 4 nside");
        let batch = integer(&v, "batch", 1);
        assert!(batch > 0, "batch must be positive");
        let channels = if v["cross"].as_bool().unwrap_or(false) { 2 } else { 1 };
        let sht = Sht::new(nside, lmax);
        #[cfg(feature = "gpu")]
        let device = match v["backend"].as_str().unwrap_or("cpu") {
            "gpu" => {
                let gpu = Arc::new(Gpu::new(integer(&v, "device", 0)).expect("CUDA initialisation"));
                Some((GpuSht::new(&gpu, &sht).expect("device transforms"), gpu))
            }
            "cpu" => None,
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        #[cfg(not(feature = "gpu"))]
        match v["backend"].as_str().unwrap_or("cpu") {
            "cpu" => {}
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Box::into_raw(Box::new(SpectrumHandle {
            sht,
            #[cfg(feature = "gpu")]
            device,
            id,
            batch,
            channels,
            masks: Vec::new(),
            alms: Vec::new(),
            current: None,
        }))
    })
}

/// Frees a spectrum handle (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_free(p: *mut SpectrumHandle) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// JSON: version, sizes, backend, number of masks and their `f_sky = mean
/// w^2`.  Free with `stlp_string_free`.
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_info(p: *mut SpectrumHandle) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let h = &*p;
        #[cfg(feature = "gpu")]
        let backend = if h.device.is_some() { "gpu" } else { "cpu" };
        #[cfg(not(feature = "gpu"))]
        let backend = "cpu";
        let fsky: Vec<f64> = h.masks.iter().map(|w| w.iter().map(|v| v * v).sum::<f64>() / w.len() as f64).collect();
        let info = json!({"version": VERSION, "upstream_revision": env!("STL_BRIDGE_REVISION"),
            "nside": h.sht.nside, "lmax": h.sht.lmax, "batch": h.batch, "channels": h.channels, "cross": h.channels == 2,
            "backend": backend, "dtype": "float64", "masks": h.masks.len(), "fsky": fsky, "iter": 0});
        CString::new(info.to_string()).unwrap().into_raw()
    })
}

/// Multiplies the maps by each of the `k` mask weights `weights` (`[k,
/// 12 nside^2]`, any finite values, e.g. apodised) before the analysis from
/// the next forward on, or by nothing (`k = 0`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_set_masks(p: *mut SpectrumHandle, weights: *const f64, k: usize) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        h.current = None;
        h.alms.clear();
        let npix = h.sht.npix;
        let w: &[f64] = if k > 0 { std::slice::from_raw_parts(weights, k * npix) } else { &[] };
        assert!(w.iter().all(|v| v.is_finite()), "mask weights must be finite");
        h.masks = w.chunks(npix).map(<[f64]>::to_vec).collect();
        0
    })
}

/// Spectra of the maps `x` (`n = batch * channels * 12 nside^2`) to `out`
/// (`batch K (lmax + 1)`).  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_forward(p: *mut SpectrumHandle, x: *const f64, n: usize, generation: u64, member: u64, objective: u64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &mut *p;
        let npix = h.sht.npix;
        assert_eq!(n, h.batch * h.channels * npix, "input shape mismatch");
        h.current = None;
        h.alms.clear();
        let x = std::slice::from_raw_parts(x, n);
        let k_count = h.averages();
        let mut maps = Vec::with_capacity(h.batch * k_count * h.channels);
        for b in 0..h.batch {
            for k in 0..k_count {
                for c in 0..h.channels {
                    let xi = &x[(b * h.channels + c) * npix..][..npix];
                    maps.push(match h.masks.get(k) {
                        Some(w) => xi.iter().zip(w).map(|(a, m)| a * m).collect(),
                        None => xi.to_vec(),
                    });
                }
            }
        }
        h.alms = h.analysis(&maps);
        let nl = h.sht.lmax + 1;
        let o = std::slice::from_raw_parts_mut(out, h.batch * k_count * nl);
        for (row, a) in o.chunks_mut(nl).zip(h.alms.chunks(h.channels)) {
            row.copy_from_slice(&if h.channels == 1 { h.sht.power(&a[0]) } else { h.sht.cross_power(&a[0], &a[1]) });
        }
        h.current = Some(h.identity(generation, member, objective));
        0
    })
}

/// Map cotangent `out` (`batch * channels * 12 nside^2`) of the spectrum
/// cotangents `cot` (`n = batch K (lmax + 1)`) after the forward of the
/// same identity.  0, or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn stlp_spec_vjp(p: *mut SpectrumHandle, cot: *const f64, n: usize, generation: u64, member: u64, objective: u64, out: *mut f64) -> i32 {
    guard(-1, || {
        let h = &*p;
        assert_eq!(h.current, Some(h.identity(generation, member, objective)), "stale, wrong-member or wrong-objective forward state");
        let (nl, k_count, npix) = (h.sht.lmax + 1, h.averages(), h.sht.npix);
        assert_eq!(n, h.batch * k_count * nl, "cotangent shape mismatch");
        let cot = std::slice::from_raw_parts(cot, n);
        let mut ca = Vec::with_capacity(h.alms.len());
        for (d, a) in cot.chunks(nl).zip(h.alms.chunks(h.channels)) {
            if h.channels == 1 {
                ca.push(h.sht.power_vjp(&a[0], d));
            } else {
                let (g0, g1) = h.sht.cross_power_vjp(&a[0], &a[1], d);
                ca.extend([g0, g1]);
            }
        }
        let g = h.analysis_vjp(&ca);
        let o = std::slice::from_raw_parts_mut(out, h.batch * h.channels * npix);
        o.fill(0.0);
        // The masks are summed in mask order.
        for b in 0..h.batch {
            for k in 0..k_count {
                for c in 0..h.channels {
                    let gi = &g[(b * k_count + k) * h.channels + c];
                    let oi = &mut o[(b * h.channels + c) * npix..][..npix];
                    match h.masks.get(k) {
                        Some(w) => oi.iter_mut().zip(gi.iter().zip(w)).for_each(|(o, (v, m))| *o += v * m),
                        None => oi.iter_mut().zip(gi).for_each(|(o, v)| *o += v),
                    }
                }
            }
        }
        0
    })
}
