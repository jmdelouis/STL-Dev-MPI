//! C ABI over the native engines and L-BFGS, for the Python ensemble driver
//! of P. Campeti (`python/stl_ensemble`, `stl_port/native.py`, `ctypes`).
//!
//! One handle owns one engine (CPU [`LocalEngine`] or [`GpuEngine`]) whose
//! batch, geometry and plan are fixed at creation: the complete plan
//! (`Family::ALL`, one channel, `fewer = false`, `dg(j) = j`, gauge cosmo).
//! Maps are `f64` NESTED `[batch, 12 nside^2]`; complex values and cotangents
//! are interleaved `(re, im)` pairs, and the gradient is that of
//! `Re sum(conj(cotangent) * values)`.  Every forward state carries an
//! identity (handle, trial generation, member, objective); `stlp_vjp` and
//! `stlp_restore` refuse a stale or foreign one.  One caller thread per
//! handle.  Panics are caught at the boundary: the call returns its failure
//! value and `stlp_last_error` the message.

use num_complex::Complex64;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::lbfgs::{minimize, LbfgsOptions, Local};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::stencil::Gauge;
#[cfg(feature = "gpu")]
use stl_core::engine::{Partition, Serial};
#[cfg(feature = "gpu")]
use stl_gpu::{ConvForm, FieldStore, GeometryCache, Gpu, GpuEngine, GpuOptions, HostForwardState};

/// Version of the C ABI and its JSON (`stlp_info`).
pub const VERSION: &str = "2026-10-30-v2";

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
    npix: usize,
    current: Option<Identity>,
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
/// `STL_GPU_FIELDS` or their defaults.  Null on failure.
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
        let plan = build_plan(j, geo.l, &cfg.j_to_dg, 1, None, &Family::ALL);
        let npix = 12 * nside * nside;
        // Safety: the boxes outlive the engine (see `Handle`).
        let g: &'static LocalGeometry = &*(&*geo as *const LocalGeometry);
        let c: &'static EngineConfig = &*(&*cfg as *const EngineConfig);
        let backend = match v["backend"].as_str().unwrap_or("cpu") {
            "cpu" => Backend::Cpu(LocalEngine::new(g, c, vec![0.0; batch * npix], batch, 1)),
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
                Backend::Gpu(GpuEngine::new(gpu, g, c, Partition::serial(g, c.levels()), batch, 1, &opts))
            }
            "gpu" => panic!("GPU backend not compiled: build stl-bridge with --features gpu"),
            other => panic!("backend must be cpu or gpu, not {other:?}"),
        };
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Box::into_raw(Box::new(Handle { backend, _geo: geo, _cfg: cfg, plan, id, nside, batch, npix, current: None }))
    })
}

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
        let plan: Vec<Value> = h
            .plan
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({"index": i, "family": c.family.name(), "native_dg": c.native_dg, "c1": c.c1, "c2": c.c2,
                       "j1": c.j1, "j2": c.j2, "j3": c.j3, "l1": c.l1, "l2": c.l2, "l3": c.l3})
            })
            .collect();
        #[allow(unused_mut)]
        let mut info = json!({"version": VERSION, "upstream_revision": env!("STL_BRIDGE_REVISION"),
            "nside": h.nside, "batch": h.batch, "channels": 1, "dtype": "float64/complex128", "fewer": false, "plan": plan});
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

/// Forward sweep of the maps `x` (`n = batch * 12 nside^2`): values to `out`
/// (`2 batch len(plan)`), and with `moments` the spatial variances of the
/// terms to `var` (`batch len(plan)`).  0, or -1 on failure.
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
        assert_eq!(n, h.batch * h.npix, "input shape mismatch");
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

/// Map cotangent `out` (`batch * 12 nside^2`) of the cotangents `cot`
/// (`2 n`, `n = batch len(plan)`) after the forward sweep of the same
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
        assert_eq!(n, h.batch * h.plan.len(), "cotangent shape mismatch");
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
        std::slice::from_raw_parts_mut(out, h.batch * h.npix).copy_from_slice(&grad);
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
