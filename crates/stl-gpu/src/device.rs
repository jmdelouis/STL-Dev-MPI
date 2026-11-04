//! One CUDA device driven by one rank: context, stream, the kernels of
//! `kernels.cu` compiled by NVRTC, transfers and launches.
//!
//! The kernels are compiled for the device's architecture with
//! `--fmad=false` and IEEE division / square root, so that the arithmetic
//! of every kernel is the one of its Rust counterpart in stl-core.

use cudarc::driver::{result, sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, PinnedHostSlice, ValidAsZeroBits};
use cudarc::nvrtc::{result as nvrtc_result, sys as nvrtc_sys, Ptx};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

const SOURCE: &str = concat!(include_str!("kernels.cu"), include_str!("harmonic.cu"), include_str!("planar.cu"), include_str!("disc.cu"));

const KERNELS: [&str; 91] = [
    "disc_terms",
    "disc_terms_rev",
    "ring_thetas",
    "geom_build",
    "geom_pack",
    "geom_wt",
    "geom_count",
    "geom_fill",
    "seg_sort",
    "conv_fwd",
    "conv_adj",
    "dia_reach",
    "flag_count",
    "flag_fill",
    "dia_build",
    "dia_slots",
    "dia_pack",
    "dia_fwd",
    "dia_gin_t",
    "dia_adj",
    "face_conv_f32",
    "face_conv_f64",
    "csr_apply",
    "modulus_at",
    "modvjp_add",
    "gather_add",
    "gather_copy",
    "fill_i32",
    "dmap_set",
    "pack_rows",
    "unpack_rows",
    "coef_partial",
    "coef_reverse",
    "kappa_pchip",
    "quad_mean",
    "detail_moments",
    "level_power_grad",
    "kurtosis_grad",
    "add_product",
    "marginal_row",
    "rs_init",
    "rs_hist",
    "scan_block",
    "scan_add",
    "rs_scatter",
    "sht_coef",
    "sht_leg_syn",
    "sht_leg_adj",
    "sht_chunk_sum",
    "sht_ring_syn",
    "sht_ring_store",
    "sht_ring_load",
    "sht_ring_ana",
    "sht_scale_m",
    "sht_add_scaled_m",
    "sht_combine",
    "leg_table",
    "leg_pack_syn",
    "leg_unpack_syn",
    "leg_pack_adj",
    "leg_unpack_adj",
    "sht_power",
    "sht_power_vjp",
    "px_sums",
    "mk_standard",
    "mk_geom",
    "mk_forward",
    "mk_vjp",
    "mk_final",
    "cic_partial",
    "cic_vjp",
    "pl_conv",
    "pl_conv_adj",
    "pl_read",
    "pl_accum",
    "pl_crop",
    "pl_embed_add",
    "pl_abs",
    "pl_modvjp",
    "pl_add",
    "pl_dec",
    "pl_dec_adj",
    "pl_scale",
    "pl_to_complex",
    "pl_transpose",
    "pl_cscale",
    "pl_real_part",
    "pl_spec_mult",
    "pl_spec_mult_sum",
    "pl_rings",
    "pl_rings_vjp",
];

/// Compiles the kernels to a native binary for the device (`sm_XY`): a
/// cubin loads on any driver of the toolkit's major version, where PTX
/// would require a driver at least as recent as NVRTC.
fn compile(major: i32, minor: i32) -> Result<Ptx, String> {
    let src = CString::new(SOURCE).unwrap();
    let prog = nvrtc_result::create_program(&src, None).map_err(|e| format!("NVRTC: {e:?}"))?;
    let options = [
        format!("--gpu-architecture=sm_{major}{minor}"),
        "--fmad=false".into(),
        "--prec-div=true".into(),
        "--prec-sqrt=true".into(),
        "--ftz=false".into(),
    ];
    unsafe {
        if let Err(e) = nvrtc_result::compile_program(prog, &options) {
            let log = nvrtc_result::get_program_log(prog).unwrap_or_default();
            let log = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
            return Err(format!("NVRTC: {e:?}\n{log}"));
        }
        let mut size = 0usize;
        nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size).result().map_err(|e| format!("NVRTC: {e:?}"))?;
        let mut buf = vec![0u8; size];
        nvrtc_sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut c_char).result().map_err(|e| format!("NVRTC: {e:?}"))?;
        nvrtc_result::destroy_program(prog).map_err(|e| format!("NVRTC: {e:?}"))?;
        Ok(Ptx::from_binary(buf))
    }
}

/// A kernel argument (passed by value).
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// A raw device address (see `Gpu::ptr`, `Gpu::at`).
    Ptr(u64),
    /// A 64-bit integer (`long long` in the kernels).
    I64(i64),
    /// A 32-bit integer (`int`).
    I32(i32),
    /// A double.
    F64(f64),
}

impl From<i64> for Arg {
    fn from(v: i64) -> Self {
        Arg::I64(v)
    }
}

impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::I64(v as i64)
    }
}

impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}

impl From<f64> for Arg {
    fn from(v: f64) -> Self {
        Arg::F64(v)
    }
}

/// The null device pointer, for optional arrays.
pub const NULL: Arg = Arg::Ptr(0);

/// `dst.copy_from_slice(src)` on the host threads (`STL_THREADS`, else the
/// available CPUs).
fn par_copy(dst: &mut [f64], src: &[f64]) {
    let threads = std::env::var("STL_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);
    let per = src.len().div_ceil(threads).max(1 << 16);
    std::thread::scope(|s| {
        for (d, c) in dst.chunks_mut(per).zip(src.chunks(per)) {
            s.spawn(move || d.copy_from_slice(c));
        }
    });
}

/// One CUDA device with its stream and the compiled kernels; every transfer
/// and launch of a rank goes through it, in stream order.
pub struct Gpu {
    /// The device's primary context.
    pub ctx: Arc<CudaContext>,
    /// The single stream of the rank: transfers and launches are ordered on it.
    pub stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    funcs: HashMap<&'static str, CudaFunction>,
    /// Device name as reported by the driver.
    pub name: String,
    /// Synchronise after every launch (`STL_GPU_SYNC` set): errors name their kernel.
    debug: bool,
}

impl Gpu {
    /// Number of visible devices.
    pub fn count() -> Result<usize, String> {
        CudaContext::device_count().map(|n| n as usize).map_err(|e| format!("CUDA: {e:?}"))
    }

    /// The device `ordinal` (its primary context) with a stream of its own.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let err = |e: cudarc::driver::DriverError| format!("CUDA device {ordinal}: {e:?}");
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        // A single stream per rank: no cross-stream event bookkeeping.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(err)?;
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let module = ctx.load_module(compile(major, minor)?).map_err(err)?;
        let mut funcs = HashMap::new();
        for name in KERNELS {
            funcs.insert(name, module.load_function(name).map_err(err)?);
        }
        let name = ctx.name().unwrap_or_default();
        let debug = std::env::var_os("STL_GPU_SYNC").is_some();
        Ok(Gpu { ctx, stream, _module: module, funcs, name, debug })
    }

    /// Device for the rank: `STL_GPU_DEVICE` if set, else the local rank
    /// (MPI launchers' environment) or `rank`, modulo the device count.
    pub fn for_rank(rank: usize) -> Result<Self, String> {
        let n = Self::count()?;
        if n == 0 {
            return Err("no CUDA device".into());
        }
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let ordinal = env("STL_GPU_DEVICE").unwrap_or_else(|| {
            env("OMPI_COMM_WORLD_LOCAL_RANK")
                .or_else(|| env("MPI_LOCALRANKID"))
                .or_else(|| env("SLURM_LOCALID"))
                .or_else(|| env("PMI_LOCAL_RANK"))
                .unwrap_or(rank)
                % n
        });
        Self::new(ordinal)
    }

    /// Ordinal of the device.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Free and total device memory, bytes.
    pub fn memory(&self) -> (usize, usize) {
        self.ctx.bind_to_thread().unwrap();
        self.ctx.mem_get_info().unwrap()
    }

    /// A new device buffer holding `v` (a null buffer if `v` is empty).
    pub fn upload<T: DeviceRepr>(&self, v: &[T]) -> CudaSlice<T> {
        if v.is_empty() {
            return self.stream.null().unwrap();
        }
        self.stream.clone_htod(v).expect("host to device copy")
    }

    /// A host copy of `s`; synchronises the stream.
    pub fn download<T: DeviceRepr>(&self, s: &CudaSlice<T>) -> Vec<T> {
        if s.is_empty() {
            return Vec::new();
        }
        let v = self.stream.clone_dtoh(s).expect("device to host copy");
        self.stream.synchronize().unwrap();
        v
    }

    /// Copies `v` into the existing buffer `s` (same length).
    pub fn copy_in<T: DeviceRepr>(&self, v: &[T], s: &mut CudaSlice<T>) {
        assert_eq!(v.len(), s.len());
        if !v.is_empty() {
            self.stream.memcpy_htod(v, s).expect("host to device copy");
        }
    }

    /// Page-locked host memory for `n` values (staged transfers).
    pub fn pinned(&self, n: usize) -> PinnedHostSlice<f64> {
        unsafe { self.ctx.alloc_pinned_with_flags::<f64>(n, 0) }.expect("page-locked host allocation")
    }

    /// A page-locked host copy of `s` (a transfer on the stream), or None
    /// if `s` is empty or the host allocation fails.
    pub fn pinned_copy<T: DeviceRepr + ValidAsZeroBits>(&self, s: &CudaSlice<T>) -> Option<PinnedHostSlice<T>> {
        if s.is_empty() {
            return None;
        }
        let mut h = unsafe { self.ctx.alloc_pinned_with_flags::<T>(s.len(), 0) }.ok()?;
        self.stream.memcpy_dtoh(s, &mut h).expect("device to host copy");
        Some(h)
    }

    /// A new device buffer holding the page-locked `h` (one transfer).
    pub fn upload_pinned<T: DeviceRepr + ValidAsZeroBits>(&self, h: &PinnedHostSlice<T>) -> CudaSlice<T> {
        let mut d = unsafe { self.stream.alloc::<T>(h.len()) }.expect("device allocation");
        self.stream.memcpy_htod(h, &mut d).expect("host to device copy");
        d
    }

    /// `x` to a new device buffer through the page-locked `stage` of the
    /// same length: a copy on the host threads, then one transfer.  (With
    /// the event tracking off, the stream is synchronised before the stage
    /// is written or read again.)
    pub fn upload_staged(&self, x: &[f64], stage: &mut PinnedHostSlice<f64>) -> CudaSlice<f64> {
        assert_eq!(stage.len(), x.len());
        self.sync();
        par_copy(stage.as_mut_slice().unwrap(), x);
        let mut d = self.zeros::<f64>(x.len());
        self.stream.memcpy_htod(stage, &mut d).expect("host to device copy");
        d
    }

    /// `s` to the host through the page-locked `stage` of the same length.
    pub fn download_staged(&self, s: &CudaSlice<f64>, stage: &mut PinnedHostSlice<f64>) -> Vec<f64> {
        assert_eq!(stage.len(), s.len());
        self.sync();
        self.stream.memcpy_dtoh(s, stage).expect("device to host copy");
        self.sync();
        let mut out = vec![0.0; s.len()];
        par_copy(&mut out, stage.as_slice().unwrap());
        out
    }

    /// A new zero-filled device buffer of `n` values (a null buffer if `n == 0`).
    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> CudaSlice<T> {
        if n == 0 {
            return self.stream.null().unwrap();
        }
        self.stream.alloc_zeros(n).expect("device allocation")
    }

    /// Zeroes `s` on the stream.
    pub fn clear<T: DeviceRepr + ValidAsZeroBits>(&self, s: &mut CudaSlice<T>) {
        if !s.is_empty() {
            self.stream.memset_zeros(s).unwrap();
        }
    }

    /// A new device buffer holding a device-to-device copy of `s`.
    pub fn copy<T: DeviceRepr>(&self, s: &CudaSlice<T>) -> CudaSlice<T> {
        if s.is_empty() {
            return self.stream.null().unwrap();
        }
        self.stream.clone_dtod(s).unwrap()
    }

    /// Waits for the stream; panics on a kernel failure.
    pub fn sync(&self) {
        self.stream.synchronize().expect("kernel failure");
    }

    /// Raw device address of a buffer (+ `offset` elements).
    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        let (p, _sync) = s.device_ptr(&self.stream);
        p
    }

    /// Kernel argument: the address of element `offset` of `s`.
    pub fn at<T>(&self, s: &CudaSlice<T>, offset: usize) -> Arg {
        Arg::Ptr(self.ptr(s) + (offset * std::mem::size_of::<T>()) as u64)
    }

    /// Kernel argument: the address of `s`.
    pub fn a<T>(&self, s: &CudaSlice<T>) -> Arg {
        Arg::Ptr(self.ptr(s))
    }

    /// Launches `name` on a `(grid.0, grid.1)` grid of `block` threads.
    pub fn launch(&self, name: &str, grid: (u32, u32), block: u32, args: &[Arg]) {
        self.launch3(name, (grid.0, grid.1, 1), block, args)
    }

    /// Launches `name` on a three-dimensional grid of `block` threads.
    pub fn launch3(&self, name: &str, grid: (u32, u32, u32), block: u32, args: &[Arg]) {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return;
        }
        let f = &self.funcs[name];
        let mut values: Vec<u64> = args
            .iter()
            .map(|a| match *a {
                Arg::Ptr(p) => p,
                Arg::I64(v) => v as u64,
                Arg::I32(v) => v as u32 as u64,
                Arg::F64(v) => v.to_bits(),
            })
            .collect();
        let mut params: Vec<*mut c_void> = values.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        self.ctx.bind_to_thread().unwrap();
        unsafe {
            result::launch_kernel(
                f.cu_function(),
                grid,
                (block, 1, 1),
                0,
                self.stream.cu_stream() as sys::CUstream,
                &mut params,
            )
        }
        .unwrap_or_else(|e| panic!("launch {name}: {e:?}"));
        if self.debug {
            self.stream.synchronize().unwrap_or_else(|e| panic!("kernel {name}: {e:?}"));
        }
    }

    /// One thread per element of `n` (1-D grid).
    pub fn launch1(&self, name: &str, n: usize, args: &[Arg]) {
        if n == 0 {
            return;
        }
        let block = 256u32;
        let grid = n.div_ceil(block as usize);
        assert!(grid <= u32::MAX as usize, "{name}: grid too large");
        self.launch(name, (grid as u32, 1), block, args);
    }
}
