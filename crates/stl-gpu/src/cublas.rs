//! The cuBLAS entry points of the batched Legendre sums (one DGEMM per m
//! and parity), loaded at run time as cuFFT (`cufft.rs`): `STL_CUBLAS` (a
//! path) if set, else `libcublas.so` and its versioned names on the library
//! path.  The C interface used here (`cublas*_v2`) is the same in cuBLAS 10
//! to 13.  DGEMM runs on the FP64 tensor cores of an A100 by default
//! (IEEE fused multiply-adds); for given sizes and stream its result is the
//! same at every run.

use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// A cuBLAS handle (`cublasHandle_t`).
pub type Handle = *mut c_void;
/// `CUBLAS_OP_N`: the operand is used as stored.
pub const OP_N: c_int = 0;
/// `CUBLAS_OP_T`: the operand is transposed.
pub const OP_T: c_int = 1;

type Dgemm = unsafe extern "C" fn(Handle, c_int, c_int, c_int, c_int, c_int, *const f64, *const f64, c_int, *const f64, c_int, *const f64, *mut f64, c_int) -> c_int;

/// The cuBLAS entry points resolved from the loaded library (see `cublas`).
pub struct CuBlas {
    _lib: Library,
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    dgemm: Dgemm,
}

// The handle is used from one thread at a time (the device's stream).
unsafe impl Send for CuBlas {}
unsafe impl Sync for CuBlas {}

fn check(code: c_int, what: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("cuBLAS {what}: error {code}"))
    }
}

fn load() -> Result<CuBlas, String> {
    let names: Vec<String> = match std::env::var("STL_CUBLAS") {
        Ok(p) => vec![p],
        Err(_) => ["libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so.10", "libcublas.so.10.0"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let lib = names
        .iter()
        .find_map(|n| unsafe { Library::new(n.as_str()) }.ok())
        .ok_or_else(|| format!("cuBLAS not found ({}): add its directory to LD_LIBRARY_PATH or set STL_CUBLAS", names.join(", ")))?;
    unsafe {
        macro_rules! sym {
            ($name:literal) => {
                *lib.get($name).map_err(|e| format!("cuBLAS {}: {e}", $name))?
            };
        }
        Ok(CuBlas {
            create: sym!("cublasCreate_v2"),
            destroy: sym!("cublasDestroy_v2"),
            set_stream: sym!("cublasSetStream_v2"),
            dgemm: sym!("cublasDgemm_v2"),
            _lib: lib,
        })
    }
}

/// The library, loaded once per process.
pub fn cublas() -> Result<&'static CuBlas, String> {
    static LIB: OnceLock<Result<CuBlas, String>> = OnceLock::new();
    LIB.get_or_init(load).as_ref().map_err(|e| e.clone())
}

impl CuBlas {
    /// A handle whose calls run on `stream`.
    pub fn handle(&self, stream: *mut c_void) -> Result<Handle, String> {
        let mut h: Handle = std::ptr::null_mut();
        unsafe {
            check((self.create)(&mut h), "create")?;
            check((self.set_stream)(h, stream), "stream")?;
        }
        Ok(h)
    }

    /// Releases a handle made by `handle` (errors are ignored).
    pub fn destroy(&self, h: Handle) {
        unsafe {
            (self.destroy)(h);
        }
    }

    /// `C = alpha op(A) op(B) + beta C`, column-major (`m x n` result, inner
    /// dimension `k`), device addresses.
    ///
    /// # Safety
    /// The addresses must hold the matrices with the given leading dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dgemm(&self, h: Handle, ta: c_int, tb: c_int, m: usize, n: usize, k: usize, alpha: f64, a: u64, lda: usize, b: u64, ldb: usize, beta: f64, c: u64, ldc: usize) -> Result<(), String> {
        check(
            (self.dgemm)(h, ta, tb, m as c_int, n as c_int, k as c_int, &alpha, a as *const f64, lda as c_int, b as *const f64, ldb as c_int, &beta, c as *mut f64, ldc as c_int),
            "dgemm",
        )
    }
}
