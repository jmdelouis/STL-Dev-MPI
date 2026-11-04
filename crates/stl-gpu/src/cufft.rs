//! The few cuFFT entry points of the harmonic transforms, loaded at run
//! time: `STL_CUFFT` (a path) if set, else `libcufft.so` and its versioned
//! names on the library path.  cudarc exposes cuFFT only to CUDA 12
//! builds, and garoupe's driver is CUDA 11.4 (its toolkit's cuFFT 10 works
//! with it); the C interface used here is the same in cuFFT 10 to 12.

use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// A cuFFT plan handle (`cufftHandle`).
pub type Handle = c_int;
/// `CUFFT_Z2Z`.
const Z2Z: c_int = 0x69;
/// `CUFFT_FORWARD`: exponent sign -1.
pub const FORWARD: c_int = -1;
/// `CUFFT_INVERSE`: exponent sign +1 (unnormalised).
pub const INVERSE: c_int = 1;

type MakePlanMany = unsafe extern "C" fn(Handle, c_int, *mut c_int, *mut c_int, c_int, c_int, *mut c_int, c_int, c_int, c_int, c_int, *mut usize) -> c_int;

/// The cuFFT entry points resolved from the loaded library (see `cufft`).
pub struct CuFft {
    _lib: Library,
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    set_auto_allocation: unsafe extern "C" fn(Handle, c_int) -> c_int,
    make_plan_many: MakePlanMany,
    set_work_area: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    exec_z2z: unsafe extern "C" fn(Handle, *mut c_void, *mut c_void, c_int) -> c_int,
}

fn check(code: c_int, what: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("cuFFT {what}: error {code}"))
    }
}

fn load() -> Result<CuFft, String> {
    let names: Vec<String> = match std::env::var("STL_CUFFT") {
        Ok(p) => vec![p],
        Err(_) => ["libcufft.so", "libcufft.so.12", "libcufft.so.11", "libcufft.so.10"].iter().map(|s| s.to_string()).collect(),
    };
    let lib = names
        .iter()
        .find_map(|n| unsafe { Library::new(n.as_str()) }.ok())
        .ok_or_else(|| format!("cuFFT not found ({}): add its directory to LD_LIBRARY_PATH or set STL_CUFFT", names.join(", ")))?;
    unsafe {
        macro_rules! sym {
            ($name:literal) => {
                *lib.get($name).map_err(|e| format!("cuFFT {}: {e}", $name))?
            };
        }
        Ok(CuFft {
            create: sym!("cufftCreate"),
            destroy: sym!("cufftDestroy"),
            set_auto_allocation: sym!("cufftSetAutoAllocation"),
            make_plan_many: sym!("cufftMakePlanMany"),
            set_work_area: sym!("cufftSetWorkArea"),
            set_stream: sym!("cufftSetStream"),
            exec_z2z: sym!("cufftExecZ2Z"),
            _lib: lib,
        })
    }
}

/// The library, loaded once per process.
pub fn cufft() -> Result<&'static CuFft, String> {
    static LIB: OnceLock<Result<CuFft, String>> = OnceLock::new();
    LIB.get_or_init(load).as_ref().map_err(|e| e.clone())
}

impl CuFft {
    /// A plan of `batch` in-place complex double transforms of length `len`,
    /// `dist` values apart, on `stream`, without its own work area; returns
    /// the plan and the work area it needs (bytes).
    pub fn plan_z2z(&self, len: usize, batch: usize, dist: usize, stream: *mut c_void) -> Result<(Handle, usize), String> {
        let mut h: Handle = 0;
        unsafe {
            check((self.create)(&mut h), "create")?;
            check((self.set_auto_allocation)(h, 0), "auto allocation")?;
            let mut n = [len as c_int];
            let mut embed = [dist as c_int];
            let mut bytes = 0usize;
            check(
                (self.make_plan_many)(h, 1, n.as_mut_ptr(), embed.as_mut_ptr(), 1, dist as c_int, embed.as_mut_ptr(), 1, dist as c_int, Z2Z, batch as c_int, &mut bytes),
                "plan",
            )?;
            check((self.set_stream)(h, stream), "stream")?;
            Ok((h, bytes))
        }
    }

    /// # Safety
    /// `area` must be a device buffer of at least the plan's work size.
    pub unsafe fn set_work_area(&self, h: Handle, area: u64) -> Result<(), String> {
        check((self.set_work_area)(h, area as *mut c_void), "work area")
    }

    /// # Safety
    /// `data` must address the plan's batch on the device.
    pub unsafe fn exec_z2z(&self, h: Handle, data: u64, direction: c_int) -> Result<(), String> {
        check((self.exec_z2z)(h, data as *mut c_void, data as *mut c_void, direction), "exec")
    }

    /// Releases a plan handle (errors are ignored).
    pub fn destroy(&self, h: Handle) {
        unsafe {
            (self.destroy)(h);
        }
    }
}
