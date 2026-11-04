//! The C ABI of `stl-bridge` as the Python driver uses it: statistics
//! against the STL-Dev fixtures, maps streamed through one handle (reload,
//! checkpoint / restore) against fresh handles, resident batch against
//! streamed members, the VJP against finite differences, L-BFGS through the
//! callback; with `--features gpu`, the device against the CPU for both
//! convolution forms and both field stores.

use std::ffi::{c_void, CStr, CString};
use stl_bridge::*;

const FIXTURE: &str = "stats/stats_n32_J4_L4_k5_C1_B1_full_b256";

fn npy(name: &str) -> Vec<f64> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(FIXTURE).join(format!("{name}.npy"));
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap().into_vec::<f64>().unwrap()
}

fn error() -> String {
    unsafe { CStr::from_ptr(stlp_last_error()).to_string_lossy().into_owned() }
}

/// A handle, or `None` (with the reason) when it cannot be built.
struct H(*mut Handle);

impl H {
    fn new(config: &str) -> Option<H> {
        let c = CString::new(config).unwrap();
        let p = unsafe { stlp_new(c.as_ptr()) };
        if p.is_null() {
            eprintln!("handle not built ({config}): {}", error());
            return None;
        }
        Some(H(p))
    }

    fn info(&self) -> serde_json::Value {
        unsafe {
            let p = stlp_info(self.0);
            assert!(!p.is_null(), "{}", error());
            let v = serde_json::from_str(CStr::from_ptr(p).to_str().unwrap()).unwrap();
            stlp_string_free(p);
            v
        }
    }

    fn ncoeff(&self) -> usize {
        self.info()["plan"].as_array().unwrap().len()
    }

    /// Interleaved values and spatial variances.
    fn forward(&self, x: &[f64], generation: u64, member: u64, objective: u64) -> (Vec<f64>, Vec<f64>) {
        let n = self.ncoeff() * x.len() / self.npix();
        let (mut out, mut var) = (vec![0.0; 2 * n], vec![0.0; n]);
        let s = unsafe { stlp_forward(self.0, x.as_ptr(), x.len(), generation, member, objective, true, out.as_mut_ptr(), var.as_mut_ptr()) };
        assert_eq!(s, 0, "{}", error());
        (out, var)
    }

    fn vjp(&self, cot: &[f64], generation: u64, member: u64, objective: u64) -> Result<Vec<f64>, String> {
        let batch = cot.len() / (2 * self.ncoeff());
        let mut g = vec![0.0; batch * self.npix()];
        let s = unsafe { stlp_vjp(self.0, cot.as_ptr(), cot.len() / 2, generation, member, objective, g.as_mut_ptr()) };
        if s == 0 {
            Ok(g)
        } else {
            Err(error())
        }
    }

    /// Values of one member: channels times pixels.
    fn npix(&self) -> usize {
        let info = self.info();
        let nside = info["nside"].as_u64().unwrap() as usize;
        info["channels"].as_u64().unwrap() as usize * 12 * nside * nside
    }
}

/// A patch handle, or `None` (with the reason) when it cannot be built.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
struct P(*mut PatchHandle);

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
impl P {
    fn new(config: &str) -> Option<P> {
        let c = CString::new(config).unwrap();
        let p = unsafe { stlp_patch_new(c.as_ptr()) };
        if p.is_null() {
            eprintln!("patch handle not built ({config}): {}", error());
            return None;
        }
        Some(P(p))
    }

    fn info(&self) -> serde_json::Value {
        unsafe {
            let s = stlp_patch_info(self.0);
            assert!(!s.is_null(), "{}", error());
            let v = serde_json::from_str(CStr::from_ptr(s).to_str().unwrap()).unwrap();
            stlp_string_free(s);
            v
        }
    }

    /// Level-0 sizes of the discs (radii in radians, core <= 0: none).
    fn select(&self, centres: &[f64], radius: f64, core: f64) -> Vec<usize> {
        let mut counts = vec![0usize; centres.len() / 3];
        assert_eq!(unsafe { stlp_patch_select(self.0, centres.as_ptr(), counts.len(), radius, core, counts.as_mut_ptr()) }, 0, "{}", error());
        counts
    }

    fn forward(&self, x: &[f64], b: usize, generation: u64) -> (Vec<f64>, Vec<f64>) {
        let n = b * self.info()["plan"].as_array().unwrap().len();
        let (mut v, mut var) = (vec![0.0; 2 * n], vec![0.0; n]);
        assert_eq!(unsafe { stlp_patch_forward(self.0, x.as_ptr(), x.len(), generation, true, v.as_mut_ptr(), var.as_mut_ptr()) }, 0, "{}", error());
        (v, var)
    }

    fn vjp(&self, cot: &[f64], len: usize, generation: u64) -> Vec<f64> {
        let mut g = vec![0.0; len];
        assert_eq!(unsafe { stlp_patch_vjp(self.0, cot.as_ptr(), cot.len() / 2, generation, g.as_mut_ptr()) }, 0, "{}", error());
        g
    }
}

impl Drop for P {
    fn drop(&mut self) {
        unsafe { stlp_patch_free(self.0) };
    }
}

impl Drop for H {
    fn drop(&mut self) {
        unsafe { stlp_free(self.0) }
    }
}

struct Snap(*mut Snapshot);

impl Drop for Snap {
    fn drop(&mut self) {
        unsafe { stlp_checkpoint_free(self.0) }
    }
}

fn checkpoint(h: &H) -> Snap {
    let p = unsafe { stlp_checkpoint(h.0) };
    assert!(!p.is_null(), "{}", error());
    Snap(p)
}

fn restore(h: &H, s: &Snap, generation: u64, member: u64, objective: u64) -> Result<(), String> {
    match unsafe { stlp_restore(h.0, s.0, generation, member, objective) } {
        0 => Ok(()),
        _ => Err(error()),
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

const N32: &str = r#"{"nside": 32, "j": 4, "batch": 1, "block_pixels": 256, "backend": "cpu"}"#;

#[test]
fn cpu_statistics_match_the_fixture() {
    let h = H::new(N32).unwrap();
    let (values, _) = h.forward(&npy("map0"), 1, 0, 7);
    let (re, im) = (npy("values_re"), npy("values_im"));
    let reference: Vec<f64> = re.iter().zip(&im).flat_map(|(&a, &b)| [a, b]).collect();
    let worst = values.iter().zip(&reference).fold(0f64, |m, (x, y)| m.max((x - y).abs() / (1e-11 + 1e-10 * y.abs())));
    assert!(worst <= 1.0, "values: {worst} x tolerance");
}

/// Maps streamed through one handle (forward of another map in between,
/// checkpoint / restore) give the statistics and gradients of fresh handles,
/// to the bit; stale identities are refused.
#[test]
fn cpu_streaming_matches_fresh_handles() {
    streaming(N32);
}

fn streaming(config: &str) {
    let Some(h) = H::new(config) else { return };
    let npix = h.npix();
    let (a, b) = (npy("map0"), noise(npix, 3));
    let cot = noise(2 * h.ncoeff(), 5);
    let fresh = |x: &[f64]| {
        let f = H::new(config).unwrap();
        let v = f.forward(x, 1, 0, 1);
        (v, f.vjp(&cot, 1, 0, 1).unwrap())
    };
    let ((va, sa), ga) = fresh(&a);
    let ((vb, _), gb) = fresh(&b);
    // a, then b, then a again: reload.
    assert_eq!(h.forward(&a, 1, 0, 1).0, va);
    assert_eq!(h.forward(&b, 1, 1, 1).0, vb);
    let (v, s) = h.forward(&a, 1, 0, 1);
    assert_eq!((v, s), (va.clone(), sa));
    assert_eq!(h.vjp(&cot, 1, 0, 1).unwrap(), ga);
    // Checkpoint of a, forward of b, restore of a.
    h.forward(&a, 2, 0, 1);
    let snap = checkpoint(&h);
    assert!(unsafe { stlp_checkpoint_bytes(snap.0) } > 8 * npix);
    h.forward(&b, 2, 1, 1);
    assert!(h.vjp(&cot, 2, 0, 1).is_err(), "vjp of a stale member accepted");
    assert!(restore(&h, &snap, 2, 1, 1).is_err(), "restore with the wrong member accepted");
    assert!(restore(&h, &snap, 3, 0, 1).is_err(), "restore with the wrong generation accepted");
    assert_eq!(h.vjp(&cot, 2, 1, 1).unwrap(), gb);
    restore(&h, &snap, 2, 0, 1).unwrap();
    assert_eq!(h.vjp(&cot, 2, 0, 1).unwrap(), ga);
    // A checkpoint of another handle is refused.
    let other = H::new(config).unwrap();
    other.forward(&a, 2, 0, 1);
    assert!(restore(&other, &snap, 2, 0, 1).is_err(), "foreign checkpoint accepted");
}

/// A resident batch of two maps against the two maps streamed one by one.
#[test]
fn cpu_resident_batch_matches_streamed_members() {
    resident(N32, &N32.replace(r#""batch": 1"#, r#""batch": 2"#));
}

fn resident(one: &str, two: &str) {
    let (Some(h1), Some(h2)) = (H::new(one), H::new(two)) else { return };
    let npix = h1.npix();
    let nc = h1.ncoeff();
    let (a, b) = (npy("map0"), noise(npix, 9));
    let cot = noise(4 * nc, 11);
    let ab: Vec<f64> = a.iter().chain(&b).copied().collect();
    let (v2, s2) = h2.forward(&ab, 1, 10, 2);
    let g2 = h2.vjp(&cot, 1, 10, 2).unwrap();
    let (va, sa) = h1.forward(&a, 1, 0, 2);
    let ga = h1.vjp(&cot[..2 * nc], 1, 0, 2).unwrap();
    let (vb, sb) = h1.forward(&b, 1, 1, 2);
    let gb = h1.vjp(&cot[2 * nc..], 1, 1, 2).unwrap();
    let cat = |x: Vec<f64>, y: Vec<f64>| x.into_iter().chain(y).collect::<Vec<f64>>();
    let (ev, es, eg) = (max_rel(&v2, &cat(va, vb)), max_rel(&s2, &cat(sa, sb)), max_rel(&g2, &cat(ga, gb)));
    eprintln!("resident vs streamed: values {ev:e}, variances {es:e}, gradient {eg:e}");
    assert!(ev <= 1e-14 && es <= 1e-13 && eg <= 1e-14);
}

/// `d/dt Re sum(conj(c) S(x + t u))` against central differences.
#[test]
fn cpu_vjp_matches_finite_differences() {
    let h = H::new(r#"{"nside": 16, "j": 4, "batch": 1, "block_pixels": 512, "backend": "cpu"}"#).unwrap();
    let npix = h.npix();
    let x: Vec<f64> = noise(npix, 13).iter().map(|v| 1.0 + v).collect();
    let u = noise(npix, 17);
    let cot = noise(2 * h.ncoeff(), 19);
    let f = |t: f64| {
        let y: Vec<f64> = x.iter().zip(&u).map(|(a, b)| a + t * b).collect();
        let (v, _) = h.forward(&y, 0, 0, 0);
        v.iter().zip(&cot).map(|(a, b)| a * b).sum::<f64>()
    };
    let step = 1e-5;
    let fd = (f(step) - f(-step)) / (2.0 * step);
    h.forward(&x, 1, 0, 0);
    let g = h.vjp(&cot, 1, 0, 0).unwrap();
    let an: f64 = g.iter().zip(&u).map(|(a, b)| a * b).sum();
    let rel = (an - fd).abs() / fd.abs();
    eprintln!("vjp {an:.12e} finite differences {fd:.12e} relative {rel:e}");
    assert!(rel <= 1e-7);
}

unsafe extern "C" fn quadratic(x: *const f64, n: usize, g: *mut f64, context: *mut c_void) -> f64 {
    let x = std::slice::from_raw_parts(x, n);
    let g = std::slice::from_raw_parts_mut(g, n);
    let calls = &mut *(context as *mut usize);
    *calls += 1;
    let mut loss = 0.0;
    for i in 0..n {
        let a = 1.0 + i as f64 / n as f64;
        let d = x[i] - (i as f64).sin();
        loss += 0.5 * a * d * d;
        g[i] = a * d;
    }
    loss
}

#[test]
fn lbfgs_through_the_callback() {
    let n = 200_000;
    let mut x = vec![0.0; n];
    let mut calls = 0usize;
    let opts = CString::new(r#"{"max_iter": 50, "history_size": 10}"#).unwrap();
    let evals = unsafe { stlp_lbfgs(x.as_mut_ptr(), n, opts.as_ptr(), quadratic, &mut calls as *mut usize as *mut c_void) };
    assert!(evals > 0, "{}", error());
    assert_eq!(evals as usize, calls);
    let err = x.iter().enumerate().fold(0f64, |m, (i, v)| m.max((v - (i as f64).sin()).abs()));
    assert!(err < 1e-6, "L-BFGS did not converge: {err:e} after {evals} evaluations");
}

#[test]
fn bad_configuration_is_an_error() {
    assert!(H::new(r#"{"nside": 12}"#).is_none());
    assert!(error().contains("power of two"), "{}", error());
    assert!(H::new(r#"{"nside": 8, "j": 3, "backend": "tpu"}"#).is_none());
}

/// Patch handle on discs of radius 180 degrees (two centres, one batch)
/// against the full-sky handle: values, variances, gradients; the cross
/// handle on `(a, a)` against the auto terms; the geometry cache.
#[test]
fn patch_full_disc_matches_the_full_sky_handle() {
    let nside = 8usize;
    let npix = 12 * nside * nside;
    for cross in [false, true] {
        let channels = if cross { 2 } else { 1 };
        let full = H::new(&format!(r#"{{"nside": {nside}, "j": 3, "batch": 2, "block_pixels": 256, "cross": {cross}}}"#)).unwrap();
        let x = noise(2 * channels * npix, 11);
        let (v_ref, var_ref) = full.forward(&x, 1, 0, 0);
        let cot = noise(v_ref.len(), 12);
        let g_ref = full.vjp(&cot, 1, 0, 0).unwrap();
        let config = CString::new(format!(r#"{{"nside": {nside}, "j": 3, "cross": {cross}, "threads": 2, "cache": 1}}"#)).unwrap();
        let p = unsafe { stlp_patch_new(config.as_ptr()) };
        assert!(!p.is_null(), "{}", error());
        let centres = [0.0, 0.0, 1.0, 1.0, 2.0, -0.5];
        let mut counts = [0usize; 2];
        for _ in 0..2 {
            assert_eq!(unsafe { stlp_patch_select(p, centres.as_ptr(), 2, std::f64::consts::PI, 0.0, counts.as_mut_ptr()) }, 0, "{}", error());
            assert_eq!(counts, [npix, npix]);
        }
        let mut ids = vec![0i64; npix];
        assert_eq!(unsafe { stlp_patch_ids(p, 1, ids.as_mut_ptr()) }, 0);
        assert!(ids.iter().enumerate().all(|(i, &v)| v == i as i64));
        let (mut v, mut var) = (vec![0.0; v_ref.len()], vec![0.0; var_ref.len()]);
        assert_eq!(unsafe { stlp_patch_forward(p, x.as_ptr(), x.len(), 5, true, v.as_mut_ptr(), var.as_mut_ptr()) }, 0, "{}", error());
        assert!(max_rel(&v, &v_ref) < 1e-10 && max_rel(&var, &var_ref) < 1e-10);
        let mut g = vec![0.0; x.len()];
        assert_ne!(unsafe { stlp_patch_vjp(p, cot.as_ptr(), cot.len() / 2, 4, g.as_mut_ptr()) }, 0, "stale generation accepted");
        assert_eq!(unsafe { stlp_patch_vjp(p, cot.as_ptr(), cot.len() / 2, 5, g.as_mut_ptr()) }, 0, "{}", error());
        assert!(max_rel(&g, &g_ref) < 1e-10);
        let info = unsafe {
            let s = stlp_patch_info(p);
            let v: serde_json::Value = serde_json::from_str(CStr::from_ptr(s).to_str().unwrap()).unwrap();
            stlp_string_free(s);
            v
        };
        assert_eq!(info["plan"], full.info()["plan"]);
        assert_eq!((info["cache"]["entries"].as_u64(), info["cache"]["hits"].as_u64(), info["cache"]["misses"].as_u64()), (Some(1), Some(1), Some(3)));
        unsafe { stlp_patch_free(p) };
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    fn config(batch: usize, conv: &str, fields: &str) -> String {
        format!(r#"{{"nside": 32, "j": 4, "batch": {batch}, "block_pixels": 256, "backend": "gpu", "conv": "{conv}", "fields": "{fields}", "cache_bytes": 0}}"#)
    }

    const FORMS: [(&str, &str); 4] = [("dia", "keep"), ("dia", "recompute"), ("stencils", "keep"), ("stencils", "recompute")];

    /// Device statistics and gradients against the CPU handle.
    #[test]
    fn device_matches_cpu() {
        let cpu = H::new(N32).unwrap();
        let a = npy("map0");
        let cot = noise(2 * cpu.ncoeff(), 23);
        let (vc, sc) = cpu.forward(&a, 1, 0, 0);
        let gc = cpu.vjp(&cot, 1, 0, 0).unwrap();
        for (conv, fields) in FORMS {
            let Some(h) = H::new(&config(1, conv, fields)) else { return };
            let info = h.info();
            assert_eq!((info["conv"].as_str().unwrap(), info["fields"].as_str().unwrap()), (conv, fields));
            let (v, s) = h.forward(&a, 1, 0, 0);
            let g = h.vjp(&cot, 1, 0, 0).unwrap();
            let (ev, es, eg) = (max_rel(&v, &vc), max_rel(&s, &sc), max_rel(&g, &gc));
            eprintln!("{conv}/{fields}: values {ev:e}, variances {es:e}, gradient {eg:e}");
            assert!(ev <= 1e-13 && es <= 1e-12 && eg <= 1e-12);
        }
    }

    /// Cross handle (two channels, cross terms only): device against CPU.
    #[test]
    fn device_cross_matches_cpu() {
        let cpu = H::new(r#"{"nside": 32, "j": 4, "batch": 2, "block_pixels": 256, "cross": true}"#).unwrap();
        let x = noise(2 * cpu.npix(), 31);
        let cot = noise(4 * cpu.ncoeff(), 32);
        let (vc, sc) = cpu.forward(&x, 1, 0, 0);
        let gc = cpu.vjp(&cot, 1, 0, 0).unwrap();
        for (conv, fields) in FORMS {
            let cfg = config(2, conv, fields).replace("\"backend\"", "\"cross\": true, \"backend\"");
            let Some(h) = H::new(&cfg) else { return };
            let (v, s) = h.forward(&x, 1, 0, 0);
            let g = h.vjp(&cot, 1, 0, 0).unwrap();
            let (ev, es, eg) = (max_rel(&v, &vc), max_rel(&s, &sc), max_rel(&g, &gc));
            eprintln!("cross {conv}/{fields}: values {ev:e}, variances {es:e}, gradient {eg:e}");
            // The cross S1 VJP divides by (|z| + 1e-8)^1.5: rounding of small
            // |z| is amplified (2.2e-12 measured on a T4).
            assert!(ev <= 1e-13 && es <= 1e-12 && eg <= 1e-11);
        }
    }

    /// Device patch handle against the CPU one: several patches per disc
    /// (groups of different sizes), with and without a core, auto and
    /// cross, the device cache below one disc (only the batch kept).
    #[test]
    fn device_patches_match_cpu() {
        let deg = std::f64::consts::PI / 180.0;
        let centres = [0.0, 0.0, 1.0, 1.0, 2.0, -0.5, 0.0, 0.0, 1.0, -1.0, 0.3, 0.2, 1.0, 2.0, -0.5];
        let b = centres.len() / 3;
        for cross in [false, true] {
            let channels = if cross { 2 } else { 1 };
            let base = format!(r#""nside": 32, "j": 4, "cross": {cross}, "threads": 2"#);
            let cpu = P::new(&format!("{{{base}}}")).unwrap();
            for (gpu_cfg, core) in [(r#""cache_bytes": 0"#, 25.0 * deg), (r#""cache": 2"#, 0.0)] {
                let Some(dev) = P::new(&format!(r#"{{{base}, "backend": "gpu", {gpu_cfg}}}"#)) else { return };
                let counts = cpu.select(&centres, 40.0 * deg, core);
                for _ in 0..2 {
                    assert_eq!(dev.select(&centres, 40.0 * deg, core), counts);
                }
                let n: usize = channels * counts.iter().sum::<usize>();
                let x = noise(n, 41);
                let (vc, sc) = cpu.forward(&x, b, 3);
                let cot = noise(vc.len(), 42);
                let gc = cpu.vjp(&cot, n, 3);
                for round in 0..2 {
                    let (v, s) = dev.forward(&x, b, 7);
                    let g = dev.vjp(&cot, n, 7);
                    let (ev, es, eg) = (max_rel(&v, &vc), max_rel(&s, &sc), max_rel(&g, &gc));
                    eprintln!("patches cross {cross} core {core:.3} round {round}: values {ev:e}, variances {es:e}, gradient {eg:e}");
                    assert!(ev <= 1e-12 && es <= 1e-11 && eg <= 1e-11);
                }
                let info = dev.info();
                eprintln!("device {}", info["device"]);
                assert_eq!(info["backend"], "gpu");
            }
        }
    }

    /// Patches read from and cotangents written to device memory: the
    /// host results.
    #[test]
    fn device_pointers_match_host_patches() {
        let deg = std::f64::consts::PI / 180.0;
        let centres = [0.0, 0.0, 1.0, 1.0, 2.0, -0.5, 0.0, 0.0, 1.0];
        let Some(dev) = P::new(r#"{"nside": 32, "j": 4, "cross": true, "backend": "gpu"}"#) else { return };
        let counts = dev.select(&centres, 30.0 * deg, 20.0 * deg);
        let n = 2 * counts.iter().sum::<usize>();
        let x = noise(n, 61);
        let (vh, sh) = dev.forward(&x, 3, 1);
        let cot = noise(vh.len(), 62);
        let gh = dev.vjp(&cot, n, 1);
        let gpu = stl_gpu::Gpu::new(0).unwrap();
        let xd = gpu.upload(&x);
        let gd = gpu.zeros::<f64>(n);
        let (mut v, mut s) = (vec![0.0; vh.len()], vec![0.0; sh.len()]);
        assert_eq!(unsafe { stlp_patch_forward_device(dev.0, gpu.ptr(&xd), n, 2, true, v.as_mut_ptr(), s.as_mut_ptr()) }, 0, "{}", error());
        assert_eq!(unsafe { stlp_patch_vjp_device(dev.0, cot.as_ptr(), cot.len() / 2, 2, gpu.ptr(&gd)) }, 0, "{}", error());
        let g = gpu.download(&gd);
        assert_eq!((v, s), (vh, sh));
        assert_eq!(g, gh);
    }

    /// A device disc of radius 180 degrees: the full-sky statistics.
    #[test]
    fn device_full_disc_matches_the_full_sky_handle() {
        let nside = 16usize;
        let npix = 12 * nside * nside;
        for cross in [false, true] {
            let channels = if cross { 2 } else { 1 };
            let full = H::new(&format!(r#"{{"nside": {nside}, "j": 3, "batch": 2, "block_pixels": 256, "cross": {cross}}}"#)).unwrap();
            let x = noise(2 * channels * npix, 51);
            let (v_ref, var_ref) = full.forward(&x, 1, 0, 0);
            let cot = noise(v_ref.len(), 52);
            let g_ref = full.vjp(&cot, 1, 0, 0).unwrap();
            let Some(p) = P::new(&format!(r#"{{"nside": {nside}, "j": 3, "cross": {cross}, "backend": "gpu"}}"#)) else { return };
            assert_eq!(p.select(&[0.0, 0.0, 1.0, 1.0, 2.0, -0.5], std::f64::consts::PI, 0.0), [npix, npix]);
            let (v, var) = p.forward(&x, 2, 1);
            let g = p.vjp(&cot, x.len(), 1);
            let (ev, es, eg) = (max_rel(&v, &v_ref), max_rel(&var, &var_ref), max_rel(&g, &g_ref));
            eprintln!("full disc cross {cross}: values {ev:e}, variances {es:e}, gradient {eg:e}");
            assert!(ev <= 1e-10 && es <= 1e-10 && eg <= 1e-10);
            assert_eq!(p.info()["plan"], full.info()["plan"]);
        }
    }

    #[test]
    fn device_streaming_matches_fresh_handles() {
        for (conv, fields) in FORMS {
            streaming(&config(1, conv, fields));
        }
    }

    #[test]
    fn device_resident_batch_matches_streamed_members() {
        for (conv, fields) in FORMS {
            resident(&config(1, conv, fields), &config(2, conv, fields));
        }
    }

    /// With recomputed fields a checkpoint holds the maps and the stored
    /// moduli only.
    #[test]
    fn recompute_checkpoints_are_smaller() {
        let bytes = |fields| {
            let h = H::new(&config(1, "dia", fields))?;
            h.forward(&npy("map0"), 1, 0, 0);
            Some(unsafe { stlp_checkpoint_bytes(checkpoint(&h).0) })
        };
        let (Some(keep), Some(recompute)) = (bytes("keep"), bytes("recompute")) else { return };
        eprintln!("checkpoint bytes: keep {keep}, recompute {recompute}");
        assert!(4 * recompute < keep);
    }
}
