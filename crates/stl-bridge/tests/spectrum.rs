//! The spectrum handle of `stl-bridge` (`stlp_spec_*`): auto and cross
//! pseudo-spectra of masked maps against the direct `Sht`, the VJP against
//! finite differences, the identity checks; with `--features gpu`, the
//! device against the CPU.

use std::ffi::{CStr, CString};
use stl_bridge::*;
use stl_core::sht::Sht;

fn error() -> String {
    unsafe { CStr::from_ptr(stlp_last_error()).to_string_lossy().into_owned() }
}

struct S(*mut SpectrumHandle);

impl S {
    fn new(config: &str) -> Option<S> {
        let c = CString::new(config).unwrap();
        let p = unsafe { stlp_spec_new(c.as_ptr()) };
        if p.is_null() {
            eprintln!("spectrum handle not built ({config}): {}", error());
            return None;
        }
        Some(S(p))
    }

    fn info(&self) -> serde_json::Value {
        unsafe {
            let p = stlp_spec_info(self.0);
            assert!(!p.is_null(), "{}", error());
            let v = serde_json::from_str(CStr::from_ptr(p).to_str().unwrap()).unwrap();
            stlp_string_free(p);
            v
        }
    }

    fn set_masks(&self, w: &[f64], k: usize) {
        assert_eq!(unsafe { stlp_spec_set_masks(self.0, w.as_ptr(), k) }, 0, "{}", error());
    }

    fn forward(&self, x: &[f64], generation: u64) -> Vec<f64> {
        let i = self.info();
        let n = i["batch"].as_u64().unwrap() * i["masks"].as_u64().unwrap().max(1) * (i["lmax"].as_u64().unwrap() + 1);
        let mut out = vec![0.0; n as usize];
        let s = unsafe { stlp_spec_forward(self.0, x.as_ptr(), x.len(), generation, 0, 7, out.as_mut_ptr()) };
        assert_eq!(s, 0, "{}", error());
        out
    }

    fn vjp(&self, cot: &[f64], generation: u64, len: usize) -> Result<Vec<f64>, String> {
        let mut g = vec![0.0; len];
        match unsafe { stlp_spec_vjp(self.0, cot.as_ptr(), cot.len(), generation, 0, 7, g.as_mut_ptr()) } {
            0 => Ok(g),
            _ => Err(error()),
        }
    }
}

impl Drop for S {
    fn drop(&mut self) {
        unsafe { stlp_spec_free(self.0) };
    }
}

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

/// Two weights `[2, npix]`: a smooth (apodised-like) positive weight and a
/// latitude cut.
fn weights(nside: usize) -> Vec<f64> {
    let npix = 12 * nside * nside;
    let ids: Vec<i64> = (0..npix as i64).collect();
    let mut v = vec![0.0; 3 * npix];
    assert_eq!(unsafe { stlp_pixel_vectors(nside, ids.as_ptr(), npix, v.as_mut_ptr()) }, 0, "{}", error());
    let mut w: Vec<f64> = v.chunks(3).map(|u| 0.5 + 0.5 * (3.0 * u[2]).tanh()).collect();
    w.extend(v.chunks(3).map(|u| if u[2].abs() > 0.3 { 1.0 } else { 0.0 }));
    w
}

/// Auto and cross spectra of masked maps against `Sht` directly; the
/// gradient against central differences; identity checks.
#[test]
fn cpu_spectra_and_gradient() {
    let (nside, lmax, batch) = (8usize, 20usize, 2usize);
    let npix = 12 * nside * nside;
    let sht = Sht::new(nside, lmax);
    let w = weights(nside);
    for cross in [false, true] {
        let ch = if cross { 2 } else { 1 };
        let h = S::new(&format!(r#"{{"nside": {nside}, "lmax": {lmax}, "batch": {batch}, "cross": {cross}}}"#)).unwrap();
        let x = noise(batch * ch * npix, 3);
        // Whole sky, then two masks.
        for k in [0usize, 2] {
            h.set_masks(&w[..k * npix], k);
            let cl = h.forward(&x, 1);
            for b in 0..batch {
                for kk in 0..k.max(1) {
                    let alm: Vec<_> = (0..ch)
                        .map(|c| {
                            let xi = &x[(b * ch + c) * npix..][..npix];
                            let y: Vec<f64> = if k == 0 { xi.to_vec() } else { xi.iter().zip(&w[kk * npix..]).map(|(a, m)| a * m).collect() };
                            sht.analysis(&y)
                        })
                        .collect();
                    let expect = if cross { sht.cross_power(&alm[0], &alm[1]) } else { sht.power(&alm[0]) };
                    let got = &cl[(b * k.max(1) + kk) * (lmax + 1)..][..lmax + 1];
                    assert!(max_rel(got, &expect) <= 1e-14, "cross {cross} masks {k}");
                }
            }
            let cot = noise(cl.len(), 5);
            assert!(h.vjp(&cot, 2, x.len()).unwrap_err().contains("stale"));
            let g = h.vjp(&cot, 1, x.len()).unwrap();
            let f = |y: &[f64]| -> f64 { h.forward(y, 9).iter().zip(&cot).map(|(a, b)| a * b).sum() };
            for t in 0..3 {
                let d = noise(x.len(), 11 + t);
                let eps = 1e-4;
                let xp: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + eps * b).collect();
                let xm: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a - eps * b).collect();
                let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
                let an: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
                eprintln!("cross {cross} masks {k}: directional derivative {an:e} vs {fd:e}");
                assert!((fd - an).abs() <= 1e-8 * an.abs().max(1e-6), "cross {cross} masks {k}: {fd} vs {an}");
            }
            h.forward(&x, 1);
        }
    }
}

#[test]
fn bad_spectrum_configuration_is_an_error() {
    assert!(S::new(r#"{"nside": 12}"#).is_none());
    assert!(S::new(r#"{"nside": 8, "backend": "tpu"}"#).is_none());
    let h = S::new(r#"{"nside": 8, "lmax": 10}"#).unwrap();
    let mut out = vec![0.0; 11];
    assert_eq!(unsafe { stlp_spec_forward(h.0, out.as_ptr(), 3, 0, 0, 0, out.as_mut_ptr()) }, -1);
    assert!(error().contains("shape"));
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    /// Device against CPU: auto and cross, two masks, batch of two, at
    /// nside 32 (lmax 2 nside - 1).
    #[test]
    fn device_spectra_match_cpu() {
        let (nside, batch) = (32usize, 2usize);
        let lmax = 2 * nside - 1;
        let npix = 12 * nside * nside;
        let w = weights(nside);
        for cross in [false, true] {
            let ch = if cross { 2 } else { 1 };
            let cfg = format!(r#"{{"nside": {nside}, "lmax": {lmax}, "batch": {batch}, "cross": {cross}, "backend": "BACKEND"}}"#);
            let cpu = S::new(&cfg.replace("BACKEND", "cpu")).unwrap();
            let Some(gpu) = S::new(&cfg.replace("BACKEND", "gpu")) else { return };
            for h in [&cpu, &gpu] {
                h.set_masks(&w, 2);
            }
            let x = noise(batch * ch * npix, 21);
            let (cc, cg) = (cpu.forward(&x, 1), gpu.forward(&x, 1));
            let cot = noise(cc.len(), 22);
            let (gc, gg) = (cpu.vjp(&cot, 1, x.len()).unwrap(), gpu.vjp(&cot, 1, x.len()).unwrap());
            let (ev, eg) = (max_rel(&cg, &cc), max_rel(&gg, &gc));
            eprintln!("spectra cross {cross}: values {ev:e}, gradient {eg:e}");
            assert!(ev <= 1e-12 && eg <= 1e-12);
        }
    }
}
