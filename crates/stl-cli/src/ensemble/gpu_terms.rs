//! The harmonic terms of the objective on the device (`--harmonic gpu`):
//! the C_l bands, the CIC PDF and the Minkowski functionals of this
//! group's physical maps through `stl_gpu::harmonic`, with the features
//! and map cotangents of `fields.rs` to rounding (orders of the sums at the
//! top of `stl-gpu/src/harmonic.cu`; the CIC histogram and the C_l are the
//! host's sums).  The maps go up once per evaluation, their cotangents
//! come down once.
//!
//! Four transforms per member and evaluation (`fields.rs` does six):
//! - the analysis of u = (x - mean) / sigma is that of x by linearity,
//!   `a(u) = (a(x) - mean a(1)) / sigma`, `a(1)` computed once;
//! - the two syntheses of the reverse pass (C_l cotangent, and the
//!   `analysis_vjp` of the Minkowski derivatives) are one: the cotangent
//!   through the standardisation is linear, and its two means over the
//!   synthesised part are dot products of `a_lm`
//!   ([`GpuSht::analysis_vjp_dot`]).
//!
//! The Minkowski states (u and the five derivative maps) and the maps stay
//! on the device between the passes when they fit `STL_MF_STATE_MB`
//! (default 8 GiB for a group's members); otherwise the reverse pass uploads
//! the map again and recomputes the derivatives from `a(u)` (one transform
//! more).  Everything is freed at the end of the reverse pass.

use super::fields::{window_cotangent, window_sums, Cic, Minkowski};
use std::sync::Arc;
use stl_core::sht::Sht;
use stl_gpu::harmonic::{cic_histogram, cic_vjp, minkowski_final, minkowski_sums, minkowski_vjp, pixel_sum, standardize, GpuSht, MinkowskiParams};
use stl_gpu::terms::add_product;
use stl_gpu::{CudaSlice, Gpu};

/// Device bytes for the states and maps of a group's members between the
/// passes: `STL_MF_STATE_MB`, default 8 GiB.
fn state_bytes() -> usize {
    std::env::var("STL_MF_STATE_MB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(8 << 30, |mb| mb << 20)
}

/// What the cotangent of the functionals needs of the forward pass.
struct MkState {
    u: CudaSlice<f64>,
    d: CudaSlice<f64>,
}

struct Member {
    /// The physical map (kept with the state).
    x: Option<CudaSlice<f64>>,
    /// Its analysis (C_l, and a(u) by linearity).
    alm: CudaSlice<f64>,
    mean: f64,
    sigma: f64,
    floor: f64,
    mk: Option<MkState>,
}

pub struct GpuTerms {
    gpu: Arc<Gpu>,
    sht: GpuSht,
    mk: MinkowskiParams,
    /// Analysis of the constant map 1.
    one: CudaSlice<f64>,
    members: Vec<Member>,
}

impl GpuTerms {
    pub fn new(gpu: Arc<Gpu>, sht: &Sht, mk: &Minkowski) -> Result<Self, String> {
        let dev = GpuSht::new(&gpu, sht)?;
        let params = MinkowskiParams { nu: gpu.upload(&mk.nu), k: mk.nu.len(), eps: mk.eps };
        let one = dev.analysis(&gpu, &gpu.upload(&vec![1.0; dev.npix]));
        Ok(GpuTerms { gpu, sht: dev, mk: params, one, members: Vec::new() })
    }

    /// `a(u) = (a(x) - mean a(1)) / sigma`.
    fn alm_u(&self, m: &Member) -> CudaSlice<f64> {
        self.sht.combine(&self.gpu, &m.alm, 1.0 / m.sigma, &self.one, -m.mean / m.sigma)
    }

    /// Features of the members' whole physical maps `kfull` `[local, npix]`:
    /// band powers `[local, nb]`, CIC `[local, bins]`, Minkowski `[local, 3 k]`;
    /// the seconds of the first two and of the third.
    pub fn forward(&mut self, kfull: &[f64], windows: &[Vec<f64>], cic: &Cic) -> (Vec<f64>, Vec<f64>, Vec<f64>, [f64; 2]) {
        let npix = self.sht.npix;
        let n = npix as f64;
        let local = kfull.len() / npix;
        let keep = local * npix * 7 * 8 <= state_bytes();
        let (mut pw, mut hist, mut mfv) = (Vec::new(), Vec::new(), Vec::new());
        let mut secs = [0.0; 2];
        self.members.clear();
        for row in kfull.chunks(npix) {
            let t = std::time::Instant::now();
            let gpu = &*self.gpu;
            let x = gpu.upload(row);
            let alm = self.sht.analysis(gpu, &x);
            pw.extend(window_sums(windows, &self.sht.power(gpu, &alm)));
            let h = cic_histogram(gpu, &x, cic.lo, cic.dx(), cic.bins);
            let norm = n * cic.dx();
            hist.extend(h.into_iter().map(|v| v / norm));
            secs[0] += t.elapsed().as_secs_f64();
            // Minkowski::forward: standardised map, derivatives of a(u), sums.
            let t = std::time::Instant::now();
            let mean = pixel_sum(gpu, gpu.a(&x), stl_gpu::device::NULL, npix, 0.0, 0) / n;
            let sigma = (pixel_sum(gpu, gpu.a(&x), stl_gpu::device::NULL, npix, mean, 1) / n + 1e-12).sqrt();
            let mut mem = Member { x: None, alm, mean, sigma, floor: 0.0, mk: None };
            let u = standardize(gpu, &x, mean, sigma);
            let d = self.sht.derivatives(gpu, &self.alm_u(&mem));
            mem.floor = 1e-4 * pixel_sum(gpu, gpu.at(&d, 0), gpu.at(&d, npix), npix, 0.0, 2) / n + 1e-12;
            let mut v = minkowski_sums(gpu, &self.mk, &u, &d, mem.floor);
            v.iter_mut().for_each(|a| *a /= n);
            mfv.extend(v);
            if keep {
                mem.x = Some(x);
                mem.mk = Some(MkState { u, d });
            }
            secs[1] += t.elapsed().as_secs_f64();
            self.members.push(mem);
        }
        (pw, hist, mfv, secs)
    }

    /// Map cotangents `[local, npix]` of the members of the last forward
    /// pass (whole maps `kfull`) from the cotangents of their band powers,
    /// CIC and functionals (rows of `cps`, `cpdf`, `cmf` for these members);
    /// the seconds of the first two and of the third.  Frees the members.
    pub fn reverse(&mut self, kfull: &[f64], windows: &[Vec<f64>], cic: &Cic, cps: &[f64], cpdf: &[f64], cmf: &[f64]) -> (Vec<f64>, [f64; 2]) {
        let gpu = &*self.gpu;
        let (nb, bins, nm) = (windows.len(), cic.bins, 3 * self.mk.k);
        let npix = self.sht.npix;
        let n = npix as f64;
        let mut out = Vec::with_capacity(self.members.len() * npix);
        let mut secs = [0.0; 2];
        for (m, mem) in std::mem::take(&mut self.members).into_iter().enumerate() {
            let t = std::time::Instant::now();
            let uploaded;
            let x = match &mem.x {
                Some(x) => x,
                None => {
                    uploaded = gpu.upload(&kfull[m * npix..(m + 1) * npix]);
                    &uploaded
                }
            };
            let dcl = window_cotangent(windows, &cps[m * nb..(m + 1) * nb], self.sht.lmax);
            let mut c = self.sht.power_vjp_coefficients(gpu, &mem.alm, &dcl);
            let mut hk = cic_vjp(gpu, x, cic.lo, cic.dx(), &cpdf[m * bins..(m + 1) * bins], n * cic.dx() * cic.dx());
            gpu.sync();
            secs[0] += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let au = self.alm_u(&mem);
            let (u, d) = match mem.mk {
                Some(s) => (s.u, s.d),
                None => (standardize(gpu, x, mem.mean, mem.sigma), self.sht.derivatives(gpu, &au)),
            };
            let (gu, gd) = minkowski_vjp(gpu, &self.mk, &u, &d, mem.floor, &cmf[m * nm..(m + 1) * nm]);
            drop(d);
            // The cotangent a of a(u); g = gu + Y(Omega a / mult) goes through
            // (g - mean(g) - u mean(g u)) / sigma: the synthesis joins that of
            // the C_l, the two means of its part are a_lm dot products.
            let a = self.sht.derivatives_vjp(gpu, &gd);
            drop(gd);
            let mg = (pixel_sum(gpu, gpu.a(&gu), stl_gpu::device::NULL, npix, 0.0, 0) + self.sht.analysis_vjp_dot(gpu, &self.one, &a)) / n;
            let mu = (pixel_sum(gpu, gpu.a(&gu), gpu.a(&u), npix, 0.0, 3) + self.sht.analysis_vjp_dot(gpu, &au, &a)) / n;
            self.sht.add_analysis_vjp_coefficients(gpu, &mut c, &a, 1.0 / mem.sigma);
            let g3 = minkowski_final(gpu, &gu, &u, mg, mu, mem.sigma);
            add_product(gpu, &mut hk, &g3, None);
            add_product(gpu, &mut hk, &self.sht.synthesis(gpu, &c), None);
            out.extend(gpu.download(&hk));
            secs[1] += t.elapsed().as_secs_f64();
        }
        (out, secs)
    }
}
