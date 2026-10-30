//! The harmonic terms of the objective on the device (`--harmonic gpu`):
//! the C_l bands, the CIC PDF and the Minkowski functionals of this
//! group's physical maps through `stl_gpu::harmonic`, with the features
//! and map cotangents of `fields.rs` to rounding (orders of the sums at the
//! top of `stl-gpu/src/harmonic.cu`; the CIC histogram and the C_l are the
//! host's sums).  The maps go up once per evaluation, their cotangents
//! come down once; the Minkowski states stay on the device between the
//! passes (recomputed above `MF_STATE_BYTES`, as on the host).

use super::fields::{window_cotangent, window_sums, Cic, Minkowski};
use std::sync::Arc;
use stl_core::sht::Sht;
use stl_gpu::harmonic::{cic_histogram, cic_vjp, minkowski_final, minkowski_sums, minkowski_vjp, pixel_sum, standardize, GpuSht, MinkowskiParams};
use stl_gpu::terms::add_product;
use stl_gpu::{CudaSlice, Gpu};

/// Device memory for the Minkowski states of a group's members (u and the
/// five derivative maps) between the passes.
const MF_STATE_BYTES: usize = 4 << 30;

/// What the cotangent of the functionals needs of the forward pass.
struct MkState {
    u: CudaSlice<f64>,
    d: CudaSlice<f64>,
    sigma: f64,
    floor: f64,
}

struct Member {
    /// The physical map.
    x: CudaSlice<f64>,
    /// Its analysis (C_l).
    alm: CudaSlice<f64>,
    mk: Option<MkState>,
}

pub struct GpuTerms {
    gpu: Arc<Gpu>,
    sht: GpuSht,
    mk: MinkowskiParams,
    members: Vec<Member>,
}

impl GpuTerms {
    pub fn new(gpu: Arc<Gpu>, sht: &Sht, mk: &Minkowski) -> Result<Self, String> {
        let dev = GpuSht::new(&gpu, sht)?;
        let params = MinkowskiParams { nu: gpu.upload(&mk.nu), k: mk.nu.len(), eps: mk.eps };
        Ok(GpuTerms { gpu, sht: dev, mk: params, members: Vec::new() })
    }

    /// `Minkowski::forward` on the device: the functionals `[3, k]` and the state.
    fn minkowski(&self, x: &CudaSlice<f64>) -> (Vec<f64>, MkState) {
        let gpu = &*self.gpu;
        let npix = x.len();
        let n = npix as f64;
        let mean = pixel_sum(gpu, gpu.a(x), stl_gpu::device::NULL, npix, 0.0, 0) / n;
        let sigma = (pixel_sum(gpu, gpu.a(x), stl_gpu::device::NULL, npix, mean, 1) / n + 1e-12).sqrt();
        let u = standardize(gpu, x, mean, sigma);
        let d = self.sht.derivatives(gpu, &self.sht.analysis(gpu, &u));
        let floor = 1e-4 * pixel_sum(gpu, gpu.at(&d, 0), gpu.at(&d, npix), npix, 0.0, 2) / n + 1e-12;
        let mut v = minkowski_sums(gpu, &self.mk, &u, &d, floor);
        v.iter_mut().for_each(|a| *a /= n);
        (v, MkState { u, d, sigma, floor })
    }

    /// Features of the members' whole physical maps `kfull` `[local, npix]`:
    /// band powers `[local, nb]`, CIC `[local, bins]`, Minkowski `[local, 3 k]`;
    /// the seconds of the first two and of the third.
    pub fn forward(&mut self, kfull: &[f64], windows: &[Vec<f64>], cic: &Cic) -> (Vec<f64>, Vec<f64>, Vec<f64>, [f64; 2]) {
        let npix = self.sht.npix;
        let local = kfull.len() / npix;
        let keep = local * npix * 6 * 8 <= MF_STATE_BYTES;
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
            let norm = npix as f64 * cic.dx();
            hist.extend(h.into_iter().map(|v| v / norm));
            secs[0] += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let (v, state) = self.minkowski(&x);
            mfv.extend(v);
            secs[1] += t.elapsed().as_secs_f64();
            self.members.push(Member { x, alm, mk: keep.then_some(state) });
        }
        (pw, hist, mfv, secs)
    }

    /// Map cotangents `[local, npix]` of the members of the last forward
    /// pass from the cotangents of their band powers, CIC and functionals
    /// (rows of `cps`, `cpdf`, `cmf` for these members); the seconds of the
    /// first two and of the third.
    pub fn reverse(&self, windows: &[Vec<f64>], cic: &Cic, cps: &[f64], cpdf: &[f64], cmf: &[f64]) -> (Vec<f64>, [f64; 2]) {
        let gpu = &*self.gpu;
        let (nb, bins, nm) = (windows.len(), cic.bins, 3 * self.mk.k);
        let npix = self.sht.npix;
        let n = npix as f64;
        let mut out = Vec::with_capacity(self.members.len() * npix);
        let mut secs = [0.0; 2];
        for (m, mem) in self.members.iter().enumerate() {
            let t = std::time::Instant::now();
            let dcl = window_cotangent(windows, &cps[m * nb..(m + 1) * nb], self.sht.lmax);
            let mut hk = self.sht.power_vjp_map(gpu, &mem.alm, &dcl);
            let g2 = cic_vjp(gpu, &mem.x, cic.lo, cic.dx(), &cpdf[m * bins..(m + 1) * bins], n * cic.dx() * cic.dx());
            add_product(gpu, &mut hk, &g2, None);
            gpu.sync();
            secs[0] += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let recomputed;
            let s = match &mem.mk {
                Some(s) => s,
                None => {
                    recomputed = self.minkowski(&mem.x).1;
                    &recomputed
                }
            };
            let (mut gu, gd) = minkowski_vjp(gpu, &self.mk, &s.u, &s.d, s.floor, &cmf[m * nm..(m + 1) * nm]);
            let back = self.sht.analysis_vjp(gpu, &self.sht.derivatives_vjp(gpu, &gd));
            add_product(gpu, &mut gu, &back, None);
            let mg = pixel_sum(gpu, gpu.a(&gu), stl_gpu::device::NULL, npix, 0.0, 0) / n;
            let mu = pixel_sum(gpu, gpu.a(&gu), gpu.a(&s.u), npix, 0.0, 3) / n;
            let g3 = minkowski_final(gpu, &gu, &s.u, mg, mu, s.sigma);
            add_product(gpu, &mut hk, &g3, None);
            out.extend(gpu.download(&hk));
            secs[1] += t.elapsed().as_secs_f64();
        }
        (out, secs)
    }
}
