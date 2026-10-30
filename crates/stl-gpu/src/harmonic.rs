//! Spherical harmonic transforms of full-sky HEALPix maps on the device
//! (port of `stl_core::sht`, same conventions, packing and NESTED maps) and
//! the pixel kernels of the harmonic terms of the ensemble objective
//! (`stl-cli/src/ensemble/fields.rs`: CIC PDF, Minkowski functionals).
//!
//! The Legendre sums run on the device (`harmonic.cu`: one thread per
//! (m, ring pair) for the synthesis, one block per (m, chunk of rings) for
//! its transpose), the ring FFTs through cuFFT (one batched plan per ring
//! length: the polar pairs and the equatorial rings).  Complex buffers are
//! `CudaSlice<f64>` of interleaved (re, im) pairs.  The orders of the sums
//! and the differences with the host are listed at the top of
//! `harmonic.cu`: the results agree with `stl_core::sht` to rounding.

use crate::cufft::{cufft, CuFft, Handle, FORWARD, INVERSE};
use crate::device::{Arg, Gpu, NULL};
use cudarc::driver::CudaSlice;
use stl_core::sht::{Sht, LIMIT, LN_SCALE, SCALE};

/// Pixels per partial sum of [`pixel_sum`] and of the Minkowski functionals.
pub const CHUNK: usize = 1 << 16;

/// Ring pairs per block of the transposed Legendre sums (`SHT_T * SHT_R`).
const ADJ_RINGS: usize = 256 * 4;

/// Legendre sums `[m][ring]` (interleaved complex): A, and B, C for the
/// derivatives.
struct Sums {
    a: CudaSlice<f64>,
    b: Option<CudaSlice<f64>>,
    c: Option<CudaSlice<f64>>,
}

/// One batched cuFFT plan: the rings of one length.
struct FftClass {
    plan: Handle,
    /// First pixel (RING order) of the first ring of the batch.
    first: usize,
}

pub struct GpuSht {
    pub nside: usize,
    pub lmax: usize,
    pub npix: usize,
    pub nalm: usize,
    nr: usize,
    pairs: usize,
    max_nphi: usize,
    cs: CudaSlice<f64>,
    sn: CudaSlice<f64>,
    lnsin: CudaSlice<f64>,
    phi0: CudaSlice<f64>,
    start: CudaSlice<i64>,
    nphi: CudaSlice<i32>,
    nest: CudaSlice<u32>,
    log_mm: CudaSlice<f64>,
    coef: CudaSlice<f64>,
    fft: &'static CuFft,
    classes: Vec<FftClass>,
    _work: CudaSlice<u8>,
    chunks: usize,
    area: f64,
}

impl Drop for GpuSht {
    fn drop(&mut self) {
        for c in &self.classes {
            self.fft.destroy(c.plan);
        }
    }
}

impl GpuSht {
    /// The transforms of `sht` on `gpu`: ring tables, recursion
    /// coefficients and one cuFFT plan per ring length.
    pub fn new(gpu: &Gpu, sht: &Sht) -> Result<Self, String> {
        let fft = cufft()?;
        let (nside, lmax, npix) = (sht.nside, sht.lmax, sht.npix);
        let nr = sht.rings.len();
        let pairs = nr / 2;
        let nalm = sht.nalm();
        let f = |g: fn(&stl_core::sht::Ring) -> f64| gpu.upload(&sht.rings.iter().map(g).collect::<Vec<f64>>());
        let cs = f(|r| r.cos);
        let sn = f(|r| r.sin);
        let lnsin = f(|r| r.sin.ln());
        let phi0 = f(|r| r.phi0);
        let start = gpu.upload(&sht.rings.iter().map(|r| r.start as i64).collect::<Vec<_>>());
        let nphi = gpu.upload(&sht.rings.iter().map(|r| r.nphi as i32).collect::<Vec<_>>());
        let max_nphi = sht.rings.iter().map(|r| r.nphi).max().unwrap_or(0);
        let nest = gpu.upload(sht.nest_of_ring());
        let log_mm = gpu.upload(sht.log_mm());
        let coef = gpu.zeros::<f64>(3 * nalm);
        gpu.launch("sht_coef", ((lmax + 1).div_ceil(256) as u32, (lmax + 1) as u32), 256, &[lmax.into(), gpu.a(&coef)]);
        // Batches of rings of one length: the polar pairs (north i, south
        // 4 nside - i), then the 2 nside + 1 equatorial rings, contiguous.
        let mut batches: Vec<(usize, usize, usize, usize)> = Vec::new(); // (length, batch, first, distance)
        for i in 1..nside {
            let (n, s) = (&sht.rings[i - 1], &sht.rings[nr - i]);
            batches.push((n.nphi, 2, n.start, s.start - n.start));
        }
        let eq = &sht.rings[nside - 1];
        batches.push((eq.nphi, 2 * nside + 1, eq.start, eq.nphi));
        gpu.ctx.bind_to_thread().map_err(|e| format!("CUDA: {e:?}"))?;
        let stream = gpu.stream.cu_stream() as *mut std::ffi::c_void;
        let mut classes = Vec::with_capacity(batches.len());
        let mut work_bytes = 0usize;
        for &(len, batch, first, dist) in &batches {
            let (plan, bytes) = fft.plan_z2z(len, batch, dist, stream)?;
            work_bytes = work_bytes.max(bytes);
            classes.push(FftClass { plan, first });
        }
        let work = gpu.zeros::<u8>(work_bytes.max(16));
        let wp = gpu.ptr(&work);
        for c in &classes {
            unsafe { fft.set_work_area(c.plan, wp)? };
        }
        Ok(GpuSht {
            nside,
            lmax,
            npix,
            nalm,
            nr,
            pairs,
            max_nphi,
            cs,
            sn,
            lnsin,
            phi0,
            start,
            nphi,
            nest,
            log_mm,
            coef,
            fft,
            classes,
            _work: work,
            chunks: (pairs + 1).div_ceil(ADJ_RINGS),
            area: sht.area(),
        })
    }

    /// Constants of the scaled recursion, as `Sht::recurse`.
    fn rc(&self) -> [Arg; 5] {
        let ln2 = std::f64::consts::LN_2;
        [(-650.0 * ln2).into(), (300.0 * ln2).into(), LN_SCALE.into(), SCALE.into(), LIMIT.into()]
    }

    fn opt(&self, gpu: &Gpu, s: &Option<CudaSlice<f64>>) -> Arg {
        s.as_ref().map_or(NULL, |v| gpu.a(v))
    }

    /// Legendre sums of `alm` (B, C when `deriv`).
    fn legendre(&self, gpu: &Gpu, alm: &CudaSlice<f64>, deriv: bool) -> Sums {
        assert_eq!(alm.len(), 2 * self.nalm);
        let n = 2 * (self.lmax + 1) * self.nr;
        let sums = Sums { a: gpu.zeros(n), b: deriv.then(|| gpu.zeros(n)), c: deriv.then(|| gpu.zeros(n)) };
        let mut args = vec![
            gpu.a(alm),
            self.lmax.into(),
            gpu.a(&self.coef),
            gpu.a(&self.log_mm),
            gpu.a(&self.cs),
            gpu.a(&self.sn),
            gpu.a(&self.lnsin),
            self.nr.into(),
            self.pairs.into(),
            (deriv as i32).into(),
        ];
        args.extend(self.rc());
        args.extend([gpu.a(&sums.a), self.opt(gpu, &sums.b), self.opt(gpu, &sums.c)]);
        gpu.launch("sht_leg_syn", ((self.pairs + 1).div_ceil(128) as u32, (self.lmax + 1) as u32), 128, &args);
        sums
    }

    /// Transpose of [`legendre`](Self::legendre): the packed `a_lm`.
    fn legendre_adjoint(&self, gpu: &Gpu, sums: &Sums) -> CudaSlice<f64> {
        let deriv = sums.b.is_some();
        let part = gpu.zeros::<f64>(2 * self.nalm * self.chunks);
        let mut args = vec![
            gpu.a(&sums.a),
            self.opt(gpu, &sums.b),
            self.opt(gpu, &sums.c),
            (deriv as i32).into(),
            self.lmax.into(),
            gpu.a(&self.coef),
            gpu.a(&self.log_mm),
            gpu.a(&self.cs),
            gpu.a(&self.sn),
            gpu.a(&self.lnsin),
            self.nr.into(),
            self.pairs.into(),
        ];
        args.extend(self.rc());
        args.extend([gpu.a(&part), self.nalm.into()]);
        gpu.launch("sht_leg_adj", ((self.lmax + 1) as u32, self.chunks as u32), 256, &args);
        if self.chunks == 1 {
            return part;
        }
        let alm = gpu.zeros::<f64>(2 * self.nalm);
        gpu.launch1("sht_chunk_sum", self.nalm, &[gpu.a(&part), self.nalm.into(), (self.chunks as i32).into(), gpu.a(&alm)]);
        alm
    }

    /// In-place FFTs of every ring of the `k` complex maps `h` (RING order).
    fn fft(&self, gpu: &Gpu, h: &CudaSlice<f64>, k: usize, inverse: bool) {
        gpu.ctx.bind_to_thread().unwrap();
        let base = gpu.ptr(h);
        let dir = if inverse { INVERSE } else { FORWARD };
        for o in 0..k {
            for c in &self.classes {
                let p = base + ((o * self.npix + c.first) * 16) as u64;
                unsafe { self.fft.exec_z2z(c.plan, p, dir) }.unwrap_or_else(|e| panic!("{e}"));
            }
        }
    }

    /// Maps of the Legendre sums (mode 0: the synthesis, one map; mode 1:
    /// the five derivative maps), `[k, npix]` NESTED.
    fn synthesise(&self, gpu: &Gpu, sums: &Sums, mode: i32) -> CudaSlice<f64> {
        let k = if mode == 0 { 1 } else { 5 };
        let h = gpu.zeros::<f64>(2 * k * self.npix);
        gpu.launch(
            "sht_ring_syn",
            (self.max_nphi.div_ceil(128) as u32, self.nr as u32),
            128,
            &[
                gpu.a(&sums.a),
                self.opt(gpu, &sums.b),
                self.opt(gpu, &sums.c),
                mode.into(),
                self.lmax.into(),
                self.nr.into(),
                gpu.a(&self.start),
                gpu.a(&self.nphi),
                gpu.a(&self.phi0),
                gpu.a(&self.cs),
                gpu.a(&self.sn),
                gpu.a(&h),
                self.npix.into(),
            ],
        );
        self.fft(gpu, &h, k, true);
        let out = gpu.zeros::<f64>(k * self.npix);
        gpu.launch1("sht_ring_store", k * self.npix, &[gpu.a(&h), k.into(), self.npix.into(), gpu.a(&self.nest), gpu.a(&out)]);
        out
    }

    /// Legendre sums of the transposed synthesis of the maps `x` `[k, npix]`
    /// (mode 0: one map; mode 1: the five derivative cotangents).
    fn analyse(&self, gpu: &Gpu, x: &CudaSlice<f64>, mode: i32, with_mult: bool) -> Sums {
        let k = if mode == 0 { 1 } else { 5 };
        assert_eq!(x.len(), k * self.npix);
        let h = gpu.zeros::<f64>(2 * k * self.npix);
        gpu.launch1("sht_ring_load", k * self.npix, &[gpu.a(x), k.into(), self.npix.into(), gpu.a(&self.nest), gpu.a(&h)]);
        self.fft(gpu, &h, k, false);
        let n = 2 * (self.lmax + 1) * self.nr;
        let deriv = mode == 1;
        let sums = Sums { a: gpu.zeros(n), b: deriv.then(|| gpu.zeros(n)), c: deriv.then(|| gpu.zeros(n)) };
        gpu.launch(
            "sht_ring_ana",
            ((self.lmax + 1).div_ceil(128) as u32, self.nr as u32),
            128,
            &[
                gpu.a(&h),
                mode.into(),
                (with_mult as i32).into(),
                self.lmax.into(),
                self.nr.into(),
                gpu.a(&self.start),
                gpu.a(&self.nphi),
                gpu.a(&self.phi0),
                gpu.a(&self.cs),
                gpu.a(&self.sn),
                self.npix.into(),
                gpu.a(&sums.a),
                self.opt(gpu, &sums.b),
                self.opt(gpu, &sums.c),
            ],
        );
        sums
    }

    fn scale_m(&self, gpu: &Gpu, alm: &mut CudaSlice<f64>, f0: f64, f1: f64) {
        gpu.launch("sht_scale_m", ((self.lmax + 1).div_ceil(256) as u32, (self.lmax + 1) as u32), 256, &[gpu.a(alm), self.lmax.into(), f0.into(), f1.into()]);
    }

    /// `Y a`: the map of `alm` (`Sht::synthesis`).
    pub fn synthesis(&self, gpu: &Gpu, alm: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.synthesise(gpu, &self.legendre(gpu, alm, false), 0)
    }

    /// `Y* x` (`Sht::adjoint_synthesis`).
    pub fn adjoint_synthesis(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.legendre_adjoint(gpu, &self.analyse(gpu, x, 0, false))
    }

    /// `Omega Y* x` (`Sht::analysis`).
    pub fn analysis(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> CudaSlice<f64> {
        let mut alm = self.adjoint_synthesis(gpu, x);
        self.scale_m(gpu, &mut alm, self.area, self.area);
        alm
    }

    /// `Omega Y(c / mult)` (`Sht::analysis_vjp`).
    pub fn analysis_vjp(&self, gpu: &Gpu, c: &CudaSlice<f64>) -> CudaSlice<f64> {
        let mut scaled = gpu.copy(c);
        self.scale_m(gpu, &mut scaled, self.area, self.area / 2.0);
        self.synthesis(gpu, &scaled)
    }

    /// `C_l` of `alm` (`Sht::power`), on the host.
    pub fn power(&self, gpu: &Gpu, alm: &CudaSlice<f64>) -> Vec<f64> {
        let cl = gpu.zeros::<f64>(self.lmax + 1);
        gpu.launch1("sht_power", self.lmax + 1, &[gpu.a(alm), self.lmax.into(), gpu.a(&cl)]);
        gpu.download(&cl)
    }

    /// Map cotangent of the analysis whose `a_lm` gave the `C_l`, from the
    /// cotangent `dcl` of the `C_l` (`Sht::power_vjp` then `analysis_vjp`).
    pub fn power_vjp_map(&self, gpu: &Gpu, alm: &CudaSlice<f64>, dcl: &[f64]) -> CudaSlice<f64> {
        assert_eq!(dcl.len(), self.lmax + 1);
        let d = gpu.upload(dcl);
        let c = gpu.zeros::<f64>(2 * self.nalm);
        gpu.launch(
            "sht_power_vjp",
            ((self.lmax + 1).div_ceil(256) as u32, (self.lmax + 1) as u32),
            256,
            &[gpu.a(alm), self.lmax.into(), gpu.a(&d), self.area.into(), (self.area / 2.0).into(), gpu.a(&c)],
        );
        self.synthesis(gpu, &c)
    }

    /// The five covariant derivative maps `[5, npix]` (`Sht::derivatives`).
    pub fn derivatives(&self, gpu: &Gpu, alm: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.synthesise(gpu, &self.legendre(gpu, alm, true), 1)
    }

    /// Cotangent of the `a_lm` from those of the derivative maps `[5, npix]`
    /// (`Sht::derivatives_vjp`).
    pub fn derivatives_vjp(&self, gpu: &Gpu, cot: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.legendre_adjoint(gpu, &self.analyse(gpu, cot, 1, true))
    }
}

/// Sum over the `n` pixels at `a` (and `b`; device addresses, see
/// [`Gpu::at`]), chunks of [`CHUNK`] in a fixed tree, then the chunks in
/// order: mode 0 `sum a`, 1 `sum (a - shift)^2`, 2 `sum a^2 + b^2`, 3 `sum a b`.
pub fn pixel_sum(gpu: &Gpu, a: Arg, b: Arg, n: usize, shift: f64, mode: i32) -> f64 {
    let chunks = n.div_ceil(CHUNK);
    let out = gpu.zeros::<f64>(chunks);
    gpu.launch("px_sums", (chunks as u32, 1), 256, &[a, b, n.into(), CHUNK.into(), shift.into(), mode.into(), gpu.a(&out)]);
    gpu.download(&out).iter().sum()
}

/// `(x - mean) / sigma`.
pub fn standardize(gpu: &Gpu, x: &CudaSlice<f64>, mean: f64, sigma: f64) -> CudaSlice<f64> {
    let u = gpu.zeros::<f64>(x.len());
    gpu.launch1("mk_standard", x.len(), &[gpu.a(x), x.len().into(), mean.into(), sigma.into(), gpu.a(&u)]);
    u
}

/// The parameters of the soft Minkowski functionals (`Minkowski` of stl-cli).
pub struct MinkowskiParams {
    pub nu: CudaSlice<f64>,
    pub k: usize,
    pub eps: f64,
}

/// Sums over the pixels of the three functionals at every threshold,
/// `[3, k]` (not yet divided by the pixel count), from the standardised
/// map `u` and the derivative maps `d` `[5, n]`.
pub fn minkowski_sums(gpu: &Gpu, p: &MinkowskiParams, u: &CudaSlice<f64>, d: &CudaSlice<f64>, floor: f64) -> Vec<f64> {
    let n = u.len();
    let (mag, curv) = (gpu.zeros::<f64>(n), gpu.zeros::<f64>(n));
    gpu.launch1("mk_geom", n, &[gpu.a(d), n.into(), floor.into(), gpu.a(&mag), gpu.a(&curv)]);
    let chunks = n.div_ceil(CHUNK);
    let out = gpu.zeros::<f64>(chunks * p.k * 3);
    let tau = 2.0 * std::f64::consts::PI;
    let cn = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * p.eps);
    gpu.launch(
        "mk_forward",
        (chunks as u32, p.k as u32),
        256,
        &[gpu.a(u), gpu.a(&mag), gpu.a(&curv), n.into(), CHUNK.into(), gpu.a(&p.nu), p.eps.into(), cn.into(), tau.into(), gpu.a(&out)],
    );
    let h = gpu.download(&out);
    let mut v = vec![0.0; 3 * p.k];
    for c in 0..chunks {
        for t in 0..p.k {
            for f in 0..3 {
                v[f * p.k + t] += h[(c * p.k + t) * 3 + f];
            }
        }
    }
    v
}

/// Per-pixel cotangents of the functionals (cotangent `cot` `[3, k]`): of
/// `u` (`[n]`, divided by n) and of the derivative maps (`[5, n]`).
pub fn minkowski_vjp(gpu: &Gpu, p: &MinkowskiParams, u: &CudaSlice<f64>, d: &CudaSlice<f64>, floor: f64, cot: &[f64]) -> (CudaSlice<f64>, CudaSlice<f64>) {
    let n = u.len();
    assert_eq!(cot.len(), 3 * p.k);
    let c = gpu.upload(cot);
    let (gu, gd) = (gpu.zeros::<f64>(n), gpu.zeros::<f64>(5 * n));
    let tau = 2.0 * std::f64::consts::PI;
    let cn = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * p.eps);
    gpu.launch1(
        "mk_vjp",
        n,
        &[gpu.a(u), gpu.a(d), n.into(), floor.into(), gpu.a(&p.nu), (p.k as i32).into(), p.eps.into(), cn.into(), tau.into(), gpu.a(&c), gpu.a(&gu), gpu.a(&gd)],
    );
    (gu, gd)
}

/// `(g - mg - u mu) / sigma`, the cotangent through the standardisation.
pub fn minkowski_final(gpu: &Gpu, g: &CudaSlice<f64>, u: &CudaSlice<f64>, mg: f64, mu: f64, sigma: f64) -> CudaSlice<f64> {
    let out = gpu.zeros::<f64>(g.len());
    gpu.launch1("mk_final", g.len(), &[gpu.a(g), gpu.a(u), g.len().into(), mg.into(), mu.into(), sigma.into(), gpu.a(&out)]);
    out
}

/// Pixels per chunk of the CIC histogram (`CHUNK` of stl-cli's fields).
pub const CIC_CHUNK: usize = 1 << 14;

/// CIC histogram of `x` on `bins <= 256` centres from `lo` by `dx` (chunks
/// of [`CIC_CHUNK`] in pixel order, added in chunk order: the host's sums).
pub fn cic_histogram(gpu: &Gpu, x: &CudaSlice<f64>, lo: f64, dx: f64, bins: usize) -> Vec<f64> {
    assert!(bins <= 256);
    let n = x.len();
    let chunks = n.div_ceil(CIC_CHUNK).max(1);
    let out = gpu.zeros::<f64>(chunks * bins);
    gpu.launch1("cic_partial", chunks, &[gpu.a(x), n.into(), CIC_CHUNK.into(), lo.into(), dx.into(), (bins as i32).into(), gpu.a(&out)]);
    let h = gpu.download(&out);
    let mut v = vec![0.0; bins];
    for part in h.chunks(bins) {
        v.iter_mut().zip(part).for_each(|(a, b)| *a += b);
    }
    v
}

/// Map cotangent of the CIC density.
pub fn cic_vjp(gpu: &Gpu, x: &CudaSlice<f64>, lo: f64, dx: f64, cot: &[f64], norm: f64) -> CudaSlice<f64> {
    let c = gpu.upload(cot);
    let g = gpu.zeros::<f64>(x.len());
    gpu.launch1("cic_vjp", x.len(), &[gpu.a(x), x.len().into(), lo.into(), dx.into(), (cot.len() as i32).into(), gpu.a(&c), norm.into(), gpu.a(&g)]);
    g
}
