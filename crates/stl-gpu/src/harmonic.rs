//! Spherical harmonic transforms of full-sky HEALPix maps on the device
//! (port of `stl_core::sht`, same conventions, packing and NESTED maps) and
//! the pixel kernels of the harmonic terms of the ensemble objective
//! (`stl-cli/src/ensemble/fields.rs`: CIC PDF, Minkowski functionals).
//!
//! The Legendre sums run on the device in one of two ways (`harmonic.cu`):
//! - as GEMMs (the default when cuBLAS loads, `cublas.rs`): for a block of
//!   m, a table of lambda_lm per parity of l - m is built once and shared
//!   by every map of a batch (`*_many`) and every sum of the transform, one
//!   DGEMM per (m, parity); blocks of at most 32 m whose tables take at most
//!   `STL_SHT_TABLE_MB` (default 1024);
//! - with `STL_SHT=kernels` (or without cuBLAS), one map at a time: one
//!   thread per (m, ring pair) for the synthesis, one block per (m, chunk
//!   of rings) for its transpose.
//!
//! The ring FFTs go through cuFFT: one batched plan per ring length (the
//! polar pairs and the equatorial rings), and for the five derivative maps
//! of a member one plan per polar ring length over the five maps.  Complex buffers are
//! `CudaSlice<f64>` of interleaved (re, im) pairs.  The orders of the sums
//! and the differences with the host are listed in `harmonic.cu`: the
//! results agree with `stl_core::sht` to rounding.

use crate::cublas::{cublas, CuBlas, OP_N, OP_T};
use crate::cufft::{cufft, CuFft, Handle, FORWARD, INVERSE};
use crate::device::{Arg, Gpu, NULL};
use crate::terms::copy_rows;
use cudarc::driver::CudaSlice;
use stl_core::sht::{Sht, LIMIT, LN_SCALE, SCALE};

/// Pixels per partial sum of [`pixel_sum`] and of the Minkowski functionals.
pub const CHUNK: usize = 1 << 16;

/// Ring pairs per block of the transposed Legendre sums (`SHT_T * SHT_R`).
const ADJ_RINGS: usize = 256 * 4;

/// At most this many m per block of the GEMM Legendre sums.
const GEMM_M: usize = 32;

/// Legendre sums `[rows][m][ring]` (interleaved complex) of `rows` maps: A,
/// and B, C for the derivatives.
struct Sums {
    a: CudaSlice<f64>,
    b: Option<CudaSlice<f64>>,
    c: Option<CudaSlice<f64>>,
    rows: usize,
}

/// One batched cuFFT plan: the rings of one length.
struct FftClass {
    plan: Handle,
    /// First pixel (RING order) of the first ring of the batch.
    first: usize,
}

/// One polar ring length for the five derivative maps of a member at once
/// (five transforms `npix` apart), run on the north and the south ring.
struct FftRings {
    plan: Handle,
    north: usize,
    south: usize,
}

/// Rows `l = m + q + 2 j <= lmax` of parity `q`.
fn leg_rows(lmax: usize, m: usize, q: usize) -> usize {
    if q == 1 {
        (lmax - m + 1) / 2
    } else {
        (lmax - m) / 2 + 1
    }
}

/// A block of m of the GEMM Legendre sums: the table rows before each
/// (m, parity), on the host and on the device.
struct MBlock {
    m0: usize,
    m1: usize,
    cum: Vec<usize>,
    cum_dev: CudaSlice<i64>,
    rows: usize,
}

/// The Legendre sums as one DGEMM per (m, parity).
struct Gemm {
    blas: &'static CuBlas,
    handle: crate::cublas::Handle,
    blocks: Vec<MBlock>,
    /// The table of one block, `[rows, P]`.
    table: CudaSlice<f64>,
    max_rows: usize,
    max_m: usize,
}

impl Gemm {
    fn new(gpu: &Gpu, lmax: usize, p: usize) -> Result<Self, String> {
        let blas = cublas()?;
        gpu.ctx.bind_to_thread().map_err(|e| format!("CUDA: {e:?}"))?;
        let handle = blas.handle(gpu.stream.cu_stream() as *mut std::ffi::c_void)?;
        let budget = std::env::var("STL_SHT_TABLE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024) << 20;
        let mut blocks = Vec::new();
        let mut m = 0;
        while m <= lmax {
            let m0 = m;
            let (mut rows, mut cum) = (0usize, Vec::new());
            while m <= lmax && (m == m0 || (m - m0 < GEMM_M && (rows + lmax + 1 - m) * p * 8 <= budget)) {
                for q in 0..2 {
                    cum.push(rows);
                    rows += leg_rows(lmax, m, q);
                }
                m += 1;
            }
            let dev: Vec<i64> = cum.iter().map(|&c| c as i64).collect();
            blocks.push(MBlock { m0, m1: m, cum_dev: gpu.upload(&dev), cum, rows });
        }
        let max_rows = blocks.iter().map(|b| b.rows).max().unwrap_or(1);
        let max_m = blocks.iter().map(|b| b.m1 - b.m0).max().unwrap_or(1);
        Ok(Gemm { blas, handle, blocks, table: gpu.zeros(max_rows * p), max_rows, max_m })
    }
}

/// Spherical harmonic transforms of full-sky NESTED maps at `nside` up to
/// `lmax` on the device (the transforms of a `stl_core::sht::Sht`).
pub struct GpuSht {
    /// Resolution of the maps.
    pub nside: usize,
    /// Largest multipole.
    pub lmax: usize,
    /// Pixels per map (`12 nside^2`).
    pub npix: usize,
    /// Number of packed `a_lm`, `m <= l <= lmax` (`(lmax + 1)(lmax + 2) / 2`).
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
    rings5: Vec<FftRings>,
    _work: CudaSlice<u8>,
    chunks: usize,
    area: f64,
    gemm: Option<Gemm>,
    /// Seconds of the phases of the transforms when `STL_SHT_PHASES` is set
    /// (each phase then ends with a synchronisation): Legendre tables, GEMMs,
    /// packing of their columns, ring FFTs.
    phases: Option<std::cell::RefCell<[f64; 4]>>,
}

/// Names of the phases timed in `GpuSht::phases`.
pub const PHASES: [&str; 4] = ["legendre_tables", "legendre_gemms", "legendre_columns", "ring_ffts"];

impl Drop for GpuSht {
    fn drop(&mut self) {
        for c in &self.classes {
            self.fft.destroy(c.plan);
        }
        for c in &self.rings5 {
            self.fft.destroy(c.plan);
        }
        if let Some(g) = &self.gemm {
            g.blas.destroy(g.handle);
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
        let mut rings5 = Vec::with_capacity(nside.saturating_sub(1));
        for i in 1..nside {
            let (n, s) = (&sht.rings[i - 1], &sht.rings[nr - i]);
            let (plan, bytes) = fft.plan_z2z(n.nphi, 5, npix, stream)?;
            work_bytes = work_bytes.max(bytes);
            rings5.push(FftRings { plan, north: n.start, south: s.start });
        }
        let work = gpu.zeros::<u8>(work_bytes.max(16));
        let wp = gpu.ptr(&work);
        for c in classes.iter().map(|c| c.plan).chain(rings5.iter().map(|c| c.plan)) {
            unsafe { fft.set_work_area(c, wp)? };
        }
        let gemm = if std::env::var("STL_SHT").as_deref() == Ok("kernels") {
            None
        } else {
            match Gemm::new(gpu, lmax, pairs + 1) {
                Ok(g) => Some(g),
                Err(e) => {
                    eprintln!("stl-gpu: Legendre sums without GEMMs ({e})");
                    None
                }
            }
        };
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
            rings5,
            _work: work,
            chunks: (pairs + 1).div_ceil(ADJ_RINGS),
            area: sht.area(),
            gemm,
            phases: std::env::var_os("STL_SHT_PHASES").map(|_| Default::default()),
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

    fn opt_at(&self, gpu: &Gpu, s: &Option<CudaSlice<f64>>, offset: usize) -> Arg {
        s.as_ref().map_or(NULL, |v| gpu.at(v, offset))
    }

    /// Whether the Legendre sums run as GEMMs.
    pub fn uses_gemm(&self) -> bool {
        self.gemm.is_some()
    }

    /// The seconds of the phases since the last call ([`PHASES`]), when
    /// `STL_SHT_PHASES` is set.
    pub fn take_phases(&self) -> Option<[f64; 4]> {
        self.phases.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut()))
    }

    /// Ends phase `i` started at `t` (when timed).
    fn phase(&self, gpu: &Gpu, i: usize, t: &mut std::time::Instant) {
        if let Some(p) = &self.phases {
            gpu.sync();
            p.borrow_mut()[i] += t.elapsed().as_secs_f64();
            *t = std::time::Instant::now();
        }
    }

    /// Doubles of one row of [`Sums`].
    fn row(&self) -> usize {
        2 * (self.lmax + 1) * self.nr
    }

    fn sums(&self, gpu: &Gpu, rows: usize, deriv: bool) -> Sums {
        let n = rows * self.row();
        Sums { a: gpu.zeros(n), b: deriv.then(|| gpu.zeros(n)), c: deriv.then(|| gpu.zeros(n)), rows }
    }

    /// Legendre sums of the `rows` maps whose packed `a_lm` are the rows of
    /// `alms` (`[rows, 2 nalm]`): A, and B, C when `deriv`.
    fn legendre(&self, gpu: &Gpu, alms: &CudaSlice<f64>, rows: usize, deriv: bool) -> Sums {
        assert_eq!(alms.len(), rows * 2 * self.nalm);
        let sums = self.sums(gpu, rows, deriv);
        match &self.gemm {
            Some(g) => self.legendre_gemm(gpu, g, alms, &sums),
            None => {
                for r in 0..rows {
                    let o = r * self.row();
                    let mut args = vec![
                        gpu.at(alms, r * 2 * self.nalm),
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
                    args.extend([gpu.at(&sums.a, o), self.opt_at(gpu, &sums.b, o), self.opt_at(gpu, &sums.c, o)]);
                    gpu.launch("sht_leg_syn", ((self.pairs + 1).div_ceil(128) as u32, (self.lmax + 1) as u32), 128, &args);
                }
            }
        }
        sums
    }

    /// The table of the block `b` in `g.table`.
    fn leg_table(&self, gpu: &Gpu, g: &Gemm, b: &MBlock) {
        let p = self.pairs + 1;
        let mut args = vec![b.m0.into(), self.lmax.into(), gpu.a(&self.coef), gpu.a(&self.log_mm), gpu.a(&self.cs), gpu.a(&self.lnsin), p.into()];
        args.extend(self.rc());
        args.extend([gpu.a(&b.cum_dev), gpu.a(&g.table)]);
        gpu.launch("leg_table", (p.div_ceil(128) as u32, (b.m1 - b.m0) as u32), 128, &args);
    }

    /// `C = op(T) X` of one (m, q) of block `b`: `t` true for the transpose
    /// (adjoint, `[n_q, cols]` from `[P, cols]`), false for the synthesis
    /// (`[P, cols]` from `[n_q, cols]`); `x` holds the `[n_q, cols]` side.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, gpu: &Gpu, g: &Gemm, b: &MBlock, m: usize, q: usize, cols: usize, x: &CudaSlice<f64>, o: &CudaSlice<f64>, t: bool) {
        let p = self.pairs + 1;
        let n = leg_rows(self.lmax, m, q);
        if n == 0 {
            return;
        }
        let c = b.cum[2 * (m - b.m0) + q];
        let table = gpu.ptr(&g.table) + (8 * p * c) as u64;
        let xp = gpu.ptr(x) + (8 * cols * c) as u64;
        let op = gpu.ptr(o) + (8 * (2 * (m - b.m0) + q) * p * cols) as u64;
        let r = unsafe {
            if t {
                g.blas.dgemm(g.handle, OP_T, OP_N, n, cols, p, 1.0, table, p, op, p, 0.0, xp, n)
            } else {
                g.blas.dgemm(g.handle, OP_N, OP_N, p, cols, n, 1.0, table, p, xp, n, 0.0, op, p)
            }
        };
        r.unwrap_or_else(|e| panic!("{e}"));
    }

    fn legendre_gemm(&self, gpu: &Gpu, g: &Gemm, alms: &CudaSlice<f64>, s: &Sums) {
        let kinds = if s.b.is_some() { 4 } else { 1 };
        let p = self.pairs + 1;
        let cols = 2 * s.rows * kinds;
        let x = gpu.zeros::<f64>(g.max_rows * cols);
        let mut o = gpu.zeros::<f64>(2 * g.max_m * p * cols);
        gpu.ctx.bind_to_thread().unwrap();
        let mut t = std::time::Instant::now();
        self.phase(gpu, 2, &mut t);
        for b in &g.blocks {
            let mb = (b.m1 - b.m0) as u32;
            self.leg_table(gpu, g, b);
            self.phase(gpu, 0, &mut t);
            gpu.launch(
                "leg_pack_syn",
                ((self.lmax + 1).div_ceil(256) as u32, mb),
                256,
                &[gpu.a(alms), self.nalm.into(), (s.rows as i32).into(), (kinds as i32).into(), b.m0.into(), self.lmax.into(), gpu.a(&self.coef), gpu.a(&b.cum_dev), gpu.a(&x)],
            );
            // (m, q) without rows keep zeros.
            gpu.clear(&mut o);
            self.phase(gpu, 2, &mut t);
            for m in b.m0..b.m1 {
                for q in 0..2 {
                    self.gemm(gpu, g, b, m, q, cols, &x, &o, false);
                }
            }
            self.phase(gpu, 1, &mut t);
            gpu.launch(
                "leg_unpack_syn",
                (p.div_ceil(128) as u32, mb),
                128,
                &[
                    gpu.a(&o),
                    (s.rows as i32).into(),
                    (kinds as i32).into(),
                    b.m0.into(),
                    p.into(),
                    self.nr.into(),
                    self.pairs.into(),
                    gpu.a(&self.cs),
                    gpu.a(&self.sn),
                    gpu.a(&s.a),
                    self.opt(gpu, &s.b),
                    self.opt(gpu, &s.c),
                    ((self.lmax + 1) * self.nr).into(),
                ],
            );
            self.phase(gpu, 2, &mut t);
        }
    }

    /// Transpose of [`legendre`](Self::legendre): the packed `a_lm` `[rows, 2 nalm]`.
    fn legendre_adjoint(&self, gpu: &Gpu, sums: &Sums) -> CudaSlice<f64> {
        let deriv = sums.b.is_some();
        let out = gpu.zeros::<f64>(sums.rows * 2 * self.nalm);
        if let Some(g) = &self.gemm {
            let kinds = if deriv { 4 } else { 1 };
            let p = self.pairs + 1;
            let cols = 2 * sums.rows * kinds;
            let y = gpu.zeros::<f64>(g.max_rows * cols);
            let t = gpu.zeros::<f64>(2 * g.max_m * p * cols);
            gpu.ctx.bind_to_thread().unwrap();
            let mut tm = std::time::Instant::now();
            self.phase(gpu, 2, &mut tm);
            for b in &g.blocks {
                let mb = (b.m1 - b.m0) as u32;
                self.leg_table(gpu, g, b);
                self.phase(gpu, 0, &mut tm);
                gpu.launch(
                    "leg_pack_adj",
                    (p.div_ceil(128) as u32, mb),
                    128,
                    &[
                        gpu.a(&sums.a),
                        self.opt(gpu, &sums.b),
                        self.opt(gpu, &sums.c),
                        ((self.lmax + 1) * self.nr).into(),
                        (sums.rows as i32).into(),
                        (kinds as i32).into(),
                        b.m0.into(),
                        p.into(),
                        self.nr.into(),
                        self.pairs.into(),
                        gpu.a(&self.cs),
                        gpu.a(&self.sn),
                        gpu.a(&t),
                    ],
                );
                self.phase(gpu, 2, &mut tm);
                for m in b.m0..b.m1 {
                    for q in 0..2 {
                        self.gemm(gpu, g, b, m, q, cols, &y, &t, true);
                    }
                }
                self.phase(gpu, 1, &mut tm);
                gpu.launch(
                    "leg_unpack_adj",
                    ((self.lmax + 1).div_ceil(256) as u32, mb),
                    256,
                    &[gpu.a(&y), (sums.rows as i32).into(), (kinds as i32).into(), b.m0.into(), self.lmax.into(), gpu.a(&self.coef), gpu.a(&b.cum_dev), gpu.a(&out), self.nalm.into()],
                );
                self.phase(gpu, 2, &mut tm);
            }
            return out;
        }
        let part = gpu.zeros::<f64>(2 * self.nalm * self.chunks);
        for r in 0..sums.rows {
            let o = r * self.row();
            let mut args = vec![
                gpu.at(&sums.a, o),
                self.opt_at(gpu, &sums.b, o),
                self.opt_at(gpu, &sums.c, o),
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
            gpu.launch1("sht_chunk_sum", self.nalm, &[gpu.a(&part), self.nalm.into(), (self.chunks as i32).into(), gpu.at(&out, r * 2 * self.nalm)]);
        }
        out
    }

    /// In-place FFTs of every ring of the `k` complex maps `h` (RING order).
    fn fft(&self, gpu: &Gpu, h: &CudaSlice<f64>, k: usize, inverse: bool) {
        gpu.ctx.bind_to_thread().unwrap();
        let mut t = std::time::Instant::now();
        if self.phases.is_some() {
            gpu.sync();
            t = std::time::Instant::now();
        }
        let base = gpu.ptr(h);
        let dir = if inverse { INVERSE } else { FORWARD };
        let exec = |plan, first: usize| unsafe { self.fft.exec_z2z(plan, base + (first * 16) as u64, dir) }.unwrap_or_else(|e| panic!("{e}"));
        if k == 5 {
            // The polar rings of the five maps by length (north, then south),
            // the equatorial rings map by map.
            for c in &self.rings5 {
                exec(c.plan, c.north);
                exec(c.plan, c.south);
            }
            let eq = self.classes.last().unwrap();
            for o in 0..k {
                exec(eq.plan, o * self.npix + eq.first);
            }
        } else {
            for o in 0..k {
                for c in &self.classes {
                    exec(c.plan, o * self.npix + c.first);
                }
            }
        }
        self.phase(gpu, 3, &mut t);
    }

    /// Maps of row `r` of the Legendre sums (mode 0: the synthesis, one
    /// map; mode 1: the five derivative maps), `[k, npix]` NESTED.
    fn synthesise(&self, gpu: &Gpu, sums: &Sums, r: usize, mode: i32) -> CudaSlice<f64> {
        let k = if mode == 0 { 1 } else { 5 };
        let off = r * self.row();
        let h = gpu.zeros::<f64>(2 * k * self.npix);
        gpu.launch(
            "sht_ring_syn",
            (self.max_nphi.div_ceil(128) as u32, self.nr as u32),
            128,
            &[
                gpu.at(&sums.a, off),
                self.opt_at(gpu, &sums.b, off),
                self.opt_at(gpu, &sums.c, off),
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

    /// Row `r` of `sums`: the Legendre sums of the transposed synthesis of
    /// the maps `x` `[k, npix]` (mode 0: one map; mode 1: the five
    /// derivative cotangents).
    fn analyse(&self, gpu: &Gpu, x: &CudaSlice<f64>, mode: i32, with_mult: bool, sums: &Sums, r: usize) {
        let k = if mode == 0 { 1 } else { 5 };
        assert_eq!(x.len(), k * self.npix);
        let off = r * self.row();
        let h = gpu.zeros::<f64>(2 * k * self.npix);
        gpu.launch1("sht_ring_load", k * self.npix, &[gpu.a(x), k.into(), self.npix.into(), gpu.a(&self.nest), gpu.a(&h)]);
        self.fft(gpu, &h, k, false);
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
                gpu.at(&sums.a, off),
                self.opt_at(gpu, &sums.b, off),
                self.opt_at(gpu, &sums.c, off),
            ],
        );
    }

    fn scale_m(&self, gpu: &Gpu, alm: &mut CudaSlice<f64>, f0: f64, f1: f64) {
        gpu.launch("sht_scale_m", ((self.lmax + 1).div_ceil(256) as u32, (self.lmax + 1) as u32), 256, &[gpu.a(alm), self.lmax.into(), f0.into(), f1.into()]);
    }

    /// The `a_lm` of a batch as the rows of one buffer.
    fn pack(&self, gpu: &Gpu, alms: &[&CudaSlice<f64>]) -> CudaSlice<f64> {
        let n = 2 * self.nalm;
        let mut out = gpu.zeros::<f64>(alms.len() * n);
        for (r, a) in alms.iter().enumerate() {
            assert_eq!(a.len(), n);
            copy_rows(gpu, a, 0, 0, &mut out, 0, r * n, n, 1);
        }
        out
    }

    /// The rows of `packed` as separate `a_lm`.
    fn unpack(&self, gpu: &Gpu, packed: CudaSlice<f64>, rows: usize) -> Vec<CudaSlice<f64>> {
        if rows == 1 {
            return vec![packed];
        }
        let n = 2 * self.nalm;
        (0..rows)
            .map(|r| {
                let mut a = gpu.zeros::<f64>(n);
                copy_rows(gpu, &packed, 0, r * n, &mut a, 0, 0, n, 1);
                a
            })
            .collect()
    }

    fn legendre_of(&self, gpu: &Gpu, alms: &[&CudaSlice<f64>], deriv: bool) -> Sums {
        if alms.len() == 1 {
            return self.legendre(gpu, alms[0], 1, deriv);
        }
        self.legendre(gpu, &self.pack(gpu, alms), alms.len(), deriv)
    }

    /// `Y a` of a batch: the maps of `alms` (`Sht::synthesis`).
    pub fn synthesis_many(&self, gpu: &Gpu, alms: &[&CudaSlice<f64>]) -> Vec<CudaSlice<f64>> {
        let sums = self.legendre_of(gpu, alms, false);
        (0..alms.len()).map(|r| self.synthesise(gpu, &sums, r, 0)).collect()
    }

    /// `Y* x` of a batch (`Sht::adjoint_synthesis`).
    pub fn adjoint_synthesis_many(&self, gpu: &Gpu, xs: &[&CudaSlice<f64>]) -> Vec<CudaSlice<f64>> {
        let sums = self.sums(gpu, xs.len(), false);
        for (r, x) in xs.iter().enumerate() {
            self.analyse(gpu, x, 0, false, &sums, r);
        }
        self.unpack(gpu, self.legendre_adjoint(gpu, &sums), xs.len())
    }

    /// `Omega Y* x` of a batch (`Sht::analysis`).
    pub fn analysis_many(&self, gpu: &Gpu, xs: &[&CudaSlice<f64>]) -> Vec<CudaSlice<f64>> {
        let mut alms = self.adjoint_synthesis_many(gpu, xs);
        for a in &mut alms {
            self.scale_m(gpu, a, self.area, self.area);
        }
        alms
    }

    /// The five covariant derivative maps `[5, npix]` of each of `alms`
    /// (`Sht::derivatives`).
    pub fn derivatives_many(&self, gpu: &Gpu, alms: &[&CudaSlice<f64>]) -> Vec<CudaSlice<f64>> {
        let sums = self.legendre_of(gpu, alms, true);
        (0..alms.len()).map(|r| self.synthesise(gpu, &sums, r, 1)).collect()
    }

    /// Cotangents of the `a_lm` from those of the derivative maps `[5, npix]`
    /// of a batch (`Sht::derivatives_vjp`).
    pub fn derivatives_vjp_many(&self, gpu: &Gpu, cots: &[&CudaSlice<f64>]) -> Vec<CudaSlice<f64>> {
        let sums = self.sums(gpu, cots.len(), true);
        for (r, c) in cots.iter().enumerate() {
            self.analyse(gpu, c, 1, true, &sums, r);
        }
        self.unpack(gpu, self.legendre_adjoint(gpu, &sums), cots.len())
    }

    /// `Y a`: the map of `alm` (`Sht::synthesis`).
    pub fn synthesis(&self, gpu: &Gpu, alm: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.synthesis_many(gpu, &[alm]).pop().unwrap()
    }

    /// `Y* x` (`Sht::adjoint_synthesis`).
    pub fn adjoint_synthesis(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.adjoint_synthesis_many(gpu, &[x]).pop().unwrap()
    }

    /// `Omega Y* x` (`Sht::analysis`).
    pub fn analysis(&self, gpu: &Gpu, x: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.analysis_many(gpu, &[x]).pop().unwrap()
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
        self.synthesis(gpu, &self.power_vjp_coefficients(gpu, alm, dcl))
    }

    /// The coefficients `c` of [`power_vjp_map`](Self::power_vjp_map) before
    /// the synthesis (`Omega (power_vjp) / mult`): the map cotangent is `Y c`.
    pub fn power_vjp_coefficients(&self, gpu: &Gpu, alm: &CudaSlice<f64>, dcl: &[f64]) -> CudaSlice<f64> {
        assert_eq!(dcl.len(), self.lmax + 1);
        let d = gpu.upload(dcl);
        let c = gpu.zeros::<f64>(2 * self.nalm);
        gpu.launch(
            "sht_power_vjp",
            ((self.lmax + 1).div_ceil(256) as u32, (self.lmax + 1) as u32),
            256,
            &[gpu.a(alm), self.lmax.into(), gpu.a(&d), self.area.into(), (self.area / 2.0).into(), gpu.a(&c)],
        );
        c
    }

    /// `c += s Omega a / mult` (the coefficients of [`analysis_vjp`](Self::analysis_vjp)
    /// of `s a`, added to those of another synthesis).
    pub fn add_analysis_vjp_coefficients(&self, gpu: &Gpu, c: &mut CudaSlice<f64>, a: &CudaSlice<f64>, s: f64) {
        assert_eq!(c.len(), 2 * self.nalm);
        assert_eq!(a.len(), 2 * self.nalm);
        gpu.launch(
            "sht_add_scaled_m",
            ((self.lmax + 1).div_ceil(256) as u32, (self.lmax + 1) as u32),
            256,
            &[gpu.a(c), gpu.a(a), self.lmax.into(), (s * self.area).into(), (s * self.area / 2.0).into()],
        );
    }

    /// `p a + q b` of two packed `a_lm`.
    pub fn combine(&self, gpu: &Gpu, a: &CudaSlice<f64>, p: f64, b: &CudaSlice<f64>, q: f64) -> CudaSlice<f64> {
        assert_eq!(a.len(), b.len());
        let out = gpu.zeros::<f64>(a.len());
        gpu.launch1("sht_combine", a.len(), &[gpu.a(a), p.into(), gpu.a(b), q.into(), a.len().into(), gpu.a(&out)]);
        out
    }

    /// `sum_p y_p (Y c)_p` for `y` whose analysis is `ay` (`Y^T y = mult
    /// Y* y = mult ay / Omega`) and `c` the coefficients of an
    /// [`analysis_vjp`](Self::analysis_vjp) of `a` (`c = Omega a / mult`): the
    /// real dot product of `ay` and `a`.
    pub fn analysis_vjp_dot(&self, gpu: &Gpu, ay: &CudaSlice<f64>, a: &CudaSlice<f64>) -> f64 {
        pixel_sum(gpu, gpu.a(ay), gpu.a(a), 2 * self.nalm, 0.0, 3)
    }

    /// The five covariant derivative maps `[5, npix]` (`Sht::derivatives`).
    pub fn derivatives(&self, gpu: &Gpu, alm: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.derivatives_many(gpu, &[alm]).pop().unwrap()
    }

    /// Cotangent of the `a_lm` from those of the derivative maps `[5, npix]`
    /// (`Sht::derivatives_vjp`).
    pub fn derivatives_vjp(&self, gpu: &Gpu, cot: &CudaSlice<f64>) -> CudaSlice<f64> {
        self.derivatives_vjp_many(gpu, &[cot]).pop().unwrap()
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
    /// The `k` thresholds, in units of the standardised map.
    pub nu: CudaSlice<f64>,
    /// Number of thresholds.
    pub k: usize,
    /// Width of the Gaussian smoothing of the thresholds (standardised units).
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
