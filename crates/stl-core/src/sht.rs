//! Spherical harmonic transforms of full-sky HEALPix maps (NESTED at the
//! interface), for the harmonic terms of the ensemble objective of
//! P. Campeti (`stl-ensemble`: C_l and Minkowski functionals; the reference
//! is `python/stl_ensemble/stl_port/harmonics.py`, on DUCC).
//!
//! Conventions, as there:
//! - `a_lm` real-field coefficients, `m >= 0`, packed m-major
//!   ([`Sht::index`], healpy's order); synthesis
//!   `f = sum_l a_l0 Y_l0 + 2 Re sum_{m > 0} a_lm Y_lm`;
//! - analysis `A x = Omega Y* x` (`Omega = 4 pi / N_pix`, no ring weights,
//!   healpy `iter = 0`), `Y*` the adjoint synthesis of DUCC
//!   (`sum_p x_p conj(Y_lm(p))`);
//! - cotangents of complex vectors under `Re sum conj(c) dv` (a coefficient
//!   counts once): the transpose of the synthesis is `mult Y*`, that of the
//!   analysis `Omega Y(c / mult)`, `mult = 1` for `m = 0`, 2 otherwise.
//!
//! Rings: the Legendre functions (Condon-Shortley phase, as healpy) come
//! from the normalised recursion in `l`
//! at fixed `m`, scaled by powers of `2^600` while they are below double
//! range (they are then negligible and skipped), north and south rings
//! together (`Y_lm(pi - theta) = (-1)^(l+m) Y_lm(theta)`); the sums over `l`
//! run in parallel over `m`, the FFTs of the rings in parallel over rings
//! (`STL_THREADS` host threads), every sum in a fixed order.
//!
//! Speed (as libsharp/DUCC): the recursions run on blocks of ring pairs at
//! once (fixed-size lanes that LLVM vectorises, compiled for AVX2 when the
//! CPU has it, without FMA), branch-free once every lane is in range; the
//! rings where the functions of `m` stay below about `1e-30` up to `lmax`
//! are skipped (libsharp's `mlim`); the transposed sums accumulate per lane
//! and reduce once per `m`; the ring FFTs are real FFTs (complex FFTs of
//! half length).  The results agree with the one-ring scalar recursion to
//! a few `1e-15` relative.

use crate::healpix::{order_of, ring2nest};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type C64 = Complex64;

/// One iso-latitude ring.
#[derive(Clone, Debug)]
pub struct Ring {
    /// `cos theta` of the ring.
    pub cos: f64,
    /// `sin theta` of the ring.
    pub sin: f64,
    /// Number of pixels of the ring.
    pub nphi: usize,
    /// Longitude of the first pixel.
    pub phi0: f64,
    /// First pixel in RING order.
    pub start: usize,
}

/// Legendre sums of a set of `a_lm` on every ring, `[m][ring]`: the values
/// `A = sum_l a_lm lambda_lm`, the theta derivatives `B = sum_l a_lm
/// d lambda_lm / d theta` and `C = sum_l l (l + 1) a_lm lambda_lm` (B, C
/// only when asked).
struct Sums {
    a: Vec<C64>,
    b: Vec<C64>,
    c: Vec<C64>,
}

/// Scaling of the Legendre recursion (shared with the device transforms).
pub const LN_SCALE: f64 = 600.0 * std::f64::consts::LN_2;
/// `2^600`, the scaling step of the Legendre recursion.
pub const SCALE: f64 = 4.149515568880993e180; // 2^600
/// `2^300`: a scaled value above this is divided by `SCALE`.
pub const LIMIT: f64 = 2.037035976334486e90; // 2^300

/// Ring pairs recursed together (the lanes of the block recursions), spin 0.
const LANES: usize = 16;
/// Lanes of the spin-`s` recursions (two chains per lane).
const SPIN_LANES: usize = 8;

/// Scale exponent `k <= 0` of a start value of log magnitude `log`.
#[inline(always)]
fn scale_exponent(log: f64) -> i32 {
    if log > -650.0 * std::f64::consts::LN_2 {
        0
    } else {
        ((log - 300.0 * std::f64::consts::LN_2) / LN_SCALE).ceil() as i32
    }
}

/// `f()` compiled for AVX2 when the CPU has it: wider vectors for the block
/// recursions, without FMA (Rust never contracts `a * b + c`), so the
/// results are those of the baseline build.
#[inline(always)]
fn simd<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn avx2<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: the CPU supports AVX2.
            return unsafe { avx2(f) };
        }
    }
    f()
}

/// Width of the per-`l` accumulators of the transposed sums: the lanes
/// are folded to `W` before they are added (fewer loads and stores per `l`).
const W: usize = 4;

/// The lanes folded to `W`, in a fixed order.
#[inline(always)]
fn fold<const NB: usize>(v: &[f64; NB]) -> [f64; W] {
    let mut o: [f64; W] = std::array::from_fn(|j| v[j]);
    for c in 1..NB / W {
        for j in 0..W {
            o[j] += v[c * W + j];
        }
    }
    o
}

/// Sum of `W` folded lanes, in a fixed tree order.
#[inline(always)]
fn lane_sum(mut v: [f64; W]) -> f64 {
    let mut w = W / 2;
    while w > 0 {
        for i in 0..w {
            v[i] += v[i + w];
        }
        w /= 2;
    }
    v[0]
}

/// `e^{i sign m phi0}` for `m = 0..count`, by products re-anchored every 32 `m`.
fn phases(phi0: f64, sign: f64, count: usize, out: &mut Vec<C64>) {
    out.clear();
    if phi0 == 0.0 {
        out.resize(count, C64::new(1.0, 0.0));
        return;
    }
    let step = C64::from_polar(1.0, sign * phi0);
    let mut ph = C64::new(1.0, 0.0);
    for m in 0..count {
        if m % 32 == 0 {
            ph = C64::from_polar(1.0, sign * (m as f64 * phi0));
        }
        out.push(ph);
        ph *= step;
    }
}

/// Highest `m` where a ring of `(cos, sin)` has spin-`s` functions above
/// about `1e-30` up to `lmax` (libsharp's `sharp_get_mlim`).
pub fn mlim(lmax: usize, s: usize, cos: f64, sin: f64) -> usize {
    let ofs = (lmax as f64 * 0.01).max(100.0);
    let b = -2.0 * s as f64 * cos.abs();
    let t1 = lmax as f64 * sin + ofs;
    let c = (s * s) as f64 - t1 * t1;
    let discr = b * b - 4.0 * c;
    if discr <= 0.0 {
        return lmax;
    }
    (((-b + discr.sqrt()) / 2.0).min(lmax as f64) + 0.5) as usize
}

/// Real FFT of an even length `n` through a complex FFT of length `n / 2`.
struct RealFft {
    n: usize,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    /// `e^{-2 pi i k / n}`, `k = 0..=n/2`.
    tw: Vec<C64>,
}

impl RealFft {
    fn new(planner: &mut FftPlanner<f64>, n: usize) -> Self {
        assert!(n % 2 == 0);
        let h = n / 2;
        let tw = (0..=h).map(|k| C64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64)).collect();
        RealFft { n, fwd: planner.plan_fft_forward(h), inv: planner.plan_fft_inverse(h), tw }
    }

    /// `F_k = sum_j x_j e^{-2 pi i j k / n}`, `k = 0..=n/2` into `out`.
    fn forward(&self, x: impl Fn(usize) -> f64, out: &mut Vec<C64>, buf: &mut Vec<C64>) {
        let h = self.n / 2;
        buf.clear();
        buf.extend((0..h).map(|j| C64::new(x(2 * j), x(2 * j + 1))));
        self.fwd.process(buf);
        out.clear();
        out.extend((0..=h).map(|k| {
            let zk = buf[k % h];
            let zc = buf[(h - k) % h].conj();
            let e = (zk + zc) * 0.5;
            let o = (zk - zc) * C64::new(0.0, -0.5);
            e + self.tw[k] * o
        }));
    }

    /// Bin `m` (any `m >= 0`) of a real spectrum from [`forward`](Self::forward).
    #[inline(always)]
    fn bin(&self, f: &[C64], m: usize) -> C64 {
        let k = m % self.n;
        if k <= self.n / 2 {
            f[k]
        } else {
            f[self.n - k].conj()
        }
    }

    /// `x_j = Re sum_k h_k e^{2 pi i j k / n}` of a complex `h` of length `n`.
    fn inverse_real(&self, hc: &[C64], out: &mut [f64], buf: &mut Vec<C64>) {
        let (n, h) = (self.n, self.n / 2);
        buf.clear();
        buf.extend((0..h).map(|k| {
            let a = (hc[k] + hc[(n - k) % n].conj()) * 0.5;
            let b = (hc[k + h] + hc[h - k].conj()) * 0.5;
            let ev = a + b;
            let od = (a - b) * self.tw[k].conj();
            ev + C64::new(0.0, 1.0) * od
        }));
        self.inv.process(buf);
        for (j, z) in buf.iter().enumerate() {
            out[2 * j] = z.re;
            out[2 * j + 1] = z.im;
        }
    }
}

/// Spherical harmonic transforms of full-sky maps of one `nside` up to
/// `lmax`: ring geometry, RING -> NESTED table and FFT plans.
pub struct Sht {
    /// HEALPix `nside`.
    pub nside: usize,
    /// Maximum multipole.
    pub lmax: usize,
    /// Number of pixels, `12 nside^2`.
    pub npix: usize,
    /// The `4 nside - 1` rings, from north to south.
    pub rings: Vec<Ring>,
    /// NESTED index of every pixel in RING order.
    nest: Vec<u32>,
    /// `ln` of the `m`-dependent constant of `lambda_mm` (without `sin^m`).
    log_mm: Vec<f64>,
    /// `ln n!`, `n = 0..=2 lmax + 1` (starts of the spin recursions).
    ln_fact: Vec<f64>,
    fft: HashMap<usize, RealFft>,
    threads: usize,
}

/// Host threads: `STL_THREADS`, else the CPUs available to the process.
fn host_threads() -> usize {
    std::env::var("STL_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1)
}

/// `f(i)` for `i < n` on `threads` threads (dynamic assignment), results in order.
fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let t = threads.min(n).max(1);
    if t == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..t)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        out.push((i, f(i)));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for part in parts.iter_mut() {
        for (i, v) in part.drain(..) {
            slots[i] = Some(v);
        }
    }
    slots.into_iter().map(|v| v.unwrap()).collect()
}

impl Sht {
    /// Transforms of maps of `nside` up to `lmax` (`3 nside - 1` by default in the callers).
    pub fn new(nside: usize, lmax: usize) -> Self {
        assert!(nside.is_power_of_two(), "nside must be a power of two");
        let npix = 12 * nside * nside;
        let ns = nside as f64;
        let nrings = 4 * nside - 1;
        let mut rings = Vec::with_capacity(nrings);
        for i in 1..=nrings {
            let north = i.min(4 * nside - i);
            let (cos, sin, nphi, shifted, start);
            if north < nside {
                // Polar caps: 1 - z = i^2 / (3 nside^2) exactly.
                let t = (north * north) as f64 / (3.0 * ns * ns);
                cos = 1.0 - t;
                sin = (t * (2.0 - t)).sqrt();
                nphi = 4 * north;
                shifted = true;
                start = if i < nside { 2 * north * (north - 1) } else { npix - 2 * north * (north + 1) };
            } else {
                let z = (2 * nside) as f64 - north as f64;
                cos = z * 2.0 / (3.0 * ns);
                sin = ((1.0 - cos) * (1.0 + cos)).sqrt();
                nphi = 4 * nside;
                shifted = (i - nside) % 2 == 0;
                start = 2 * nside * (nside - 1) + (i - nside) * 4 * nside;
            }
            let cos = if i > 2 * nside { -cos } else { cos };
            let phi0 = if shifted { std::f64::consts::PI / nphi as f64 } else { 0.0 };
            rings.push(Ring { cos, sin, nphi, phi0, start });
        }
        let order = order_of(nside as i64);
        let nest = (0..npix).map(|p| ring2nest(p as i64, order) as u32).collect();
        let mut log_mm = vec![0.0; lmax + 1];
        let mut acc = 0.0;
        for m in 0..=lmax {
            if m > 0 {
                acc += ((2 * m - 1) as f64 / (2 * m) as f64).ln();
            }
            log_mm[m] = 0.5 * (((2 * m + 1) as f64) / (4.0 * std::f64::consts::PI)).ln() + 0.5 * acc;
        }
        let mut ln_fact = vec![0.0; 2 * lmax + 2];
        for n in 1..ln_fact.len() {
            ln_fact[n] = ln_fact[n - 1] + (n as f64).ln();
        }
        let mut planner = FftPlanner::new();
        let mut fft = HashMap::new();
        for r in &rings {
            fft.entry(r.nphi).or_insert_with(|| RealFft::new(&mut planner, r.nphi));
        }
        Sht { nside, lmax, npix, rings, nest, log_mm, ln_fact, fft, threads: host_threads() }
    }

    /// Packed index of `(l, m)`, `m <= l <= lmax`.
    pub fn index(&self, l: usize, m: usize) -> usize {
        m * (self.lmax + 1) - m * (m.saturating_sub(1)) / 2 + (l - m)
    }

    /// Number of packed coefficients.
    pub fn nalm(&self) -> usize {
        (self.lmax + 1) * (self.lmax + 2) / 2
    }

    /// `(l, m)` of every packed coefficient.
    pub fn lm(&self) -> Vec<(usize, usize)> {
        (0..=self.lmax).flat_map(|m| (m..=self.lmax).map(move |l| (l, m))).collect()
    }

    /// Pixel solid angle `Omega = 4 pi / N_pix`, steradians.
    pub fn area(&self) -> f64 {
        4.0 * std::f64::consts::PI / self.npix as f64
    }

    /// NESTED index of every pixel in RING order.
    pub fn nest_of_ring(&self) -> &[u32] {
        &self.nest
    }

    /// `ln` of the `m`-dependent constant of `lambda_mm`, `m = 0..=lmax`
    /// (the device transforms start their recursions from it).
    pub fn log_mm(&self) -> &[f64] {
        &self.log_mm
    }

    fn pairs(&self) -> usize {
        self.rings.len() / 2
    }

    /// North rows (the equator included) where the spin-`s` functions of `m`
    /// are significant ([`mlim`]), in blocks of `NB` lanes: `(rows, count)`,
    /// the last block padded with its last row.
    fn blocks<const NB: usize>(&self, m: usize, s: usize) -> Vec<([usize; NB], usize)> {
        let rows: Vec<usize> = (0..=self.pairs()).filter(|&p| m <= mlim(self.lmax, s, self.rings[p].cos, self.rings[p].sin)).collect();
        rows.chunks(NB)
            .map(|c| {
                let mut r = [c[c.len() - 1]; NB];
                r[..c.len()].copy_from_slice(c);
                (r, c.len())
            })
            .collect()
    }

    /// `(cos, sin)` of the lanes of a block.
    fn lanes<const NB: usize>(&self, rows: &[usize; NB]) -> ([f64; NB], [f64; NB]) {
        (std::array::from_fn(|i| self.rings[rows[i]].cos), std::array::from_fn(|i| self.rings[rows[i]].sin))
    }

    /// The Legendre recursion of `m` on `NB` rings of `(cos, sin)` at once:
    /// `f(l, lambda_l, lambda_{l-1})` for `l = m..=lmax`, the lanes whose
    /// value is not yet in double range (below `2^-300`) set to zero.  While
    /// a lane is scaled the recursion checks the range; afterwards it runs
    /// without branches.
    #[inline(always)]
    fn recurse_block<const NB: usize>(&self, m: usize, coef: &[(f64, f64, f64)], cos: &[f64; NB], sin: &[f64; NB], mut f: impl FnMut(usize, &[f64; NB], &[f64; NB])) {
        let mut cur = [0.0; NB];
        let mut prev = [0.0; NB];
        let mut k = [0i32; NB];
        // Condon-Shortley phase (healpy, DUCC).
        let sign = if m % 2 == 1 { -1.0 } else { 1.0 };
        for i in 0..NB {
            let log = self.log_mm[m] + m as f64 * sin[i].ln();
            k[i] = scale_exponent(log);
            cur[i] = (log - k[i] as f64 * LN_SCALE).exp() * sign;
        }
        let mut l = m;
        if k.iter().any(|&k| k < 0) {
            let (mut lam, mut pre) = ([0.0; NB], [0.0; NB]);
            loop {
                let mut done = true;
                for i in 0..NB {
                    let on = if k[i] == 0 { 1.0 } else { 0.0 };
                    lam[i] = cur[i] * on;
                    pre[i] = prev[i] * on;
                    done &= k[i] == 0;
                }
                if done {
                    break;
                }
                f(l, &lam, &pre);
                if l == self.lmax {
                    return;
                }
                l += 1;
                let (a, b, _) = coef[l - m];
                for i in 0..NB {
                    let next = a * (cos[i] * cur[i] - b * prev[i]);
                    let up = k[i] < 0 && next.abs() > LIMIT;
                    // 2^-600: exact, as the division by SCALE.
                    let s = if up { 1.0 / SCALE } else { 1.0 };
                    prev[i] = cur[i] * s;
                    cur[i] = next * s;
                    k[i] += up as i32;
                }
            }
        }
        loop {
            f(l, &cur, &prev);
            if l == self.lmax {
                return;
            }
            l += 1;
            let (a, b, _) = coef[l - m];
            for i in 0..NB {
                let next = a * (cos[i] * cur[i] - b * prev[i]);
                prev[i] = cur[i];
                cur[i] = next;
            }
        }
    }

    /// Recursion coefficients of `m`: `(a_l, b_l, c_l)` for `l = m..=lmax`.
    fn coefficients(&self, m: usize) -> Vec<(f64, f64, f64)> {
        let mf = m as f64;
        (m..=self.lmax)
            .map(|l| {
                if l == m {
                    return (0.0, 0.0, 0.0);
                }
                let lf = l as f64;
                let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
                let l1 = lf - 1.0;
                let b = ((l1 * l1 - mf * mf) / (4.0 * l1 * l1 - 1.0)).sqrt();
                let c = ((2.0 * lf + 1.0) * (lf * lf - mf * mf) / (2.0 * lf - 1.0)).sqrt();
                (a, b, c)
            })
            .collect()
    }

    /// Legendre sums `[m][ring]` of `alm` (B, C when `deriv`).
    fn legendre(&self, alm: &[C64], deriv: bool) -> Sums {
        const NB: usize = LANES;
        assert_eq!(alm.len(), self.nalm());
        let nr = self.rings.len();
        let pairs = self.pairs();
        let per_m = par_map(self.lmax + 1, self.threads, |m| simd(|| {
            let coef = self.coefficients(m);
            let a = &alm[self.index(m, m)..self.index(m, m) + self.lmax + 1 - m];
            let (mut va, mut vb, mut vc) = (vec![C64::default(); nr], vec![C64::default(); if deriv { nr } else { 0 }], vec![C64::default(); if deriv { nr } else { 0 }]);
            for (rows, count) in self.blocks::<NB>(m, 0) {
                let (cos, sin) = self.lanes(&rows);
                // [re, im][even, odd in l + m][lane].
                let mut sa = [[[0.0; NB]; 2]; 2];
                let mut sb = [[[0.0; NB]; 2]; 2];
                let mut sc = [[[0.0; NB]; 2]; 2];
                self.recurse_block(m, &coef, &cos, &sin, |l, lam, pre| {
                    let par = (l + m) & 1;
                    let x = a[l - m];
                    for i in 0..NB {
                        sa[0][par][i] += x.re * lam[i];
                        sa[1][par][i] += x.im * lam[i];
                    }
                    if deriv {
                        let (c, lf, ll) = (coef[l - m].2, l as f64, (l * (l + 1)) as f64);
                        for i in 0..NB {
                            let d = (lf * cos[i] * lam[i] - c * pre[i]) / sin[i];
                            let w = ll * lam[i];
                            sb[0][par][i] += x.re * d;
                            sb[1][par][i] += x.im * d;
                            sc[0][par][i] += x.re * w;
                            sc[1][par][i] += x.im * w;
                        }
                    }
                });
                for (i, &p) in rows[..count].iter().enumerate() {
                    let get = |v: &[[[f64; NB]; 2]; 2], par: usize| C64::new(v[0][par][i], v[1][par][i]);
                    va[p] = get(&sa, 0) + get(&sa, 1);
                    if deriv {
                        vb[p] = get(&sb, 0) + get(&sb, 1);
                        vc[p] = get(&sc, 0) + get(&sc, 1);
                    }
                    if p < pairs {
                        let s = nr - 1 - p;
                        va[s] = get(&sa, 0) - get(&sa, 1);
                        if deriv {
                            vb[s] = get(&sb, 1) - get(&sb, 0);
                            vc[s] = get(&sc, 0) - get(&sc, 1);
                        }
                    }
                }
            }
            (va, vb, vc)
        }));
        let mut sums = Sums { a: Vec::with_capacity((self.lmax + 1) * nr), b: Vec::new(), c: Vec::new() };
        for (va, vb, vc) in per_m {
            sums.a.extend(va);
            sums.b.extend(vb);
            sums.c.extend(vc);
        }
        sums
    }

    /// Transpose of [`legendre`](Self::legendre): `a_lm = sum_rings lambda A +
    /// lambda' B + l (l + 1) lambda C` (B, C when present).
    fn legendre_adjoint(&self, sums: &Sums) -> Vec<C64> {
        const NB: usize = LANES;
        let nr = self.rings.len();
        let pairs = self.pairs();
        let deriv = !sums.b.is_empty();
        let per_m = par_map(self.lmax + 1, self.threads, |m| simd(|| {
            let coef = self.coefficients(m);
            // [l - m][re, im][lane], summed over the lanes at the end.
            let mut acc = vec![[[0.0; W]; 2]; self.lmax + 1 - m];
            let row = |v: &[C64], r: usize| v[m * nr + r];
            for (rows, count) in self.blocks::<NB>(m, 0) {
                let (cos, sin) = self.lanes(&rows);
                // [re, im][even, odd in l + m][lane], zero on the padding lanes.
                let mut ta = [[[0.0; NB]; 2]; 2];
                let mut tb = [[[0.0; NB]; 2]; 2];
                let mut tc = [[[0.0; NB]; 2]; 2];
                for (i, &p) in rows[..count].iter().enumerate() {
                    let s = nr - 1 - p;
                    let south = p < pairs;
                    let both = |v: &[C64]| (row(v, p), if south { row(v, s) } else { C64::default() });
                    let put = |t: &mut [[[f64; NB]; 2]; 2], v: [C64; 2]| {
                        for par in 0..2 {
                            t[0][par][i] = v[par].re;
                            t[1][par][i] = v[par].im;
                        }
                    };
                    let (n, sv) = both(&sums.a);
                    put(&mut ta, [n + sv, n - sv]);
                    if deriv {
                        let (n, sv) = both(&sums.b);
                        put(&mut tb, [n - sv, n + sv]);
                        let (n, sv) = both(&sums.c);
                        put(&mut tc, [n + sv, n - sv]);
                    }
                }
                self.recurse_block(m, &coef, &cos, &sin, |l, lam, pre| {
                    let par = (l + m) & 1;
                    let (mut re, mut im) = ([0.0; NB], [0.0; NB]);
                    if deriv {
                        let (c, lf, ll) = (coef[l - m].2, l as f64, (l * (l + 1)) as f64);
                        for i in 0..NB {
                            let d = (lf * cos[i] * lam[i] - c * pre[i]) / sin[i];
                            let w = ll * lam[i];
                            re[i] = ta[0][par][i] * lam[i] + tb[0][par][i] * d + tc[0][par][i] * w;
                            im[i] = ta[1][par][i] * lam[i] + tb[1][par][i] * d + tc[1][par][i] * w;
                        }
                    } else {
                        for i in 0..NB {
                            re[i] = ta[0][par][i] * lam[i];
                            im[i] = ta[1][par][i] * lam[i];
                        }
                    }
                    let (fr, fi) = (fold(&re), fold(&im));
                    let [ar, ai] = &mut acc[l - m];
                    for j in 0..W {
                        ar[j] += fr[j];
                        ai[j] += fi[j];
                    }
                });
            }
            acc.into_iter().map(|[re, im]| C64::new(lane_sum(re), lane_sum(im))).collect::<Vec<_>>()
        }));
        per_m.into_iter().flatten().collect()
    }

    /// Ring values `Re sum_m mult_m H_m e^{i m phi}` for each output, `H`
    /// from `combine(ring, m, A_m, B_m, C_m)`; maps in NESTED order.
    fn synthesise<const K: usize>(&self, sums: &Sums, combine: impl Fn(&Ring, usize, C64, C64, C64) -> [C64; K] + Sync) -> [Vec<f64>; K] {
        let nr = self.rings.len();
        let deriv = !sums.b.is_empty();
        let per_ring = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let n = ring.nphi;
            let mut ph = Vec::new();
            phases(ring.phi0, 1.0, self.lmax + 1, &mut ph);
            let mut h = vec![vec![C64::default(); n]; K];
            for m in 0..=self.lmax {
                let i = m * nr + r;
                let (b, c) = if deriv { (sums.b[i], sums.c[i]) } else { (C64::default(), C64::default()) };
                let v = combine(ring, m, sums.a[i], b, c);
                let w = if m == 0 { 1.0 } else { 2.0 };
                let shift = ph[m] * w;
                for k in 0..K {
                    h[k][m % n] += v[k] * shift;
                }
            }
            let rf = &self.fft[&n];
            let mut buf = Vec::new();
            h.iter()
                .map(|x| {
                    let mut v = vec![0.0; n];
                    rf.inverse_real(x, &mut v, &mut buf);
                    v
                })
                .collect::<Vec<_>>()
        });
        let mut out: [Vec<f64>; K] = std::array::from_fn(|_| vec![0.0; self.npix]);
        for (r, values) in per_ring.into_iter().enumerate() {
            let start = self.rings[r].start;
            for (k, v) in values.into_iter().enumerate() {
                for (j, x) in v.into_iter().enumerate() {
                    out[k][self.nest[start + j] as usize] = x;
                }
            }
        }
        out
    }

    /// Transpose of [`synthesise`](Self::synthesise) without the `mult`
    /// factor (`with_mult = false`, the adjoint synthesis of DUCC) or with it:
    /// the Legendre sums `[m][ring]` of `split(ring, m, F^1_m, ..., F^K_m)`,
    /// `F^k_m = sum_j x^k_j e^{-i m phi_j}` (`(A, B, C)`; B and C when `deriv`).
    fn analyse<const K: usize>(&self, maps: [&[f64]; K], deriv: bool, with_mult: bool, split: impl Fn(&Ring, usize, [C64; K]) -> [C64; 3] + Sync) -> Sums {
        maps.iter().for_each(|x| assert_eq!(x.len(), self.npix));
        let nr = self.rings.len();
        let per_ring = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let n = ring.nphi;
            let rf = &self.fft[&n];
            let mut buf = Vec::new();
            let spectra: Vec<Vec<C64>> = maps
                .iter()
                .map(|x| {
                    let mut out = Vec::new();
                    rf.forward(|j| x[self.nest[ring.start + j] as usize], &mut out, &mut buf);
                    out
                })
                .collect();
            let mut ph = Vec::new();
            phases(ring.phi0, -1.0, self.lmax + 1, &mut ph);
            (0..=self.lmax)
                .map(|m| {
                    let w = if with_mult && m > 0 { 2.0 } else { 1.0 };
                    let shift = ph[m] * w;
                    let f: [C64; K] = std::array::from_fn(|k| rf.bin(&spectra[k], m) * shift);
                    split(ring, m, f)
                })
                .collect::<Vec<[C64; 3]>>()
        });
        let take = |k: usize| -> Vec<C64> { (0..=self.lmax).flat_map(|m| per_ring.iter().map(move |v| v[m][k])).collect() };
        Sums { a: take(0), b: if deriv { take(1) } else { Vec::new() }, c: if deriv { take(2) } else { Vec::new() } }
    }

    /// `Y a`: the real map of `alm` (NESTED).
    pub fn synthesis(&self, alm: &[C64]) -> Vec<f64> {
        let [x] = self.synthesise(&self.legendre(alm, false), |_, _, a, _, _| [a]);
        x
    }

    /// `Y* x`, DUCC's adjoint synthesis.
    pub fn adjoint_synthesis(&self, x: &[f64]) -> Vec<C64> {
        self.legendre_adjoint(&self.analyse([x], false, false, |_, _, [f]| [f, C64::default(), C64::default()]))
    }

    /// Analysis `Omega Y* x`.
    pub fn analysis(&self, x: &[f64]) -> Vec<C64> {
        let area = self.area();
        self.adjoint_synthesis(x).into_iter().map(|v| v * area).collect()
    }

    /// Map cotangent of the analysis: `Omega Y(c / mult)`.
    pub fn analysis_vjp(&self, c: &[C64]) -> Vec<f64> {
        let area = self.area();
        let scaled: Vec<C64> = self.lm().iter().zip(c).map(|(&(_, m), v)| v * if m == 0 { area } else { area / 2.0 }).collect();
        self.synthesis(&scaled)
    }

    /// `C_l = sum_m mult |a_lm|^2 / (2 l + 1)`.
    pub fn power(&self, alm: &[C64]) -> Vec<f64> {
        let mut cl = vec![0.0; self.lmax + 1];
        for (&(l, m), a) in self.lm().iter().zip(alm) {
            cl[l] += if m == 0 { 1.0 } else { 2.0 } * a.norm_sqr();
        }
        cl.iter_mut().enumerate().for_each(|(l, v)| *v /= (2 * l + 1) as f64);
        cl
    }

    /// Cotangent of the `a_lm` from that of the `C_l`.
    pub fn power_vjp(&self, alm: &[C64], dcl: &[f64]) -> Vec<C64> {
        self.lm().iter().zip(alm).map(|(&(l, m), a)| a * (2.0 * if m == 0 { 1.0 } else { 2.0 } * dcl[l] / (2 * l + 1) as f64)).collect()
    }

    /// Cross spectrum `C_l = sum_m mult Re(a_lm conj(b_lm)) / (2 l + 1)`.
    pub fn cross_power(&self, a: &[C64], b: &[C64]) -> Vec<f64> {
        let mut cl = vec![0.0; self.lmax + 1];
        for (&(l, m), (x, y)) in self.lm().iter().zip(a.iter().zip(b)) {
            cl[l] += if m == 0 { 1.0 } else { 2.0 } * (x * y.conj()).re;
        }
        cl.iter_mut().enumerate().for_each(|(l, v)| *v /= (2 * l + 1) as f64);
        cl
    }

    /// Cotangents of `a` and `b` from that of their cross `C_l`:
    /// `mult dcl b / (2 l + 1)` and `mult dcl a / (2 l + 1)`.
    pub fn cross_power_vjp(&self, a: &[C64], b: &[C64], dcl: &[f64]) -> (Vec<C64>, Vec<C64>) {
        let f: Vec<f64> = self.lm().iter().map(|&(l, m)| if m == 0 { 1.0 } else { 2.0 } * dcl[l] / (2 * l + 1) as f64).collect();
        (b.iter().zip(&f).map(|(y, s)| y * s).collect(), a.iter().zip(&f).map(|(x, s)| x * s).collect())
    }

    /// Recursion coefficients of the Wigner functions `d^l_{m, +-s}`, `l =
    /// l0..=lmax` (`l0 = max(m, s)`): `(A_l, B_l, C_l)` of `d^{l+1} = A_l (x
    /// -+ B_l) d^l - C_l d^{l-1}` (`B_l` enters with the sign of `-b`).
    fn spin_coefficients(&self, m: usize, s: usize) -> Vec<(f64, f64, f64)> {
        let (a2, b2) = ((m * m) as f64, (s * s) as f64);
        let r = |l: f64| ((l * l - a2).max(0.0) * (l * l - b2).max(0.0)).sqrt();
        (m.max(s)..=self.lmax)
            .map(|l| {
                let lf = l as f64;
                let rn = r(lf + 1.0);
                ((2.0 * lf + 1.0) * (lf + 1.0) / rn, (m * s) as f64 / (lf * (lf + 1.0)), (lf + 1.0) * r(lf) / (lf * rn))
            })
            .collect()
    }

    /// The spin-`s` functions of `m` on `NB` rings of `(cos, sin)` at once
    /// (north hemisphere or equator): `f(l, lambda+, lambda-)` for `l =
    /// max(m, s)..=lmax` with `lambda+- = (_s lambda_lm +- _{-s} lambda_lm) /
    /// 2`, `_s lambda_lm = (-1)^s sqrt((2l+1)/4pi) d^l_{m,-s}`, `_{-s}
    /// lambda_lm = sqrt((2l+1)/4pi) d^l_{m,s}` (healpy's `map2alm_spin`), a
    /// function not yet in double range counted as zero.
    #[inline(always)]
    fn recurse_spin_block<const NB: usize>(&self, m: usize, s: usize, coef: &[(f64, f64, f64)], cos: &[f64; NB], sin: &[f64; NB], mut f: impl FnMut(usize, &[f64; NB], &[f64; NB])) {
        let j = m.max(s);
        let lf = |n: usize| self.ln_fact[n];
        // [b = -s, b = +s][lane]: current, previous and scale exponent.
        let mut cur = [[0.0; NB]; 2];
        let mut prev = [[0.0; NB]; 2];
        let mut k = [[0i32; NB]; 2];
        for i in 0..NB {
            // Half-angle cosine and sine without cancellation.
            let (ch, sh) = if cos[i] >= 0.0 {
                let ch = ((1.0 + cos[i]) / 2.0).sqrt();
                (ch, sin[i] / (2.0 * ch))
            } else {
                let sh = ((1.0 - cos[i]) / 2.0).sqrt();
                (sin[i] / (2.0 * sh), sh)
            };
            let (lch, lsh) = (ch.ln(), sh.ln());
            // d^j_{m,b} at j = max(m, s): (log magnitude, sign).
            let start = |b: i64| -> (f64, f64) {
                let (jj, a) = (j as i64, m as i64);
                if m >= s {
                    let log = 0.5 * (lf(2 * j) - lf((jj + b) as usize) - lf((jj - b) as usize)) + (jj + b) as f64 * lch + (jj - b) as f64 * lsh;
                    (log, if (jj - b) % 2 == 0 { 1.0 } else { -1.0 })
                } else {
                    let base = 0.5 * (lf(2 * j) - lf((jj + a) as usize) - lf((jj - a) as usize));
                    if b > 0 {
                        (base + (jj + a) as f64 * lch + (jj - a) as f64 * lsh, 1.0)
                    } else {
                        (base + (jj - a) as f64 * lch + (jj + a) as f64 * lsh, if (a + jj) % 2 == 0 { 1.0 } else { -1.0 })
                    }
                }
            };
            for (t, b) in [-(s as i64), s as i64].into_iter().enumerate() {
                let (log, sign) = start(b);
                k[t][i] = scale_exponent(log);
                cur[t][i] = (log - k[t][i] as f64 * LN_SCALE).exp() * sign;
            }
        }
        let sgn = if s % 2 == 0 { 1.0 } else { -1.0 };
        let (mut lp, mut lm) = ([0.0; NB], [0.0; NB]);
        let mut scaled = k.iter().flatten().any(|&k| k < 0);
        let mut l = j;
        loop {
            let n = 0.5 * (((2 * l + 1) as f64) / (4.0 * std::f64::consts::PI)).sqrt();
            if scaled {
                let mut done = true;
                for i in 0..NB {
                    let ls = if k[0][i] == 0 { sgn * cur[0][i] } else { 0.0 };
                    let lv = if k[1][i] == 0 { cur[1][i] } else { 0.0 };
                    lp[i] = n * (ls + lv);
                    lm[i] = n * (ls - lv);
                    done &= k[0][i] == 0 && k[1][i] == 0;
                }
                scaled = !done;
            } else {
                for i in 0..NB {
                    let (ls, lv) = (sgn * cur[0][i], cur[1][i]);
                    lp[i] = n * (ls + lv);
                    lm[i] = n * (ls - lv);
                }
            }
            f(l, &lp, &lm);
            if l == self.lmax {
                return;
            }
            l += 1;
            let (a, b, c) = coef[l - 1 - j];
            for (t, bt) in [-b, b].into_iter().enumerate() {
                for i in 0..NB {
                    let next = a * (cos[i] - bt) * cur[t][i] - c * prev[t][i];
                    if scaled {
                        let up = k[t][i] < 0 && next.abs() > LIMIT;
                        // 2^-600: exact, as the division by SCALE.
                        let sc = if up { 1.0 / SCALE } else { 1.0 };
                        prev[t][i] = cur[t][i] * sc;
                        cur[t][i] = next * sc;
                        k[t][i] += up as i32;
                    } else {
                        prev[t][i] = cur[t][i];
                        cur[t][i] = next;
                    }
                }
            }
        }
    }

    /// Spin-`s` analysis (`s >= 1`) of the pairs `(Q, U)` (NESTED), healpy's
    /// `map2alm_spin` without iteration: `E = -(a_s + a_{-s}) / 2`, `B = i
    /// (a_s - a_{-s}) / 2`, `a_{+-s} = Omega sum_p (Q +- iU) conj(_{+-s}Y_lm)`;
    /// the `a_lm` with `l < s` are zero.  The recursions are shared by the
    /// pairs.
    pub fn analysis_spin(&self, s: usize, maps: &[(&[f64], &[f64])]) -> Vec<(Vec<C64>, Vec<C64>)> {
        const NB: usize = SPIN_LANES;
        assert!(s >= 1 && s <= self.lmax, "spin must be in 1..=lmax (spin 0: analysis)");
        maps.iter().for_each(|(q, u)| assert!(q.len() == self.npix && u.len() == self.npix));
        let nr = self.rings.len();
        let nm = maps.len();
        let pairs = self.pairs();
        // Ring spectra [ring][m][map] of Q and U (phase of the first pixel included).
        let spectra = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let rf = &self.fft[&ring.nphi];
            let (mut buf, mut fq, mut fu, mut ph) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            phases(ring.phi0, -1.0, self.lmax + 1, &mut ph);
            let mut out = vec![(C64::default(), C64::default()); (self.lmax + 1) * nm];
            for (i, (q, u)) in maps.iter().enumerate() {
                rf.forward(|j| q[self.nest[ring.start + j] as usize], &mut fq, &mut buf);
                rf.forward(|j| u[self.nest[ring.start + j] as usize], &mut fu, &mut buf);
                for m in 0..=self.lmax {
                    out[m * nm + i] = (rf.bin(&fq, m) * ph[m], rf.bin(&fu, m) * ph[m]);
                }
            }
            out
        });
        let area = self.area();
        let per_m = par_map(self.lmax + 1, self.threads, |m| simd(|| {
            let coef = self.spin_coefficients(m, s);
            let len = self.lmax + 1 - m;
            // [map, l - m][E re, E im, B re, B im][lane], summed over the lanes at the end.
            let mut acc = vec![[[0.0; W]; 4]; nm * len];
            // [map][Q even, Q odd, U even, U odd][re, im][lane] of the ring pairs.
            let mut x = vec![[[[0.0; NB]; 2]; 4]; nm];
            for (rows, count) in self.blocks::<NB>(m, s) {
                let (cos, sin) = self.lanes(&rows);
                x.iter_mut().for_each(|v| *v = [[[0.0; NB]; 2]; 4]);
                for (lane, &p) in rows[..count].iter().enumerate() {
                    for (i, v) in x.iter_mut().enumerate() {
                        let (qn, un) = spectra[p][m * nm + i];
                        let (qs, us) = if p < pairs { spectra[nr - 1 - p][m * nm + i] } else { (C64::default(), C64::default()) };
                        for (t, z) in [qn + qs, qn - qs, un + us, un - us].into_iter().enumerate() {
                            v[t][0][lane] = z.re;
                            v[t][1][lane] = z.im;
                        }
                    }
                }
                self.recurse_spin_block(m, s, &coef, &cos, &sin, |l, lp, lm| {
                    // lambda+ is even in l + m + s about the equator, lambda- odd.
                    let par = (l + m + s) & 1;
                    for (i, v) in x.iter().enumerate() {
                        // e -= v[par] lp + i lm v[3 - par], b -= v[2 + par] lp - i lm v[1 - par].
                        let mut t = [[0.0; NB]; 4];
                        for k in 0..NB {
                            t[0][k] = v[par][0][k] * lp[k] - lm[k] * v[3 - par][1][k];
                            t[1][k] = v[par][1][k] * lp[k] + lm[k] * v[3 - par][0][k];
                            t[2][k] = v[2 + par][0][k] * lp[k] + lm[k] * v[1 - par][1][k];
                            t[3][k] = v[2 + par][1][k] * lp[k] - lm[k] * v[1 - par][0][k];
                        }
                        let a = &mut acc[i * len + l - m];
                        for c in 0..4 {
                            let f = fold(&t[c]);
                            for j in 0..W {
                                a[c][j] -= f[j];
                            }
                        }
                    }
                });
            }
            let sum = |r: usize, i: usize| C64::new(lane_sum(acc[r][i]), lane_sum(acc[r][i + 1]));
            ((0..nm * len).map(|r| sum(r, 0)).collect::<Vec<_>>(), (0..nm * len).map(|r| sum(r, 2)).collect::<Vec<_>>())
        }));
        (0..nm)
            .map(|i| {
                let mut e = Vec::with_capacity(self.nalm());
                let mut b = Vec::with_capacity(self.nalm());
                for (m, (pe, pb)) in per_m.iter().enumerate() {
                    let len = self.lmax + 1 - m;
                    e.extend(pe[i * len..(i + 1) * len].iter().map(|v| v * area));
                    b.extend(pb[i * len..(i + 1) * len].iter().map(|v| v * area));
                }
                (e, b)
            })
            .collect()
    }

    /// Map cotangents `(Q, U)` of [`analysis_spin`](Self::analysis_spin) from
    /// the cotangents of `(E, B)` (each coefficient counted once).
    pub fn analysis_spin_vjp(&self, s: usize, cots: &[(&[C64], &[C64])]) -> Vec<(Vec<f64>, Vec<f64>)> {
        const NB: usize = SPIN_LANES;
        assert!(s >= 1 && s <= self.lmax, "spin must be in 1..=lmax (spin 0: analysis_vjp)");
        cots.iter().for_each(|(e, b)| assert!(e.len() == self.nalm() && b.len() == self.nalm()));
        let nr = self.rings.len();
        let nm = cots.len();
        let pairs = self.pairs();
        // Per m: [ring][map] sums G^Q, G^U.
        let per_m = par_map(self.lmax + 1, self.threads, |m| simd(|| {
            let coef = self.spin_coefficients(m, s);
            let base = self.index(m, m);
            let mut g = vec![(C64::default(), C64::default()); nr * nm];
            // [map][parity][cE l+, cB l-, cB l+, cE l-][re, im][lane].
            let mut acc = vec![[[[[0.0; NB]; 2]; 4]; 2]; nm];
            for (rows, count) in self.blocks::<NB>(m, s) {
                let (cos, sin) = self.lanes(&rows);
                acc.iter_mut().for_each(|a| *a = [[[[0.0; NB]; 2]; 4]; 2]);
                self.recurse_spin_block(m, s, &coef, &cos, &sin, |l, lp, lm| {
                    let par = (l + m + s) & 1;
                    for (i, (ce, cb)) in cots.iter().enumerate() {
                        let (e, b) = (ce[base + l - m], cb[base + l - m]);
                        let a = &mut acc[i][par];
                        for k in 0..NB {
                            a[0][0][k] += e.re * lp[k];
                            a[0][1][k] += e.im * lp[k];
                            a[1][0][k] += b.re * lm[k];
                            a[1][1][k] += b.im * lm[k];
                            a[2][0][k] += b.re * lp[k];
                            a[2][1][k] += b.im * lp[k];
                            a[3][0][k] += e.re * lm[k];
                            a[3][1][k] += e.im * lm[k];
                        }
                    }
                });
                for (lane, &p) in rows[..count].iter().enumerate() {
                    for (i, a) in acc.iter().enumerate() {
                        // The terms cE l+, i cB l-, cB l+, -i cE l- of a parity.
                        let term = |par: usize, t: usize| {
                            let z = C64::new(a[par][t][0][lane], a[par][t][1][lane]);
                            match t {
                                1 => C64::new(-z.im, z.re),
                                3 => C64::new(z.im, -z.re),
                                _ => z,
                            }
                        };
                        let north = |t: usize| term(0, t) + term(1, t);
                        g[p * nm + i] = (north(0) + north(1), north(2) + north(3));
                        if p < pairs {
                            // South ring: lambda+ times (-1)^(l+m+s), lambda- times -(-1)^(l+m+s).
                            let south = |t: usize| term(0, t) - term(0, t + 1) - term(1, t) + term(1, t + 1);
                            g[(nr - 1 - p) * nm + i] = (south(0), south(2));
                        }
                    }
                }
            }
            g
        }));
        let area = self.area();
        let per_ring = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let n = ring.nphi;
            let rf = &self.fft[&n];
            let (mut buf, mut ph) = (Vec::new(), Vec::new());
            phases(ring.phi0, 1.0, self.lmax + 1, &mut ph);
            (0..nm)
                .map(|i| {
                    let (mut hq, mut hu) = (vec![C64::default(); n], vec![C64::default(); n]);
                    for (m, g) in per_m.iter().enumerate() {
                        let (q, u) = g[r * nm + i];
                        hq[m % n] += q * ph[m];
                        hu[m % n] += u * ph[m];
                    }
                    let (mut q, mut u) = (vec![0.0; n], vec![0.0; n]);
                    rf.inverse_real(&hq, &mut q, &mut buf);
                    rf.inverse_real(&hu, &mut u, &mut buf);
                    q.iter_mut().chain(u.iter_mut()).for_each(|v| *v *= -area);
                    (q, u)
                })
                .collect::<Vec<_>>()
        });
        let mut out = vec![(vec![0.0; self.npix], vec![0.0; self.npix]); nm];
        for (r, values) in per_ring.into_iter().enumerate() {
            let start = self.rings[r].start;
            for (i, (q, u)) in values.into_iter().enumerate() {
                for (j, (a, b)) in q.into_iter().zip(u).enumerate() {
                    let p = self.nest[start + j] as usize;
                    out[i].0[p] = a;
                    out[i].1[p] = b;
                }
            }
        }
        out
    }

    /// First and second covariant derivatives of the map of `alm` in the
    /// orthonormal `(theta, phi)` frame: `d_theta f`, `d_phi f / sin`,
    /// `H_theta_theta`, `H_phi_phi`, `H_theta_phi` (`derivative_maps` of the
    /// reference).
    pub fn derivatives(&self, alm: &[C64]) -> [Vec<f64>; 5] {
        let sums = self.legendre(alm, true);
        self.synthesise(&sums, |ring, m, a, b, c| {
            let [ka, kb, kc] = derivative_coefficients(ring, m);
            std::array::from_fn(|k| ka[k] * a + kb[k] * b + kc[k] * c)
        })
    }

    /// Cotangent of the `a_lm` from those of the five derivative maps.
    pub fn derivatives_vjp(&self, cot: [&[f64]; 5]) -> Vec<C64> {
        let sums = self.analyse(cot, true, true, |ring, m, f| {
            let [ka, kb, kc] = derivative_coefficients(ring, m);
            let dot = |kk: [C64; 5]| (0..5).map(|k| kk[k].conj() * f[k]).sum::<C64>();
            [dot(ka), dot(kb), dot(kc)]
        });
        self.legendre_adjoint(&sums)
    }
}

/// The derivative maps as `R[ka A + kb B + kc C]` on a ring, `R[G] =
/// Re sum_m mult G_m e^{i m phi}`: per output (d1, d2, d11, d22, d12) the
/// factors of A, B and C.
fn derivative_coefficients(ring: &Ring, m: usize) -> [[C64; 5]; 3] {
    let (x, s) = (ring.cos, ring.sin);
    let mf = m as f64;
    let im = C64::new(0.0, mf);
    let r = |v: f64| C64::new(v, 0.0);
    let ka = [r(0.0), im / s, r(mf * mf / (s * s)), r(-mf * mf / (s * s)), -im * x / (s * s)];
    let kb = [r(1.0), r(0.0), r(-x / s), r(x / s), im / s];
    let kc = [r(0.0), r(0.0), r(-1.0), r(0.0), r(0.0)];
    [ka, kb, kc]
}

/// Harmonic E and B maps of `(Q, U)` pairs (NESTED), the linear map of
/// `stl_port.denoise.eb_maps`: healpy `map2alm([0, Q, U], pol=True, iter)`
/// then `alm2map` of `a_E` and of `a_B` (scalar maps), up to `lmax`; with
/// its exact transpose.
///
/// The iterated analysis is `a_0 = A x`, `a_{k+1} = a_k + A (x - S a_k)`,
/// i.e. `a = sum_{j <= iter} (I - A S)^j A x`, with `A` the spin-2
/// analysis ([`Sht::analysis_spin`], `Omega S^dagger`) and `S` healpy's
/// `alm2map_spin` synthesis, obtained from the transpose of `A` as `S a =
/// A^T (mult a) / Omega` (`mult` = 1 for `m = 0`, 2 otherwise, under the
/// convention of a coefficient counted once).  The transpose is `A^T sum_j
/// (I - S^T A^T)^j Y^T`, `S^T = mult A / Omega`, `Y^T = mult Y*`.
pub struct EbTransform {
    /// The transforms (`nside`, `lmax`).
    pub sht: Sht,
    /// Iterations of the analysis (healpy's `iter`).
    pub iter: usize,
    mult: Vec<f64>,
}

impl EbTransform {
    /// E/B maps of `nside` up to `lmax` (`3 nside - 1` as `eb_maps`) with
    /// `iter` analysis iterations (3 as `eb_maps`).
    pub fn new(nside: usize, lmax: usize, iter: usize) -> Self {
        let sht = Sht::new(nside, lmax);
        let mult = sht.lm().iter().map(|&(_, m)| if m == 0 { 1.0 } else { 2.0 }).collect();
        EbTransform { sht, iter, mult }
    }

    fn scaled(&self, a: &[C64], s: f64) -> Vec<C64> {
        a.iter().zip(&self.mult).map(|(v, w)| v * (w * s)).collect()
    }

    /// healpy's `alm2map_spin` (spin 2) of the `(a_E, a_B)` pairs.
    pub fn synthesis_spin2(&self, alms: &[(Vec<C64>, Vec<C64>)]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let inv = 1.0 / self.sht.area();
        let scaled: Vec<(Vec<C64>, Vec<C64>)> = alms.iter().map(|(e, b)| (self.scaled(e, inv), self.scaled(b, inv))).collect();
        let refs: Vec<(&[C64], &[C64])> = scaled.iter().map(|(e, b)| (e.as_slice(), b.as_slice())).collect();
        self.sht.analysis_spin_vjp(2, &refs)
    }

    /// Transpose of [`synthesis_spin2`](Self::synthesis_spin2): `mult A y / Omega`.
    fn synthesis_spin2_t(&self, maps: &[(Vec<f64>, Vec<f64>)]) -> Vec<(Vec<C64>, Vec<C64>)> {
        let inv = 1.0 / self.sht.area();
        let refs: Vec<(&[f64], &[f64])> = maps.iter().map(|(q, u)| (q.as_slice(), u.as_slice())).collect();
        self.sht.analysis_spin(2, &refs).into_iter().map(|(e, b)| (self.scaled(&e, inv), self.scaled(&b, inv))).collect()
    }

    fn analysis(&self, maps: &[(Vec<f64>, Vec<f64>)]) -> Vec<(Vec<C64>, Vec<C64>)> {
        let refs: Vec<(&[f64], &[f64])> = maps.iter().map(|(q, u)| (q.as_slice(), u.as_slice())).collect();
        self.sht.analysis_spin(2, &refs)
    }

    fn analysis_t(&self, cots: &[(Vec<C64>, Vec<C64>)]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let refs: Vec<(&[C64], &[C64])> = cots.iter().map(|(e, b)| (e.as_slice(), b.as_slice())).collect();
        self.sht.analysis_spin_vjp(2, &refs)
    }

    fn pairs(&self, x: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let n = self.sht.npix;
        assert_eq!(x.len() % (2 * n), 0, "pairs [b, 2, npix]");
        x.chunks(2 * n).map(|p| (p[..n].to_vec(), p[n..].to_vec())).collect()
    }

    /// `(a_E, a_B)` of every pair: healpy's iterated `map2alm(pol=True)`.
    pub fn alms(&self, x: &[f64]) -> Vec<(Vec<C64>, Vec<C64>)> {
        let maps = self.pairs(x);
        let mut a = self.analysis(&maps);
        for _ in 0..self.iter {
            let back = self.synthesis_spin2(&a);
            let res: Vec<(Vec<f64>, Vec<f64>)> = maps
                .iter()
                .zip(&back)
                .map(|((q, u), (bq, bu))| (q.iter().zip(bq).map(|(x, y)| x - y).collect(), u.iter().zip(bu).map(|(x, y)| x - y).collect()))
                .collect();
            for ((e, b), (de, db)) in a.iter_mut().zip(self.analysis(&res)) {
                e.iter_mut().zip(&de).for_each(|(v, d)| *v += d);
                b.iter_mut().zip(&db).for_each(|(v, d)| *v += d);
            }
        }
        a
    }

    /// The E and B maps `[b, 2, npix]` of the pairs `x` `[b, 2, npix]`.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len());
        for (e, b) in self.alms(x) {
            out.extend(self.sht.synthesis(&e));
            out.extend(self.sht.synthesis(&b));
        }
        out
    }

    /// The transpose of [`apply`](Self::apply): map cotangents of `(Q, U)`
    /// `[b, 2, npix]` from those of `(E, B)` `[b, 2, npix]`.
    pub fn adjoint(&self, g: &[f64]) -> Vec<f64> {
        // Y^T: alm cotangents of the scalar syntheses.
        let c: Vec<(Vec<C64>, Vec<C64>)> = self
            .pairs(g)
            .iter()
            .map(|(ge, gb)| (self.scaled(&self.sht.adjoint_synthesis(ge), 1.0), self.scaled(&self.sht.adjoint_synthesis(gb), 1.0)))
            .collect();
        // sum_j (I - S^T A^T)^j c, then A^T.
        let mut v = c.clone();
        let mut acc = c;
        for _ in 0..self.iter {
            let st = self.synthesis_spin2_t(&self.analysis_t(&v));
            for (((ve, vb), (se, sb)), (ae, ab)) in v.iter_mut().zip(&st).zip(acc.iter_mut()) {
                ve.iter_mut().zip(se).for_each(|(x, y)| *x -= y);
                vb.iter_mut().zip(sb).for_each(|(x, y)| *x -= y);
                ae.iter_mut().zip(ve.iter()).for_each(|(x, y)| *x += y);
                ab.iter_mut().zip(vb.iter()).for_each(|(x, y)| *x += y);
            }
        }
        let mut out = Vec::with_capacity(g.len());
        for (q, u) in self.analysis_t(&acc) {
            out.extend(q);
            out.extend(u);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::healpix::pix2ang_nest;

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    fn random_alm(sht: &Sht, seed: u64) -> Vec<C64> {
        let re = noise(sht.nalm(), seed);
        let im = noise(sht.nalm(), seed + 1);
        sht.lm().iter().enumerate().map(|(i, &(_, m))| C64::new(re[i], if m == 0 { 0.0 } else { im[i] })).collect()
    }

    /// Ring centres against the NESTED pixel centres.
    #[test]
    fn rings_match_pixel_centres() {
        for nside in [1usize, 2, 4, 16] {
            let sht = Sht::new(nside, 3 * nside - 1);
            for ring in &sht.rings {
                for j in 0..ring.nphi {
                    let (theta, phi) = pix2ang_nest(sht.nest[ring.start + j] as i64, nside as i64);
                    assert!((theta.cos() - ring.cos).abs() < 1e-14, "nside {nside}: cos");
                    let dphi = (phi - ring.phi0 - j as f64 * 2.0 * std::f64::consts::PI / ring.nphi as f64).rem_euclid(2.0 * std::f64::consts::PI);
                    assert!(dphi.min(2.0 * std::f64::consts::PI - dphi) < 1e-12, "nside {nside}: phi");
                }
            }
        }
    }

    /// `Y` against the closed forms of Y_00, Y_10, Y_11, Y_20.
    #[test]
    fn synthesis_of_low_multipoles() {
        let nside = 8;
        let sht = Sht::new(nside, 2 * nside);
        let pi = std::f64::consts::PI;
        let mut alm = vec![C64::default(); sht.nalm()];
        alm[sht.index(0, 0)] = C64::new(0.3, 0.0);
        alm[sht.index(1, 0)] = C64::new(-0.7, 0.0);
        alm[sht.index(1, 1)] = C64::new(0.2, 0.5);
        alm[sht.index(2, 0)] = C64::new(1.1, 0.0);
        let x = sht.synthesis(&alm);
        for (p, v) in x.iter().enumerate() {
            let (t, ph) = pix2ang_nest(p as i64, nside as i64);
            // Normalised Legendre functions with the Condon-Shortley phase.
            let y00 = (1.0 / (4.0 * pi)).sqrt();
            let y10 = (3.0 / (4.0 * pi)).sqrt() * t.cos();
            let y11 = -(3.0 / (8.0 * pi)).sqrt() * t.sin();
            let y20 = (5.0 / (16.0 * pi)).sqrt() * (3.0 * t.cos() * t.cos() - 1.0);
            let e = 0.3 * y00 - 0.7 * y10 + 2.0 * (C64::new(0.2, 0.5) * C64::from_polar(y11, ph)).re + 1.1 * y20;
            assert!((v - e).abs() < 1e-13, "pixel {p}: {v} vs {e}");
        }
    }

    /// `<Y a, x> = Re <a, mult Y* x>` and `<D a, c> = Re <a, D* c>` for the
    /// derivative maps; the analysis inverts the synthesis at low l.
    #[test]
    fn adjoints_and_inverse() {
        for (nside, lmax) in [(4usize, 11usize), (16, 47), (16, 20)] {
            let sht = Sht::new(nside, lmax);
            let a = random_alm(&sht, 3);
            let x = noise(sht.npix, 5);
            let ya = sht.synthesis(&a);
            let lhs: f64 = ya.iter().zip(&x).map(|(p, q)| p * q).sum();
            let ys = sht.adjoint_synthesis(&x);
            let rhs: f64 = sht.lm().iter().zip(a.iter().zip(&ys)).map(|(&(_, m), (p, q))| if m == 0 { 1.0 } else { 2.0 } * (p.conj() * q).re).sum();
            assert!((lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0), "nside {nside}: synthesis adjoint {lhs} {rhs}");
            let d = sht.derivatives(&a);
            let c: Vec<Vec<f64>> = (0..5).map(|k| noise(sht.npix, 7 + k as u64)).collect();
            let lhs: f64 = (0..5).map(|k| d[k].iter().zip(&c[k]).map(|(p, q)| p * q).sum::<f64>()).sum();
            let g = sht.derivatives_vjp([&c[0], &c[1], &c[2], &c[3], &c[4]]);
            let rhs: f64 = a.iter().zip(&g).map(|(p, q)| (p.conj() * q).re).sum();
            assert!((lhs - rhs).abs() <= 1e-11 * lhs.abs().max(1.0), "nside {nside}: derivative adjoint {lhs} {rhs}");
        }
        let sht = Sht::new(32, 16);
        let a = random_alm(&sht, 11);
        let back = sht.analysis(&sht.synthesis(&a));
        let err = a.iter().zip(&back).fold(0f64, |e, (p, q)| e.max((p - q).norm()));
        assert!(err < 1e-2, "analysis of a band-limited map (iter = 0 quadrature): {err}");
    }

    /// Spin analysis: its transpose (`<A x, c> = <x, A^T c>`, each
    /// coefficient once) for spins 1 to 3, and the spin-weighted
    /// orthonormality of a band-limited pure-E field (`E = Y_20` read back).
    #[test]
    fn spin_adjoint() {
        for (nside, lmax) in [(4usize, 11usize), (8, 20)] {
            let sht = Sht::new(nside, lmax);
            for s in 1..=3 {
                let (q, u) = (noise(sht.npix, 21), noise(sht.npix, 22));
                let (e, b) = sht.analysis_spin(s, &[(&q, &u)]).pop().unwrap();
                let (ce, cb) = (random_alm(&sht, 23), random_alm(&sht, 25));
                let lhs: f64 = e.iter().zip(&ce).chain(b.iter().zip(&cb)).map(|(v, c)| (c.conj() * v).re).sum();
                let (gq, gu) = sht.analysis_spin_vjp(s, &[(&ce, &cb)]).pop().unwrap();
                let rhs: f64 = gq.iter().zip(&q).chain(gu.iter().zip(&u)).map(|(g, x)| g * x).sum();
                assert!((lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0), "nside {nside} spin {s}: {lhs} {rhs}");
                for (l, m) in sht.lm() {
                    if l < s {
                        assert!(e[sht.index(l, m)].norm() == 0.0 && b[sht.index(l, m)].norm() == 0.0);
                    }
                }
            }
        }
    }

    /// E/B maps: the transpose (`<F x, y> = <x, F^T y>`, iter 0 and 3), the
    /// spin-2 synthesis as healpy's (`A S a = a` on a band-limited pure E
    /// and pure B, to the quadrature error, which the iterations reduce).
    #[test]
    fn eb_transform_adjoint_and_iterations() {
        for iter in [0usize, 3] {
            let f = EbTransform::new(8, 23, iter);
            let n = f.sht.npix;
            let x: Vec<f64> = noise(4 * n, 31 + iter as u64);
            let y: Vec<f64> = noise(4 * n, 41 + iter as u64);
            let fx = f.apply(&x);
            let fty = f.adjoint(&y);
            let lhs: f64 = fx.iter().zip(&y).map(|(a, b)| a * b).sum();
            let rhs: f64 = x.iter().zip(&fty).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0), "iter {iter}: {lhs} {rhs}");
        }
        let f0 = EbTransform::new(16, 20, 0);
        let f3 = EbTransform::new(16, 20, 3);
        let mut e = random_alm(&f0.sht, 51);
        let mut b = random_alm(&f0.sht, 53);
        for (i, (l, _)) in f0.sht.lm().into_iter().enumerate() {
            if l < 2 {
                e[i] = C64::default();
                b[i] = C64::default();
            }
        }
        let (q, u) = f0.synthesis_spin2(&[(e.clone(), b.clone())]).pop().unwrap();
        let x: Vec<f64> = q.into_iter().chain(u).collect();
        let err = |f: &EbTransform| {
            let (ae, ab) = f.alms(&x).pop().unwrap();
            ae.iter().zip(&e).chain(ab.iter().zip(&b)).fold(0f64, |m, (p, q)| m.max((p - q).norm()))
        };
        let (e0, e3) = (err(&f0), err(&f3));
        assert!(e0 < 1e-2 && e3 < 1e-2 * e0, "read back: iter 0 {e0}, iter 3 {e3}");
    }

    /// healpy's spin-2 synthesis (`EbTransform::synthesis_spin2`) from the
    /// covariant Hessians of `psi_X = Y(N_l a_X)`, the formula of the device
    /// transforms (`stl-gpu` `harmonic.cu`, `eb_from_hessians`): `Q = -(H_tt
    /// - H_pp)[psi_E] + 2 H_tp[psi_B]`, `U = -2 H_tp[psi_E] - (H_tt -
    /// H_pp)[psi_B]`.
    #[test]
    fn spin2_synthesis_from_hessians() {
        let f = EbTransform::new(16, 40, 0);
        let sht = &f.sht;
        let mut e = random_alm(sht, 61);
        let mut b = random_alm(sht, 63);
        let mut ne = e.clone();
        let mut nb = b.clone();
        for (i, (l, _)) in sht.lm().into_iter().enumerate() {
            if l < 2 {
                e[i] = C64::default();
                b[i] = C64::default();
            }
            let n = if l < 2 { 0.0 } else { 1.0 / (((l - 1) * l * (l + 1) * (l + 2)) as f64).sqrt() };
            ne[i] = e[i] * n;
            nb[i] = b[i] * n;
        }
        let (q, u) = f.synthesis_spin2(&[(e, b)]).pop().unwrap();
        let de = sht.derivatives(&ne);
        let db = sht.derivatives(&nb);
        let mut err = 0f64;
        for p in 0..sht.npix {
            let qq = -(de[2][p] - de[3][p]) + 2.0 * db[4][p];
            let uu = -2.0 * de[4][p] - (db[2][p] - db[3][p]);
            err = err.max((qq - q[p]).abs()).max((uu - u[p]).abs());
        }
        let scale = q.iter().chain(&u).fold(0f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-12 * scale, "{}", err / scale);
    }

    /// Derivative maps of `Y_20 + Y_21` against finite differences of the
    /// synthesised field at the pixel centres.
    #[test]
    fn derivatives_match_finite_differences() {
        let nside = 16;
        let sht = Sht::new(nside, 3 * nside - 1);
        let mut alm = vec![C64::default(); sht.nalm()];
        alm[sht.index(2, 0)] = C64::new(0.8, 0.0);
        alm[sht.index(2, 1)] = C64::new(0.3, -0.4);
        alm[sht.index(3, 2)] = C64::new(-0.2, 0.1);
        let pi = std::f64::consts::PI;
        // Direct field: f(theta, phi) from the closed forms.
        let field = |t: f64, p: f64| {
            let (x, s) = (t.cos(), t.sin());
            let y20 = (5.0 / (16.0 * pi)).sqrt() * (3.0 * x * x - 1.0);
            let y21 = -(15.0 / (8.0 * pi)).sqrt() * s * x;
            let y32 = (105.0 / (32.0 * pi)).sqrt() * s * s * x;
            0.8 * y20 + 2.0 * (C64::new(0.3, -0.4) * C64::from_polar(y21, p)).re + 2.0 * (C64::new(-0.2, 0.1) * C64::from_polar(y32, 2.0 * p)).re
        };
        let d = sht.derivatives(&alm);
        let h = 1e-4;
        for p in (0..sht.npix).step_by(37) {
            let (t, ph) = pix2ang_nest(p as i64, nside as i64);
            let s = t.sin();
            let ft = (field(t + h, ph) - field(t - h, ph)) / (2.0 * h);
            let fp = (field(t, ph + h) - field(t, ph - h)) / (2.0 * h);
            let ftt = (field(t + h, ph) - 2.0 * field(t, ph) + field(t - h, ph)) / (h * h);
            let fpp = (field(t, ph + h) - 2.0 * field(t, ph) + field(t, ph - h)) / (h * h);
            let ftp = (field(t + h, ph + h) - field(t + h, ph - h) - field(t - h, ph + h) + field(t - h, ph - h)) / (4.0 * h * h);
            let expect = [ft, fp / s, ftt, fpp / (s * s) + t.cos() / s * ft, ftp / s - t.cos() / (s * s) * fp];
            for k in 0..5 {
                assert!((d[k][p] - expect[k]).abs() < 1e-5, "pixel {p} derivative {k}: {} vs {}", d[k][p], expect[k]);
            }
        }
    }

    /// The scaled recursion at high m on polar rings: Y_mm against its
    /// closed form where it is in range.
    #[test]
    fn scaled_recursion() {
        let sht = Sht::new(64, 1000);
        const NB: usize = LANES;
        // Lanes of mixed latitudes: a polar ring (scaled at high m) and nearer the equator.
        let rows: [usize; NB] = std::array::from_fn(|i| 3 + 8 * i);
        let (cos, sin) = sht.lanes(&rows);
        for m in [0usize, 10, 200, 999] {
            let coef = sht.coefficients(m);
            let mut first = [None; NB];
            sht.recurse_block(m, &coef, &cos, &sin, |l, lam, _| {
                for i in 0..NB {
                    if first[i].is_none() && lam[i] != 0.0 {
                        first[i] = Some((l, lam[i]));
                    }
                }
            });
            for i in 0..NB {
                if let Some((l, lam)) = first[i] {
                    if l == m {
                        let expect = (sht.log_mm[m] + m as f64 * sin[i].ln()).exp();
                        assert!((lam.abs() - expect).abs() <= 1e-12 * expect, "m {m} lane {i}");
                    }
                }
            }
        }
    }
}
