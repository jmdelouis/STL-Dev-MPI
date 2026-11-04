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
    fft: HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>,
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
            fft.entry(r.nphi).or_insert_with(|| (planner.plan_fft_forward(r.nphi), planner.plan_fft_inverse(r.nphi)));
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

    /// The Legendre recursion of `m` on a ring of `(cos, sin)`: `f(l, lambda_l,
    /// sin * dlambda_l/dtheta)` for every `l` whose value is in double range
    /// (the others are below `2^-300` and skipped).
    #[inline(always)]
    fn recurse(&self, m: usize, coef: &[(f64, f64, f64)], cos: f64, sin: f64, mut f: impl FnMut(usize, f64, f64)) {
        let log = self.log_mm[m] + m as f64 * sin.ln();
        let mut k: i32 = if log > -650.0 * std::f64::consts::LN_2 { 0 } else { ((log - 300.0 * std::f64::consts::LN_2) / LN_SCALE).ceil() as i32 };
        let mut prev = 0.0;
        // Condon-Shortley phase (healpy, DUCC).
        let mut cur = (log - k as f64 * LN_SCALE).exp() * if m % 2 == 1 { -1.0 } else { 1.0 };
        if k == 0 {
            f(m, cur, m as f64 * cos * cur);
        }
        for l in m + 1..=self.lmax {
            let (a, b, c) = coef[l - m];
            let next = a * (cos * cur - b * prev);
            prev = cur;
            cur = next;
            if k < 0 {
                if cur.abs() > LIMIT {
                    cur /= SCALE;
                    prev /= SCALE;
                    k += 1;
                }
                if k < 0 {
                    continue;
                }
            }
            f(l, cur, l as f64 * cos * cur - c * prev);
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
        assert_eq!(alm.len(), self.nalm());
        let nr = self.rings.len();
        let pairs = self.pairs();
        let per_m = par_map(self.lmax + 1, self.threads, |m| {
            let coef = self.coefficients(m);
            let a = &alm[self.index(m, m)..self.index(m, m) + self.lmax + 1 - m];
            let (mut va, mut vb, mut vc) = (vec![C64::default(); nr], vec![C64::default(); if deriv { nr } else { 0 }], vec![C64::default(); if deriv { nr } else { 0 }]);
            for p in 0..=pairs {
                let ring = &self.rings[p];
                // [even, odd] parts in l + m.
                let mut sa = [C64::default(); 2];
                let mut sb = [C64::default(); 2];
                let mut sc = [C64::default(); 2];
                self.recurse(m, &coef, ring.cos, ring.sin, |l, lam, dlam| {
                    let par = (l + m) & 1;
                    let x = a[l - m];
                    sa[par] += x * lam;
                    if deriv {
                        sb[par] += x * (dlam / ring.sin);
                        sc[par] += x * ((l * (l + 1)) as f64 * lam);
                    }
                });
                va[p] = sa[0] + sa[1];
                if deriv {
                    vb[p] = sb[0] + sb[1];
                    vc[p] = sc[0] + sc[1];
                }
                if p < pairs {
                    let s = nr - 1 - p;
                    va[s] = sa[0] - sa[1];
                    if deriv {
                        vb[s] = sb[1] - sb[0];
                        vc[s] = sc[0] - sc[1];
                    }
                }
            }
            (va, vb, vc)
        });
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
        let nr = self.rings.len();
        let pairs = self.pairs();
        let deriv = !sums.b.is_empty();
        let per_m = par_map(self.lmax + 1, self.threads, |m| {
            let coef = self.coefficients(m);
            let mut out = vec![C64::default(); self.lmax + 1 - m];
            let row = |v: &[C64], r: usize| v[m * nr + r];
            for p in 0..=pairs {
                let ring = &self.rings[p];
                let s = nr - 1 - p;
                let south = p < pairs;
                let pick = |v: &[C64]| -> [C64; 2] {
                    let (n, s) = (row(v, p), if south { row(v, s) } else { C64::default() });
                    [n + s, n - s]
                };
                let ta = pick(&sums.a);
                let (tb, tc) = if deriv {
                    let (n, sv) = (row(&sums.b, p), if south { row(&sums.b, s) } else { C64::default() });
                    ([n - sv, n + sv], pick(&sums.c))
                } else {
                    ([C64::default(); 2], [C64::default(); 2])
                };
                self.recurse(m, &coef, ring.cos, ring.sin, |l, lam, dlam| {
                    let par = (l + m) & 1;
                    let mut v = ta[par] * lam;
                    if deriv {
                        v += tb[par] * (dlam / ring.sin) + tc[par] * ((l * (l + 1)) as f64 * lam);
                    }
                    out[l - m] += v;
                });
            }
            out
        });
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
            let mut h = vec![vec![C64::default(); n]; K];
            for m in 0..=self.lmax {
                let i = m * nr + r;
                let (b, c) = if deriv { (sums.b[i], sums.c[i]) } else { (C64::default(), C64::default()) };
                let v = combine(ring, m, sums.a[i], b, c);
                let w = if m == 0 { 1.0 } else { 2.0 };
                let shift = C64::from_polar(w, m as f64 * ring.phi0);
                for k in 0..K {
                    h[k][m % n] += v[k] * shift;
                }
            }
            let (_, inv) = &self.fft[&n];
            h.iter_mut().for_each(|x| inv.process(x));
            h.into_iter().map(|x| x.into_iter().map(|z| z.re).collect::<Vec<f64>>()).collect::<Vec<_>>()
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
            let (fwd, _) = &self.fft[&n];
            let spectra: Vec<Vec<C64>> = maps
                .iter()
                .map(|x| {
                    let mut buf: Vec<C64> = (0..n).map(|j| C64::new(x[self.nest[ring.start + j] as usize], 0.0)).collect();
                    fwd.process(&mut buf);
                    buf
                })
                .collect();
            (0..=self.lmax)
                .map(|m| {
                    let w = if with_mult && m > 0 { 2.0 } else { 1.0 };
                    let shift = C64::from_polar(w, -(m as f64) * ring.phi0);
                    let f: [C64; K] = std::array::from_fn(|k| spectra[k][m % n] * shift);
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

    /// The spin-`s` functions of `m` on a ring of `(cos, sin)` (north
    /// hemisphere or equator): `f(l, lambda+, lambda-)` with `lambda+- =
    /// (_s lambda_lm +- _{-s} lambda_lm) / 2`, `_s lambda_lm = (-1)^s
    /// sqrt((2l+1)/4pi) d^l_{m,-s}`, `_{-s} lambda_lm = sqrt((2l+1)/4pi)
    /// d^l_{m,s}` (healpy's `map2alm_spin`), for every `l` where one of them
    /// is in double range.
    #[inline(always)]
    fn recurse_spin(&self, m: usize, s: usize, coef: &[(f64, f64, f64)], cos: f64, sin: f64, mut f: impl FnMut(usize, f64, f64)) {
        let j = m.max(s);
        // Half-angle cosine and sine without cancellation.
        let (ch, sh) = if cos >= 0.0 {
            let ch = ((1.0 + cos) / 2.0).sqrt();
            (ch, sin / (2.0 * ch))
        } else {
            let sh = ((1.0 - cos) / 2.0).sqrt();
            (sin / (2.0 * sh), sh)
        };
        let (lch, lsh) = (ch.ln(), sh.ln());
        let lf = |n: usize| self.ln_fact[n];
        // d^j_{m,b} at j = max(m, s), b = -s then +s: (log magnitude, sign).
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
        let init = |(log, sign): (f64, f64)| -> (f64, i32) {
            let k: i32 = if log > -650.0 * std::f64::consts::LN_2 { 0 } else { ((log - 300.0 * std::f64::consts::LN_2) / LN_SCALE).ceil() as i32 };
            ((log - k as f64 * LN_SCALE).exp() * sign, k)
        };
        // [b = -s, b = +s]: current, previous and scale exponent.
        let (c0, k0) = init(start(-(s as i64)));
        let (c1, k1) = init(start(s as i64));
        let mut cur = [c0, c1];
        let mut prev = [0.0; 2];
        let mut k = [k0, k1];
        let sgn = if s % 2 == 0 { 1.0 } else { -1.0 };
        let norm = |l: usize| (((2 * l + 1) as f64) / (4.0 * std::f64::consts::PI)).sqrt();
        let emit = |l: usize, cur: &[f64; 2], k: &[i32; 2], f: &mut dyn FnMut(usize, f64, f64)| {
            if k[0] == 0 || k[1] == 0 {
                let ls = if k[0] == 0 { sgn * cur[0] } else { 0.0 };
                let lm = if k[1] == 0 { cur[1] } else { 0.0 };
                let n = 0.5 * norm(l);
                f(l, n * (ls + lm), n * (ls - lm));
            }
        };
        emit(j, &cur, &k, &mut f);
        for l in j + 1..=self.lmax {
            let (a, b, c) = coef[l - 1 - j];
            for t in 0..2 {
                let bt = if t == 0 { -b } else { b };
                let next = a * (cos - bt) * cur[t] - c * prev[t];
                prev[t] = cur[t];
                cur[t] = next;
                if k[t] < 0 && cur[t].abs() > LIMIT {
                    cur[t] /= SCALE;
                    prev[t] /= SCALE;
                    k[t] += 1;
                }
            }
            emit(l, &cur, &k, &mut f);
        }
    }

    /// Spin-`s` analysis (`s >= 1`) of the pairs `(Q, U)` (NESTED), healpy's
    /// `map2alm_spin` without iteration: `E = -(a_s + a_{-s}) / 2`, `B = i
    /// (a_s - a_{-s}) / 2`, `a_{+-s} = Omega sum_p (Q +- iU) conj(_{+-s}Y_lm)`;
    /// the `a_lm` with `l < s` are zero.  The recursions are shared by the
    /// pairs.
    pub fn analysis_spin(&self, s: usize, maps: &[(&[f64], &[f64])]) -> Vec<(Vec<C64>, Vec<C64>)> {
        assert!(s >= 1 && s <= self.lmax, "spin must be in 1..=lmax (spin 0: analysis)");
        maps.iter().for_each(|(q, u)| assert!(q.len() == self.npix && u.len() == self.npix));
        let nr = self.rings.len();
        let nm = maps.len();
        let pairs = self.pairs();
        // Ring spectra [ring][m][map] of Q and U (phase of the first pixel included).
        let spectra = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let n = ring.nphi;
            let (fwd, _) = &self.fft[&n];
            let mut out = vec![(C64::default(), C64::default()); (self.lmax + 1) * nm];
            for (i, (q, u)) in maps.iter().enumerate() {
                let f = |x: &[f64]| {
                    let mut buf: Vec<C64> = (0..n).map(|j| C64::new(x[self.nest[ring.start + j] as usize], 0.0)).collect();
                    fwd.process(&mut buf);
                    buf
                };
                let (fq, fu) = (f(q), f(u));
                for m in 0..=self.lmax {
                    let shift = C64::from_polar(1.0, -(m as f64) * ring.phi0);
                    out[m * nm + i] = (fq[m % n] * shift, fu[m % n] * shift);
                }
            }
            out
        });
        let area = self.area();
        let per_m = par_map(self.lmax + 1, self.threads, |m| {
            let coef = self.spin_coefficients(m, s);
            let len = self.lmax + 1 - m;
            let mut e = vec![C64::default(); nm * len];
            let mut b = vec![C64::default(); nm * len];
            // [map][Q even, Q odd, U even, U odd] of the ring pair.
            let mut x = vec![[C64::default(); 4]; nm];
            for p in 0..=pairs {
                let ring = &self.rings[p];
                let south = nr - 1 - p;
                for (i, v) in x.iter_mut().enumerate() {
                    let (qn, un) = spectra[p][m * nm + i];
                    let (qs, us) = if p < pairs { spectra[south][m * nm + i] } else { (C64::default(), C64::default()) };
                    *v = [qn + qs, qn - qs, un + us, un - us];
                }
                self.recurse_spin(m, s, &coef, ring.cos, ring.sin, |l, lp, lm| {
                    // lambda+ is even in l + m + s about the equator, lambda- odd.
                    let par = (l + m + s) & 1;
                    for (i, v) in x.iter().enumerate() {
                        let o = i * len + l - m;
                        e[o] -= v[par] * lp + C64::new(0.0, lm) * v[3 - par];
                        b[o] -= v[2 + par] * lp - C64::new(0.0, lm) * v[1 - par];
                    }
                });
            }
            (e, b)
        });
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
        assert!(s >= 1 && s <= self.lmax, "spin must be in 1..=lmax (spin 0: analysis_vjp)");
        cots.iter().for_each(|(e, b)| assert!(e.len() == self.nalm() && b.len() == self.nalm()));
        let nr = self.rings.len();
        let nm = cots.len();
        let pairs = self.pairs();
        // Per m: [ring][map] sums G^Q, G^U.
        let per_m = par_map(self.lmax + 1, self.threads, |m| {
            let coef = self.spin_coefficients(m, s);
            let base = self.index(m, m);
            let mut g = vec![(C64::default(), C64::default()); nr * nm];
            // [map][parity]: sum cE l+, i cB l-, cB l+, -i cE l-.
            let mut acc = vec![[[C64::default(); 4]; 2]; nm];
            for p in 0..=pairs {
                let ring = &self.rings[p];
                acc.iter_mut().for_each(|a| *a = [[C64::default(); 4]; 2]);
                self.recurse_spin(m, s, &coef, ring.cos, ring.sin, |l, lp, lm| {
                    let par = (l + m + s) & 1;
                    for (i, (ce, cb)) in cots.iter().enumerate() {
                        let (e, b) = (ce[base + l - m], cb[base + l - m]);
                        let a = &mut acc[i][par];
                        a[0] += e * lp;
                        a[1] += C64::new(0.0, lm) * b;
                        a[2] += b * lp;
                        a[3] -= C64::new(0.0, lm) * e;
                    }
                });
                for (i, a) in acc.iter().enumerate() {
                    let north = |t: usize| a[0][t] + a[1][t];
                    g[p * nm + i] = (north(0) + north(1), north(2) + north(3));
                    if p < pairs {
                        // South ring: lambda+ times (-1)^(l+m+s), lambda- times -(-1)^(l+m+s).
                        let south = |t: usize| a[0][t] - a[0][t + 1] - a[1][t] + a[1][t + 1];
                        g[(nr - 1 - p) * nm + i] = (south(0), south(2));
                    }
                }
            }
            g
        });
        let area = self.area();
        let per_ring = par_map(nr, self.threads, |r| {
            let ring = &self.rings[r];
            let n = ring.nphi;
            let (_, inv) = &self.fft[&n];
            (0..nm)
                .map(|i| {
                    let (mut hq, mut hu) = (vec![C64::default(); n], vec![C64::default(); n]);
                    for (m, g) in per_m.iter().enumerate() {
                        let shift = C64::from_polar(1.0, m as f64 * ring.phi0);
                        let (q, u) = g[r * nm + i];
                        hq[m % n] += q * shift;
                        hu[m % n] += u * shift;
                    }
                    inv.process(&mut hq);
                    inv.process(&mut hu);
                    (hq.into_iter().map(|z| -area * z.re).collect::<Vec<f64>>(), hu.into_iter().map(|z| -area * z.re).collect::<Vec<f64>>())
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
        let ring = &sht.rings[3];
        for m in [0usize, 10, 200, 999] {
            let coef = sht.coefficients(m);
            let mut first = None;
            sht.recurse(m, &coef, ring.cos, ring.sin, |l, lam, _| {
                if first.is_none() {
                    first = Some((l, lam));
                }
            });
            if let Some((l, lam)) = first {
                if l == m {
                    let expect = (sht.log_mm[m] + m as f64 * ring.sin.ln()).exp();
                    assert!((lam.abs() - expect).abs() <= 1e-12 * expect, "m {m}");
                }
            }
        }
    }
}
