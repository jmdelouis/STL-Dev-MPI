//! Spin-2 front end of the scattering engine: the polarisation `P = Q + iU`
//! of level 0 becomes, at every pyramid level `j`, the spin-0 field
//!
//!     chi_j = chi_E + i chi_B = -(d_x - i d_y)^2 P_j      (pixel units of level j)
//!
//! where `P_j` is the mean of the four NESTED children of `P_{j-1}`, each
//! rotated into the parent's frame by parallel transport, and the
//! derivative is a local stencil in the tangent plane of the pixel (axes
//! `e_theta`, `e_phi`) on the pixels within 2 rings, each rotated into the
//! frame of the centre, with weights exact on the polynomials of degree 4
//! at the actual pixel positions ([`eth_bar2`]; a bilinear interpolation
//! of the pixels at the taps of a regular grid makes an O(1) relative
//! error on second derivatives, ~8 % E -> B leakage).  In
//! the flat limit `chi = (k h_j)^2 (E + i B)`: a local, frame-independent
//! E / B decomposition weighted by `(l / l_j)^2`, of order one in the band
//! of the wavelet of level `j`.  The maps `chi_E`, `chi_B` of every level
//! go to the scalar engine, which then skips its own decimation of the
//! maps (the stored moduli remain scalar).
//!
//! Both operators are complex-linear in `P` and stored as complex sparse
//! rows ([`SpinCsr`]); fields are real `[R, 2, n]` (Q, U) or (E, B).

use crate::healpix::{npix, pix2ang_nest};
use num_complex::Complex64;

/// Sign of the spin rotation: a polarisation of angle `beta` from
/// `e_theta` towards `e_phi` in the frame of pixel `m` has the angle
/// `beta - psi` in a frame whose first axis is at `psi` in that of `m`
/// (`P -> P e^{-2 i psi}`), HEALPix (COSMO) convention.
const SPIN: f64 = 2.0;

type V3 = [f64; 3];

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Centre, `e_theta` and `e_phi` of a NESTED pixel.
pub fn frame(pix: i64, nside: i64) -> (V3, V3, V3) {
    let (t, p) = pix2ang_nest(pix, nside);
    let (st, ct, sp, cp) = (t.sin(), t.cos(), p.sin(), p.cos());
    ([st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, 0.0])
}

/// Angle, in the frame of pixel `m` (from its `e_theta` towards its
/// `e_phi`), of the tangent vector `v` at `a` transported along the
/// geodesic from `a` to the centre of `m`.
pub fn transported_angle(a: V3, v: V3, m: i64, nside: i64) -> f64 {
    let (b, et, ep) = frame(m, nside);
    let axis = cross(a, b);
    let s = dot(axis, axis).sqrt();
    let c = dot(a, b);
    let w = if s < 1e-15 {
        v
    } else {
        // Rodrigues rotation of v about axis / s by the angle atan2(s, c).
        let k = [axis[0] / s, axis[1] / s, axis[2] / s];
        let kv = cross(k, v);
        let kd = dot(k, v);
        [
            v[0] * c + kv[0] * s + k[0] * kd * (1.0 - c),
            v[1] * c + kv[1] * s + k[1] * kd * (1.0 - c),
            v[2] * c + kv[2] * s + k[2] * kd * (1.0 - c),
        ]
    };
    dot(w, ep).atan2(dot(w, et))
}

/// `e^{-2 i psi}`: a polarisation of the frame of `m` expressed in the
/// transported frame (see [`SPIN`]).
#[inline]
fn spin_rotation(psi: f64) -> Complex64 {
    Complex64::from_polar(1.0, -SPIN * psi)
}

/// Complex sparse rows: `out[r] = sum_e w[e] P[col[e]]`, with the transpose
/// for the adjoint.
#[derive(Clone, Debug)]
pub struct SpinCsr {
    /// Number of input pixels.
    pub n_in: usize,
    /// Row pointers `[rows + 1]`.
    pub ptr: Vec<usize>,
    /// Input pixel of each entry.
    pub col: Vec<u32>,
    /// Complex weight of each entry.
    pub w: Vec<Complex64>,
    t_ptr: Vec<usize>,
    t_row: Vec<u32>,
    t_w: Vec<Complex64>,
}

impl SpinCsr {
    fn new(n_in: usize, rows: Vec<Vec<(u32, Complex64)>>) -> Self {
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        ptr.push(0);
        let (mut col, mut w) = (Vec::new(), Vec::new());
        for row in &rows {
            for &(c, v) in row {
                col.push(c);
                w.push(v);
            }
            ptr.push(col.len());
        }
        let mut count = vec![0usize; n_in + 1];
        for &c in &col {
            count[c as usize + 1] += 1;
        }
        for i in 0..n_in {
            count[i + 1] += count[i];
        }
        let t_ptr = count.clone();
        let mut t_row = vec![0u32; col.len()];
        let mut t_w = vec![Complex64::new(0.0, 0.0); col.len()];
        for r in 0..rows.len() {
            for e in ptr[r]..ptr[r + 1] {
                let slot = &mut count[col[e] as usize];
                t_row[*slot] = r as u32;
                t_w[*slot] = w[e].conj();
                *slot += 1;
            }
        }
        SpinCsr { n_in, ptr, col, w, t_ptr, t_row, t_w }
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Bytes held (both directions).
    pub fn nbytes(&self) -> usize {
        (self.ptr.len() + self.t_ptr.len()) * 8 + (self.col.len() + self.t_row.len()) * 4 + (self.w.len() + self.t_w.len()) * 16
    }

    /// `[R, 2, n_in] -> [R, 2, rows]`.
    pub fn apply(&self, x: &[f64], threads: usize) -> Vec<f64> {
        product(&self.ptr, &self.col, &self.w, self.n_in, x, threads)
    }

    /// Transpose (complex adjoint, `Re <g, A x>` real inner product):
    /// `[R, 2, rows] -> [R, 2, n_in]`.
    pub fn adjoint(&self, g: &[f64], threads: usize) -> Vec<f64> {
        product(&self.t_ptr, &self.t_row, &self.t_w, self.rows(), g, threads)
    }
}

/// `out[r] = sum_e w[e] P[col[e]]` on every `(Q, U)` row pair of `x`.
fn product(ptr: &[usize], col: &[u32], w: &[Complex64], n_in: usize, x: &[f64], threads: usize) -> Vec<f64> {
    let rows = ptr.len() - 1;
    assert_eq!(x.len() % (2 * n_in), 0, "field length");
    let r_count = x.len() / (2 * n_in);
    let mut out = vec![0.0; r_count * 2 * rows];
    let threads = threads.max(1).min(rows.max(1));
    let chunk = rows.div_ceil(threads).max(1);
    for (r, dst) in out.chunks_mut(2 * rows).enumerate() {
        let (q, u) = x[r * 2 * n_in..][..2 * n_in].split_at(n_in);
        let (dq, du) = dst.split_at_mut(rows);
        std::thread::scope(|s| {
            for (i, (oq, ou)) in dq.chunks_mut(chunk).zip(du.chunks_mut(chunk)).enumerate() {
                s.spawn(move || {
                    let r0 = i * chunk;
                    for k in 0..oq.len() {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for e in ptr[r0 + k]..ptr[r0 + k + 1] {
                            let c = col[e] as usize;
                            acc += w[e] * Complex64::new(q[c], u[c]);
                        }
                        oq[k] = acc.re;
                        ou[k] = acc.im;
                    }
                });
            }
        });
    }
    out
}

/// `f(p)` for `p < n`, in order, on all available threads.
fn par_rows<T: Send>(n: i64, f: impl Fn(i64) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n.max(1) as usize);
    let chunk = (n as usize).div_ceil(threads).max(1) as i64;
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk as usize)
            .map(|start| s.spawn(move || (start..(start + chunk).min(n)).map(f).collect::<Vec<T>>()))
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("spin operator thread")).collect()
    })
}

/// Transported mean of the four NESTED children: level of `nside_fine` to
/// its parents.
pub fn spin_decimation(nside_fine: i64) -> SpinCsr {
    let nc = npix(nside_fine / 2);
    let rows = (0..nc)
        .map(|p| {
            let (c, et, _) = frame(p, nside_fine / 2);
            (0..4)
                .map(|k| {
                    let child = 4 * p + k;
                    let psi = transported_angle(c, et, child, nside_fine);
                    (child as u32, 0.25 * spin_rotation(psi))
                })
                .collect()
        })
        .collect();
    SpinCsr::new(npix(nside_fine) as usize, rows)
}

/// Monomials `x^a y^b` (`a + b <= degree`) and their image under
/// `-(d_x - i d_y)^2 = -d_xx + d_yy + 2 i d_xy` at the origin.
fn monomials(degree: usize) -> Vec<(i32, i32, Complex64)> {
    let mut out = Vec::new();
    for t in 0..=degree as i32 {
        for a in (0..=t).rev() {
            let b = t - a;
            let image = match (a, b) {
                (2, 0) => Complex64::new(-2.0, 0.0),
                (0, 2) => Complex64::new(2.0, 0.0),
                (1, 1) => Complex64::new(0.0, 2.0),
                _ => Complex64::new(0.0, 0.0),
            };
            out.push((a, b, image));
        }
    }
    out
}

/// Solves the symmetric positive-definite `a x = b` (`n x n`, row-major,
/// complex right-hand side) by Cholesky; `None` if not positive definite.
fn cholesky_solve(mut a: Vec<f64>, n: usize, b: &[Complex64]) -> Option<Vec<Complex64>> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 1e-12 * a[j * n + j].abs().max(1e-300) {
            return None;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    let mut y = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            let t = y[k] * a[i * n + k];
            y[i] -= t;
        }
        y[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            let t = y[k] * a[k * n + i];
            y[i] -= t;
        }
        y[i] /= a[i * n + i];
    }
    Some(y)
}

/// Neighbours of `p` within `rings` rings (all pixels near enough when the
/// level is too small for the ring neighbourhood).
fn stencil_pixels(p: i64, nside: i64, rings: u32) -> Vec<i64> {
    if (rings as i64) < nside {
        return crate::neighbours::kth_neighbourhood(p, nside, rings);
    }
    let (c, _, _) = frame(p, nside);
    let h = (4.0 * std::f64::consts::PI / npix(nside) as f64).sqrt();
    let cut = ((rings as f64 + 1.0) * h).min(1.2).cos();
    (0..npix(nside)).filter(|&m| dot(frame(m, nside).0, c) >= cut).collect()
}

/// The operator `-(d_x - i d_y)^2` at `nside` (pixel units: lengths in
/// `h = sqrt(4 pi / N)`), one row per pixel, reading the pixels within
/// `rings` rings, each rotated into the frame of the centre by parallel
/// transport.  Weights per pixel: exact on the polynomials of degree
/// `2 rings` (degree 2 where too few pixels) of the gnomonic coordinates
/// along `(e_theta, e_phi)`, with the least weighted norm
/// `sum |w_m|^2 e^{r_m^2 / rings^2}` (polynomial finite differences on the
/// actual pixel positions).
pub fn eth_bar2(nside: i64, kernel_sz: usize) -> SpinCsr {
    assert!(kernel_sz % 2 == 1 && (3..=9).contains(&kernel_sz), "spin stencil: kernel 3, 5, 7 or 9");
    let rings = (kernel_sz / 2) as u32;
    let n = npix(nside);
    let h = (4.0 * std::f64::consts::PI / n as f64).sqrt();
    let rows = par_rows(n, |p| {
            let (c, et, ep) = frame(p, nside);
            let mut cells = stencil_pixels(p, nside, rings);
            cells.sort_unstable();
            let pts: Vec<(f64, f64)> = cells
                .iter()
                .map(|&m| {
                    let v = frame(m, nside).0;
                    let z = dot(v, c);
                    (dot(v, et) / z / h, dot(v, ep) / z / h)
                })
                .collect();
            let weight: Vec<f64> = pts.iter().map(|&(x, y)| (-(x * x + y * y) / (rings * rings) as f64).exp()).collect();
            let mut degree = 2 * rings as usize;
            let w = loop {
                let mono = monomials(degree);
                let k = mono.len();
                // A [k, M]: monomials at the pixels; solve (A D A^T) l = b, w = D A^T l.
                let rows_a: Vec<Vec<f64>> = mono.iter().map(|&(a, b, _)| pts.iter().map(|&(x, y)| x.powi(a) * y.powi(b)).collect()).collect();
                let mut g = vec![0.0; k * k];
                for i in 0..k {
                    for j in 0..=i {
                        let v: f64 = (0..pts.len()).map(|m| rows_a[i][m] * weight[m] * rows_a[j][m]).sum();
                        g[i * k + j] = v;
                        g[j * k + i] = v;
                    }
                }
                let rhs: Vec<Complex64> = mono.iter().map(|m| m.2).collect();
                if let Some(l) = cholesky_solve(g, k, &rhs) {
                    break (0..pts.len())
                        .map(|m| (0..k).fold(Complex64::new(0.0, 0.0), |s, i| s + l[i] * rows_a[i][m]) * weight[m])
                        .collect::<Vec<_>>();
                }
                assert!(degree > 2, "spin stencil: singular system at pixel {p}");
                degree = 2;
            };
            cells
                .iter()
                .zip(w)
                .map(|(&m, wm)| (m as u32, wm * spin_rotation(transported_angle(c, et, m, nside))))
                .collect()
        });
    SpinCsr::new(n as usize, rows)
}

/// The front end of every level: `levels` stencils and `levels - 1`
/// transported means from `nside0`.
#[derive(Clone, Debug)]
pub struct SpinPyramid {
    /// `nside` of level 0.
    pub nside0: i64,
    /// Stencil side (3 or 5).
    pub kernel_sz: usize,
    /// `down[j]`: level `j` -> `j + 1`.
    pub down: Vec<SpinCsr>,
    /// `eth[j]`: `P_j -> chi_j`.
    pub eth: Vec<SpinCsr>,
    /// Threads of the sparse products.
    pub threads: usize,
}

impl SpinPyramid {
    /// Operators of `levels` levels from `nside0`.
    pub fn new(nside0: i64, levels: usize, kernel_sz: usize, threads: usize) -> Self {
        assert!(levels >= 1 && (nside0 >> (levels - 1)) >= 1, "too many levels for nside");
        let down = (0..levels - 1).map(|j| spin_decimation(nside0 >> j)).collect();
        let eth = (0..levels).map(|j| eth_bar2(nside0 >> j, kernel_sz)).collect();
        SpinPyramid { nside0, kernel_sz, down, eth, threads: threads.max(1) }
    }

    /// Number of levels.
    pub fn levels(&self) -> usize {
        self.eth.len()
    }

    /// Bytes held by the operators.
    pub fn nbytes(&self) -> usize {
        self.down.iter().chain(&self.eth).map(SpinCsr::nbytes).sum()
    }

    /// `(Q, U)` of every level, `[R, 2, N_j]`, from level 0 `[R, 2, N_0]`.
    pub fn polarisation(&self, qu: &[f64]) -> Vec<Vec<f64>> {
        let mut out = vec![qu.to_vec()];
        for d in &self.down {
            let next = d.apply(out.last().unwrap(), self.threads);
            out.push(next);
        }
        out
    }

    /// `(chi_E, chi_B)` of every level, `[R, 2, N_j]`.
    pub fn forward(&self, qu: &[f64]) -> Vec<Vec<f64>> {
        self.polarisation(qu).iter().zip(&self.eth).map(|(p, e)| e.apply(p, self.threads)).collect()
    }

    /// Cotangent of level-0 `(Q, U)` from those of `chi` of every level.
    pub fn adjoint(&self, g_chi: &[Vec<f64>]) -> Vec<f64> {
        assert_eq!(g_chi.len(), self.levels());
        let mut g = self.eth.last().unwrap().adjoint(g_chi.last().unwrap(), self.threads);
        for j in (0..self.levels() - 1).rev() {
            let mut gj = self.eth[j].adjoint(&g_chi[j], self.threads);
            for (a, b) in gj.iter_mut().zip(self.down[j].adjoint(&g, self.threads)) {
                *a += b;
            }
            g = gj;
        }
        g
    }

    /// Pixels of every level whose `chi` only reads valid data, from the
    /// valid pixels of level 0 (nonzero `valid0`): a parent is valid when
    /// its four children are, and a row of the stencil when every pixel it
    /// reads is (erosion by the stencil radius at every level).
    pub fn valid_rows(&self, valid0: &[f64]) -> Vec<Vec<f64>> {
        let mut data: Vec<bool> = valid0.iter().map(|&v| v != 0.0).collect();
        let mut out = Vec::with_capacity(self.levels());
        for j in 0..self.levels() {
            if j > 0 {
                data = data.chunks(4).map(|c| c.iter().all(|&v| v)).collect();
            }
            let e = &self.eth[j];
            out.push((0..e.rows()).map(|r| f64::from(u8::from(e.col[e.ptr[r]..e.ptr[r + 1]].iter().all(|&c| data[c as usize])))).collect());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn adjoints() {
        let p = SpinPyramid::new(8, 3, 5, 2);
        let mut s = 7u64;
        let x: Vec<f64> = (0..2 * 2 * 768).map(|_| lcg(&mut s)).collect();
        let y = p.forward(&x);
        let g: Vec<Vec<f64>> = y.iter().map(|v| v.iter().map(|_| lcg(&mut s)).collect()).collect();
        let lhs: f64 = y.iter().zip(&g).map(|(a, b)| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>()).sum();
        let gx = p.adjoint(&g);
        let rhs: f64 = x.iter().zip(&gx).map(|(u, v)| u * v).sum();
        assert!((lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0), "{lhs} {rhs}");
    }

    #[test]
    fn complex_linearity() {
        // A uniform rotation of the local frames (P -> P e^{2 i a} at every
        // pixel) is not a physical rotation; but a field and its local
        // rotation by the same angle everywhere must give chi rotated by
        // the same phase: chi is complex-linear in P.
        let e = eth_bar2(4, 5);
        let mut s = 3u64;
        let x: Vec<f64> = (0..2 * 192).map(|_| lcg(&mut s)).collect();
        let a = 0.3f64;
        let r = Complex64::from_polar(1.0, 2.0 * a);
        let xr: Vec<f64> = {
            let (q, u) = x.split_at(192);
            let rot: Vec<Complex64> = q.iter().zip(u).map(|(&q, &u)| Complex64::new(q, u) * r).collect();
            rot.iter().map(|v| v.re).chain(rot.iter().map(|v| v.im)).collect()
        };
        let (y, yr) = (e.apply(&x, 1), e.apply(&xr, 1));
        for i in 0..192 {
            let c = Complex64::new(y[i], y[192 + i]) * r;
            assert!((c.re - yr[i]).abs() < 1e-12 && (c.im - yr[192 + i]).abs() < 1e-12);
        }
    }

    #[test]
    fn exact_on_quadratic_monomials() {
        // Weighted least-norm solution satisfies the constraints exactly.
        let mono = monomials(4);
        assert_eq!(mono.len(), 15);
        assert_eq!(mono.iter().filter(|m| m.2 != Complex64::new(0.0, 0.0)).count(), 3);
    }
}
