//! Transport of an image by a displacement field, with local operators only
//! (`docs/lagrangian_synthesis.md`, scheme P): the planar prototype of a
//! step that can be ported to the HEALPix tangent-plane stencil.
//!
//! - [`bicubic`]: `y(p) = x(p + u(p))` by Keys' cubic convolution
//!   (`a = -1/2`, Catmull-Rom; a 4 x 4 stencil, C1 in `u`), periodic; its
//!   VJP in `x` ([`bicubic_vjp`]) and its derivative in `u`
//!   ([`bicubic_du`], diagonal: `y(p)` depends on `u(p)` only).
//! - [`gradient`]: centred differences, the derivative of [`bicubic`] at
//!   `u = 0`.
//! - [`Smoother`]: the multi-scale operator
//!   `K = sum_l w_l 4^l (D^T)^l D^l`, `D` the stride-2 filter
//!   `[1 3 3 1] / 8` per axis (periodic), symmetric positive semi-definite,
//!   `K 1 = (sum_l w_l) 1`.
//!
//! Images are row-major `[h][w]` planes; row displacement `ur`, column
//! displacement `uc`, in pixels.

/// Keys' weights of the 4 taps `-1, 0, 1, 2` at the fraction `t` in [0, 1).
#[inline]
pub fn keys(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Derivatives in `t` of [`keys`].
#[inline]
pub fn keys_dt(t: f64) -> [f64; 4] {
    let t2 = t * t;
    [-1.5 * t2 + 2.0 * t - 0.5, 4.5 * t2 - 5.0 * t, -4.5 * t2 + 4.0 * t + 0.5, 1.5 * t2 - t]
}

#[inline]
fn wrap(i: i64, n: usize) -> usize {
    i.rem_euclid(n as i64) as usize
}

/// Base index and fraction of the coordinate `q`.
#[inline]
fn split(q: f64) -> (i64, f64) {
    let f = q.floor();
    (f as i64, q - f)
}

/// The 16 taps (index, row weight, column weight, row derivative, column
/// derivative) of the sample at `(i + ur, j + uc)`.
#[inline]
fn taps(h: usize, w: usize, i: usize, j: usize, ur: f64, uc: f64, mut f: impl FnMut(usize, f64, f64, f64, f64)) {
    let (r0, tr) = split(i as f64 + ur);
    let (c0, tc) = split(j as f64 + uc);
    let (wr, wc, dr, dc) = (keys(tr), keys(tc), keys_dt(tr), keys_dt(tc));
    for a in 0..4 {
        let row = wrap(r0 + a as i64 - 1, h) * w;
        for b in 0..4 {
            f(row + wrap(c0 + b as i64 - 1, w), wr[a], wc[b], dr[a], dc[b]);
        }
    }
}

/// `y(p) = x(p + u(p))`.
pub fn bicubic(x: &[f64], h: usize, w: usize, ur: &[f64], uc: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; h * w];
    for i in 0..h {
        for j in 0..w {
            let p = i * w + j;
            let mut s = 0.0;
            taps(h, w, i, j, ur[p], uc[p], |q, a, b, _, _| s += a * b * x[q]);
            y[p] = s;
        }
    }
    y
}

/// `(dy/dx)^T g`: the cotangent of `x` (scatter of the stencil).
pub fn bicubic_vjp(g: &[f64], h: usize, w: usize, ur: &[f64], uc: &[f64]) -> Vec<f64> {
    let mut gx = vec![0.0; h * w];
    for i in 0..h {
        for j in 0..w {
            let p = i * w + j;
            taps(h, w, i, j, ur[p], uc[p], |q, a, b, _, _| gx[q] += a * b * g[p]);
        }
    }
    gx
}

/// `(dy(p)/dur(p), dy(p)/duc(p))`.
pub fn bicubic_du(x: &[f64], h: usize, w: usize, ur: &[f64], uc: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut gr, mut gc) = (vec![0.0; h * w], vec![0.0; h * w]);
    for i in 0..h {
        for j in 0..w {
            let p = i * w + j;
            let (mut sr, mut sc) = (0.0, 0.0);
            taps(h, w, i, j, ur[p], uc[p], |q, a, b, da, db| {
                sr += da * b * x[q];
                sc += a * db * x[q];
            });
            gr[p] = sr;
            gc[p] = sc;
        }
    }
    (gr, gc)
}

/// Centred differences `((x[i+1] - x[i-1]) / 2, (x[j+1] - x[j-1]) / 2)`, periodic.
pub fn gradient(x: &[f64], h: usize, w: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut gr, mut gc) = (vec![0.0; h * w], vec![0.0; h * w]);
    for i in 0..h {
        for j in 0..w {
            let p = i * w + j;
            gr[p] = 0.5 * (x[wrap(i as i64 + 1, h) * w + j] - x[wrap(i as i64 - 1, h) * w + j]);
            gc[p] = 0.5 * (x[i * w + wrap(j as i64 + 1, w)] - x[i * w + wrap(j as i64 - 1, w)]);
        }
    }
    (gr, gc)
}

const FILTER: [f64; 4] = [0.125, 0.375, 0.375, 0.125];

/// `D`: `[h][w] -> [h/2][w/2]`, `y[i][j] = sum_ab f_a f_b x[2i + a - 1][2j + b - 1]`.
pub fn down(x: &[f64], h: usize, w: usize) -> Vec<f64> {
    let (hc, wc) = (h / 2, w / 2);
    let mut y = vec![0.0; hc * wc];
    for i in 0..hc {
        for j in 0..wc {
            let mut s = 0.0;
            for a in 0..4 {
                let row = wrap(2 * i as i64 + a as i64 - 1, h) * w;
                for b in 0..4 {
                    s += FILTER[a] * FILTER[b] * x[row + wrap(2 * j as i64 + b as i64 - 1, w)];
                }
            }
            y[i * wc + j] = s;
        }
    }
    y
}

/// `D^T`: `[h/2][w/2] -> [h][w]`.
pub fn down_t(y: &[f64], h: usize, w: usize) -> Vec<f64> {
    let (hc, wc) = (h / 2, w / 2);
    let mut x = vec![0.0; h * w];
    for i in 0..hc {
        for j in 0..wc {
            let v = y[i * wc + j];
            for a in 0..4 {
                let row = wrap(2 * i as i64 + a as i64 - 1, h) * w;
                for b in 0..4 {
                    x[row + wrap(2 * j as i64 + b as i64 - 1, w)] += FILTER[a] * FILTER[b] * v;
                }
            }
        }
    }
    x
}

/// Pyramid bands of a plane: `G_j = (4 D^T)^j D^j x`, bands
/// `G_j - G_(j+1)`, `j < levels`, then `G_levels`; they sum to `x`.
pub fn bands(x: &[f64], h: usize, w: usize, levels: usize) -> Vec<Vec<f64>> {
    let mut low = vec![x.to_vec()];
    let mut coarse = x.to_vec();
    let (mut hc, mut wc) = (h, w);
    for l in 1..=levels {
        coarse = down(&coarse, hc, wc);
        hc /= 2;
        wc /= 2;
        let mut up = coarse.clone();
        let (mut hu, mut wu) = (hc, wc);
        for _ in 0..l {
            up = down_t(&up, 2 * hu, 2 * wu).into_iter().map(|v| 4.0 * v).collect();
            hu *= 2;
            wu *= 2;
        }
        low.push(up);
    }
    let mut out: Vec<Vec<f64>> = low.windows(2).map(|p| p[0].iter().zip(&p[1]).map(|(a, b)| a - b).collect()).collect();
    out.push(low.pop().unwrap());
    out
}

/// `K = sum_l w_l 4^l (D^T)^l D^l` on `[h][w]` planes.
#[derive(Clone, Debug)]
pub struct Smoother {
    /// Rows of a plane.
    pub h: usize,
    /// Columns of a plane.
    pub w: usize,
    /// `w_l`, `l = 0..len` (level `l` at `h / 2^l`).
    pub weights: Vec<f64>,
}

impl Smoother {
    /// `levels` levels (as many as the halvings allow, down to 4 x 4),
    /// `w_l = 2^(slope l)`.
    pub fn new(h: usize, w: usize, levels: usize, slope: f64) -> Self {
        let mut n = 1;
        while n < levels && (h >> n) >= 4 && (w >> n) >= 4 && (h >> n) << n == h && (w >> n) << n == w {
            n += 1;
        }
        Smoother { h, w, weights: (0..n).map(|l| (slope * l as f64).exp2()).collect() }
    }

    /// `K v` on one plane.
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        let mut out: Vec<f64> = v.iter().map(|x| self.weights[0] * x).collect();
        let mut coarse = v.to_vec();
        let (mut h, mut w) = (self.h, self.w);
        for (l, &wl) in self.weights.iter().enumerate().skip(1) {
            coarse = down(&coarse, h, w);
            h /= 2;
            w /= 2;
            // (D^T)^l back to the full size.
            let mut up = coarse.clone();
            let (mut hu, mut wu) = (h, w);
            for _ in 0..l {
                up = down_t(&up, 2 * hu, 2 * wu);
                hu *= 2;
                wu *= 2;
            }
            let s = wl * 4f64.powi(l as i32);
            out.iter_mut().zip(&up).for_each(|(o, u)| *o += s * u);
        }
        out
    }
}
