//! Periodic grids of the planar pyramid: windows, partial convolutions and
//! decimation pieces (`LocalGrid2D` of `docs/Pyramide_gradient_par_patch.md`
//! in STL-Dev, `pbc=True`).
//!
//! A level of `H x W` pixels is periodic.  A [`Window`] is a rectangle of
//! rows `start .. start + len` (any integers, taken modulo the level size)
//! by columns; fields over a window are `[planes, rows, cols]`, row-major.
//! Reading a window from a level field and adding a window into it go
//! through the modulo, so a window longer than the level (a tile plus its
//! halos on a coarse level) simply repeats pixels: the read is still the
//! periodic field, and the addition the adjoint of the read.
//!
//! A decimation piece pushes a field supported on a window (zero
//! elsewhere) to the next level: the coarse rows touched are those whose
//! stencil `2 i - 2 .. 2 i + 3` meets the fine rows, folded to the whole
//! axis as soon as they would wrap ([`Span::coarse`]), so that the pixels
//! of a piece window are distinct and the pushed values exact.

use super::kernels::{K, KS, PAD_BEFORE, SMOOTH};
use num_complex::Complex64;

/// `len` consecutive pixels from `start` of a periodic axis of `size` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: i64,
    pub len: usize,
    pub size: usize,
}

impl Span {
    pub fn new(start: i64, len: usize, size: usize) -> Self {
        Span { start, len, size }
    }

    pub fn full(size: usize) -> Self {
        Span { start: 0, len: size, size }
    }

    /// The span with `h` more pixels on each side (may exceed the axis).
    pub fn grow(self, h: usize) -> Self {
        Span { start: self.start - h as i64, len: self.len + 2 * h, size: self.size }
    }

    /// Axis pixel of position `p`.
    #[inline]
    pub fn pixel(&self, p: usize) -> usize {
        (self.start + p as i64).rem_euclid(self.size as i64) as usize
    }

    /// Position of the axis pixel `r` (any integer) in a span of distinct
    /// pixels (`len <= size`), if it holds it.
    #[inline]
    pub fn pos(&self, r: i64) -> Option<usize> {
        debug_assert!(self.len <= self.size);
        let p = (r - self.start).rem_euclid(self.size as i64) as usize;
        (p < self.len).then_some(p)
    }

    /// Pixels of the next level whose decimation stencil meets this span:
    /// `ceil((start - 3) / 2) ..= floor((start + len + 1) / 2)`, or the
    /// whole coarse axis when that would wrap.
    pub fn coarse(&self) -> Self {
        assert!(self.size % 2 == 0, "an axis of {} pixels cannot be decimated", self.size);
        let half = self.size / 2;
        let lo = (self.start - 2).div_euclid(2);
        let hi = (self.start + self.len as i64 + 1).div_euclid(2);
        let len = (hi - lo + 1) as usize;
        if len >= half {
            Span::full(half)
        } else {
            Span { start: lo, len, size: half }
        }
    }

    /// `(tap, fine position)` read by each coarse pixel of `coarse`: the
    /// taps `a` of `2 i + a - 2` that fall in this span.
    fn taps(&self, coarse: &Span) -> Vec<Vec<(usize, usize)>> {
        (0..coarse.len)
            .map(|i| {
                let ci = coarse.start + i as i64;
                (0..KS).filter_map(|a| self.pos(2 * ci + a as i64 - PAD_BEFORE).map(|p| (a, p))).collect()
            })
            .collect()
    }
}

/// Rows by columns of one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub rows: Span,
    pub cols: Span,
}

impl Window {
    pub fn full(h: usize, w: usize) -> Self {
        Window { rows: Span::full(h), cols: Span::full(w) }
    }

    pub fn len(&self) -> usize {
        self.rows.len * self.cols.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn grow(&self, h: usize) -> Self {
        Window { rows: self.rows.grow(h), cols: self.cols.grow(h) }
    }

    pub fn coarse(&self) -> Self {
        Window { rows: self.rows.coarse(), cols: self.cols.coarse() }
    }

    /// Level size `(H, W)`.
    pub fn level(&self) -> (usize, usize) {
        (self.rows.size, self.cols.size)
    }
}

/// `planes` planes of the level field `[planes, H, W]` on `win`.
pub fn read(field: &[f64], win: &Window, planes: usize) -> Vec<f64> {
    let (h, w) = win.level();
    assert_eq!(field.len(), planes * h * w);
    let cols: Vec<usize> = (0..win.cols.len).map(|q| win.cols.pixel(q)).collect();
    let mut out = Vec::with_capacity(planes * win.len());
    for p in 0..planes {
        for r in 0..win.rows.len {
            let row = &field[(p * h + win.rows.pixel(r)) * w..][..w];
            out.extend(cols.iter().map(|&c| row[c]));
        }
    }
    out
}

/// Adds `values` `[planes, win]` into the level field (adjoint of [`read`]).
pub fn accumulate(field: &mut [f64], win: &Window, values: &[f64], planes: usize) {
    let (h, w) = win.level();
    assert_eq!(field.len(), planes * h * w);
    assert_eq!(values.len(), planes * win.len());
    let cols: Vec<usize> = (0..win.cols.len).map(|q| win.cols.pixel(q)).collect();
    let mut v = values.iter();
    for p in 0..planes {
        for r in 0..win.rows.len {
            let row = &mut field[(p * h + win.rows.pixel(r)) * w..][..w];
            for &c in &cols {
                row[c] += v.next().unwrap();
            }
        }
    }
}

/// `values` `[planes, inner]` placed in a zero field over `outer`, `inner`
/// being `outer` shrunk by `h` on every side.
pub fn embed<T: Copy + Default>(values: &[T], inner: &Window, h: usize, planes: usize) -> Vec<T> {
    let (ih, iw) = (inner.rows.len, inner.cols.len);
    let (oh, ow) = (ih + 2 * h, iw + 2 * h);
    let mut out = vec![T::default(); planes * oh * ow];
    for p in 0..planes {
        for r in 0..ih {
            out[(p * oh + r + h) * ow + h..][..iw].copy_from_slice(&values[(p * ih + r) * iw..][..iw]);
        }
    }
    out
}

/// `values` `[planes, outer]` restricted to the window shrunk by `h`.
pub fn crop<T: Copy>(values: &[T], outer: &Window, h: usize, planes: usize) -> Vec<T> {
    let (oh, ow) = (outer.rows.len, outer.cols.len);
    let (ih, iw) = (oh - 2 * h, ow - 2 * h);
    let mut out = Vec::with_capacity(planes * ih * iw);
    for p in 0..planes {
        for r in 0..ih {
            out.extend_from_slice(&values[(p * oh + r + h) * ow + h..][..iw]);
        }
    }
    out
}

/// Valid cross-correlation of one real plane of `(oh + K - 1) x (ow + K - 1)`
/// pixels with complex taps: `out[i][j] = sum_ab k[a][b] x[i + a][j + b]`
/// (`F.conv2d` without padding).
pub fn conv_valid(x: &[f64], oh: usize, ow: usize, k: &[Complex64; K * K]) -> Vec<Complex64> {
    let iw = ow + K - 1;
    assert_eq!(x.len(), (oh + K - 1) * iw);
    let mut re = vec![0.0; oh * ow];
    let mut im = vec![0.0; oh * ow];
    for i in 0..oh {
        let (ro, io) = (&mut re[i * ow..][..ow], &mut im[i * ow..][..ow]);
        for a in 0..K {
            let row = &x[(i + a) * iw..][..iw];
            for b in 0..K {
                let (kr, ki) = (k[a * K + b].re, k[a * K + b].im);
                let src = &row[b..b + ow];
                for j in 0..ow {
                    ro[j] += kr * src[j];
                    io[j] += ki * src[j];
                }
            }
        }
    }
    re.into_iter().zip(im).map(|(r, i)| Complex64::new(r, i)).collect()
}

/// Adjoint of [`conv_valid`] for a real input: adds
/// `Re(g[i][j] conj(k[a][b]))` at `x[i + a][j + b]` of `out`
/// `(oh + K - 1) x (ow + K - 1)`.
pub fn conv_valid_adjoint(g: &[Complex64], oh: usize, ow: usize, k: &[Complex64; K * K], out: &mut [f64]) {
    let iw = ow + K - 1;
    assert_eq!(g.len(), oh * ow);
    assert_eq!(out.len(), (oh + K - 1) * iw);
    let gr: Vec<f64> = g.iter().map(|v| v.re).collect();
    let gi: Vec<f64> = g.iter().map(|v| v.im).collect();
    for i in 0..oh {
        let (sr, si) = (&gr[i * ow..][..ow], &gi[i * ow..][..ow]);
        for a in 0..K {
            let row = &mut out[(i + a) * iw..][..iw];
            for b in 0..K {
                let (kr, ki) = (k[a * K + b].re, k[a * K + b].im);
                let dst = &mut row[b..b + ow];
                for j in 0..ow {
                    dst[j] += sr[j] * kr + si[j] * ki;
                }
            }
        }
    }
}

/// Decimation of `planes` planes of a field supported on `win` (zero
/// elsewhere): the window of the next level it touches and the values there.
pub fn decimate(x: &[f64], win: &Window, planes: usize) -> (Window, Vec<f64>) {
    assert_eq!(x.len(), planes * win.len());
    let out = win.coarse();
    let (rt, ct) = (win.rows.taps(&out.rows), win.cols.taps(&out.cols));
    let (fh, fw) = (win.rows.len, win.cols.len);
    let mut y = vec![0.0; planes * out.len()];
    for p in 0..planes {
        let xp = &x[p * fh * fw..][..fh * fw];
        let yp = &mut y[p * out.len()..][..out.len()];
        for (i, ri) in rt.iter().enumerate() {
            for (j, cj) in ct.iter().enumerate() {
                let mut s = 0.0;
                for &(a, pr) in ri {
                    for &(b, pc) in cj {
                        s += SMOOTH[a][b] * xp[pr * fw + pc];
                    }
                }
                yp[i * out.cols.len + j] = s;
            }
        }
    }
    (out, y)
}

/// Adjoint of [`decimate`]: the cotangent `g` on `win.coarse()` pulled back
/// to `win`.
pub fn decimate_adjoint(g: &[f64], win: &Window, planes: usize) -> Vec<f64> {
    let out = win.coarse();
    assert_eq!(g.len(), planes * out.len());
    let (rt, ct) = (win.rows.taps(&out.rows), win.cols.taps(&out.cols));
    let (fh, fw) = (win.rows.len, win.cols.len);
    let mut x = vec![0.0; planes * win.len()];
    for p in 0..planes {
        let xp = &mut x[p * fh * fw..][..fh * fw];
        let gp = &g[p * out.len()..][..out.len()];
        for (i, ri) in rt.iter().enumerate() {
            for (j, cj) in ct.iter().enumerate() {
                let v = gp[i * out.cols.len + j];
                for &(a, pr) in ri {
                    for &(b, pc) in cj {
                        xp[pr * fw + pc] += SMOOTH[a][b] * v;
                    }
                }
            }
        }
    }
    x
}

/// Whole-level decimation of `[planes, H, W]` (`_downsample_tensor`,
/// circular padding, one step).
pub fn decimate_level(x: &[f64], h: usize, w: usize, planes: usize) -> Vec<f64> {
    decimate(x, &Window::full(h, w), planes).1
}

/// Adjoint of [`decimate_level`].
pub fn decimate_level_adjoint(g: &[f64], h: usize, w: usize, planes: usize) -> Vec<f64> {
    decimate_adjoint(g, &Window::full(h, w), planes)
}
