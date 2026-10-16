//! Convolution geometry on a partial grid: rotated stencils, bilinear
//! interpolation, binding on the grid with renormalised weights
//! (port of `TorchGeometry.TorchConv` and `bind_support`), and the linear
//! convolution forward of `AdjointOps.convolution_forward`.
//!
//! The geometry is computed on the fly for one domain (a block plus its
//! halo) and never stored for the full sky.

use crate::healpix::{deg2rad, interp_weights_nested, order_of, pix2ang_nest, rad2deg};
use crate::stencil::{apply, local_kernel_grid, rotation_matrices, Gauge};
use std::f64::consts::PI;

/// Bound stencil of every pixel of a sorted partial grid.
///
/// Layout follows `TorchConv._pos_safe` / `_w_norm`: entry
/// `((n * G + g) * K + k) * P + p` for bilinear neighbour `n < 4`, gauge
/// `g`, output pixel `k` and stencil tap `p`.  Positions index the grid.
#[derive(Clone, Debug)]
pub struct ConvGeometry {
    pub nside: i64,
    pub kernel_sz: usize,
    /// Number of gauges (orientations), `G`.
    pub g: usize,
    /// Number of pixels of the grid, `K`.
    pub k: usize,
    /// Number of stencil taps, `P = kernel_sz^2`.
    pub p: usize,
    pub pos: Vec<usize>,
    pub w: Vec<f64>,
    /// `complete[k]`: every read of pixel `k` (all taps, gauges and bilinear
    /// neighbours) is inside the grid, so its row equals the full-sky one.
    pub complete: Vec<bool>,
}

impl ConvGeometry {
    /// Geometry of the grid `ids` (NESTED, sorted, unique) at `nside`.
    pub fn new(ids: &[i64], nside: i64, kernel_sz: usize, g_count: usize, gauge: Gauge) -> Self {
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "cell ids must be sorted and unique");
        let order = order_of(nside);
        let (kk, pp, gg) = (ids.len(), kernel_sz * kernel_sz, g_count);
        let m = kk * pp;
        let stencil = local_kernel_grid(kernel_sz, nside);
        // Raw interpolation, layout [G, 4, K * P] as in bind_support.
        let mut idx = vec![0i64; gg * 4 * m];
        let mut wt = vec![0f64; gg * 4 * m];
        for (k, &id) in ids.iter().enumerate() {
            let (theta, phi) = pix2ang_nest(id, nside);
            // The library rounds pixel centres through degrees (healpix_geo output).
            let theta = deg2rad(90.0 - rad2deg(PI / 2.0 - theta));
            let phi = deg2rad(rad2deg(phi));
            for (g, r) in rotation_matrices(theta, phi, gg, gauge).iter().enumerate() {
                for (p, v) in stencil.iter().enumerate() {
                    let (pix, w) = interp_weights_nested(apply(r, v), order);
                    for n in 0..4 {
                        let at = (g * 4 + n) * m + k * pp + p;
                        idx[at] = pix[n];
                        wt[at] = w[n];
                    }
                }
            }
        }
        bind_support(idx, wt, ids, kernel_sz, kk, gg, nside)
    }

    #[inline]
    pub fn at(&self, n: usize, g: usize, k: usize, p: usize) -> usize {
        ((n * self.g + g) * self.k + k) * self.p + p
    }
}

/// Position of `v` in `sorted` if present.
#[inline]
fn search(sorted: &[i64], v: i64) -> Option<usize> {
    let pos = sorted.partition_point(|&s| s < v);
    (pos < sorted.len() && sorted[pos] == v).then_some(pos)
}

/// `_bind_support_batched` of healpix-analyse on raw `[G, 4, K * P]` buffers.
fn bind_support(
    mut idx: Vec<i64>,
    mut wt: Vec<f64>,
    ids: &[i64],
    kernel_sz: usize,
    kk: usize,
    gg: usize,
    nside: i64,
) -> ConvGeometry {
    let pp = kernel_sz * kernel_sz;
    let m = kk * pp;
    let p_ref = (kernel_sz / 2) * (kernel_sz + 1);
    let mut present = vec![false; gg * 4 * m];
    let mut pos = vec![0usize; gg * 4 * m];
    for (i, &v) in idx.iter().enumerate() {
        if let Some(q) = search(ids, v) {
            present[i] = true;
            pos[i] = q;
        }
    }
    for g in 0..gg {
        for col in 0..m {
            let any = (0..4).any(|n| present[(g * 4 + n) * m + col]);
            if any {
                continue;
            }
            // Empty tap: fall back to the central tap of the same pixel.
            let r = (col / pp) * pp + p_ref;
            for n in 0..4 {
                let (a, b) = ((g * 4 + n) * m + col, (g * 4 + n) * m + r);
                idx[a] = idx[b];
                wt[a] = wt[b];
                let q = ids.partition_point(|&s| s < idx[a]).min(ids.len().saturating_sub(1));
                present[a] = q < ids.len() && ids[q] == idx[a];
                pos[a] = q;
            }
        }
    }
    let mut w_norm = vec![0f64; gg * 4 * m];
    for g in 0..gg {
        for col in 0..m {
            let at = |n: usize| (g * 4 + n) * m + col;
            let mut w = [0f64; 4];
            for n in 0..4 {
                w[n] = wt[at(n)] * if present[at(n)] { 1.0 } else { 0.0 };
            }
            let mut colsum = w[0] + w[1] + w[2] + w[3];
            if colsum == 0.0 {
                w[0] = if present[at(0)] { 1.0 } else { 0.0 };
                colsum = w[0] + w[1] + w[2] + w[3];
            }
            for n in 0..4 {
                w_norm[at(n)] = w[n] / colsum.max(1e-12);
            }
        }
    }
    // Rows whose every read is present.
    let mut complete = vec![true; kk];
    for g in 0..gg {
        for n in 0..4 {
            for col in 0..m {
                if !present[(g * 4 + n) * m + col] {
                    complete[col / pp] = false;
                }
            }
        }
    }
    // Permute [G, 4, K*P] -> [4, G, K*P] like TorchConv; absent reads point at 0.
    let mut pos_out = vec![0usize; gg * 4 * m];
    let mut w_out = vec![0f64; gg * 4 * m];
    for g in 0..gg {
        for n in 0..4 {
            let (src, dst) = ((g * 4 + n) * m, (n * gg + g) * m);
            for col in 0..m {
                pos_out[dst + col] = if present[src + col] { pos[src + col] } else { 0 };
                w_out[dst + col] = w_norm[src + col];
            }
        }
    }
    ConvGeometry { nside, kernel_sz, g: gg, k: kk, p: pp, pos: pos_out, w: w_out, complete }
}

/// Convolution kernel `[G, C_in, C_out, P]`, cast through `f32` like
/// `HealPixConv.set_kernel` so that results match the library.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub g: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub p: usize,
    pub w: Vec<f64>,
}

impl Kernel {
    /// From `[C_in, C_out, P]` weights shared by every gauge.
    pub fn shared(weights: &[f64], g: usize, c_in: usize, c_out: usize, p: usize) -> Self {
        assert_eq!(weights.len(), c_in * c_out * p);
        let one: Vec<f64> = weights.iter().map(|&v| v as f32 as f64).collect();
        let w = (0..g).flat_map(|_| one.iter().copied()).collect();
        Kernel { g, c_in, c_out, p, w }
    }

    #[inline]
    pub fn at(&self, g: usize, c: usize, o: usize, p: usize) -> f64 {
        self.w[((g * self.c_in + c) * self.c_out + o) * self.p + p]
    }
}

/// Transpose of [`convolution_forward`] (`AdjointOps.convolution_adjoint`):
/// `g` `[B, G * C_out, K]` -> `[B, C_in, K]`, scattering in the reference
/// order (gauge, bilinear neighbour, pixel, tap).
pub fn convolution_adjoint(geo: &ConvGeometry, ker: &Kernel, g: &[f64], batch: usize) -> Vec<f64> {
    let (kk, pp, gg) = (geo.k, geo.p, geo.g);
    assert_eq!(ker.g, gg);
    assert_eq!(ker.p, pp);
    let co = ker.c_out;
    assert_eq!(g.len(), batch * gg * co * kk);
    let mut out = vec![0f64; batch * ker.c_in * kk];
    let mut taps = vec![0f64; kk * pp];
    for b in 0..batch {
        for gi in 0..gg {
            for c in 0..ker.c_in {
                for k in 0..kk {
                    for p in 0..pp {
                        let mut s = 0.0;
                        for o in 0..co {
                            s += g[(b * gg * co + gi * co + o) * kk + k] * ker.at(gi, c, o, p);
                        }
                        taps[k * pp + p] = s;
                    }
                }
                let dst = &mut out[(b * ker.c_in + c) * kk..][..kk];
                for n in 0..4 {
                    let base = geo.at(n, gi, 0, 0);
                    for (i, &t) in taps.iter().enumerate() {
                        dst[geo.pos[base + i]] += t * geo.w[base + i];
                    }
                }
            }
        }
    }
    out
}

/// Linear convolution of `x` `[B, C_in, K]` -> `[B, G * C_out, K]` (no bias).
pub fn convolution_forward(geo: &ConvGeometry, ker: &Kernel, x: &[f64], batch: usize) -> Vec<f64> {
    let (kk, pp, gg) = (geo.k, geo.p, geo.g);
    assert_eq!(ker.g, gg);
    assert_eq!(ker.p, pp);
    assert_eq!(x.len(), batch * ker.c_in * kk);
    let co = ker.c_out;
    let mut out = vec![0f64; batch * gg * co * kk];
    let mut gathered = vec![0f64; pp];
    for b in 0..batch {
        for g in 0..gg {
            for c in 0..ker.c_in {
                let xs = &x[(b * ker.c_in + c) * kk..][..kk];
                for k in 0..kk {
                    for (p, slot) in gathered.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for n in 0..4 {
                            let at = geo.at(n, g, k, p);
                            acc += xs[geo.pos[at]] * geo.w[at];
                        }
                        *slot = acc;
                    }
                    for o in 0..co {
                        let mut s = 0.0;
                        for (p, &v) in gathered.iter().enumerate() {
                            s += v * ker.at(g, c, o, p);
                        }
                        out[(b * gg * co + g * co + o) * kk + k] += s;
                    }
                }
            }
        }
    }
    out
}
