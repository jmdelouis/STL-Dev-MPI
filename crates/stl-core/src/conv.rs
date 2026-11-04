//! Convolution geometry on a partial grid: rotated stencils, bilinear
//! interpolation, binding on the grid with renormalised weights
//! (port of `TorchGeometry.TorchConv` and `bind_support`), and the linear
//! convolution forward / adjoint of `AdjointOps`.
//!
//! The geometry is computed on the fly for one domain (a block plus its
//! halo) and never stored for the full sky.  Memory: the binding is done
//! pixel by pixel (no `[G, 4, K * P]` raw buffers), positions are `u32`,
//! and the geometry covers only the output rows that are used, so a bound
//! stencil costs `4 G P (4 + 8)` bytes per output row (2.4 kB at `L = 2`,
//! kernel 5).  [`ConvPlan`] can also rebuild it by sub-blocks of rows at
//! every application, bounding it by the sub-block size.

use crate::healpix::{deg2rad, interp_weights_nested, order_of, pix2ang_nest, rad2deg};
use crate::stencil::{apply, local_kernel_grid, rotation_matrices, Gauge};
use std::f64::consts::PI;

/// Bound stencils of output rows `rows` reading the sorted grid `grid`.
///
/// Layout follows `TorchConv._pos_safe` / `_w_norm`: entry
/// `((n * G + g) * K + k) * P + p` for bilinear neighbour `n < 4`, gauge
/// `g`, output row `k` and stencil tap `p`.  Positions index the grid.
#[derive(Clone, Debug)]
pub struct ConvGeometry {
    /// `nside` of the level the rows and grid belong to.
    pub nside: i64,
    /// Stencil side in taps (`kernel_sz x kernel_sz` stencil).
    pub kernel_sz: usize,
    /// Number of gauges (orientations), `G`.
    pub g: usize,
    /// Number of output rows, `K`.
    pub k: usize,
    /// Number of grid cells (inputs).
    pub n_in: usize,
    /// Number of stencil taps, `P = kernel_sz^2`.
    pub p: usize,
    /// Grid position read by each entry (layout above).
    pub pos: Vec<u32>,
    /// Bilinear weight of each entry, renormalised over the reads found in the grid.
    pub w: Vec<f64>,
    /// `complete[k]`: every read of row `k` (all taps, gauges and bilinear
    /// neighbours) is inside the grid, so its row equals the full-sky one.
    pub complete: Vec<bool>,
}

/// Position of `v` in `sorted` if present.
#[inline]
fn search(sorted: &[i64], v: i64) -> Option<usize> {
    let pos = sorted.partition_point(|&s| s < v);
    (pos < sorted.len() && sorted[pos] == v).then_some(pos)
}

impl ConvGeometry {
    /// Geometry of the grid `ids` (NESTED, sorted, unique) at `nside`, one
    /// output row per grid cell (`TorchConv(ids, ...)`).
    pub fn new(ids: &[i64], nside: i64, kernel_sz: usize, g_count: usize, gauge: Gauge) -> Self {
        Self::rows(ids, ids, nside, kernel_sz, g_count, gauge)
    }

    /// Geometry of the output rows `rows` (any NESTED cells, in the order of
    /// the output) reading the grid `grid` (sorted, unique).  Row `k` is the
    /// row of `rows[k]` in `TorchConv(grid, ...)` when `rows[k]` is in `grid`.
    pub fn rows(grid: &[i64], rows: &[i64], nside: i64, kernel_sz: usize, g_count: usize, gauge: Gauge) -> Self {
        Self::rows_eps(grid, rows, nside, kernel_sz, g_count, gauge, 0.0)
    }

    /// [`rows`](Self::rows) where, with `eps > 0`, a cell of the bilinear
    /// interpolation counts as found only if its weight exceeds `eps`.  On a
    /// partial grid the binding then does not depend on cells of rounding
    /// weight (a direction on a ring boundary lists a neighbour ring with a
    /// weight of 0 or 1e-16 depending on the arithmetic), so that host and
    /// device geometries agree on incomplete rows too (disc patches).
    pub fn rows_eps(grid: &[i64], rows: &[i64], nside: i64, kernel_sz: usize, g_count: usize, gauge: Gauge, eps: f64) -> Self {
        assert!(grid.windows(2).all(|w| w[0] < w[1]), "cell ids must be sorted and unique");
        assert!(grid.len() < u32::MAX as usize, "grid too large for u32 positions");
        let order = order_of(nside);
        let (kk, pp, gg) = (rows.len(), kernel_sz * kernel_sz, g_count);
        let p_ref = (kernel_sz / 2) * (kernel_sz + 1);
        let stencil = local_kernel_grid(kernel_sz, nside);
        let at = |n: usize, g: usize, k: usize, p: usize| ((n * gg + g) * kk + k) * pp + p;
        let mut pos = vec![0u32; 4 * gg * kk * pp];
        let mut wout = vec![0f64; 4 * gg * kk * pp];
        let mut complete = vec![true; kk];
        // Per (row, gauge): raw interpolation of the P taps, then binding
        // (`_bind_support_batched` of healpix-analyse, one column at a time).
        let mut idx = vec![[0i64; 4]; pp];
        let mut wt = vec![[0f64; 4]; pp];
        for (k, &id) in rows.iter().enumerate() {
            let (theta, phi) = pix2ang_nest(id, nside);
            // The library rounds pixel centres through degrees (healpix_geo output).
            let theta = deg2rad(90.0 - rad2deg(PI / 2.0 - theta));
            let phi = deg2rad(rad2deg(phi));
            for (g, r) in rotation_matrices(theta, phi, gg, gauge).iter().enumerate() {
                for (p, v) in stencil.iter().enumerate() {
                    (idx[p], wt[p]) = interp_weights_nested(apply(r, v), order);
                }
                for p in 0..pp {
                    let (mut ix, mut wx) = (idx[p], wt[p]);
                    let bind = |ix: [i64; 4], wx: [f64; 4]| -> [Option<usize>; 4] {
                        let mut f = ix.map(|v| search(grid, v));
                        if eps > 0.0 {
                            for n in 0..4 {
                                if wx[n] <= eps {
                                    f[n] = None;
                                }
                            }
                        }
                        f
                    };
                    let mut found = bind(ix, wx);
                    if found.iter().all(Option::is_none) {
                        // Empty tap: fall back to the central tap of the same row.
                        (ix, wx) = (idx[p_ref], wt[p_ref]);
                        found = bind(ix, wx);
                    }
                    let mut w = [0f64; 4];
                    for n in 0..4 {
                        w[n] = wx[n] * if found[n].is_some() { 1.0 } else { 0.0 };
                    }
                    let mut colsum = w[0] + w[1] + w[2] + w[3];
                    if colsum == 0.0 {
                        w[0] = if found[0].is_some() { 1.0 } else { 0.0 };
                        colsum = w[0] + w[1] + w[2] + w[3];
                    }
                    for n in 0..4 {
                        let a = at(n, g, k, p);
                        wout[a] = w[n] / colsum.max(1e-12);
                        pos[a] = found[n].unwrap_or(0) as u32;
                        complete[k] &= found[n].is_some();
                    }
                }
            }
        }
        ConvGeometry { nside, kernel_sz, g: gg, k: kk, n_in: grid.len(), p: pp, pos, w: wout, complete }
    }

    /// Flat index of entry `(n, g, k, p)` in `pos` / `w`.
    #[inline]
    pub fn at(&self, n: usize, g: usize, k: usize, p: usize) -> usize {
        ((n * self.g + g) * self.k + k) * self.p + p
    }

    /// Bytes held by the bound stencils.
    pub fn nbytes(&self) -> usize {
        self.pos.len() * 4 + self.w.len() * 8 + self.complete.len()
    }
}

/// Convolution kernel `[G, C_in, C_out, P]`, cast through `f32` like
/// `HealPixConv.set_kernel` so that results match the library.
#[derive(Clone, Debug)]
pub struct Kernel {
    /// Number of gauges `G`.
    pub g: usize,
    /// Number of input channels `C_in`.
    pub c_in: usize,
    /// Number of output channels `C_out`.
    pub c_out: usize,
    /// Number of stencil taps `P`.
    pub p: usize,
    /// Weights `[G, C_in, C_out, P]`, row-major.
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

    /// Weight of gauge `g`, input channel `c`, output channel `o`, tap `p`.
    #[inline]
    pub fn at(&self, g: usize, c: usize, o: usize, p: usize) -> f64 {
        self.w[((g * self.c_in + c) * self.c_out + o) * self.p + p]
    }
}

/// Transpose of [`convolution_forward`] (`AdjointOps.convolution_adjoint`):
/// `g` `[B, G * C_out, K]` -> `[B, C_in, n_in]`, scattering in the reference
/// order (gauge, bilinear neighbour, row, tap).
pub fn convolution_adjoint(geo: &ConvGeometry, ker: &Kernel, g: &[f64], batch: usize) -> Vec<f64> {
    let mut out = vec![0f64; batch * ker.c_in * geo.n_in];
    adjoint_add(geo, ker, g, batch, None, 0, geo.k, &mut out);
    out
}

/// Rows of the geometry to apply: `sel` (indices) or all of them.
fn selected(geo: &ConvGeometry, sel: Option<&[usize]>) -> Vec<usize> {
    sel.map_or_else(|| (0..geo.k).collect(), <[usize]>::to_vec)
}

/// Adds the adjoint of the geometry rows `sel` (all by default), read from
/// rows `r0 + i` of a `[B, G * C_out, k_total]` cotangent, into `out`
/// `[B, C_in, n_in]`.
#[allow(clippy::too_many_arguments)]
fn adjoint_add(
    geo: &ConvGeometry,
    ker: &Kernel,
    g: &[f64],
    batch: usize,
    sel: Option<&[usize]>,
    r0: usize,
    k_total: usize,
    out: &mut [f64],
) {
    let (pp, gg, ni) = (geo.p, geo.g, geo.n_in);
    assert_eq!(ker.g, gg);
    assert_eq!(ker.p, pp);
    let co = ker.c_out;
    assert_eq!(g.len(), batch * gg * co * k_total);
    assert_eq!(out.len(), batch * ker.c_in * ni);
    let rows = selected(geo, sel);
    let mut taps = vec![0f64; rows.len() * pp];
    for b in 0..batch {
        for gi in 0..gg {
            for c in 0..ker.c_in {
                for i in 0..rows.len() {
                    for p in 0..pp {
                        let mut s = 0.0;
                        for o in 0..co {
                            s += g[(b * gg * co + gi * co + o) * k_total + r0 + i] * ker.at(gi, c, o, p);
                        }
                        taps[i * pp + p] = s;
                    }
                }
                let dst = &mut out[(b * ker.c_in + c) * ni..][..ni];
                for n in 0..4 {
                    for (i, &k) in rows.iter().enumerate() {
                        let base = geo.at(n, gi, k, 0);
                        for p in 0..pp {
                            dst[geo.pos[base + p] as usize] += taps[i * pp + p] * geo.w[base + p];
                        }
                    }
                }
            }
        }
    }
}

/// Linear convolution of `x` `[B, C_in, n_in]` -> `[B, G * C_out, K]` (no bias).
pub fn convolution_forward(geo: &ConvGeometry, ker: &Kernel, x: &[f64], batch: usize) -> Vec<f64> {
    let mut out = vec![0f64; batch * ker.g * ker.c_out * geo.k];
    forward_into(geo, ker, x, batch, None, 0, geo.k, &mut out);
    out
}

/// Geometry rows `sel` (all by default) of the forward into rows `r0 + i`
/// of `out` `[B, G * C_out, k_total]`.
#[allow(clippy::too_many_arguments)]
fn forward_into(
    geo: &ConvGeometry,
    ker: &Kernel,
    x: &[f64],
    batch: usize,
    sel: Option<&[usize]>,
    r0: usize,
    k_total: usize,
    out: &mut [f64],
) {
    let (pp, gg, ni) = (geo.p, geo.g, geo.n_in);
    assert_eq!(ker.g, gg);
    assert_eq!(ker.p, pp);
    assert_eq!(x.len(), batch * ker.c_in * ni);
    let co = ker.c_out;
    let rows = selected(geo, sel);
    let mut gathered = vec![0f64; pp];
    for b in 0..batch {
        for g in 0..gg {
            for c in 0..ker.c_in {
                let xs = &x[(b * ker.c_in + c) * ni..][..ni];
                for (i, &k) in rows.iter().enumerate() {
                    for (p, slot) in gathered.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for n in 0..4 {
                            let at = geo.at(n, g, k, p);
                            acc += xs[geo.pos[at] as usize] * geo.w[at];
                        }
                        *slot = acc;
                    }
                    for o in 0..co {
                        let mut s = 0.0;
                        for (p, &v) in gathered.iter().enumerate() {
                            s += v * ker.at(g, c, o, p);
                        }
                        out[(b * gg * co + g * co + o) * k_total + r0 + i] += s;
                    }
                }
            }
        }
    }
}

/// A convolution from a grid to output rows, with its geometry either held
/// for all rows or rebuilt by sub-blocks of `chunk` rows at every
/// application (geometry memory bounded by the sub-block, at the price of
/// recomputing it).  Both give the same forward values; the adjoint adds
/// the sub-blocks in turn, which may change the last bits.
pub enum ConvPlan {
    /// Geometry bound once for all output rows.
    Whole(ConvGeometry),
    /// Geometry rebuilt by sub-blocks of `chunk` rows from the stored grid,
    /// rows and stencil parameters at every application.
    Chunked {
        /// Grid read (sorted NESTED ids).
        grid: Vec<i64>,
        /// Output rows (NESTED ids, in output order).
        rows: Vec<i64>,
        /// Output rows per sub-block.
        chunk: usize,
        /// `nside` of the level.
        nside: i64,
        /// Stencil side in taps.
        kernel_sz: usize,
        /// Number of gauges `G`.
        g: usize,
        /// Orientation convention.
        gauge: Gauge,
        /// Binding threshold ([`ConvGeometry::rows_eps`]).
        eps: f64,
    },
}

impl ConvPlan {
    /// Holds the whole geometry, or the `Chunked` form when `chunk` is
    /// smaller than the number of `rows` (sub-blocks of at least one row).
    pub fn new(grid: &[i64], rows: &[i64], nside: i64, kernel_sz: usize, g: usize, gauge: Gauge, chunk: Option<usize>) -> Self {
        Self::new_eps(grid, rows, nside, kernel_sz, g, gauge, chunk, 0.0)
    }

    /// [`new`](Self::new) with the binding threshold `eps` of
    /// [`ConvGeometry::rows_eps`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_eps(grid: &[i64], rows: &[i64], nside: i64, kernel_sz: usize, g: usize, gauge: Gauge, chunk: Option<usize>, eps: f64) -> Self {
        match chunk {
            Some(c) if c < rows.len() => ConvPlan::Chunked {
                grid: grid.to_vec(),
                rows: rows.to_vec(),
                chunk: c.max(1),
                nside,
                kernel_sz,
                g,
                gauge,
                eps,
            },
            _ => ConvPlan::Whole(ConvGeometry::rows_eps(grid, rows, nside, kernel_sz, g, gauge, eps)),
        }
    }

    /// Number of output rows.
    pub fn n_rows(&self) -> usize {
        match self {
            ConvPlan::Whole(geo) => geo.k,
            ConvPlan::Chunked { rows, .. } => rows.len(),
        }
    }

    /// Number of grid cells (inputs).
    pub fn n_in(&self) -> usize {
        match self {
            ConvPlan::Whole(geo) => geo.n_in,
            ConvPlan::Chunked { grid, .. } => grid.len(),
        }
    }

    /// Calls `f(geometry, selection, r0)` for the rows `sel` (all by
    /// default), whose results are rows `r0..` of the output.
    fn each_chunk(&self, sel: Option<&[usize]>, mut f: impl FnMut(&ConvGeometry, Option<&[usize]>, usize)) {
        match self {
            ConvPlan::Whole(geo) => f(geo, sel, 0),
            ConvPlan::Chunked { grid, rows, chunk, nside, kernel_sz, g, gauge, eps } => {
                let ids: Vec<i64> = sel.map_or_else(|| rows.clone(), |s| s.iter().map(|&i| rows[i]).collect());
                for (i, part) in ids.chunks(*chunk).enumerate() {
                    let geo = ConvGeometry::rows_eps(grid, part, *nside, *kernel_sz, *g, *gauge, *eps);
                    f(&geo, None, i * chunk);
                }
            }
        }
    }

    /// `[B, C_in, n_in] -> [B, G * C_out, n_rows]`.
    pub fn forward(&self, ker: &Kernel, x: &[f64], batch: usize) -> Vec<f64> {
        self.forward_rows(ker, x, batch, None)
    }

    /// Output rows `sel` only: `[B, C_in, n_in] -> [B, G * C_out, |sel|]`.
    pub fn forward_rows(&self, ker: &Kernel, x: &[f64], batch: usize, sel: Option<&[usize]>) -> Vec<f64> {
        let k = sel.map_or(self.n_rows(), <[usize]>::len);
        let mut out = vec![0f64; batch * ker.g * ker.c_out * k];
        self.each_chunk(sel, |geo, s, r0| forward_into(geo, ker, x, batch, s, r0, k, &mut out));
        out
    }

    /// `[B, G * C_out, n_rows] -> [B, C_in, n_in]`.
    pub fn adjoint(&self, ker: &Kernel, g: &[f64], batch: usize) -> Vec<f64> {
        self.adjoint_rows(ker, g, batch, None)
    }

    /// Adjoint of [`forward_rows`](Self::forward_rows): `[B, G * C_out, |sel|] -> [B, C_in, n_in]`.
    pub fn adjoint_rows(&self, ker: &Kernel, g: &[f64], batch: usize, sel: Option<&[usize]>) -> Vec<f64> {
        let k = sel.map_or(self.n_rows(), <[usize]>::len);
        let mut out = vec![0f64; batch * ker.c_in * self.n_in()];
        self.each_chunk(sel, |geo, s, r0| adjoint_add(geo, ker, g, batch, s, r0, k, &mut out));
        out
    }
}
