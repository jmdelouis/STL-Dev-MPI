//! Patch-local pyramid operations on one block (port of
//! `STL_main/LocalGeometry.py`): the wavelet on a block and its adjoint,
//! the exact block-wise decimation of a level and its transpose, and the
//! decimation pieces of a field supported on a block with their transposes.
//!
//! Halos (pixels of the level): convolution `(kernel_size - 1) / 2 + 2`;
//! decimation `ceil(radius / pixel) + 1`, doubled for pieces and
//! transposes.  Levels are full sky in canonical NESTED order, so a cell id
//! is also an array position.
//!
//! Fields are row-major with the pixel axis last: a real field over `ids`
//! is `[R, |ids|]` for `R` leading rows (batch x channels x ...), a
//! wavelet field is complex `[R, L, |ids|]`.

use crate::conv::{ConvPlan, Kernel};
use crate::decimation::{smooth_decimation, Csr, DecimationParams};
use crate::healpix::npix;
use crate::neighbours::{dilate_fine, parents};
use crate::stencil::Gauge;
use crate::wavelet::wavelet_weights;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Index of each `ids` entry in `sorted` (all must be present).
pub fn positions(sorted: &[i64], ids: &[i64]) -> Vec<usize> {
    ids.iter()
        .map(|&v| {
            let q = sorted.partition_point(|&s| s < v);
            assert!(q < sorted.len() && sorted[q] == v, "cell id {v} is missing from the domain");
            q
        })
        .collect()
}

/// `x[r, pos[i]]` for every row: `[R, n] -> [R, |pos|]`.
fn gather_rows(x: &[f64], n: usize, pos: &[usize]) -> Vec<f64> {
    assert_eq!(x.len() % n.max(1), 0);
    x.chunks(n).flat_map(|row| pos.iter().map(move |&p| row[p])).collect()
}

/// Partial decimation of one domain: coarse ids and the forward matrix.
#[derive(Clone, Debug)]
pub struct Down {
    /// Sorted coarse NESTED ids (parents of the domain, level `dg + 1`).
    pub out_ids: Vec<i64>,
    /// Forward matrix `[N_out, N_in]`, columns indexing the domain.
    pub forward: Csr,
    /// Transpose of `forward`.
    pub transposed: Csr,
}

impl Down {
    /// `[R, N_in] -> [R, N_out]`.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        x.chunks(self.forward.n_cols).flat_map(|row| self.forward.apply(row)).collect()
    }

    /// `[R, N_out] -> [R, N_in]`.
    pub fn adjoint(&self, g: &[f64]) -> Vec<f64> {
        g.chunks(self.transposed.n_cols).flat_map(|row| self.transposed.apply(row)).collect()
    }
}

/// Fine wavelet on a block with what its adjoint needs again.
pub struct Wave {
    /// `[R, L, |block|]`.
    pub z: Vec<Complex64>,
    /// Grid read by the convolution: the block plus its convolution halo,
    /// sorted NESTED ids.
    pub domain: Vec<i64>,
    /// Convolution from `domain` to the block rows.
    pub conv: ConvPlan,
}

/// Geometry parameters of the engine; operators are built per domain.
#[derive(Clone, Debug)]
pub struct LocalGeometry {
    /// `nside` of level 0.
    pub nside0: i64,
    /// Stencil side in taps.
    pub kernel_sz: usize,
    /// Number of orientations `L` (convolution gauges).
    pub l: usize,
    /// Orientation convention of the stencils.
    pub gauge: Gauge,
    /// Wavelet kernel `[L, 1, 2, P]` (real, imaginary), the same at every level.
    pub kernel: Kernel,
    /// Pixels per block.
    pub block_pixels: usize,
    /// Convolution halo in rings, `(kernel_sz - 1) / 2 + 2`.
    pub conv_rings: u32,
    /// Decimation parameters (radius, sigma, normalisation).
    pub down_params: DecimationParams,
    /// Rebuild the convolution geometry by sub-blocks of this many output
    /// rows at every application instead of holding it for the whole block.
    pub conv_chunk: Option<usize>,
}

impl LocalGeometry {
    /// Geometry of a pyramid with level 0 at `nside0`, wavelet stencil of
    /// `kernel_sz` taps per side, `l` orientations, default decimation.
    pub fn new(nside0: i64, kernel_sz: usize, l: usize, gauge: Gauge, block_pixels: usize) -> Self {
        assert!(block_pixels > 0);
        let kernel = Kernel::shared(&wavelet_weights(kernel_sz), l, 1, 2, kernel_sz * kernel_sz);
        LocalGeometry {
            nside0,
            kernel_sz,
            l,
            gauge,
            kernel,
            block_pixels,
            conv_rings: ((kernel_sz - 1) / 2 + 2) as u32,
            down_params: DecimationParams::default(),
            conv_chunk: None,
        }
    }

    /// `nside` of level `dg`, `nside0 >> dg`.
    pub fn nside(&self, dg: usize) -> i64 {
        self.nside0 >> dg
    }

    /// Number of pixels of level `dg`.
    pub fn pixels(&self, dg: usize) -> usize {
        npix(self.nside(dg)) as usize
    }

    /// Cone radius of the decimation from level `dg`, in fine pixels, + 1
    /// (same floating-point path as `LocalGeometry.down_rings`).
    pub fn down_rings(&self, dg: usize) -> u32 {
        let pix_deg = (4.0 * PI / self.pixels(dg) as f64).sqrt() * (180.0 / PI);
        let radius = self.down_params.radius_deg.unwrap_or(3.0 * pix_deg);
        (radius / pix_deg).ceil() as u32 + 1
    }

    /// Contiguous blocks of `block_pixels` covering level `dg`.
    pub fn blocks(&self, dg: usize) -> Vec<Vec<i64>> {
        let n = self.pixels(dg) as i64;
        let size = self.block_pixels as i64;
        (0..n).step_by(self.block_pixels).map(|s| (s..(s + size).min(n)).collect()).collect()
    }


    /// Partial decimation of the fine `domain` of level `dg` (sorted NESTED
    /// ids) to its parents, with the transposed matrix.
    pub fn down(&self, dg: usize, domain: &[i64]) -> Down {
        let (out_ids, forward) = smooth_decimation(domain, self.nside(dg), &self.down_params);
        let transposed = forward.transpose();
        Down { out_ids, forward, transposed }
    }

    /// Convolution of level `dg` from the grid `grid` to the output `rows`
    /// (geometry by sub-blocks of `conv_chunk` rows when set).
    pub fn conv_plan(&self, dg: usize, grid: &[i64], rows: &[i64]) -> ConvPlan {
        ConvPlan::new(grid, rows, self.nside(dg), self.kernel_sz, self.l, self.gauge, self.conv_chunk)
    }

    /// [`conv_plan`](Self::conv_plan) with the binding threshold `eps` of
    /// [`crate::conv::ConvGeometry::rows_eps`].
    pub fn conv_plan_eps(&self, dg: usize, grid: &[i64], rows: &[i64], eps: f64) -> ConvPlan {
        ConvPlan::new_eps(grid, rows, self.nside(dg), self.kernel_sz, self.l, self.gauge, self.conv_chunk, eps)
    }

    /// Wavelet of `x_level` `[R, N_dg]` on `block`: `z` `[R, L, |block|]`.
    pub fn wave(&self, x_level: &[f64], block: &[i64], dg: usize) -> Wave {
        let n = self.pixels(dg);
        let rows = x_level.len() / n;
        let domain = dilate_fine(block, self.nside(dg), self.conv_rings);
        let conv = self.conv_plan(dg, &domain, block);
        let x: Vec<f64> = x_level.chunks(n).flat_map(|row| domain.iter().map(move |&i| row[i as usize])).collect();
        let y = conv.forward(&self.kernel, &x, rows);
        let (l, m) = (self.l, block.len());
        let mut z = Vec::with_capacity(rows * l * m);
        for r in 0..rows {
            for o in 0..l {
                let re = &y[(r * 2 * l + 2 * o) * m..][..m];
                let im = &y[(r * 2 * l + 2 * o + 1) * m..][..m];
                z.extend(re.iter().zip(im).map(|(&a, &b)| Complex64::new(a, b)));
            }
        }
        Wave { z, domain, conv }
    }

    /// Adjoint of [`wave`](Self::wave): `gz` `[R, L, |block|]` -> `[R, |domain|]`.
    pub fn wave_adjoint(&self, gz: &[Complex64], block: &[i64], wave: &Wave) -> Vec<f64> {
        let (l, m) = (self.l, block.len());
        let rows = gz.len() / (l * m);
        let mut g = vec![0f64; rows * 2 * l * m];
        for r in 0..rows {
            for o in 0..l {
                let src = &gz[(r * l + o) * m..][..m];
                for (i, v) in src.iter().enumerate() {
                    g[(r * 2 * l + 2 * o) * m + i] = v.re;
                    g[(r * 2 * l + 2 * o + 1) * m + i] = v.im;
                }
            }
        }
        wave.conv.adjoint(&self.kernel, &g, rows)
    }

    /// `maps[dg] -> maps[dg + 1]` with the full-sky weights, block by block:
    /// `[R, N_dg] -> [R, N_{dg+1}]`.
    pub fn decimate(&self, x: &[f64], dg: usize) -> Vec<f64> {
        let (n, nc) = (self.pixels(dg), self.pixels(dg + 1));
        let rows = x.len() / n;
        let mut out = vec![0f64; rows * nc];
        let rings = 2 * self.down_rings(dg);
        for block in self.blocks(dg) {
            let domain = dilate_fine(&block, self.nside(dg), rings);
            let down = self.down(dg, &domain);
            let y = down.apply(&gather_rows(x, n, &domain.iter().map(|&i| i as usize).collect::<Vec<_>>()));
            let coarse = parents(&block);
            let pos = positions(&down.out_ids, &coarse);
            let no = down.out_ids.len();
            for r in 0..rows {
                for (&c, &p) in coarse.iter().zip(&pos) {
                    out[r * nc + c as usize] = y[r * no + p];
                }
            }
        }
        out
    }

    /// Transpose of [`decimate`](Self::decimate): `[R, N_{dg+1}] -> [R, N_dg]`, exact.
    pub fn decimate_adjoint(&self, g: &[f64], dg: usize) -> Vec<f64> {
        let (n, nc) = (self.pixels(dg), self.pixels(dg + 1));
        let rows = g.len() / nc;
        let mut out = vec![0f64; rows * n];
        let rings = 2 * self.down_rings(dg);
        for block in self.blocks(dg) {
            let domain = dilate_fine(&block, self.nside(dg), rings);
            let down = self.down(dg, &domain);
            let out_pos: Vec<usize> = down.out_ids.iter().map(|&i| i as usize).collect();
            let x = down.adjoint(&gather_rows(g, nc, &out_pos));
            let pos = positions(&domain, &block);
            let kd = domain.len();
            for r in 0..rows {
                for (&b, &p) in block.iter().zip(&pos) {
                    out[r * n + b as usize] = x[r * kd + p];
                }
            }
        }
        out
    }

    /// Contribution of a field `a` `[R, |support|]` supported on `support`
    /// (level `dg`) to level `dg + 1`: `(y, next_support)`, `y` exact on
    /// `next_support`.
    pub fn piece_forward(&self, a: &[f64], support: &[i64], dg: usize) -> (Vec<f64>, Vec<i64>) {
        let rings = self.down_rings(dg);
        let domain = dilate_fine(support, self.nside(dg), 2 * rings);
        let down = self.down(dg, &domain);
        let (m, kd) = (support.len(), domain.len());
        let rows = a.len() / m;
        let pos = positions(&domain, support);
        let mut padded = vec![0f64; rows * kd];
        for r in 0..rows {
            for (i, &p) in pos.iter().enumerate() {
                padded[r * kd + p] = a[r * m + i];
            }
        }
        let y = down.apply(&padded);
        let next_support = parents(&dilate_fine(support, self.nside(dg), rings));
        let out = gather_rows(&y, down.out_ids.len(), &positions(&down.out_ids, &next_support));
        (out, next_support)
    }

    /// Transpose of [`piece_forward`](Self::piece_forward), exact on `support`:
    /// `g` `[R, |next_support|]` -> `[R, |support|]`.
    pub fn piece_adjoint(&self, g: &[f64], support: &[i64], next_support: &[i64], dg: usize) -> Vec<f64> {
        let domain = dilate_fine(support, self.nside(dg), 2 * self.down_rings(dg));
        let down = self.down(dg, &domain);
        let (mn, no) = (next_support.len(), down.out_ids.len());
        let rows = g.len() / mn;
        let pos = positions(&down.out_ids, next_support);
        let mut padded = vec![0f64; rows * no];
        for r in 0..rows {
            for (i, &p) in pos.iter().enumerate() {
                padded[r * no + p] = g[r * mn + i];
            }
        }
        let x = down.adjoint(&padded);
        gather_rows(&x, domain.len(), &positions(&domain, support))
    }
}

/// Derivative of `|z|` applied to a real cotangent (`AdjointOps.modulus_vjp`),
/// with the zero subgradient at `z = 0`.
pub fn modulus_vjp(z: &[Complex64], g: &[f64]) -> Vec<Complex64> {
    assert_eq!(z.len(), g.len());
    z.iter()
        .zip(g)
        .map(|(&v, &gi)| {
            let mag = v.norm();
            gi * v / if mag == 0.0 { 1.0 } else { mag }
        })
        .collect()
}
