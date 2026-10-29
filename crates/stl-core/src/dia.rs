//! Diagonal (DIA) form of the wavelet convolution in face coordinates.
//!
//! A HEALPix face is an `nside x nside` image in `(x, y)` (the NESTED bits
//! de-interleaved).  Every cell read by the bound stencil of a pixel
//! `(x, y, f)` lies, in the frame of face `f` extended across its edges and
//! corners, at a small offset `(dx, dy)` from it: the convolution is
//! `y[i] = sum_d a_d[i] x[cell(i, off_d)]`, with the bilinear weights and the
//! kernel folded into per-row coefficients `a_d` on a fixed set of
//! diagonals `off_d` ([`Diagonals`]).  The adjoint gathers the same
//! coefficients from the rows at the opposite offsets, so it needs no
//! transposition.
//!
//! Across a face edge the frames of the two faces differ by a rotation of a
//! multiple of 90 degrees (HEALPix C++ neighbour tables), so the opposite
//! offset seen from the cell has the same max-norm as the offset seen from
//! the row.  At the eight vertices where three faces meet, the quadrant
//! between the two other faces is empty in the extended frame
//! ([`ext_pix`] returns `None`): every pair of faces there shares an edge,
//! so every read still has an offset, only larger than in the interior.

use crate::conv::{ConvGeometry, Kernel};
use crate::healpix::{nest2xyf, xyf2nest};

/// HEALPix C++ `nb_facearray`: the face beyond each side or corner of a
/// face, indexed by `4 + xdir + 3 ydir` (`xdir`, `ydir` in -1, 0, 1); -1
/// at the vertices where only three faces meet.
const NB_FACE: [[i8; 12]; 9] = [
    [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],
    [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],
    [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],
    [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],
    [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],
    [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],
    [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],
];

/// HEALPix C++ `nb_swaparray`, by direction and face row (`face / 4`):
/// bit 1 flips x, bit 2 flips y, bit 4 then swaps x and y.
const NB_SWAP: [[u8; 3]; 9] = [[0, 0, 3], [0, 0, 6], [0, 0, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 0, 0], [6, 0, 0], [3, 0, 0]];

/// Pixel at the extended coordinates `(ex, ey)` of `face`
/// (`-nside <= ex, ey < 2 nside`); `None` in the quadrant missing at a
/// vertex of three faces.
pub fn ext_pix(face: i64, ex: i64, ey: i64, nside: i64, order: u32) -> Option<i64> {
    let (mut x, mut y, mut nb) = (ex, ey, 4usize);
    if x < 0 {
        x += nside;
        nb -= 1;
    } else if x >= nside {
        x -= nside;
        nb += 1;
    }
    if y < 0 {
        y += nside;
        nb -= 3;
    } else if y >= nside {
        y -= nside;
        nb += 3;
    }
    let f = NB_FACE[nb][face as usize];
    if f < 0 {
        return None;
    }
    let bits = NB_SWAP[nb][(face >> 2) as usize];
    if bits & 1 != 0 {
        x = nside - x - 1;
    }
    if bits & 2 != 0 {
        y = nside - y - 1;
    }
    if bits & 4 != 0 {
        std::mem::swap(&mut x, &mut y);
    }
    Some(xyf2nest(x, y, f as i64, order))
}

/// Offset of `cell` from `row` in the extended frame of `row`'s face;
/// `None` when `cell` is on neither that face nor one of its neighbours.
pub fn rel_offset(row: i64, cell: i64, nside: i64, order: u32) -> Option<(i64, i64)> {
    let (x, y, f) = nest2xyf(row, order);
    let (xq, yq, fq) = nest2xyf(cell, order);
    if fq == f {
        return Some((xq - x, yq - y));
    }
    let nb = (0..9).find(|&nb| nb != 4 && NB_FACE[nb][f as usize] as i64 == fq)?;
    let bits = NB_SWAP[nb][(f >> 2) as usize];
    let (mut a, mut b) = (xq, yq);
    if bits & 4 != 0 {
        std::mem::swap(&mut a, &mut b);
    }
    if bits & 2 != 0 {
        b = nside - b - 1;
    }
    if bits & 1 != 0 {
        a = nside - a - 1;
    }
    let (xdir, ydir) = (nb as i64 % 3 - 1, nb as i64 / 3 - 1);
    Some((a + xdir * nside - x, b + ydir * nside - y))
}

/// The diagonals: a set of offsets within the square window `[-w, w]^2`,
/// numbered row by row in `dy` then `dx`.
#[derive(Clone, Debug)]
pub struct Diagonals {
    pub w: i64,
    pub offsets: Vec<(i64, i64)>,
    /// `index[(dy + w) * (2 w + 1) + dx + w]`: diagonal of an offset, or -1.
    pub index: Vec<i32>,
}

impl Diagonals {
    /// The offsets of `keep` within `[-w, w]^2`.
    pub fn new(w: i64, keep: impl Fn(i64, i64) -> bool) -> Self {
        let side = 2 * w + 1;
        let mut offsets = Vec::new();
        let mut index = vec![-1i32; (side * side) as usize];
        for dy in -w..=w {
            for dx in -w..=w {
                if keep(dx, dy) {
                    index[((dy + w) * side + dx + w) as usize] = offsets.len() as i32;
                    offsets.push((dx, dy));
                }
            }
        }
        Diagonals { w, offsets, index }
    }

    /// The offsets read with a nonzero weight by the full-sky geometries at
    /// `nsides` (all rows), closed under the half turn.  For kernel 5 and
    /// `G = 4`, gauge cosmo, the set is the same at every `nside` from 16 to
    /// 1024: 81 diagonals with `max(|dx|, |dy|) <= 5`
    /// (`examples/dia_offsets.rs`); nside 8 reads a slightly different set.
    pub fn observed(nsides: &[i64], kernel_sz: usize, g: usize, gauge: crate::stencil::Gauge) -> Self {
        let mut seen = std::collections::BTreeSet::new();
        let mut w = 0;
        for &nside in nsides {
            let order = crate::healpix::order_of(nside);
            let ids: Vec<i64> = (0..crate::healpix::npix(nside)).collect();
            let geo = ConvGeometry::rows(&ids, &ids, nside, kernel_sz, g, gauge);
            for (k, &row) in ids.iter().enumerate() {
                for gi in 0..g {
                    for p in 0..geo.p {
                        for n in 0..4 {
                            let a = geo.at(n, gi, k, p);
                            if geo.w[a] != 0.0 {
                                let (dx, dy) =
                                    rel_offset(row, ids[geo.pos[a] as usize], nside, order).expect("read beyond the neighbour faces");
                                w = w.max(dx.abs()).max(dy.abs());
                                seen.insert((dx, dy));
                                seen.insert((-dx, -dy));
                            }
                        }
                    }
                }
            }
        }
        Self::new(w, |dx, dy| seen.contains(&(dx, dy)))
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Diagonal of the offset `(dx, dy)`, if any.
    #[inline]
    pub fn find(&self, dx: i64, dy: i64) -> Option<usize> {
        if dx.abs() > self.w || dy.abs() > self.w {
            return None;
        }
        let i = self.index[((dy + self.w) * (2 * self.w + 1) + dx + self.w) as usize];
        (i >= 0).then_some(i as usize)
    }
}

/// The convolution of [`ConvGeometry`] in DIA form, on the CPU (reference
/// of the device kernels): coefficients `coef[((g * C_out + o) * D + d) *
/// K + k]` of output row `k`, gauge `g`, output channel `o` and diagonal
/// `d`; the cell of a row on a diagonal is found from the NESTED ids.
#[derive(Clone, Debug)]
pub struct DiaGeometry {
    pub nside: i64,
    pub g: usize,
    pub c_out: usize,
    pub k: usize,
    pub diagonals: Diagonals,
    pub coef: Vec<f64>,
}

impl DiaGeometry {
    /// Folds the kernel `ker` (`C_in = 1`) into the bound stencils of
    /// `geo`, whose rows are `rows` and grid `grid`: the coefficient of a
    /// diagonal adds `ker[g, o, p] * w` over the taps `p` and then the
    /// bilinear neighbours `n` reading its cell with a nonzero weight `w`
    /// (incomplete rows keep their renormalised binding).  `None` if such a
    /// read is off the diagonals.
    pub fn from_conv(geo: &ConvGeometry, ker: &Kernel, rows: &[i64], grid: &[i64], diagonals: &Diagonals) -> Option<Self> {
        assert_eq!(ker.c_in, 1);
        assert_eq!(rows.len(), geo.k);
        let order = crate::healpix::order_of(geo.nside);
        let (gg, co, dd, kk) = (geo.g, ker.c_out, diagonals.len(), geo.k);
        let mut coef = vec![0f64; gg * co * dd * kk];
        for (k, &row) in rows.iter().enumerate() {
            for g in 0..gg {
                for p in 0..geo.p {
                    for n in 0..4 {
                        let a = geo.at(n, g, k, p);
                        if geo.w[a] == 0.0 {
                            continue;
                        }
                        let cell = grid[geo.pos[a] as usize];
                        let (dx, dy) = rel_offset(row, cell, geo.nside, order)?;
                        let d = diagonals.find(dx, dy)?;
                        for o in 0..co {
                            coef[((g * co + o) * dd + d) * kk + k] += ker.at(g, 0, o, p) * geo.w[a];
                        }
                    }
                }
            }
        }
        Some(DiaGeometry { nside: geo.nside, g: gg, c_out: co, k: kk, diagonals: diagonals.clone(), coef })
    }

    /// Cell of `row` on diagonal `d`, if it exists.
    #[inline]
    fn cell(&self, row: i64, d: usize, order: u32) -> Option<i64> {
        let (x, y, f) = nest2xyf(row, order);
        let (dx, dy) = self.diagonals.offsets[d];
        ext_pix(f, x + dx, y + dy, self.nside, order)
    }

    /// `x` `[B, n_grid]` on `grid` -> `[B, G * C_out, K]`, diagonals summed
    /// in increasing order.
    pub fn forward(&self, rows: &[i64], grid: &[i64], x: &[f64], batch: usize) -> Vec<f64> {
        let order = crate::healpix::order_of(self.nside);
        let (gg, co, dd, kk, ni) = (self.g, self.c_out, self.diagonals.len(), self.k, grid.len());
        assert_eq!(x.len(), batch * ni);
        let mut out = vec![0f64; batch * gg * co * kk];
        for (k, &row) in rows.iter().enumerate() {
            let pos: Vec<Option<usize>> = (0..dd).map(|d| self.cell(row, d, order).and_then(|c| grid.binary_search(&c).ok())).collect();
            for b in 0..batch {
                for go in 0..gg * co {
                    let mut s = 0.0;
                    for (d, p) in pos.iter().enumerate() {
                        if let Some(p) = p {
                            s += self.coef[(go * dd + d) * kk + k] * x[b * ni + p];
                        }
                    }
                    out[(b * gg * co + go) * kk + k] = s;
                }
            }
        }
        out
    }

    /// Transpose of [`forward`](Self::forward): `[B, G * C_out, K]` ->
    /// `[B, n_grid]`, gathered per grid cell over the rows at the opposite
    /// offsets (window order), then gauges and channels.
    pub fn adjoint(&self, rows: &[i64], grid: &[i64], gy: &[f64], batch: usize) -> Vec<f64> {
        let order = crate::healpix::order_of(self.nside);
        let (gg, co, dd, kk, ni) = (self.g, self.c_out, self.diagonals.len(), self.k, grid.len());
        assert_eq!(gy.len(), batch * gg * co * kk);
        let w = self.diagonals.w;
        let mut out = vec![0f64; batch * ni];
        for (j, &cell) in grid.iter().enumerate() {
            let (x, y, f) = nest2xyf(cell, order);
            for vy in -w..=w {
                for vx in -w..=w {
                    let Some(row) = ext_pix(f, x + vx, y + vy, self.nside, order) else { continue };
                    let Ok(k) = rows.binary_search(&row) else { continue };
                    let Some((dx, dy)) = rel_offset(row, cell, self.nside, order) else { continue };
                    let Some(d) = self.diagonals.find(dx, dy) else { continue };
                    for b in 0..batch {
                        let mut s = 0.0;
                        for go in 0..gg * co {
                            s += self.coef[(go * dd + d) * kk + k] * gy[(b * gg * co + go) * kk + k];
                        }
                        out[b * ni + j] += s;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::healpix::{npix, order_of};
    use crate::neighbours::kth_neighbourhood;

    #[test]
    fn unit_offsets_are_the_neighbours() {
        for nside in [2i64, 4, 8, 16] {
            let order = order_of(nside);
            for pix in 0..npix(nside) {
                let (x, y, f) = nest2xyf(pix, order);
                let mut ours: Vec<i64> =
                    (-1..=1).flat_map(|dy| (-1..=1).filter_map(move |dx| ext_pix(f, x + dx, y + dy, nside, order))).collect();
                ours.sort_unstable();
                let mut theirs = kth_neighbourhood(pix, nside, 1);
                theirs.sort_unstable();
                assert_eq!(ours, theirs, "nside {nside} pixel {pix}");
            }
        }
    }

    #[test]
    fn offsets_round_trip() {
        let nside = 16i64;
        let order = order_of(nside);
        for pix in 0..npix(nside) {
            let (x, y, f) = nest2xyf(pix, order);
            for dy in -6..=6 {
                for dx in -6..=6 {
                    let Some(q) = ext_pix(f, x + dx, y + dy, nside, order) else { continue };
                    assert_eq!(rel_offset(pix, q, nside, order), Some((dx, dy)), "nside {nside} pixel {pix} ({dx}, {dy})");
                    // Seen from the cell, the row is at an offset of the same max-norm.
                    let (bx, by) = rel_offset(q, pix, nside, order).unwrap();
                    assert_eq!(bx.abs().max(by.abs()), dx.abs().max(dy.abs()));
                }
            }
        }
    }
}
