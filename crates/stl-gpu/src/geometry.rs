//! Convolution geometry on the device (port of `stl_core::conv::ConvGeometry`):
//! rotated stencils, bilinear interpolation and binding computed per output
//! row in a kernel, then transposed (entries grouped by the grid cell they
//! read, in increasing entry order) so that the adjoint is a gather with a
//! fixed summation order.
//!
//! Entry `((g * 4 + n) * K + k) * P + p` for gauge `g`, bilinear neighbour
//! `n`, output row `k` and tap `p`: the transposed groups list the entries
//! of a cell by gauge, neighbour, row, tap, the scatter order of
//! `conv::convolution_adjoint`.  Memory: `4 G P (4 + 8 + 4)` bytes per row
//! (positions, weights, transposed indices).

use crate::device::{Arg, Gpu, NULL};
use cudarc::driver::CudaSlice;
use stl_core::healpix::order_of;
use stl_core::stencil::{local_kernel_grid, Gauge};

/// A sorted set of cells read by the convolution, on the device.  Cells of
/// the contiguous range `[own.0, own.1)` sit at positions `own.2 + (id -
/// own.0)` (no search needed).
pub struct Grid {
    pub ids: CudaSlice<i64>,
    pub n: usize,
    pub own: (i64, i64, i64),
}

impl Grid {
    pub fn new(gpu: &Gpu, ids: &[i64], own: Option<(i64, i64)>) -> Self {
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "grid ids must be sorted and unique");
        assert!(ids.len() < u32::MAX as usize);
        let own = match own {
            Some((s, e)) if s < e => {
                let p0 = ids.partition_point(|&v| v < s);
                assert!(p0 + ((e - s) as usize) <= ids.len() && ids[p0] == s && ids[p0 + (e - s - 1) as usize] == e - 1);
                (s, e, p0 as i64)
            }
            _ => (0, 0, 0),
        };
        Grid { ids: gpu.upload(ids), n: ids.len(), own }
    }
}

pub struct DeviceGeometry {
    /// Rows, gauges, taps.
    pub k: usize,
    pub g: usize,
    pub p: usize,
    pub pos: CudaSlice<u32>,
    pub w: CudaSlice<f64>,
    /// `complete[k * G + g]` when requested.
    pub complete: Option<CudaSlice<u8>>,
    /// Transposed: grid positions read (`dest`), their entry ranges
    /// (`offsets`, `|dest| + 1`) and the entries (`idx`).
    pub dest: CudaSlice<u32>,
    pub offsets: CudaSlice<u32>,
    pub idx: CudaSlice<u32>,
    pub n_dest: usize,
}

impl DeviceGeometry {
    /// Bound stencils of the output rows `rows` (`k` cells at `nside`) on `grid`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &Gpu,
        grid: &Grid,
        rows: &CudaSlice<i64>,
        k: usize,
        nside: i64,
        kernel_sz: usize,
        g: usize,
        gauge: Gauge,
        complete: bool,
    ) -> Self {
        let order = order_of(nside);
        let p = kernel_sz * kernel_sz;
        let n_entries = 4 * g * k * p;
        assert!(n_entries < u32::MAX as usize, "block too large for u32 entry indices");
        let stencil: Vec<f64> = local_kernel_grid(kernel_sz, nside).into_iter().flatten().collect();
        let stencil = gpu.upload(&stencil);
        let pos = gpu.zeros::<u32>(n_entries);
        let w = gpu.zeros::<f64>(n_entries);
        let comp = complete.then(|| gpu.zeros::<u8>(k * g));
        let p_ref = (kernel_sz / 2) * (kernel_sz + 1);
        gpu.launch1(
            "geom_build",
            k * g,
            &[
                gpu.a(&grid.ids),
                Arg::I64(grid.n as i64),
                Arg::I64(grid.own.0),
                Arg::I64(grid.own.1),
                Arg::I64(grid.own.2),
                gpu.a(rows),
                k.into(),
                nside.into(),
                (order as i32).into(),
                (g as i32).into(),
                (p as i32).into(),
                (match gauge {
                    Gauge::Cosmo => 0,
                    Gauge::Phi => 1,
                })
                .into(),
                gpu.a(&stencil),
                (p_ref as i32).into(),
                gpu.a(&pos),
                gpu.a(&w),
                comp.as_ref().map_or(NULL, |c| gpu.a(c)),
            ],
        );
        // Transpose: count per grid cell, offsets on the host, fill, sort.
        let count = gpu.zeros::<u32>(grid.n);
        let stride_grid = |n: usize| (n.div_ceil(256).min(1 << 16) as u32, 1u32);
        gpu.launch("geom_count", stride_grid(n_entries), 256, &[gpu.a(&pos), Arg::I64(n_entries as i64), gpu.a(&count)]);
        let counts = gpu.download(&count);
        let mut dest = Vec::new();
        let mut offsets = vec![0u32];
        let mut cursor = vec![0u32; grid.n];
        let mut total = 0u32;
        for (j, &c) in counts.iter().enumerate() {
            if c > 0 {
                cursor[j] = total;
                dest.push(j as u32);
                total += c;
                offsets.push(total);
            }
        }
        assert_eq!(total as usize, n_entries);
        let mut cursor_d = count;
        gpu.copy_in(&cursor, &mut cursor_d);
        let idx = gpu.zeros::<u32>(n_entries);
        gpu.launch("geom_fill", stride_grid(n_entries), 256, &[gpu.a(&pos), Arg::I64(n_entries as i64), gpu.a(&cursor_d), gpu.a(&idx)]);
        let n_dest = dest.len();
        let offsets = gpu.upload(&offsets);
        gpu.launch("seg_sort", (n_dest.clamp(1, 1 << 20) as u32, 1), 256, &[gpu.a(&offsets), Arg::I64(n_dest as i64), gpu.a(&idx)]);
        let dest = gpu.upload(&dest);
        DeviceGeometry { k, g, p, pos, w, complete: comp, dest, offsets, idx, n_dest }
    }

    pub fn bytes(&self) -> usize {
        self.pos.len() * 4 + self.w.len() * 8 + self.idx.len() * 4 + self.dest.len() * 4 + self.offsets.len() * 4
    }

    /// Bytes per output row (`G` gauges, `P` taps).
    pub fn bytes_per_row(g: usize, p: usize) -> usize {
        4 * g * p * 16 + 8
    }

    /// `out[((r G + g) 2 + o) out_stride + opos(i)]` for the rows `row0 + i`,
    /// `i < n`, from `x[r x_stride + .]`, `r < rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        ker: &CudaSlice<f64>,
        x: Arg,
        x_stride: usize,
        rows: usize,
        row0: usize,
        n: usize,
        opos: Arg,
        out: Arg,
        out_stride: usize,
    ) {
        assert!(row0 + n <= self.k);
        gpu.launch1(
            "conv_fwd",
            n * self.g * rows,
            &[
                gpu.a(&self.pos),
                gpu.a(&self.w),
                (self.g as i32).into(),
                self.k.into(),
                (self.p as i32).into(),
                gpu.a(ker),
                x,
                x_stride.into(),
                (rows as i32).into(),
                row0.into(),
                n.into(),
                opos,
                out_stride.into(),
                out,
            ],
        );
    }

    /// Adjoint of [`forward`](Self::forward) for the rows `row0 + i`, `i <
    /// n`: `out[r out_stride + dmap(j)] (+)= sum` over the entries reading
    /// grid cell `j`.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint(
        &self,
        gpu: &Gpu,
        ker: &CudaSlice<f64>,
        gin: Arg,
        gin_stride: usize,
        rows: usize,
        row0: usize,
        n: usize,
        dmap: Arg,
        out: Arg,
        out_stride: usize,
        accumulate: bool,
    ) {
        gpu.launch1(
            "conv_adj",
            self.n_dest * rows,
            &[
                gpu.a(&self.dest),
                gpu.a(&self.offsets),
                self.n_dest.into(),
                gpu.a(&self.idx),
                gpu.a(&self.w),
                (self.g as i32).into(),
                self.k.into(),
                (self.p as i32).into(),
                gpu.a(ker),
                gin,
                gin_stride.into(),
                (rows as i32).into(),
                row0.into(),
                n.into(),
                dmap,
                out,
                out_stride.into(),
                (accumulate as i32).into(),
            ],
        );
    }
}
