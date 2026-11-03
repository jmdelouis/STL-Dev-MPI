//! Convolution geometry on the device (port of `stl_core::conv::ConvGeometry`):
//! rotated stencils, bilinear interpolation and binding computed per output
//! row in a kernel, then transposed (entries grouped by the grid cell they
//! read, in a fixed order) so that the adjoint is a deterministic gather.
//!
//! Entry `((g * 4 + n) * P + p) * K + k` for gauge `g`, bilinear neighbour
//! `n`, tap `p` and output row `k` (rows fastest: coalesced reads).  The
//! transposed groups list the entries of a cell by gauge, neighbour, row,
//! tap (the scatter order of `conv::convolution_adjoint`), each as `k << 11
//! | (g * 4 + n) << 6 | p` with its weight.
//!
//! Memory per output row: `4 G P` entries of 24 bytes (position and weight
//! by row, packed entry and weight by cell).  The transposition alone (4
//! bytes per entry) can be kept while the weights are dropped and
//! recomputed ([`DeviceGeometry::strip`], [`DeviceGeometry::refresh`]):
//! the stencils kernel is a fraction of the cost of the transposition.

use crate::device::{Arg, Gpu, NULL};
use cudarc::driver::CudaSlice;
use stl_core::healpix::order_of;
use stl_core::stencil::{local_kernel_grid, Gauge};

/// Input rows per thread of the convolution kernels (`RC` in kernels.cu).
const RC: usize = 4;

/// Largest number of output rows of one geometry (packed entry format).
pub const MAX_ROWS: usize = 1 << 21;

/// A sorted set of cells read by the convolution, on the device.  Cells of
/// the contiguous range `[own.0, own.1)` sit at positions `own.2 + (id -
/// own.0)` (no search needed).
pub struct Grid {
    /// NESTED ids of the cells, sorted and unique.
    pub ids: CudaSlice<i64>,
    /// Number of cells.
    pub n: usize,
    /// The contiguous range `[own.0, own.1)` and the position `own.2` of its
    /// first cell (`(0, 0, 0)` if none).
    pub own: (i64, i64, i64),
}

impl Grid {
    /// Uploads the sorted, unique `ids`; `own`, if given, must be a range of
    /// consecutive ids all present in `ids`.
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

/// What a geometry is computed from.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Resolution of the level.
    pub nside: i64,
    /// Side of the stencil (`kernel_size`); `P = kernel_sz^2` taps.
    pub kernel_sz: usize,
    /// Number of gauges `G`.
    pub g: usize,
    /// Gauge of the stencil rotation.
    pub gauge: Gauge,
}

/// The convolution geometry of `K` output rows on a [`Grid`]: positions and
/// weights by row, and their transposition by grid cell for the adjoint.
pub struct DeviceGeometry {
    /// Number of output rows `K`.
    pub k: usize,
    /// Number of gauges `G`.
    pub g: usize,
    /// Number of taps `P`.
    pub p: usize,
    /// Positions and weights by row (`None` once stripped).
    pub pos: Option<CudaSlice<u32>>,
    /// Bilinear weights, same layout as `pos`.
    pub w: Option<CudaSlice<f64>>,
    /// `complete[k * G + g]` when requested.
    pub complete: Option<CudaSlice<u8>>,
    /// Transposed: grid positions read (`dest`), their entry ranges
    /// (`offsets`, `|dest| + 1`), the packed entries (`kgp`) and their
    /// weights (`wt`, `None` once stripped).
    pub dest: CudaSlice<u32>,
    /// Entry ranges of the cells of `dest` (`|dest| + 1`).
    pub offsets: CudaSlice<u32>,
    /// Packed entries `k << 11 | (g * 4 + n) << 6 | p`, grouped by cell.
    pub kgp: CudaSlice<u32>,
    /// Weights of the packed entries (`None` once stripped).
    pub wt: Option<CudaSlice<f64>>,
    /// Number of grid cells read (`|dest|`).
    pub n_dest: usize,
}

/// With `STL_GPU_PROFILE` set, prints the time since its creation (after a
/// synchronisation) at every call.
pub(crate) fn lapper(gpu: &Gpu) -> impl Fn(&str) + '_ {
    let profile = std::env::var_os("STL_GPU_PROFILE").is_some();
    let started = std::time::Instant::now();
    move |what: &str| {
        if profile {
            gpu.sync();
            eprintln!("  geometry {what}: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
    }
}

fn stride_grid(n: usize) -> (u32, u32) {
    (n.div_ceil(256).clamp(1, 1 << 16) as u32, 1u32)
}

/// Positions and weights by row of the output rows `rows` (`k` cells) on `grid`.
fn stencils(gpu: &Gpu, grid: &Grid, rows: &CudaSlice<i64>, k: usize, s: &Spec, complete: bool) -> (CudaSlice<u32>, CudaSlice<f64>, Option<CudaSlice<u8>>) {
    let order = order_of(s.nside);
    let p = s.kernel_sz * s.kernel_sz;
    let n_entries = 4 * s.g * k * p;
    let stencil: Vec<f64> = local_kernel_grid(s.kernel_sz, s.nside).into_iter().flatten().collect();
    let stencil = gpu.upload(&stencil);
    let pos = gpu.zeros::<u32>(n_entries);
    let w = gpu.zeros::<f64>(n_entries);
    let comp = complete.then(|| gpu.zeros::<u8>(k * s.g));
    let p_ref = (s.kernel_sz / 2) * (s.kernel_sz + 1);
    let n_rings = 4 * s.nside as usize;
    let thetas = gpu.zeros::<f64>(n_rings);
    gpu.launch1("ring_thetas", n_rings, &[s.nside.into(), gpu.a(&thetas)]);
    gpu.launch1(
        "geom_build",
        k * s.g,
        &[
            gpu.a(&grid.ids),
            Arg::I64(grid.n as i64),
            Arg::I64(grid.own.0),
            Arg::I64(grid.own.1),
            Arg::I64(grid.own.2),
            gpu.a(rows),
            k.into(),
            s.nside.into(),
            (order as i32).into(),
            (s.g as i32).into(),
            (p as i32).into(),
            (match s.gauge {
                Gauge::Cosmo => 0,
                Gauge::Phi => 1,
            })
            .into(),
            gpu.a(&stencil),
            (p_ref as i32).into(),
            gpu.a(&thetas),
            gpu.a(&pos),
            gpu.a(&w),
            comp.as_ref().map_or(NULL, |c| gpu.a(c)),
        ],
    );
    (pos, w, comp)
}

impl DeviceGeometry {
    /// Bound stencils of the output rows `rows` (`k` cells at `nside`) on
    /// `grid`, and their transposition.
    pub fn build(gpu: &Gpu, grid: &Grid, rows: &CudaSlice<i64>, k: usize, s: &Spec, complete: bool) -> Self {
        let p = s.kernel_sz * s.kernel_sz;
        let n_entries = 4 * s.g * k * p;
        assert!(n_entries < u32::MAX as usize, "block too large for u32 entry indices");
        assert!(k <= MAX_ROWS && s.g <= 8 && p <= 64, "block, gauges or taps beyond the packed entry format");
        let lap = lapper(gpu);
        let (pos, w, comp) = stencils(gpu, grid, rows, k, s, complete);
        lap("stencils");
        // Transpose: count per grid cell, offsets on the host, fill, sort.
        let count = gpu.zeros::<u32>(grid.n);
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
        drop(cursor_d);
        let n_dest = dest.len();
        let offsets = gpu.upload(&offsets);
        lap("count + fill");
        gpu.launch(
            "seg_sort",
            (n_dest.clamp(1, 1 << 20) as u32, 1),
            256,
            &[gpu.a(&offsets), Arg::I64(n_dest as i64), gpu.a(&idx), k.into(), (p as i32).into()],
        );
        lap("sort");
        let kgp = gpu.zeros::<u32>(n_entries);
        let wt = gpu.zeros::<f64>(n_entries);
        gpu.launch(
            "geom_pack",
            stride_grid(n_entries),
            256,
            &[gpu.a(&idx), Arg::I64(n_entries as i64), gpu.a(&w), k.into(), (p as i32).into(), gpu.a(&kgp), gpu.a(&wt)],
        );
        drop(idx);
        lap("pack");
        let dest = gpu.upload(&dest);
        DeviceGeometry { k, g: s.g, p, pos: Some(pos), w: Some(w), complete: comp, dest, offsets, kgp, wt: Some(wt), n_dest }
    }

    /// Drops the weights and positions, keeps the transposition.
    pub fn strip(&mut self) {
        self.pos = None;
        self.w = None;
        self.wt = None;
    }

    /// Whether the weights were dropped by [`strip`](Self::strip).
    pub fn is_stripped(&self) -> bool {
        self.w.is_none()
    }

    /// Recomputes what [`strip`](Self::strip) dropped (the same values).
    pub fn refresh(&mut self, gpu: &Gpu, grid: &Grid, rows: &CudaSlice<i64>, s: &Spec) {
        if !self.is_stripped() {
            return;
        }
        let lap = lapper(gpu);
        let (pos, w, _) = stencils(gpu, grid, rows, self.k, s, false);
        lap("stencils (refresh)");
        let n_entries = self.kgp.len();
        let wt = gpu.zeros::<f64>(n_entries);
        gpu.launch(
            "geom_wt",
            stride_grid(n_entries),
            256,
            &[gpu.a(&self.kgp), Arg::I64(n_entries as i64), gpu.a(&w), self.k.into(), (self.p as i32).into(), gpu.a(&wt)],
        );
        lap("weights by cell (refresh)");
        self.pos = Some(pos);
        self.w = Some(w);
        self.wt = Some(wt);
    }

    /// Device bytes held now (weights and positions if kept, plus the transposition).
    pub fn bytes(&self) -> usize {
        let n = self.kgp.len();
        let weights = if self.is_stripped() { 0 } else { n * 20 };
        weights + Self::stripped_bytes(self)
    }

    /// Bytes once stripped (the transposition).
    pub fn stripped_bytes(&self) -> usize {
        self.kgp.len() * 4 + self.dest.len() * 4 + self.offsets.len() * 4
    }

    /// Bytes per output row (`G` gauges, `P` taps): everything, or the
    /// transposition only.
    pub fn bytes_per_row(g: usize, p: usize) -> usize {
        4 * g * p * 24 + 8
    }

    /// Bytes per output row of the transposition alone (`G` gauges, `P` taps).
    pub fn transposition_bytes_per_row(g: usize, p: usize) -> usize {
        4 * g * p * 4 + 8
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
        self.forward_mapped(gpu, ker, x, NULL, x_stride, rows, row0, n, opos, out, out_stride);
    }

    /// [`forward`](Self::forward) reading grid cell `j` at `x[r x_stride +
    /// xmap[j]]` (`i32`; -1: a zero input), e.g. a field held on a block's
    /// convolution rows only.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mapped(
        &self,
        gpu: &Gpu,
        ker: &CudaSlice<f64>,
        x: Arg,
        xmap: Arg,
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
            n * self.g * rows.div_ceil(RC),
            &[
                gpu.a(self.pos.as_ref().expect("stripped geometry")),
                gpu.a(self.w.as_ref().expect("stripped geometry")),
                (self.g as i32).into(),
                self.k.into(),
                (self.p as i32).into(),
                gpu.a(ker),
                x,
                xmap,
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
            32 * self.n_dest * rows.div_ceil(RC),
            &[
                gpu.a(&self.dest),
                gpu.a(&self.offsets),
                self.n_dest.into(),
                gpu.a(&self.kgp),
                gpu.a(self.wt.as_ref().expect("stripped geometry")),
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
