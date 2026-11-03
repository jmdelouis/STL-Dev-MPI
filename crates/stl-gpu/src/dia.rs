//! DIA form of the convolution on the device (port of `stl_core::dia`).
//!
//! The cell a row reads on diagonal `d` is found from the row's face
//! coordinates and the offset `off_d` (`ext_pix`), so no position is
//! stored; the kernel and the bound bilinear weights are folded into
//! per-row coefficients, computed dense on all `D` diagonals (a transient
//! of `2 G D` doubles per row) and kept for the diagonals the row uses
//! (that receive a term; ~44 of 85 at `L = 4`): a mask per row (`3 x u32`) and
//! `coef[((s * K + k) * G + g) * 2 + o]` for its `s`-th used diagonal (the
//! gauges and channels of a row together, rows next, `S` slots: the most
//! diagonals a row of the block uses).  The adjoint gathers, for each grid
//! cell, the rows at the opposite offsets: no transposition, no sort; the
//! cotangents are first copied row by row (`dia_gin_t`, a transient of the
//! size of the cotangents) so that a row's values are contiguous.  What is
//! kept is the coefficients, the masks (12 bytes per row) and the list of
//! the cells the rows reach (`dest`, 4 bytes per cell); [`DiaGeometry::
//! strip`] keeps the masks and cells, from which [`DiaGeometry::refresh`]
//! accumulates the coefficients directly into their slots.
//!
//! Summation orders: the coefficient of a diagonal adds the taps then the
//! bilinear neighbours reading its cell; the forward adds the diagonals in
//! increasing order (as `stl_core::dia::DiaGeometry`); the adjoint sums,
//! per cell, each row of the window `[-W, W]^2` over gauges and channels,
//! then the rows by [`lanes`] lane-strided partial sums combined by a fixed
//! shuffle tree (the CPU adds them in window order, as one lane does).

use crate::device::{Arg, Gpu, NULL};
use crate::geometry::{Grid, Spec};
use cudarc::driver::CudaSlice;
use stl_core::dia::Diagonals;
use stl_core::healpix::order_of;
use stl_core::stencil::{local_kernel_grid, Gauge};

/// Input rows and gauges per thread (`RC`, `GB` in kernels.cu).
const RC: usize = 4;
const GB: usize = 4;

/// Lanes per cell of the adjoint (`STL_DIA_LANES`, a power of two <= 32).
pub fn lanes() -> usize {
    let l = std::env::var("STL_DIA_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize);
    assert!(l.is_power_of_two() && l <= 32, "STL_DIA_LANES must be a power of two <= 32");
    l
}

/// The diagonals on the device: offsets `(dx, dy)` and the index of the
/// window `[-w, w]^2`.
pub struct DeviceDiagonals {
    /// Half-width `W` of the window `[-W, W]^2`.
    pub w: i64,
    /// Number of diagonals `D`.
    pub d: usize,
    offsets: CudaSlice<i32>,
    index: CudaSlice<i32>,
}

impl DeviceDiagonals {
    /// Uploads the offsets and the window index of `diag`.
    pub fn new(gpu: &Gpu, diag: &Diagonals) -> Self {
        let offsets: Vec<i32> = diag.offsets.iter().flat_map(|&(dx, dy)| [dx as i32, dy as i32]).collect();
        DeviceDiagonals { w: diag.w, d: diag.len(), offsets: gpu.upload(&offsets), index: gpu.upload(&diag.index) }
    }
}

/// The block's own rows among the (sorted) geometry rows: `rows[f0 + i] =
/// start + i` for `i < len`, found without a search.
#[derive(Clone, Copy, Debug)]
pub struct OwnRows {
    /// First id of the run.
    pub start: i64,
    /// Length of the run (0 if none).
    pub len: i64,
    /// Position of `start` among the rows.
    pub f0: i64,
}

impl OwnRows {
    /// The longest run of consecutive ids in `rows` (sorted).
    pub fn longest_run(rows: &[i64]) -> Self {
        let (mut best, mut i) = (OwnRows { start: 0, len: 0, f0: 0 }, 0);
        while i < rows.len() {
            let mut j = i + 1;
            while j < rows.len() && rows[j] == rows[j - 1] + 1 {
                j += 1;
            }
            if (j - i) as i64 > best.len {
                best = OwnRows { start: rows[i], len: (j - i) as i64, f0: i as i64 };
            }
            i = j;
        }
        best
    }
}

/// The DIA coefficients of `K` output rows on a `Grid`, their masks of
/// used diagonals and the grid cells they reach.
pub struct DiaGeometry {
    /// Number of output rows `K`.
    pub k: usize,
    /// Number of gauges `G`.
    pub g: usize,
    /// Number of diagonals `D`.
    pub d: usize,
    nside: i64,
    order: u32,
    w: i64,
    /// `[S, K, G, 2]`, slot `s` of row `k` its `s`-th used diagonal
    /// (`None` once stripped).
    pub coef: Option<CudaSlice<f64>>,
    /// `[K, 3]`: bit `b` of word `w` for diagonal `32 w + b`.
    pub mask: CudaSlice<u32>,
    /// `S`, the most diagonals a row uses.
    pub slots: usize,
    /// Grid positions reached by the rows' diagonals (increasing).
    pub dest: CudaSlice<u32>,
    /// Number of grid cells reached (`|dest|`).
    pub n_dest: usize,
    own: OwnRows,
    /// Nonzero reads off the diagonals at the last build (must be 0).
    pub misses: u32,
}

/// Packed coefficients of the `k` rows, with their masks and slot count
/// (computed through the dense coefficients, or `packed` as in a previous
/// build), and the count of reads off the diagonals.
#[allow(clippy::too_many_arguments)]
fn coefficients(
    gpu: &Gpu,
    grid: &Grid,
    rows: &CudaSlice<i64>,
    k: usize,
    s: &Spec,
    diag: &DeviceDiagonals,
    ker: &CudaSlice<f64>,
    packed: Option<(&CudaSlice<u32>, usize)>,
) -> (CudaSlice<f64>, Option<(CudaSlice<u32>, usize)>, u32) {
    assert!(diag.d <= 96, "at most 96 diagonals (row masks of 3 words)");
    let lap = crate::geometry::lapper(gpu);
    let order = order_of(s.nside);
    let p = s.kernel_sz * s.kernel_sz;
    let stencil: Vec<f64> = local_kernel_grid(s.kernel_sz, s.nside).into_iter().flatten().collect();
    let stencil = gpu.upload(&stencil);
    let p_ref = (s.kernel_sz / 2) * (s.kernel_sz + 1);
    let n_rings = 4 * s.nside as usize;
    let thetas = gpu.zeros::<f64>(n_rings);
    gpu.launch1("ring_thetas", n_rings, &[s.nside.into(), gpu.a(&thetas)]);
    let out = gpu.zeros::<f64>(packed.map_or(diag.d, |p| p.1) * s.g * 2 * k);
    let mask = gpu.zeros::<u32>(if packed.is_some() { 0 } else { 3 * k });
    let errors = gpu.zeros::<u32>(1);
    gpu.launch1(
        "dia_build",
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
            gpu.a(ker),
            gpu.a(&diag.index),
            (diag.w as i32).into(),
            (diag.d as i32).into(),
            gpu.a(&out),
            if packed.is_some() { NULL } else { gpu.a(&mask) },
            packed.map_or(NULL, |p| gpu.a(p.0)),
            gpu.a(&errors),
        ],
    );
    let misses = gpu.download(&errors)[0];
    if packed.is_some() {
        lap("DIA packed coefficients");
        return (out, None, misses);
    }
    let dense = out;
    lap("DIA dense coefficients");
    let slots = gpu.zeros::<u32>(1);
    gpu.launch1("dia_slots", k, &[gpu.a(&mask), k.into(), gpu.a(&slots)]);
    let n_slots = gpu.download(&slots)[0] as usize;
    let coef = gpu.zeros::<f64>(n_slots * k * s.g * 2);
    gpu.launch1(
        "dia_pack",
        k,
        &[gpu.a(&dense), k.into(), (s.g as i32).into(), (diag.d as i32).into(), gpu.a(&mask), gpu.a(&coef)],
    );
    lap("DIA packed");
    (coef, Some((mask, n_slots)), misses)
}

impl DiaGeometry {
    /// Coefficients of the output rows `rows` (`k` sorted cells at
    /// `nside`, among them the block's own rows `own`) on `grid`, with the
    /// kernel `ker` (`[G, 1, 2, P]`) folded in, and the cells they reach.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &Gpu,
        grid: &Grid,
        rows: &CudaSlice<i64>,
        k: usize,
        s: &Spec,
        diag: &DeviceDiagonals,
        ker: &CudaSlice<f64>,
        own: OwnRows,
    ) -> Self {
        let order = order_of(s.nside);
        assert!(s.nside > 2 * diag.w, "DIA needs nside > 2 W (the window stays within the neighbour faces)");
        let flag = gpu.zeros::<u8>(grid.n);
        gpu.launch1(
            "dia_reach",
            k * diag.d,
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
                gpu.a(&diag.offsets),
                (diag.d as i32).into(),
                gpu.a(&flag),
            ],
        );
        // Compacted on the device (in increasing order): only the chunk
        // counts cross to the host.
        let n_chunks = grid.n.div_ceil(1024);
        let counts = gpu.zeros::<u32>(n_chunks);
        gpu.launch("flag_count", (n_chunks as u32, 1), 256, &[gpu.a(&flag), Arg::I64(grid.n as i64), gpu.a(&counts)]);
        let mut offsets = gpu.download(&counts);
        let mut total = 0u32;
        for c in offsets.iter_mut() {
            (*c, total) = (total, total + *c);
        }
        let offsets = gpu.upload(&offsets);
        let dest = gpu.zeros::<u32>(total as usize);
        gpu.launch("flag_fill", (n_chunks as u32, 1), 256, &[gpu.a(&flag), Arg::I64(grid.n as i64), gpu.a(&offsets), gpu.a(&dest)]);
        drop(flag);
        crate::geometry::lapper(gpu)("DIA cells reached");
        let (coef, mask, misses) = coefficients(gpu, grid, rows, k, s, diag, ker, None);
        let (mask, slots) = mask.unwrap();
        DiaGeometry {
            k,
            g: s.g,
            d: diag.d,
            nside: s.nside,
            order,
            w: diag.w,
            coef: Some(coef),
            mask,
            slots,
            n_dest: total as usize,
            dest,
            own,
            misses,
        }
    }

    /// Drops the coefficients, keeps the masks and the cells reached.
    pub fn strip(&mut self) {
        self.coef = None;
    }

    /// Whether the coefficients were dropped by [`strip`](Self::strip).
    pub fn is_stripped(&self) -> bool {
        self.coef.is_none()
    }

    /// Recomputes the coefficients (the same values).
    #[allow(clippy::too_many_arguments)]
    pub fn refresh(&mut self, gpu: &Gpu, grid: &Grid, rows: &CudaSlice<i64>, s: &Spec, diag: &DeviceDiagonals, ker: &CudaSlice<f64>) {
        if self.is_stripped() {
            let (coef, _, misses) = coefficients(gpu, grid, rows, self.k, s, diag, ker, Some((&self.mask, self.slots)));
            self.coef = Some(coef);
            self.misses = misses;
        }
    }

    /// Device bytes held now (coefficients if kept, plus masks and cells).
    pub fn bytes(&self) -> usize {
        self.coef.as_ref().map_or(0, |c| c.len() * 8) + self.stripped_bytes()
    }

    /// Bytes once stripped (masks and cells reached).
    pub fn stripped_bytes(&self) -> usize {
        self.mask.len() * 4 + self.n_dest * 4
    }

    /// Bytes per row once stripped: the mask and about one reached cell.
    pub const STRIPPED_BYTES_PER_ROW: usize = 16;

    /// Bytes per row of the dense coefficients of a first build (`G`
    /// gauges, `D` diagonals).
    pub fn dense_bytes_per_row(g: usize, d: usize) -> usize {
        g * 2 * d * 8
    }

    /// Bytes per row kept whole, for budgets: a row uses ~58 % of the
    /// diagonals at most (49 of 85 at `L = 4`, 45 of 77 at `L = 2`).
    pub fn bytes_per_row_estimate(g: usize, d: usize) -> usize {
        (d * 3).div_ceil(5) * g * 2 * 8 + Self::STRIPPED_BYTES_PER_ROW
    }

    /// The coefficients on all diagonals, `[D, K, G, 2]` (host, for tests).
    pub fn dense(&self, gpu: &Gpu) -> Vec<f64> {
        let (packed, mask) = (gpu.download(self.coef.as_ref().expect("stripped geometry")), gpu.download(&self.mask));
        let row = self.g * 2;
        let mut out = vec![0f64; self.d * self.k * row];
        for k in 0..self.k {
            let mut s = 0;
            for d in 0..self.d {
                if (mask[3 * k + d / 32] >> (d % 32)) & 1 != 0 {
                    out[(d * self.k + k) * row..][..row].copy_from_slice(&packed[(s * self.k + k) * row..][..row]);
                    s += 1;
                }
            }
        }
        out
    }

    /// `out[((r G + g) 2 + o) out_stride + opos(i)]` for the rows `row0 + i`,
    /// `i < n`, from `x[r x_stride + .]` on `grid`, `r < rows_in`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        grid: &Grid,
        rows: &CudaSlice<i64>,
        diag: &DeviceDiagonals,
        x: Arg,
        x_stride: usize,
        rows_in: usize,
        row0: usize,
        n: usize,
        opos: Arg,
        out: Arg,
        out_stride: usize,
    ) {
        self.forward_mapped(gpu, grid, rows, diag, x, NULL, x_stride, rows_in, row0, n, opos, out, out_stride);
    }

    /// [`forward`](Self::forward) reading grid cell `j` at `x[r x_stride +
    /// xmap[j]]` (`i32`; -1: skipped), e.g. a field held on a block's
    /// convolution rows only.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mapped(
        &self,
        gpu: &Gpu,
        grid: &Grid,
        rows: &CudaSlice<i64>,
        diag: &DeviceDiagonals,
        x: Arg,
        xmap: Arg,
        x_stride: usize,
        rows_in: usize,
        row0: usize,
        n: usize,
        opos: Arg,
        out: Arg,
        out_stride: usize,
    ) {
        assert!(row0 + n <= self.k);
        gpu.launch1(
            "dia_fwd",
            n * self.g.div_ceil(GB) * rows_in.div_ceil(RC),
            &[
                gpu.a(self.coef.as_ref().expect("stripped geometry")),
                gpu.a(&self.mask),
                (self.g as i32).into(),
                self.k.into(),
                gpu.a(&diag.offsets),
                gpu.a(rows),
                self.nside.into(),
                (self.order as i32).into(),
                gpu.a(&grid.ids),
                Arg::I64(grid.n as i64),
                Arg::I64(grid.own.0),
                Arg::I64(grid.own.1),
                Arg::I64(grid.own.2),
                x,
                xmap,
                x_stride.into(),
                (rows_in as i32).into(),
                row0.into(),
                n.into(),
                opos,
                out_stride.into(),
                out,
            ],
        );
    }

    /// Adjoint of [`forward`](Self::forward) for the rows `row0 + i`, `i <
    /// n`: `out[r out_stride + dmap(j)] (+)= sum` over the rows reading
    /// grid cell `j`.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint(
        &self,
        gpu: &Gpu,
        grid: &Grid,
        rows: &CudaSlice<i64>,
        diag: &DeviceDiagonals,
        gin: Arg,
        gin_stride: usize,
        rows_in: usize,
        row0: usize,
        n: usize,
        dmap: Arg,
        out: Arg,
        out_stride: usize,
        accumulate: bool,
    ) {
        let rg2 = rows_in * self.g * 2;
        let gt = gpu.zeros::<f64>(n * rg2);
        gpu.launch1("dia_gin_t", n * rg2, &[gin, gin_stride.into(), (rg2 as i32).into(), n.into(), gpu.a(&gt)]);
        let lpc = lanes();
        gpu.launch1(
            "dia_adj",
            lpc * self.n_dest * rows_in.div_ceil(RC),
            &[
                gpu.a(&self.dest),
                self.n_dest.into(),
                gpu.a(&grid.ids),
                gpu.a(self.coef.as_ref().expect("stripped geometry")),
                gpu.a(&self.mask),
                (self.g as i32).into(),
                self.k.into(),
                gpu.a(&diag.index),
                (self.w as i32).into(),
                gpu.a(rows),
                Arg::I64(self.own.start),
                Arg::I64(self.own.len),
                Arg::I64(self.own.f0),
                self.nside.into(),
                (self.order as i32).into(),
                gpu.a(&gt),
                (rows_in as i32).into(),
                row0.into(),
                n.into(),
                dmap,
                out,
                out_stride.into(),
                (accumulate as i32).into(),
                (lpc as i32).into(),
            ],
        );
    }
}
