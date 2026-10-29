//! DIA form of the convolution on the device (port of `stl_core::dia`).
//!
//! The cell a row reads on diagonal `d` is found from the row's face
//! coordinates and the offset `off_d` (`ext_pix`), so no position is
//! stored; the kernel and the bound bilinear weights are folded into
//! coefficients `coef[((g * 2 + o) * D + d) * K + k]` (rows fastest).  The
//! adjoint gathers, for each grid cell, the rows at the opposite offsets:
//! no transposition, no sort.  What is kept is the coefficients (`2 G D`
//! doubles per row) and the list of the cells the rows reach (`dest`, 4
//! bytes per cell), which [`DiaGeometry::strip`] keeps alone.
//!
//! Summation orders: the coefficient of a diagonal adds the taps then the
//! bilinear neighbours reading its cell; the forward adds the diagonals in
//! increasing order; the adjoint adds, per cell, the rows of the window
//! `[-W, W]^2` (`vy` then `vx` increasing), each summed over gauges and
//! channels first.  The same orders as `stl_core::dia::DiaGeometry`.

use crate::device::{Arg, Gpu};
use crate::geometry::{Grid, Spec};
use cudarc::driver::CudaSlice;
use stl_core::dia::Diagonals;
use stl_core::healpix::order_of;
use stl_core::stencil::{local_kernel_grid, Gauge};

/// Input rows and gauges per thread (`RC`, `GB` in kernels.cu).
const RC: usize = 4;
const GB: usize = 4;

/// The diagonals on the device: offsets `(dx, dy)` and the index of the
/// window `[-w, w]^2`.
pub struct DeviceDiagonals {
    pub w: i64,
    pub d: usize,
    offsets: CudaSlice<i32>,
    index: CudaSlice<i32>,
}

impl DeviceDiagonals {
    pub fn new(gpu: &Gpu, diag: &Diagonals) -> Self {
        let offsets: Vec<i32> = diag.offsets.iter().flat_map(|&(dx, dy)| [dx as i32, dy as i32]).collect();
        DeviceDiagonals { w: diag.w, d: diag.len(), offsets: gpu.upload(&offsets), index: gpu.upload(&diag.index) }
    }
}

/// The block's own rows among the (sorted) geometry rows: `rows[f0 + i] =
/// start + i` for `i < len`, found without a search.
#[derive(Clone, Copy, Debug)]
pub struct OwnRows {
    pub start: i64,
    pub len: i64,
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

pub struct DiaGeometry {
    pub k: usize,
    pub g: usize,
    pub d: usize,
    nside: i64,
    order: u32,
    w: i64,
    /// `None` once stripped.
    pub coef: Option<CudaSlice<f64>>,
    /// Grid positions reached by the rows' diagonals (increasing).
    pub dest: CudaSlice<u32>,
    pub n_dest: usize,
    own: OwnRows,
    /// Nonzero reads off the diagonals at the last build (must be 0).
    pub misses: u32,
}

/// Coefficients of the `k` rows (and the count of reads off the diagonals).
fn coefficients(
    gpu: &Gpu,
    grid: &Grid,
    rows: &CudaSlice<i64>,
    k: usize,
    s: &Spec,
    diag: &DeviceDiagonals,
    ker: &CudaSlice<f64>,
) -> (CudaSlice<f64>, u32) {
    let order = order_of(s.nside);
    let p = s.kernel_sz * s.kernel_sz;
    let stencil: Vec<f64> = local_kernel_grid(s.kernel_sz, s.nside).into_iter().flatten().collect();
    let stencil = gpu.upload(&stencil);
    let p_ref = (s.kernel_sz / 2) * (s.kernel_sz + 1);
    let n_rings = 4 * s.nside as usize;
    let thetas = gpu.zeros::<f64>(n_rings);
    gpu.launch1("ring_thetas", n_rings, &[s.nside.into(), gpu.a(&thetas)]);
    let coef = gpu.zeros::<f64>(s.g * 2 * diag.d * k);
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
            gpu.a(&coef),
            gpu.a(&errors),
        ],
    );
    let misses = gpu.download(&errors)[0];
    (coef, misses)
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
        let dest: Vec<u32> = gpu.download(&flag).iter().enumerate().filter(|(_, &f)| f != 0).map(|(j, _)| j as u32).collect();
        drop(flag);
        let (coef, misses) = coefficients(gpu, grid, rows, k, s, diag, ker);
        DiaGeometry {
            k,
            g: s.g,
            d: diag.d,
            nside: s.nside,
            order,
            w: diag.w,
            coef: Some(coef),
            n_dest: dest.len(),
            dest: gpu.upload(&dest),
            own,
            misses,
        }
    }

    /// Drops the coefficients, keeps the cells reached.
    pub fn strip(&mut self) {
        self.coef = None;
    }

    pub fn is_stripped(&self) -> bool {
        self.coef.is_none()
    }

    /// Recomputes the coefficients (the same values).
    #[allow(clippy::too_many_arguments)]
    pub fn refresh(&mut self, gpu: &Gpu, grid: &Grid, rows: &CudaSlice<i64>, s: &Spec, diag: &DeviceDiagonals, ker: &CudaSlice<f64>) {
        if self.is_stripped() {
            let (coef, misses) = coefficients(gpu, grid, rows, self.k, s, diag, ker);
            self.coef = Some(coef);
            self.misses = misses;
        }
    }

    pub fn bytes(&self) -> usize {
        self.coef.as_ref().map_or(0, |c| c.len() * 8) + self.n_dest * 4
    }

    /// Coefficient bytes per output row (`G` gauges, `D` diagonals).
    pub fn bytes_per_row(g: usize, d: usize) -> usize {
        g * 2 * d * 8
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
        assert!(row0 + n <= self.k);
        gpu.launch1(
            "dia_fwd",
            n * self.g.div_ceil(GB) * rows_in.div_ceil(RC),
            &[
                gpu.a(self.coef.as_ref().expect("stripped geometry")),
                (self.g as i32).into(),
                self.k.into(),
                (self.d as i32).into(),
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
        gpu.launch1(
            "dia_adj",
            self.n_dest * rows_in.div_ceil(RC),
            &[
                gpu.a(&self.dest),
                self.n_dest.into(),
                gpu.a(&grid.ids),
                gpu.a(self.coef.as_ref().expect("stripped geometry")),
                (self.g as i32).into(),
                self.k.into(),
                (self.d as i32).into(),
                gpu.a(&diag.index),
                (self.w as i32).into(),
                gpu.a(rows),
                Arg::I64(self.own.start),
                Arg::I64(self.own.len),
                Arg::I64(self.own.f0),
                self.nside.into(),
                (self.order as i32).into(),
                gin,
                gin_stride.into(),
                (rows_in as i32).into(),
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
