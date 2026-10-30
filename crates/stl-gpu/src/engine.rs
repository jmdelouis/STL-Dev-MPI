//! The level sweeps of `stl_core::engine::LocalEngine` on one device.
//!
//! Every level's maps and stored moduli `m[d][n]` live on the device over
//! the rank's layout, and in the reverse sweep their cotangents and the map
//! cotangents.  The wavelet fields `z_n` and moduli `|z_n|` (on the
//! convolution rows of the blocks) are either kept from the forward sweep
//! for the reverse one ([`FieldStore::Keep`]) or, as in the CPU engine,
//! recomputed by every block of the reverse sweep from the maps
//! ([`FieldStore::Recompute`]: the forward sweep holds `|z_n|` of the level
//! being swept only, for its decimations); both give the same values.  A
//! level is processed by device blocks (contiguous owned cells, one
//! convolution geometry each).
//!
//! Decimations use the full-sky rows of the owned cells (`topology`):
//! `maps[d+1] = D maps[d]`, `m[d][n] = D m[d][n-1]` (`m[d][d+1] = D |z_d|`)
//! after a halo exchange of the finer field, the same values as the
//! CPU pieces (exact rows); the reverse chain of a level is accumulated by
//! Horner's scheme through the transposed rows down to the next coarser
//! level, the last transposed rows being applied by every block to its own
//! cells.  The other halo and ghost exchanges are those of the CPU engine.
//!
//! The convolution geometry of a block is either the masked DIA form
//! ([`ConvForm::Dia`], the default, `dia`) on the levels where it applies
//! (nside > 2 W), or the bound stencils with their transposition
//! (`geometry`): the same operator to rounding; the DIA takes a third of
//! the memory and no transposition.
//!
//! Summation orders: convolution and decimation sums in the CPU order;
//! coefficient partial sums by chunks of `CHUNK` pixels in a fixed tree,
//! chunks, blocks and levels in increasing order; cotangents of one target
//! accumulated in plan order.
//!
//! An engine evaluates any plan and batch on its decomposition: several
//! [`GpuEvaluator`]s (the target statistics, the synthesis, the statistics
//! of a transformed map) share one engine, hence its topology, decimation
//! rows, geometry cache and level fields.  A plan without S3 or S4 reads
//! neither `|z_n|` nor the stored moduli: they are neither computed nor
//! allocated for it.

use crate::device::{Arg, Gpu, NULL};
use crate::dia::{DeviceDiagonals, DiaGeometry, OwnRows};
use crate::geometry::{DeviceGeometry, Grid, Spec};
use crate::topology::{HostCsr, Topology};
use cudarc::driver::{CudaSlice, DeviceRepr};
use num_complex::Complex64;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use stl_core::engine::{EngineConfig, Evaluator, Exchange, Partition};
use stl_core::local::LocalGeometry;
use stl_core::plan::{Coefficient, Family};

/// Pixels per partial sum of a coefficient (then summed in chunk order).
const CHUNK: usize = 1 << 16;

/// What is kept of the convolution geometry of the blocks between sweeps
/// (it depends on the decomposition only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryCache {
    /// Everything when it fits, else the transposition (below).
    Auto,
    /// Positions, weights and transposition (24 bytes per entry), for the
    /// blocks that fit; the others keep their transposition if it fits.
    Full,
    /// The transposition only (4 bytes per entry; the weights are
    /// recomputed at every sweep), for the blocks that fit.
    Transposition,
    /// Nothing: everything recomputed at every sweep.
    None,
}

impl GeometryCache {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(GeometryCache::Auto),
            "full" | "yes" => Some(GeometryCache::Full),
            "transposition" => Some(GeometryCache::Transposition),
            "none" | "no" => Some(GeometryCache::None),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GeometryCache::Auto => "auto",
            GeometryCache::Full => "full",
            GeometryCache::Transposition => "transposition",
            GeometryCache::None => "none",
        }
    }
}

/// The form of the convolution geometry of the blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvForm {
    /// Bound stencils (positions and weights by row) and their
    /// transposition sorted by grid cell (`geometry`).
    Stencils,
    /// Masked DIA (`dia`, `docs/dia_convolution.md`) on the levels with
    /// nside > 2 W, the bound stencils on the coarser ones.  "Full" keeps
    /// the coefficients, "transposition" the row masks and the cells
    /// reached (16 bytes per row).
    Dia,
}

impl ConvForm {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "stencils" => Some(ConvForm::Stencils),
            "dia" => Some(ConvForm::Dia),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ConvForm::Stencils => "stencils",
            ConvForm::Dia => "dia",
        }
    }
}

/// What the forward sweep keeps of the wavelet fields for the reverse one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldStore {
    /// `Keep` when its level fields, with the decimation rows and the
    /// topology, take at most half of the free device memory, else
    /// `Recompute`.
    Auto,
    /// `z_n` and `|z_n|` of every level, `3 L` doubles per held cell and
    /// map.
    Keep,
    /// `z_n` and `|z_n|` of a block recomputed on its convolution rows in
    /// the reverse sweep (one more wavelet convolution of the maps per
    /// block); the forward sweep holds `|z_n|` of the level being swept.
    Recompute,
}

impl FieldStore {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(FieldStore::Auto),
            "keep" => Some(FieldStore::Keep),
            "recompute" => Some(FieldStore::Recompute),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FieldStore::Auto => "auto",
            FieldStore::Keep => "keep",
            FieldStore::Recompute => "recompute",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Owned cells per device block; 0: from the device memory.
    pub block_rows: usize,
    /// Form of the convolution geometry (`STL_GPU_CONV`: `dia`, the
    /// default, or `stencils`).
    pub conv: ConvForm,
    /// Wavelet fields kept or recomputed (`STL_GPU_FIELDS`: `auto`, the
    /// default, `keep` or `recompute`).
    pub fields: FieldStore,
    pub cache: GeometryCache,
    /// Bytes for the geometry cache; 0: from the device memory.
    pub cache_bytes: usize,
    /// Host threads for the setup.
    pub threads: usize,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            block_rows: 0,
            conv: std::env::var("STL_GPU_CONV")
                .ok()
                .map(|v| ConvForm::parse(&v).unwrap_or_else(|| panic!("STL_GPU_CONV: 'stencils' or 'dia', not {v:?}")))
                .unwrap_or(ConvForm::Dia),
            fields: std::env::var("STL_GPU_FIELDS")
                .ok()
                .map(|v| FieldStore::parse(&v).unwrap_or_else(|| panic!("STL_GPU_FIELDS: 'auto', 'keep' or 'recompute', not {v:?}")))
                .unwrap_or(FieldStore::Auto),
            cache: GeometryCache::Auto,
            cache_bytes: 0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }
}

/// The convolution geometry of one block.
enum BlockGeometry {
    Stencils(DeviceGeometry),
    Dia(DiaGeometry),
}

impl BlockGeometry {
    fn bytes(&self) -> usize {
        match self {
            BlockGeometry::Stencils(g) => g.bytes(),
            BlockGeometry::Dia(g) => g.bytes(),
        }
    }

    /// Bytes once stripped.
    fn stripped_bytes(&self) -> usize {
        match self {
            BlockGeometry::Stencils(g) => DeviceGeometry::stripped_bytes(g),
            BlockGeometry::Dia(g) => g.stripped_bytes(),
        }
    }

    fn strip(&mut self) {
        match self {
            BlockGeometry::Stencils(g) => g.strip(),
            BlockGeometry::Dia(g) => g.strip(),
        }
    }

    fn is_stripped(&self) -> bool {
        match self {
            BlockGeometry::Stencils(g) => g.is_stripped(),
            BlockGeometry::Dia(g) => g.is_stripped(),
        }
    }
}

/// Sparse rows on the device.
struct DevCsr {
    indptr: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    vals: CudaSlice<f64>,
    n_rows: usize,
}

impl DevCsr {
    fn new(gpu: &Gpu, h: &HostCsr) -> Self {
        DevCsr { indptr: gpu.upload(&h.indptr), cols: gpu.upload(&h.cols), vals: gpu.upload(&h.vals), n_rows: h.n_rows() }
    }

    fn bytes(&self) -> usize {
        self.indptr.len() * 4 + self.cols.len() * 4 + self.vals.len() * 8
    }

    /// `y[r y_stride + y0 + i] (+)= sum_q vals x[r x_stride + cols]`.
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, gpu: &Gpu, x: Arg, x_stride: usize, rows: usize, y: Arg, y_stride: usize, y0: usize, accumulate: bool) {
        self.apply_rows(gpu, 0, self.n_rows, x, x_stride, rows, y, y_stride, y0, accumulate);
    }

    /// [`apply`](Self::apply) for the sparse rows `row0 + i`, `i < n`, only
    /// (`y` at `y0 + i`).
    #[allow(clippy::too_many_arguments)]
    fn apply_rows(&self, gpu: &Gpu, row0: usize, n: usize, x: Arg, x_stride: usize, rows: usize, y: Arg, y_stride: usize, y0: usize, accumulate: bool) {
        assert!(row0 + n <= self.n_rows);
        gpu.launch1(
            "csr_apply",
            n * rows,
            &[
                gpu.at(&self.indptr, row0),
                gpu.a(&self.cols),
                gpu.a(&self.vals),
                n.into(),
                x,
                x_stride.into(),
                (rows as i32).into(),
                y,
                y_stride.into(),
                y0.into(),
                (accumulate as i32).into(),
            ],
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Operand {
    ptr: u64,
    im_off: i64,
}

const NONE: Operand = Operand { ptr: 0, im_off: -1 };

/// Where a block reads a field of its level: base address, row stride and
/// position of the block's first owned cell (a level field: held cells and
/// layout position; the block's own: convolution rows and `f0`).
#[derive(Clone, Copy, Debug)]
struct Field {
    ptr: u64,
    stride: usize,
    pos: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ForwardItem {
    kind: i32,
    pad: i32,
    a: Operand,
    b: Operand,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Contrib {
    kind: i32,
    pad: i32,
    a: Operand,
    b: Operand,
    g_re: f64,
    g_im: f64,
    n_total: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Target {
    ptr: u64,
    im_off: i64,
    start: u32,
    end: u32,
}

unsafe impl DeviceRepr for ForwardItem {}
unsafe impl DeviceRepr for Contrib {}
unsafe impl DeviceRepr for Target {}

const F_MEAN: i32 = 0;
const F_SQ: i32 = 1;
const F_ABS: i32 = 2;
const F_S1X: i32 = 3;
const F_PROD: i32 = 4;

const R_MEAN: i32 = 0;
const R_VAR: i32 = 1;
const R_S1SAME: i32 = 2;
const R_S2SAME: i32 = 3;
const R_S1X_A: i32 = 4;
const R_S1X_B: i32 = 5;
const R_GA: i32 = 6;
const R_GB: i32 = 7;

/// One process's sweeps on its device.
pub struct GpuEngine<'a> {
    pub gpu: Arc<Gpu>,
    pub geo: &'a LocalGeometry,
    pub cfg: &'a EngineConfig,
    pub part: Partition,
    pub topo: Topology,
    pub batch: usize,
    pub channels: usize,
    ker: CudaSlice<f64>,
    grids: Vec<Grid>,
    a1_ids: Vec<Vec<CudaSlice<i64>>>,
    a1_pos: Vec<Vec<CudaSlice<u32>>>,
    down: Vec<DevCsr>,
    up: Vec<DevCsr>,
    /// The diagonals of the DIA geometry (`ConvForm::Dia`).
    diag: Option<DeviceDiagonals>,
    cache: Vec<Vec<Option<BlockGeometry>>>,
    cache_mode: GeometryCache,
    /// Budget and use of the geometry cache, bytes.
    cache_bytes: usize,
    cached_bytes: usize,
    /// Part of the budget for whole geometries under `Transposition`.
    whole_budget: usize,
    whole_used: usize,
    dmap: CudaSlice<i32>,
    /// `maps[n]` `[B C, |layout n|]`.
    pub maps: Vec<CudaSlice<f64>>,
    /// `m[d][n]` `[B C L, |layout n|]` (allocated by the first plan with S3 or S4).
    pub m: BTreeMap<(usize, usize), CudaSlice<f64>>,
    /// Kept or recomputed `z_n` and `|z_n|` (`Keep` or `Recompute`).
    fields: FieldStore,
    /// Peak bytes of the level fields at the batch given to `new` (plans
    /// with S3 or S4).
    field_bytes: usize,
    /// `|z_n|` `[B C L, |layout n|]` (plans with S3 or S4; with
    /// `Recompute`, level `n` while the forward sweep is on it, empty
    /// otherwise) and `z_n` `[B C L, 2, |layout n|]` (`Keep` only) on the
    /// blocks' convolution rows (and halo for `|z_n|`).
    absz: Vec<CudaSlice<f64>>,
    z: Vec<CudaSlice<f64>>,
    pub timers: BTreeMap<&'static str, f64>,
    block_rows: usize,
    /// Largest device memory in use seen at the phase ends (whole device).
    peak_used: usize,
    /// The evaluator whose forward sweep the fields hold (`GpuEvaluator::id`).
    holder: usize,
}

impl<'a> GpuEngine<'a> {
    pub fn new(
        gpu: Arc<Gpu>,
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        part: Partition,
        batch: usize,
        channels: usize,
        opts: &GpuOptions,
    ) -> Self {
        let levels = cfg.levels();
        assert_eq!(part.layouts.len(), levels);
        let (l, p) = (geo.l, geo.kernel_sz * geo.kernel_sz);
        let bc = batch * channels;
        let held: Vec<usize> = part.layouts.iter().map(|x| x.len()).collect();
        let (free, _total) = gpu.memory();
        // Held cells of the levels with a decimation: about 30 + 7.5 nonzero
        // weights (12 bytes) per coarse / fine row, plus the block topology.
        let csr_bytes: usize =
            held[..levels - 1].iter().sum::<usize>() * 200 + held.iter().sum::<usize>() * 32 + (512 << 20);
        let fields = match opts.fields {
            FieldStore::Auto if Self::field_doubles(&held, bc, l, FieldStore::Keep) * 8 + csr_bytes <= free / 2 => FieldStore::Keep,
            FieldStore::Auto => FieldStore::Recompute,
            f => f,
        };
        let field_doubles = Self::field_doubles(&held, bc, l, fields);
        let fixed = field_doubles * 8 + csr_bytes;
        let avail = free.saturating_sub(fixed);
        let diag = (opts.conv == ConvForm::Dia).then(|| {
            let d = stl_core::dia::Diagonals::observed(&[8, 16, 32], geo.kernel_sz, l, geo.gauge);
            DeviceDiagonals::new(&gpu, &d)
        });
        // Transients of one block: its geometry while built (and the
        // transposition temporaries, or the dense DIA coefficients and the
        // row-ordered copy of a cotangent; the coarse levels that keep the
        // bound stencils are smaller than a block), second wavelets and
        // their cotangents, the cotangents of z_n, |z_n| and the chain, and
        // with `Recompute` the block's z_n and |z_n|.
        let geometry_transient = match &diag {
            None => DeviceGeometry::bytes_per_row(l, p) * 3 / 2,
            Some(dg) => DiaGeometry::dense_bytes_per_row(l, dg.d) + DiaGeometry::bytes_per_row_estimate(l, dg.d) + bc * l * l * 16,
        };
        let own_fields = if fields == FieldStore::Recompute { bc * l * 24 } else { 0 };
        let per_row = geometry_transient + levels * bc * l * l * 32 + bc * l * 56 + own_fields;
        // At most 2^20 owned cells: the convolution rows (with their halo)
        // must stay within geometry::MAX_ROWS (packed transposed entries).
        let max_rows = crate::geometry::MAX_ROWS / 2;
        let block_rows = if opts.block_rows > 0 {
            opts.block_rows.min(max_rows)
        } else {
            ((avail as f64 * 0.3 / per_row as f64) as usize).clamp(1024, max_rows)
        };
        let started = Instant::now();
        let topo = Topology::new(geo, &part, block_rows, opts.threads);
        let setup = started.elapsed().as_secs_f64();
        let ker = gpu.upload(&geo.kernel.w);
        let grids: Vec<Grid> = topo.levels.iter().map(|lv| Grid::new(&gpu, &lv.ids, Some(lv.owned))).collect();
        let a1_ids = topo.levels.iter().map(|lv| lv.blocks.iter().map(|b| gpu.upload(&b.a1)).collect()).collect();
        let a1_pos = topo.levels.iter().map(|lv| lv.blocks.iter().map(|b| gpu.upload(&b.a1pos)).collect()).collect();
        let down = topo.down.iter().map(|h| DevCsr::new(&gpu, h)).collect();
        let up = topo.up.iter().map(|h| DevCsr::new(&gpu, h)).collect();
        let largest = topo.levels.iter().flat_map(|lv| lv.blocks.iter().map(|b| b.a1.len())).max().unwrap_or(0);
        // Bytes of every block's geometry, whole and stripped.
        let (mut whole_total, mut stripped_total) = (0usize, 0usize);
        for lv in &topo.levels {
            let rows: usize = lv.blocks.iter().map(|b| b.a1.len()).sum();
            match &diag {
                Some(dg) if lv.nside > 2 * dg.w => {
                    whole_total += rows * DiaGeometry::bytes_per_row_estimate(l, dg.d);
                    stripped_total += rows * DiaGeometry::STRIPPED_BYTES_PER_ROW;
                }
                _ => {
                    whole_total += rows * DeviceGeometry::bytes_per_row(l, p);
                    stripped_total += rows * DeviceGeometry::transposition_bytes_per_row(l, p);
                }
            }
        }
        // Cache budget from what is free once the topology is on the device
        // (the level fields are allocated at the first sweeps).
        let cache_bytes = if opts.cache_bytes > 0 {
            opts.cache_bytes
        } else {
            let (free_now, _) = gpu.memory();
            free_now.saturating_sub(field_doubles * 8 + largest * per_row + (512 << 20)) * 4 / 5
        };
        let cache_mode = match opts.cache {
            GeometryCache::Auto if whole_total <= cache_bytes => GeometryCache::Full,
            GeometryCache::Auto => GeometryCache::Transposition,
            m => m,
        };
        let cache = topo.levels.iter().map(|lv| lv.blocks.iter().map(|_| None).collect()).collect();
        let dmap = gpu.zeros::<i32>(held.iter().copied().max().unwrap_or(0));
        let mut timers = BTreeMap::new();
        timers.insert("gpu_setup_topology", setup);
        GpuEngine {
            gpu,
            geo,
            cfg,
            part,
            topo,
            batch,
            channels,
            ker,
            grids,
            a1_ids,
            a1_pos,
            down,
            up,
            diag,
            cache,
            cache_mode,
            cache_bytes,
            cached_bytes: 0,
            whole_budget: cache_bytes.saturating_sub(stripped_total),
            whole_used: 0,
            dmap,
            maps: Vec::new(),
            m: BTreeMap::new(),
            fields,
            field_bytes: field_doubles * 8,
            absz: Vec::new(),
            z: Vec::new(),
            timers,
            block_rows,
            peak_used: 0,
            holder: 0,
        }
    }

    /// Evaluates `batch` maps from now on: the level fields are reallocated
    /// at the next sweep if the batch changes.  The block size and the
    /// cache budget stay those of the batch given to [`new`](Self::new),
    /// which should be the largest.
    pub fn set_batch(&mut self, batch: usize) {
        if batch != self.batch {
            self.batch = batch;
            self.maps.clear();
            self.z.clear();
            self.absz.clear();
            self.m.clear();
        }
    }

    /// Doubles of the level fields at their peak for `held` cells per
    /// level (`bc` maps and channels, `l` orientations), assuming S3 or S4.
    /// Between the sweeps: maps and stored moduli; in the reverse sweep
    /// besides, their cotangents and the chain on the two levels below the
    /// finest; `Keep` adds z and |z| on every level, `Recompute` |z_0| in
    /// the forward sweep.
    fn field_doubles(held: &[usize], bc: usize, l: usize, fields: FieldStore) -> usize {
        let kept: usize = held.iter().enumerate().map(|(n, &h)| h * (bc + n * bc * l)).sum();
        let reverse = 2 * kept + bc * l * held.iter().skip(1).take(2).sum::<usize>();
        match fields {
            FieldStore::Recompute => (kept + bc * l * held[0]).max(reverse),
            _ => reverse + held.iter().map(|&h| 3 * bc * l * h).sum::<usize>(),
        }
    }

    /// Kept or recomputed wavelet fields (`Keep` or `Recompute`).
    pub fn field_store(&self) -> FieldStore {
        self.fields
    }

    /// Peak bytes of the level fields (estimate, plans with S3 or S4) and
    /// bytes of the decimation rows and their transposes on the device.
    pub fn memory_parts(&self) -> (usize, usize) {
        (self.field_bytes, self.down.iter().chain(&self.up).map(DevCsr::bytes).sum())
    }

    /// Owned cells per device block.
    pub fn block_rows(&self) -> usize {
        self.block_rows
    }

    /// Peak and total device memory, bytes.
    pub fn device_memory(&self) -> (usize, usize) {
        let (free, total) = self.gpu.memory();
        (self.peak_used.max(total - free), total)
    }

    pub fn levels(&self) -> usize {
        self.cfg.levels()
    }

    fn held(&self, n: usize) -> usize {
        self.topo.levels[n].len()
    }

    /// Device blocks per level.
    pub fn blocks_per_level(&self) -> Vec<usize> {
        self.topo.levels.iter().map(|lv| lv.blocks.len()).collect()
    }

    /// The form of the convolution geometry (DIA on the levels with nside >
    /// 2 W only).
    pub fn conv_form(&self) -> ConvForm {
        if self.diag.is_some() {
            ConvForm::Dia
        } else {
            ConvForm::Stencils
        }
    }

    /// The geometry cache policy, its budget and its use, bytes.
    pub fn geometry_cache(&self) -> (GeometryCache, usize, usize) {
        (self.cache_mode, self.cache_bytes, self.cached_bytes)
    }

    fn tick(&mut self, name: &'static str, since: Instant) {
        self.gpu.sync();
        *self.timers.entry(name).or_default() += since.elapsed().as_secs_f64();
        let (free, total) = self.gpu.memory();
        self.peak_used = self.peak_used.max(total - free);
    }

    // -- exchanges ----------------------------------------------------------------

    fn halo(&self, ex: &dyn Exchange, level: usize, f: &mut CudaSlice<f64>, rows: usize) {
        if ex.trivial() {
            return;
        }
        let mut h = self.gpu.download(f);
        ex.halo(level, &mut h, rows);
        self.gpu.copy_in(&h, f);
    }

    fn reduce_ghosts(&self, ex: &dyn Exchange, level: usize, f: &mut CudaSlice<f64>, rows: usize) {
        if ex.trivial() {
            return;
        }
        let mut h = self.gpu.download(f);
        ex.reduce_ghosts(level, &mut h, rows);
        self.gpu.copy_in(&h, f);
    }

    // -- geometry -------------------------------------------------------------------

    fn spec(&self, n: usize) -> Spec {
        Spec { nside: self.topo.levels[n].nside, kernel_sz: self.geo.kernel_sz, g: self.geo.l, gauge: self.geo.gauge }
    }

    /// Whether level `n` takes the DIA geometry.
    fn dia_level(&self, n: usize) -> bool {
        self.diag.as_ref().is_some_and(|d| self.topo.levels[n].nside > 2 * d.w)
    }

    fn take_geometry(&mut self, n: usize, b: usize) -> BlockGeometry {
        let spec = self.spec(n);
        if let Some(mut g) = self.cache[n][b].take() {
            self.cached_bytes -= g.bytes();
            if !g.is_stripped() && self.cache_mode != GeometryCache::Full {
                self.whole_used -= g.bytes() - g.stripped_bytes();
            }
            if g.is_stripped() {
                let t = Instant::now();
                let (grid, rows) = (&self.grids[n], &self.a1_ids[n][b]);
                match &mut g {
                    BlockGeometry::Stencils(s) => s.refresh(&self.gpu, grid, rows, &spec),
                    BlockGeometry::Dia(d) => {
                        d.refresh(&self.gpu, grid, rows, &spec, self.diag.as_ref().unwrap(), &self.ker);
                        assert_eq!(d.misses, 0, "DIA coefficients off the masks of their first build");
                    }
                }
                self.tick("gpu_geometry_weights", t);
            }
            return g;
        }
        let t = Instant::now();
        let blk = &self.topo.levels[n].blocks[b];
        let k = blk.a1.len();
        let mut g = None;
        if self.dia_level(n) {
            let own = OwnRows { start: blk.start, len: blk.len as i64, f0: blk.f0 as i64 };
            let d = DiaGeometry::build(&self.gpu, &self.grids[n], &self.a1_ids[n][b], k, &spec, self.diag.as_ref().unwrap(), &self.ker, own);
            if d.misses == 0 {
                g = Some(BlockGeometry::Dia(d));
            } else {
                eprintln!("level {n} block {b}: {} reads off the DIA diagonals, bound stencils instead", d.misses);
            }
        }
        let g = g.unwrap_or_else(|| BlockGeometry::Stencils(DeviceGeometry::build(&self.gpu, &self.grids[n], &self.a1_ids[n][b], k, &spec, false)));
        self.tick("gpu_geometry", t);
        g
    }

    /// `out[((r G + g) 2 + o) out_stride + opos(i)]` for the geometry rows
    /// `row0 + i`, `i < n_rows`, of block `b` of level `n`, from `x[r
    /// x_stride + .]`, `r < rows` (grid cell `j` at `xmap[j]` if given).
    #[allow(clippy::too_many_arguments)]
    fn conv_forward(
        &self,
        geom: &BlockGeometry,
        n: usize,
        b: usize,
        x: Arg,
        xmap: Arg,
        x_stride: usize,
        rows: usize,
        row0: usize,
        n_rows: usize,
        opos: Arg,
        out: Arg,
        out_stride: usize,
    ) {
        match geom {
            BlockGeometry::Stencils(g) => g.forward_mapped(&self.gpu, &self.ker, x, xmap, x_stride, rows, row0, n_rows, opos, out, out_stride),
            BlockGeometry::Dia(g) => g.forward_mapped(
                &self.gpu,
                &self.grids[n],
                &self.a1_ids[n][b],
                self.diag.as_ref().unwrap(),
                x,
                xmap,
                x_stride,
                rows,
                row0,
                n_rows,
                opos,
                out,
                out_stride,
            ),
        }
    }

    /// Adjoint of [`conv_forward`](Self::conv_forward): `out[r out_stride +
    /// dmap(j)] += sum` over the rows `row0 + i`, `i < n_rows`, reading grid
    /// cell `j`.
    #[allow(clippy::too_many_arguments)]
    fn conv_adjoint(
        &self,
        geom: &BlockGeometry,
        n: usize,
        b: usize,
        gin: Arg,
        gin_stride: usize,
        rows: usize,
        row0: usize,
        n_rows: usize,
        dmap: Arg,
        out: Arg,
        out_stride: usize,
    ) {
        match geom {
            BlockGeometry::Stencils(g) => g.adjoint(&self.gpu, &self.ker, gin, gin_stride, rows, row0, n_rows, dmap, out, out_stride, true),
            BlockGeometry::Dia(g) => g.adjoint(
                &self.gpu,
                &self.grids[n],
                &self.a1_ids[n][b],
                self.diag.as_ref().unwrap(),
                gin,
                gin_stride,
                rows,
                row0,
                n_rows,
                dmap,
                out,
                out_stride,
                true,
            ),
        }
    }

    /// Keeps the geometry of a block as the policy and the budget allow:
    /// with `Transposition`, the budget left once every block's
    /// transposition is counted keeps whole geometries, first come.
    fn put_geometry(&mut self, n: usize, b: usize, mut g: BlockGeometry) {
        if self.cache_mode == GeometryCache::None {
            return;
        }
        let whole = g.bytes();
        let mut stripped = g.stripped_bytes();
        let keep_whole = match self.cache_mode {
            GeometryCache::Full => self.cached_bytes + whole <= self.cache_bytes,
            _ => self.whole_used + (whole - stripped) <= self.whole_budget,
        };
        if keep_whole {
            if self.cache_mode != GeometryCache::Full {
                self.whole_used += whole - stripped;
            }
            stripped = whole;
        } else {
            g.strip();
        }
        if self.cached_bytes + stripped <= self.cache_bytes {
            self.cached_bytes += stripped;
            self.cache[n][b] = Some(g);
        }
    }

    // -- maps -----------------------------------------------------------------------

    /// Loads the level-0 field `x` `[B, C, |layout 0|]` (owned cells read),
    /// exchanges its halo and decimates it to every level.
    pub fn load(&mut self, ex: &dyn Exchange, x: &[f64]) {
        let started = Instant::now();
        let x0 = self.gpu.upload(x);
        self.tick("gpu_load", started);
        self.load_device(ex, x0);
    }

    /// [`load`](Self::load) of a field already on the device, which becomes
    /// the level-0 map (see [`take_map`](Self::take_map)).
    pub fn load_device(&mut self, ex: &dyn Exchange, mut x0: CudaSlice<f64>) {
        let started = Instant::now();
        let bc = self.batch * self.channels;
        assert_eq!(x0.len(), bc * self.held(0));
        self.halo(ex, 0, &mut x0, bc);
        self.maps = vec![x0];
        for d in 0..self.levels() - 1 {
            let mut next = self.gpu.zeros::<f64>(bc * self.held(d + 1));
            self.down[d].apply(
                &self.gpu,
                self.gpu.a(&self.maps[d]),
                self.held(d),
                bc,
                self.gpu.a(&next),
                self.held(d + 1),
                self.topo.levels[d + 1].own_pos0,
                false,
            );
            self.halo(ex, d + 1, &mut next, bc);
            self.maps.push(next);
        }
        self.tick("gpu_load", started);
    }

    /// The level-0 map of the last load, taken from the engine (a later
    /// sweep needs a new load).
    pub fn take_map(&mut self) -> CudaSlice<f64> {
        std::mem::replace(&mut self.maps[0], self.gpu.zeros::<f64>(0))
    }

    /// `maps[n]` on the host.
    pub fn map_host(&self, n: usize) -> Vec<f64> {
        self.gpu.download(&self.maps[n])
    }

    pub fn m_host(&self, d: usize, n: usize) -> Vec<f64> {
        self.gpu.download(&self.m[&(d, n)])
    }

    // -- operands -------------------------------------------------------------------

    fn groups(plan: &[Coefficient]) -> BTreeMap<usize, Vec<usize>> {
        let mut g: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, c) in plan.iter().enumerate() {
            g.entry(c.native_dg).or_default().push(i);
        }
        g
    }

    /// Whether the plan reads `|z_n|` or the stored moduli (S3, S4).
    fn needs_moduli(plan: &[Coefficient]) -> bool {
        plan.iter().any(|c| matches!(c.family, Family::S3 | Family::S4))
    }

    /// Levels `d` of the second wavelets read by the coefficients `idx`.
    fn branch_levels(&self, plan: &[Coefficient], idx: &[usize]) -> BTreeSet<usize> {
        let dg = &self.cfg.j_to_dg;
        let mut out = BTreeSet::new();
        for &i in idx {
            let c = &plan[i];
            match c.family {
                Family::S3 => {
                    out.insert(dg[c.j1.unwrap()]);
                }
                Family::S4 => {
                    if !self.cfg.fewer {
                        out.insert(dg[c.j1.unwrap()]);
                    }
                    out.insert(dg[c.j2.unwrap()]);
                }
                _ => {}
            }
        }
        out
    }

    /// Second wavelets of the block on its owned cells: `[B C L(l1), L(l3),
    /// 2, |F|]`, from the stored moduli and, for `d = n`, from `|z_n|` read
    /// as `own` (address, row stride, map of the grid cells: none for the
    /// level field, the block's `dmap` for its own).
    fn branches(
        &self,
        n: usize,
        b: usize,
        geom: &BlockGeometry,
        which: &BTreeSet<usize>,
        own: Option<(Arg, usize, Arg)>,
    ) -> BTreeMap<usize, CudaSlice<f64>> {
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let blk = &self.topo.levels[n].blocks[b];
        let ln = self.held(n);
        which
            .iter()
            .map(|&d| {
                let out = self.gpu.zeros::<f64>(bc * l * l * 2 * blk.len);
                let (x, stride, xmap) = if d == n { own.expect("|z_n| of the block") } else { (self.gpu.a(&self.m[&(d, n)]), ln, NULL) };
                self.conv_forward(geom, n, b, x, xmap, stride, bc * l, blk.f0, blk.len, NULL, self.gpu.a(&out), blk.len);
                (d, out)
            })
            .collect()
    }

    /// Operands of the coefficients of a block (batch item `bi`), with
    /// `z_n` and `|z_n|` read at `z` and `absz` (plans with S3 or S4).
    #[allow(clippy::too_many_arguments)]
    fn operands(
        &self,
        n: usize,
        b: usize,
        c: &Coefficient,
        bi: usize,
        branch: &BTreeMap<usize, CudaSlice<f64>>,
        z: Field,
        absz: Option<Field>,
    ) -> (Operand, Operand) {
        let (ch, l) = (self.channels, self.geo.l);
        let lv = &self.topo.levels[n];
        let blk = &lv.blocks[b];
        let nf = blk.len;
        let posf = |level: usize| -> usize {
            let t = &self.topo.levels[level];
            t.own_pos0 + (blk.start - t.owned.0) as usize
        };
        let ln = self.held(n);
        let f64s = std::mem::size_of::<f64>() as u64;
        let z_op = |cc: usize, li: usize| Operand {
            ptr: z.ptr + ((((bi * ch + cc) * l + li) * 2 * z.stride + z.pos) as u64) * f64s,
            im_off: z.stride as i64,
        };
        let map_op = |level: usize, cc: usize| Operand {
            ptr: self.gpu.ptr(&self.maps[level]) + (((bi * ch + cc) * self.held(level) + posf(level)) as u64) * f64s,
            im_off: -1,
        };
        let branch_op = |d: usize, cc: usize, l1: usize, l3: usize| Operand {
            ptr: self.gpu.ptr(&branch[&d]) + (((((bi * ch + cc) * l + l1) * l + l3) * 2 * nf) as u64) * f64s,
            im_off: nf as i64,
        };
        let mod_op = |d: usize, cc: usize, l1: usize| {
            let f = if d == n {
                absz.expect("|z_n| of the block")
            } else {
                Field { ptr: self.gpu.ptr(&self.m[&(d, n)]), stride: ln, pos: posf(n) }
            };
            Operand { ptr: f.ptr + ((((bi * ch + cc) * l + l1) * f.stride + f.pos) as u64) * f64s, im_off: -1 }
        };
        let dg = &self.cfg.j_to_dg;
        match c.family {
            Family::Mean | Family::Var => (map_op(0, c.c1), NONE),
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                (z_op(c.c1, l1), z_op(c.c2, l1))
            }
            Family::S3 => {
                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                if !self.cfg.fewer {
                    (branch_op(d1, c.c1, l1, l3), z_op(c.c2, l3))
                } else {
                    (map_op(n, c.c1), branch_op(d1, c.c2, l1, l3))
                }
            }
            Family::S4 => {
                let (d1, d2) = (dg[c.j1.unwrap()], dg[c.j2.unwrap()]);
                let (l1, l2, l3) = (c.l1.unwrap(), c.l2.unwrap(), c.l3.unwrap());
                let a = if self.cfg.fewer { mod_op(d1, c.c1, l1) } else { branch_op(d1, c.c1, l1, l3) };
                (a, branch_op(d2, c.c2, l2, l3))
            }
        }
    }

    // -- forward --------------------------------------------------------------------

    /// Coefficient values `[B, len(plan)]` (summed over the processes) and,
    /// with `moments`, the spatial variance of their terms; requires
    /// [`load`](Self::load).  Fills, for a plan with S3 or S4, the stored
    /// moduli, and with `Keep`, `z` and `|z|`.
    pub fn forward(&mut self, ex: &dyn Exchange, plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let started = Instant::now();
        let levels = self.levels();
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let rows = bc * l;
        let groups = Self::groups(plan);
        let moduli = Self::needs_moduli(plan);
        let keep = self.fields == FieldStore::Keep;
        let mut values = vec![Complex64::new(0.0, 0.0); self.batch * plan.len()];
        let mut second = vec![0.0; if moments { self.batch * plan.len() } else { 0 }];
        // Level fields.
        if keep && self.z.len() != levels {
            self.z = (0..levels).map(|n| self.gpu.zeros::<f64>(rows * 2 * self.held(n))).collect();
        }
        if moduli && self.absz.len() != levels {
            self.absz = (0..levels).map(|n| self.gpu.zeros::<f64>(if keep { rows * self.held(n) } else { 0 })).collect();
        }
        if moduli && self.m.is_empty() {
            for d in 0..levels {
                for n in d + 1..levels {
                    let f = self.gpu.zeros::<f64>(rows * self.held(n));
                    self.m.insert((d, n), f);
                }
            }
        }
        for n in 0..levels {
            let ln = self.held(n);
            let idx: Vec<usize> = groups.get(&n).cloned().unwrap_or_default();
            let which = self.branch_levels(plan, &idx);
            let n_total = self.geo.pixels(n) as f64;
            if moduli && !keep {
                self.absz[n] = self.gpu.zeros::<f64>(rows * ln);
            }
            for b in 0..self.topo.levels[n].blocks.len() {
                let geom = self.take_geometry(n, b);
                let blk = &self.topo.levels[n].blocks[b];
                let (na1, nf, f0) = (blk.a1.len(), blk.len, blk.f0);
                let lvn = &self.topo.levels[n];
                let posf_n = lvn.own_pos0 + (blk.start - lvn.owned.0) as usize;
                let t = Instant::now();
                // z_n on the block's convolution rows: in the level field
                // (`Keep`) or the block's own.
                let own_z = (!keep).then(|| self.gpu.zeros::<f64>(rows * 2 * na1));
                let (z_rows, z_stride, z_pos, zf) = match &own_z {
                    Some(z) => (self.gpu.a(z), na1, NULL, Field { ptr: self.gpu.ptr(z), stride: na1, pos: f0 }),
                    None => (
                        self.gpu.a(&self.z[n]),
                        ln,
                        self.gpu.a(&self.a1_pos[n][b]),
                        Field { ptr: self.gpu.ptr(&self.z[n]), stride: ln, pos: posf_n },
                    ),
                };
                self.conv_forward(&geom, n, b, self.gpu.a(&self.maps[n]), NULL, ln, bc, 0, na1, z_pos, z_rows, z_stride);
                let mut af = None;
                if moduli {
                    self.gpu.launch1(
                        "modulus_at",
                        na1 * rows,
                        &[
                            z_rows,
                            z_stride.into(),
                            z_pos,
                            na1.into(),
                            (rows as i32).into(),
                            self.gpu.a(&self.absz[n]),
                            ln.into(),
                            self.gpu.a(&self.a1_pos[n][b]),
                        ],
                    );
                    af = Some(Field { ptr: self.gpu.ptr(&self.absz[n]), stride: ln, pos: posf_n });
                }
                let own = moduli.then(|| (self.gpu.a(&self.absz[n]), ln, NULL));
                let branch = self.branches(n, b, &geom, &which, own);
                self.tick("gpu_forward_conv", t);
                let t = Instant::now();
                if !idx.is_empty() && nf > 0 {
                    let mut items = Vec::with_capacity(idx.len() * self.batch);
                    for &i in &idx {
                        let c = &plan[i];
                        let kind = match c.family {
                            Family::Mean => F_MEAN,
                            Family::Var => F_SQ,
                            Family::S1 if c.c1 == c.c2 => F_ABS,
                            Family::S2 if c.c1 == c.c2 => F_SQ,
                            Family::S1 => F_S1X,
                            _ => F_PROD,
                        };
                        for bi in 0..self.batch {
                            let (a, bo) = self.operands(n, b, c, bi, &branch, zf, af);
                            items.push(ForwardItem { kind, pad: 0, a, b: bo });
                        }
                    }
                    *self.timers.entry("gpu_host_lists").or_default() += t.elapsed().as_secs_f64();
                    let sums = self.partial_sums(&items, nf, moments);
                    for (k, &i) in idx.iter().enumerate() {
                        for bi in 0..self.batch {
                            let (re, im, sq) = sums[k * self.batch + bi];
                            values[bi * plan.len() + i] += Complex64::new(re, im) / n_total;
                            if moments {
                                second[bi * plan.len() + i] += sq / n_total;
                            }
                        }
                    }
                }
                self.tick("gpu_forward_coefficients", t);
                drop(branch);
                drop(own_z);
                self.put_geometry(n, b, geom);
            }
            // The moduli of level n are complete: halo, then the decimation
            // chain to every coarser level.
            if !moduli {
                continue;
            }
            let t = Instant::now();
            let mut a = std::mem::replace(&mut self.absz[n], self.gpu.zeros::<f64>(0));
            self.halo(ex, n, &mut a, rows);
            self.absz[n] = a;
            for n2 in n + 1..levels {
                let mut target = self.m.remove(&(n, n2)).unwrap();
                let src = if n2 == n + 1 { &self.absz[n] } else { &self.m[&(n, n2 - 1)] };
                self.down[n2 - 1].apply(
                    &self.gpu,
                    self.gpu.a(src),
                    self.held(n2 - 1),
                    rows,
                    self.gpu.a(&target),
                    self.held(n2),
                    self.topo.levels[n2].own_pos0,
                    false,
                );
                self.halo(ex, n2, &mut target, rows);
                self.m.insert((n, n2), target);
            }
            self.tick("gpu_forward_decimation", t);
            if !keep {
                self.absz[n] = self.gpu.zeros::<f64>(0);
            }
        }
        let mut flat: Vec<f64> = values.iter().flat_map(|v| [v.re, v.im]).collect();
        ex.allreduce(&mut flat);
        if moments {
            ex.allreduce(&mut second);
        }
        let values: Vec<Complex64> = flat.chunks(2).map(|p| Complex64::new(p[0], p[1])).collect();
        let second = if moments {
            values.iter().zip(&second).map(|(s, m2)| (m2 - s.norm_sqr()).max(0.0)).collect()
        } else {
            Vec::new()
        };
        self.tick("forward", started);
        (values, second)
    }

    /// `(sum re, sum im, sum |f|^2)` per item over `n` pixels.
    fn partial_sums(&self, items: &[ForwardItem], n: usize, moments: bool) -> Vec<(f64, f64, f64)> {
        let n_chunks = n.div_ceil(CHUNK);
        let mut out = Vec::with_capacity(items.len());
        for part in items.chunks(65535) {
            let dev_items = self.gpu.upload(part);
            let buf = self.gpu.zeros::<f64>(part.len() * n_chunks * 3);
            self.gpu.launch(
                "coef_partial",
                (n_chunks as u32, part.len() as u32),
                256,
                &[self.gpu.a(&dev_items), n.into(), CHUNK.into(), (moments as i32).into(), self.gpu.a(&buf)],
            );
            let h = self.gpu.download(&buf);
            for k in 0..part.len() {
                let (mut re, mut im, mut sq) = (0.0, 0.0, 0.0);
                for c in 0..n_chunks {
                    let o = &h[(k * n_chunks + c) * 3..][..3];
                    re += o[0];
                    im += o[1];
                    sq += o[2];
                }
                out.push((re, im, sq));
            }
        }
        out
    }

    // -- reverse --------------------------------------------------------------------

    /// Map cotangents of every level on the device (owned cells valid, halo
    /// zero) for the coefficient cotangent `gradient` `[B, len(plan)]`;
    /// requires [`forward`](Self::forward).
    pub fn reverse(&mut self, ex: &dyn Exchange, plan: &[Coefficient], gradient: &[Complex64]) -> Vec<CudaSlice<f64>> {
        let started = Instant::now();
        assert_eq!(gradient.len(), self.batch * plan.len());
        let zero = Complex64::new(0.0, 0.0);
        let active: Vec<bool> = (0..plan.len()).map(|i| (0..self.batch).any(|b| gradient[b * plan.len() + i] != zero)).collect();
        let groups = Self::groups(plan);
        let moduli = Self::needs_moduli(plan);
        let keep = self.fields == FieldStore::Keep;
        let levels = self.levels();
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let rows = bc * l;
        // Stored-moduli cotangents (none without S3 / S4: the chain is zero).
        let mut gm: BTreeMap<(usize, usize), CudaSlice<f64>> = BTreeMap::new();
        for d in 0..if moduli { levels } else { 0 } {
            for n in d + 1..levels {
                gm.insert((d, n), self.gpu.zeros::<f64>(rows * self.held(n)));
            }
        }
        let mut mg: Vec<CudaSlice<f64>> = (0..levels).map(|n| self.gpu.zeros::<f64>(bc * self.held(n))).collect();
        for n in (0..levels).rev() {
            let ln = self.held(n);
            let t = Instant::now();
            let chain = if moduli { self.chain(ex, n, &gm) } else { None };
            self.tick("gpu_reverse_chain", t);
            let idx: Vec<usize> = groups.get(&n).map(|v| v.iter().copied().filter(|&i| active[i]).collect()).unwrap_or_default();
            let which = self.branch_levels(plan, &idx);
            let n_total = self.geo.pixels(n) as f64;
            for b in 0..self.topo.levels[n].blocks.len() {
                let geom = self.take_geometry(n, b);
                let blk = self.topo.levels[n].blocks[b].clone();
                let (na1, nf) = (blk.a1.len(), blk.len);
                let lvn = &self.topo.levels[n];
                let posf_n = lvn.own_pos0 + (blk.start - lvn.owned.0) as usize;
                let t = Instant::now();
                // Positions of the grid cells on the block's convolution rows.
                if !which.is_empty() {
                    self.gpu.launch1("fill_i32", ln, &[self.gpu.a(&self.dmap), ln.into(), Arg::I32(-1)]);
                    self.gpu.launch1("dmap_set", na1, &[self.gpu.a(&self.dmap), self.gpu.a(&self.a1_pos[n][b]), na1.into()]);
                }
                // z_n and |z_n| on the convolution rows: kept in the level
                // fields, or recomputed from the maps as in the forward sweep.
                let (own_z, own_absz) = if keep {
                    (None, None)
                } else {
                    let z = self.gpu.zeros::<f64>(rows * 2 * na1);
                    self.conv_forward(&geom, n, b, self.gpu.a(&self.maps[n]), NULL, ln, bc, 0, na1, NULL, self.gpu.a(&z), na1);
                    let a = moduli.then(|| {
                        let a = self.gpu.zeros::<f64>(rows * na1);
                        self.gpu.launch1(
                            "modulus_at",
                            na1 * rows,
                            &[self.gpu.a(&z), na1.into(), NULL, na1.into(), (rows as i32).into(), self.gpu.a(&a), na1.into(), NULL],
                        );
                        a
                    });
                    (Some(z), a)
                };
                let (z_rows, z_stride, z_pos, zf) = match &own_z {
                    Some(z) => (self.gpu.a(z), na1, NULL, Field { ptr: self.gpu.ptr(z), stride: na1, pos: blk.f0 }),
                    None => (
                        self.gpu.a(&self.z[n]),
                        ln,
                        self.gpu.a(&self.a1_pos[n][b]),
                        Field { ptr: self.gpu.ptr(&self.z[n]), stride: ln, pos: posf_n },
                    ),
                };
                let (af, own) = match &own_absz {
                    Some(a) => (Some(Field { ptr: self.gpu.ptr(a), stride: na1, pos: blk.f0 }), Some((self.gpu.a(a), na1, self.gpu.a(&self.dmap)))),
                    None if moduli => (
                        Some(Field { ptr: self.gpu.ptr(&self.absz[n]), stride: ln, pos: posf_n }),
                        Some((self.gpu.a(&self.absz[n]), ln, NULL)),
                    ),
                    None => (None, None),
                };
                let branch = self.branches(n, b, &geom, &which, own);
                self.tick("gpu_reverse_conv", t);
                let t = Instant::now();
                let gz = self.gpu.zeros::<f64>(rows * 2 * na1);
                let mut gbranch: BTreeMap<usize, CudaSlice<f64>> = BTreeMap::new();
                let mut gmod: BTreeMap<usize, CudaSlice<f64>> = BTreeMap::new();
                // Targets of the coefficient cotangents, contributions in plan order.
                let mut targets: Vec<(u64, i64, Vec<Contrib>)> = Vec::new();
                let mut index: HashMap<u64, usize> = HashMap::new();
                let f64s = std::mem::size_of::<f64>() as u64;
                let dg = self.cfg.j_to_dg.clone();
                let ch = self.channels;
                for &i in &idx {
                    let c = plan[i];
                    for d in [
                        (c.family == Family::S3).then(|| dg[c.j1.unwrap()]),
                        (c.family == Family::S4 && !self.cfg.fewer).then(|| dg[c.j1.unwrap()]),
                        (c.family == Family::S4).then(|| dg[c.j2.unwrap()]),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        gbranch.entry(d).or_insert_with(|| self.gpu.zeros::<f64>(rows * l * 2 * nf));
                    }
                    if c.family == Family::S4 && self.cfg.fewer {
                        gmod.entry(dg[c.j1.unwrap()]).or_insert_with(|| self.gpu.zeros::<f64>(rows * na1));
                    }
                    for bi in 0..self.batch {
                        let g = gradient[bi * plan.len() + i];
                        let (a, bo) = self.operands(n, b, &c, bi, &branch, zf, af);
                        let tz = |cc: usize, li: usize| -> (u64, i64) {
                            (self.gpu.ptr(&gz) + ((((bi * ch + cc) * l + li) * 2 * na1 + blk.f0) as u64) * f64s, na1 as i64)
                        };
                        let tb = |d: usize, cc: usize, l1: usize, l3: usize| -> (u64, i64) {
                            (self.gpu.ptr(&gbranch[&d]) + (((((bi * ch + cc) * l + l1) * l + l3) * 2 * nf) as u64) * f64s, nf as i64)
                        };
                        let tmap = |level: usize, cc: usize| -> (u64, i64) {
                            let lv = &self.topo.levels[level];
                            let p = lv.own_pos0 + (blk.start - lv.owned.0) as usize;
                            (self.gpu.ptr(&mg[level]) + (((bi * ch + cc) * self.held(level) + p) as u64) * f64s, -1)
                        };
                        let contrib = |kind: i32| Contrib { kind, pad: 0, a, b: bo, g_re: g.re, g_im: g.im, n_total };
                        let mut push = |(ptr, im): (u64, i64), kind: i32| {
                            let at = *index.entry(ptr).or_insert_with(|| {
                                targets.push((ptr, im, Vec::new()));
                                targets.len() - 1
                            });
                            targets[at].2.push(contrib(kind));
                        };
                        match c.family {
                            Family::Mean => push(tmap(0, c.c1), R_MEAN),
                            Family::Var => push(tmap(0, c.c1), R_VAR),
                            Family::S1 | Family::S2 => {
                                let l1 = c.l1.unwrap();
                                if c.c1 == c.c2 {
                                    push(tz(c.c1, l1), if c.family == Family::S1 { R_S1SAME } else { R_S2SAME });
                                } else if c.family == Family::S1 {
                                    push(tz(c.c1, l1), R_S1X_A);
                                    push(tz(c.c2, l1), R_S1X_B);
                                } else {
                                    push(tz(c.c1, l1), R_GA);
                                    push(tz(c.c2, l1), R_GB);
                                }
                            }
                            Family::S3 => {
                                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                                if self.cfg.fewer {
                                    push(tmap(n, c.c1), R_GA);
                                    push(tb(d1, c.c2, l1, l3), R_GB);
                                } else {
                                    push(tb(d1, c.c1, l1, l3), R_GA);
                                    push(tz(c.c2, l3), R_GB);
                                }
                            }
                            Family::S4 => {
                                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                                if self.cfg.fewer {
                                    let p = self.gpu.ptr(&gmod[&d1]) + ((((bi * ch + c.c1) * l + l1) * na1 + blk.f0) as u64) * f64s;
                                    push((p, -1), R_GA);
                                } else {
                                    push(tb(d1, c.c1, l1, l3), R_GA);
                                }
                                push(tb(dg[c.j2.unwrap()], c.c2, c.l2.unwrap(), l3), R_GB);
                            }
                        }
                    }
                }
                if !targets.is_empty() && nf > 0 {
                    let mut flat_t = Vec::with_capacity(targets.len());
                    let mut flat_c = Vec::new();
                    for (ptr, im, cs) in &targets {
                        let start = flat_c.len() as u32;
                        flat_c.extend_from_slice(cs);
                        flat_t.push(Target { ptr: *ptr, im_off: *im, start, end: flat_c.len() as u32 });
                    }
                    // Host time building the contribution lists (no sync).
                    let lists = t.elapsed().as_secs_f64();
                    *self.timers.entry("gpu_host_lists").or_default() += lists;
                    let dc = self.gpu.upload(&flat_c);
                    for part in flat_t.chunks(65535) {
                        let dt = self.gpu.upload(part);
                        self.gpu.launch(
                            "coef_reverse",
                            (nf.div_ceil(256) as u32, part.len() as u32),
                            256,
                            &[self.gpu.a(&dt), self.gpu.a(&dc), nf.into()],
                        );
                    }
                }
                self.tick("gpu_reverse_coefficients", t);
                drop(branch);
                drop(own_absz);
                let t = Instant::now();
                // Second-wavelet cotangents -> modulus cotangents on a1 (the
                // levels of `which`: dmap is set).
                for (&d, g) in &gbranch {
                    let slot = gmod.entry(d).or_insert_with(|| self.gpu.zeros::<f64>(rows * na1));
                    self.conv_adjoint(&geom, n, b, self.gpu.a(g), nf, rows, blk.f0, nf, self.gpu.a(&self.dmap), self.gpu.a(slot), na1);
                }
                drop(gbranch);
                // Own level: modulus VJP; finer levels: stored-moduli cotangents.
                for (&d, g) in &gmod {
                    if d == n {
                        self.gpu.launch1(
                            "modvjp_add",
                            na1 * rows,
                            &[
                                z_rows,
                                z_stride.into(),
                                0usize.into(),
                                z_pos,
                                self.gpu.a(g),
                                na1.into(),
                                0usize.into(),
                                NULL,
                                self.gpu.a(&gz),
                                na1.into(),
                                0usize.into(),
                                na1.into(),
                                (rows as i32).into(),
                            ],
                        );
                    } else {
                        self.gpu.launch1(
                            "gather_add",
                            na1 * rows,
                            &[
                                self.gpu.a(g),
                                na1.into(),
                                0usize.into(),
                                NULL,
                                self.gpu.a(&gm[&(d, n)]),
                                ln.into(),
                                0usize.into(),
                                self.gpu.a(&self.a1_pos[n][b]),
                                na1.into(),
                                (rows as i32).into(),
                            ],
                        );
                    }
                }
                drop(gmod);
                // The block's own chain to the coarser levels: the last
                // transposed decimation on its cells.
                if let Some(coarse) = &chain {
                    let off = (blk.start - self.topo.levels[n].owned.0) as usize;
                    let cb = self.gpu.zeros::<f64>(rows * nf);
                    self.up[n].apply_rows(&self.gpu, off, nf, self.gpu.a(coarse), self.held(n + 1), rows, self.gpu.a(&cb), nf, 0, false);
                    self.gpu.launch1(
                        "modvjp_add",
                        nf * rows,
                        &[
                            Arg::Ptr(zf.ptr),
                            zf.stride.into(),
                            zf.pos.into(),
                            NULL,
                            self.gpu.a(&cb),
                            nf.into(),
                            0usize.into(),
                            NULL,
                            self.gpu.a(&gz),
                            na1.into(),
                            blk.f0.into(),
                            nf.into(),
                            (rows as i32).into(),
                        ],
                    );
                }
                drop(own_z);
                // Wavelet adjoint on the a1 rows -> map cotangent.
                self.conv_adjoint(&geom, n, b, self.gpu.a(&gz), na1, bc, 0, na1, NULL, self.gpu.a(&mg[n]), ln);
                self.tick("gpu_reverse_adjoint", t);
                self.put_geometry(n, b, geom);
            }
            let t = Instant::now();
            for d in 0..if moduli { n } else { 0 } {
                let mut f = gm.remove(&(d, n)).unwrap();
                self.reduce_ghosts(ex, n, &mut f, rows);
                self.halo(ex, n, &mut f, rows);
                gm.insert((d, n), f);
            }
            self.reduce_ghosts(ex, n, &mut mg[n], bc);
            self.tick("gpu_reverse_exchange", t);
        }
        self.tick("reverse", started);
        mg
    }

    /// Cotangent of `|z_n|` from the stored moduli it feeds, `sum_{n2 > n}
    /// (D^T)^(n2 - n) gm[n][n2]`, by Horner's scheme from the coarsest level
    /// down to level `n + 1`: `sum_{n2 > n} (D^T)^(n2 - n - 1) gm[n][n2]`
    /// `[B C L, |layout n + 1|]` (halo exchanged), to which every block of
    /// level `n` applies the last transposed rows, its own.
    fn chain(&self, ex: &dyn Exchange, n: usize, gm: &BTreeMap<(usize, usize), CudaSlice<f64>>) -> Option<CudaSlice<f64>> {
        let top = self.levels() - 1;
        if n >= top {
            return None;
        }
        let rows = self.batch * self.channels * self.geo.l;
        let mut cur = self.gpu.copy(&gm[&(n, top)]);
        for lev in (n + 1..top).rev() {
            let llev = self.held(lev);
            let mut t = self.gpu.zeros::<f64>(rows * llev);
            let p0 = self.topo.levels[lev].own_pos0;
            self.up[lev].apply(&self.gpu, self.gpu.a(&cur), self.held(lev + 1), rows, self.gpu.a(&t), llev, p0, false);
            let own = self.topo.levels[lev].owned_len();
            self.gpu.launch1(
                "gather_add",
                own * rows,
                &[
                    self.gpu.a(&gm[&(n, lev)]),
                    llev.into(),
                    p0.into(),
                    NULL,
                    self.gpu.a(&t),
                    llev.into(),
                    p0.into(),
                    NULL,
                    own.into(),
                    (rows as i32).into(),
                ],
            );
            self.halo(ex, lev, &mut t, rows);
            cur = t;
        }
        Some(cur)
    }

    /// Chains the map cotangents of all levels back to level 0 through the
    /// transposed decimations: the level-0 cotangent over the layout, owned
    /// cells valid.
    pub fn analysis_vjp(&mut self, ex: &dyn Exchange, mut mg: Vec<CudaSlice<f64>>) -> CudaSlice<f64> {
        let started = Instant::now();
        let bc = self.batch * self.channels;
        for d in (1..self.levels()).rev() {
            self.halo(ex, d, &mut mg[d], bc);
            let (fine, coarse) = mg.split_at_mut(d);
            self.up[d - 1].apply(
                &self.gpu,
                self.gpu.a(&coarse[0]),
                self.held(d),
                bc,
                self.gpu.a(&fine[d - 1]),
                self.held(d - 1),
                self.topo.levels[d - 1].own_pos0,
                true,
            );
        }
        self.tick("analysis_vjp", started);
        mg.swap_remove(0)
    }

    /// Band cotangents of the Laplacian pyramid from the level-0 cotangent
    /// `g0` (host; `LocalEngine::analysis_vjp`).
    pub fn bands(&self, ex: &dyn Exchange, g0: Vec<f64>) -> Vec<Vec<f64>> {
        let bc = self.batch * self.channels;
        let mut bands = vec![g0];
        for d in 0..self.levels() - 1 {
            let fine = bands.last().unwrap();
            let (lf, lc) = (&self.part.layouts[d], &self.part.layouts[d + 1]);
            let (s, e) = self.part.owned[d];
            let mut coarse = vec![0.0; bc * lc.len()];
            for r in 0..bc {
                for p in s..e {
                    coarse[r * lc.len() + lc.pos(p / 4)] += fine[r * lf.len() + lf.pos(p)];
                }
            }
            ex.reduce_ghosts(d + 1, &mut coarse, bc);
            bands.push(coarse);
        }
        bands
    }
}

/// Identifiers of the evaluators (0: none).
static EVALUATORS: AtomicUsize = AtomicUsize::new(1);

/// [`Evaluator`] on a device: the same objective as the CPU sweeps, for one
/// plan and batch.  Evaluators made by [`share`](Self::share) use the same
/// engine: an evaluation (forward then gradient) of one must not be
/// interleaved with a forward of another, which the objectives never do.
pub struct GpuEvaluator<'a> {
    engine: Rc<RefCell<GpuEngine<'a>>>,
    plan: Vec<Coefficient>,
    batch: usize,
    id: usize,
    reported: BTreeMap<&'static str, f64>,
}

impl<'a> GpuEvaluator<'a> {
    /// `plan` on `engine` at the engine's batch.
    pub fn new(engine: GpuEngine<'a>, plan: &[Coefficient]) -> Self {
        let batch = engine.batch;
        Self::on(Rc::new(RefCell::new(engine)), plan, batch)
    }

    fn on(engine: Rc<RefCell<GpuEngine<'a>>>, plan: &[Coefficient], batch: usize) -> Self {
        let id = EVALUATORS.fetch_add(1, Ordering::Relaxed);
        GpuEvaluator { engine, plan: plan.to_vec(), batch, id, reported: BTreeMap::new() }
    }

    /// `plan` at `batch` maps on the same engine (its topology, geometry
    /// cache, decimation rows and level fields): no device memory beyond
    /// the fields of the larger batch and the moduli of a plan with S3 or
    /// S4.
    pub fn share(&self, plan: &[Coefficient], batch: usize) -> Self {
        Self::on(self.engine.clone(), plan, batch)
    }

    /// The engine (shared with the other evaluators made by [`share`](Self::share)).
    pub fn engine(&self) -> std::cell::Ref<'_, GpuEngine<'a>> {
        self.engine.borrow()
    }

    /// The engine's device.
    pub fn gpu(&self) -> Arc<Gpu> {
        self.engine.borrow().gpu.clone()
    }

    /// [`Evaluator::forward`] of maps already on the device (`[B, C, |layout
    /// 0|]`), which the engine keeps as its level-0 map until
    /// [`take_map`](Self::take_map).
    pub fn forward_device(&mut self, ex: &dyn Exchange, x: CudaSlice<f64>, moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let mut e = self.engine.borrow_mut();
        e.set_batch(self.batch);
        e.holder = self.id;
        e.load_device(ex, x);
        e.forward(ex, &self.plan, moments)
    }

    /// [`Evaluator::gradient`] left on the device: the level-0 cotangent
    /// over the layout (owned cells valid).
    pub fn gradient_device(&mut self, ex: &dyn Exchange, cotangent: &[Complex64]) -> CudaSlice<f64> {
        let mut e = self.engine.borrow_mut();
        assert_eq!(e.holder, self.id, "gradient without the forward sweep of this evaluator");
        let mg = e.reverse(ex, &self.plan, cotangent);
        e.analysis_vjp(ex, mg)
    }

    /// The level-0 map of this evaluator's last forward sweep, back from the
    /// engine (after its gradient).
    pub fn take_map(&mut self) -> CudaSlice<f64> {
        let mut e = self.engine.borrow_mut();
        assert_eq!(e.holder, self.id, "map of another evaluator's sweep");
        e.take_map()
    }
}

impl Evaluator for GpuEvaluator<'_> {
    fn forward(&mut self, ex: &dyn Exchange, x: &[f64], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let mut e = self.engine.borrow_mut();
        e.set_batch(self.batch);
        e.holder = self.id;
        e.load(ex, x);
        e.forward(ex, &self.plan, moments)
    }

    fn gradient(&mut self, ex: &dyn Exchange, cotangent: &[Complex64]) -> Vec<f64> {
        let mut e = self.engine.borrow_mut();
        assert_eq!(e.holder, self.id, "gradient without the forward sweep of this evaluator");
        let mg = e.reverse(ex, &self.plan, cotangent);
        let g0 = e.analysis_vjp(ex, mg);
        let started = Instant::now();
        let out = e.gpu.download(&g0);
        e.tick("gpu_download", started);
        out
    }

    /// The engine's phases (of every evaluator sharing it).
    fn take_timers(&mut self) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        for (&k, &v) in &self.engine.borrow().timers {
            let before = self.reported.insert(k, v).unwrap_or(0.0);
            out.push((k, v - before));
        }
        out
    }

    fn device_memory(&self) -> Option<(usize, usize)> {
        Some(self.engine.borrow().device_memory())
    }
}
