//! The level sweeps of `stl_core::engine::LocalEngine` on one device.
//!
//! Every level's fields live on the device over the rank's layout: maps,
//! wavelet fields `z_n` and moduli `|z_n|` (on the convolution rows of the
//! blocks, kept from the forward sweep for the reverse one), stored moduli
//! `m[d][n]` and, in the reverse sweep, their cotangents and the map
//! cotangents.  A level is processed by device blocks (contiguous owned
//! cells, one convolution geometry each).
//!
//! Decimations use the full-sky rows of the owned cells (`topology`):
//! `maps[d+1] = D maps[d]`, `m[d][n] = D m[d][n-1]` (`m[d][d+1] = D |z_d|`)
//! after a halo exchange of the finer field, the same values as the
//! CPU pieces (exact rows); the reverse chain of a level is accumulated by
//! Horner's scheme through the transposed rows.  The other halo and ghost
//! exchanges are those of the CPU engine.
//!
//! Summation orders: convolution and decimation sums in the CPU order;
//! coefficient partial sums by chunks of `CHUNK` pixels in a fixed tree,
//! chunks, blocks and levels in increasing order; cotangents of one target
//! accumulated in plan order.

use crate::device::{Arg, Gpu, NULL};
use crate::geometry::{DeviceGeometry, Grid, Spec};
use crate::topology::{HostCsr, Topology};
use cudarc::driver::{CudaSlice, DeviceRepr};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Owned cells per device block; 0: from the device memory.
    pub block_rows: usize,
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
            cache: GeometryCache::Auto,
            cache_bytes: 0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
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

    /// `y[r y_stride + y0 + i] (+)= sum_q vals x[r x_stride + cols]`.
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, gpu: &Gpu, x: Arg, x_stride: usize, rows: usize, y: Arg, y_stride: usize, y0: usize, accumulate: bool) {
        gpu.launch1(
            "csr_apply",
            self.n_rows * rows,
            &[
                gpu.a(&self.indptr),
                gpu.a(&self.cols),
                gpu.a(&self.vals),
                self.n_rows.into(),
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
    cache: Vec<Vec<Option<DeviceGeometry>>>,
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
    /// `m[d][n]` `[B C L, |layout n|]`.
    pub m: BTreeMap<(usize, usize), CudaSlice<f64>>,
    /// `|z_n|` `[B C L, |layout n|]` and `z_n` `[B C L, 2, |layout n|]` on the
    /// blocks' convolution rows (and halo for `|z_n|`).
    absz: Vec<CudaSlice<f64>>,
    z: Vec<CudaSlice<f64>>,
    pub timers: BTreeMap<&'static str, f64>,
    block_rows: usize,
    /// Largest device memory in use seen at the phase ends (whole device).
    peak_used: usize,
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
        // Level fields: maps, z, |z|, m (and in the reverse sweep gm, map
        // cotangents, chain): doubles per held cell.
        let field_doubles: usize =
            (0..levels).map(|n| held[n] * (2 * bc + 3 * bc * l + 2 * n * bc * l + bc * l)).sum();
        let (free, _total) = gpu.memory();
        // Held cells of the levels with a decimation: about 30 + 7.5 nonzero
        // weights (12 bytes) per coarse / fine row, plus the block topology.
        let csr_bytes: usize = held[..levels - 1].iter().sum::<usize>() * 200;
        let fixed = field_doubles * 8 + csr_bytes + held.iter().sum::<usize>() * 32 + (512 << 20);
        let avail = free.saturating_sub(fixed);
        // Transients of one block: its geometry while built (and the
        // transposition temporaries), second wavelets and their cotangents.
        let per_row = DeviceGeometry::bytes_per_row(l, p) * 3 / 2 + levels * bc * l * l * 32 + bc * l * 48;
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
        let rows_total: usize = topo.levels.iter().flat_map(|lv| lv.blocks.iter().map(|b| b.a1.len())).sum();
        let largest = topo.levels.iter().flat_map(|lv| lv.blocks.iter().map(|b| b.a1.len())).max().unwrap_or(0);
        // Cache budget from what is free once the topology is on the device
        // (the level fields are allocated at the first sweeps).
        let cache_bytes = if opts.cache_bytes > 0 {
            opts.cache_bytes
        } else {
            let (free_now, _) = gpu.memory();
            free_now.saturating_sub(field_doubles * 8 + largest * per_row + (512 << 20)) * 4 / 5
        };
        let cache_mode = match opts.cache {
            GeometryCache::Auto if rows_total * DeviceGeometry::bytes_per_row(l, p) <= cache_bytes => GeometryCache::Full,
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
            cache,
            cache_mode,
            cache_bytes,
            cached_bytes: 0,
            whole_budget: cache_bytes.saturating_sub(rows_total * DeviceGeometry::transposition_bytes_per_row(l, p)),
            whole_used: 0,
            dmap,
            maps: Vec::new(),
            m: BTreeMap::new(),
            absz: Vec::new(),
            z: Vec::new(),
            timers,
            block_rows,
            peak_used: 0,
        }
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

    fn take_geometry(&mut self, n: usize, b: usize) -> DeviceGeometry {
        let spec = self.spec(n);
        if let Some(mut g) = self.cache[n][b].take() {
            self.cached_bytes -= g.bytes();
            if !g.is_stripped() && self.cache_mode != GeometryCache::Full {
                self.whole_used -= g.bytes() - DeviceGeometry::stripped_bytes(&g);
            }
            if g.is_stripped() {
                let t = Instant::now();
                g.refresh(&self.gpu, &self.grids[n], &self.a1_ids[n][b], &spec);
                self.tick("gpu_geometry_weights", t);
            }
            return g;
        }
        let t = Instant::now();
        let k = self.topo.levels[n].blocks[b].a1.len();
        let g = DeviceGeometry::build(&self.gpu, &self.grids[n], &self.a1_ids[n][b], k, &spec, false);
        self.tick("gpu_geometry", t);
        g
    }

    /// Keeps the geometry of a block as the policy and the budget allow:
    /// with `Transposition`, the budget left once every block's
    /// transposition is counted keeps whole geometries, first come.
    fn put_geometry(&mut self, n: usize, b: usize, mut g: DeviceGeometry) {
        if self.cache_mode == GeometryCache::None {
            return;
        }
        let whole = g.bytes();
        let mut stripped = DeviceGeometry::stripped_bytes(&g);
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
        let bc = self.batch * self.channels;
        assert_eq!(x.len(), bc * self.held(0));
        let mut x0 = self.gpu.upload(x);
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

    /// Second wavelets of the block on its owned cells: `[B C L(l1), L(l3), 2, |F|]`.
    fn branches(&self, n: usize, b: usize, geom: &DeviceGeometry, which: &BTreeSet<usize>) -> BTreeMap<usize, CudaSlice<f64>> {
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let blk = &self.topo.levels[n].blocks[b];
        let ln = self.held(n);
        which
            .iter()
            .map(|&d| {
                let out = self.gpu.zeros::<f64>(bc * l * l * 2 * blk.len);
                let input = if d == n { &self.absz[n] } else { &self.m[&(d, n)] };
                geom.forward(&self.gpu, &self.ker, self.gpu.a(input), ln, bc * l, blk.f0, blk.len, NULL, self.gpu.a(&out), blk.len);
                (d, out)
            })
            .collect()
    }

    /// Operands of the coefficients of a block (batch item `bi`).
    #[allow(clippy::too_many_arguments)]
    fn operands(
        &self,
        n: usize,
        b: usize,
        c: &Coefficient,
        bi: usize,
        branch: &BTreeMap<usize, CudaSlice<f64>>,
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
            ptr: self.gpu.ptr(&self.z[n]) + ((((bi * ch + cc) * l + li) * 2 * ln + posf(n)) as u64) * f64s,
            im_off: ln as i64,
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
            let f = if d == n { &self.absz[n] } else { &self.m[&(d, n)] };
            Operand { ptr: self.gpu.ptr(f) + ((((bi * ch + cc) * l + l1) * ln + posf(n)) as u64) * f64s, im_off: -1 }
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
    /// [`load`](Self::load).  Fills `z`, `|z|` and the stored moduli.
    pub fn forward(&mut self, ex: &dyn Exchange, plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let started = Instant::now();
        let levels = self.levels();
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let rows = bc * l;
        let groups = Self::groups(plan);
        let mut values = vec![Complex64::new(0.0, 0.0); self.batch * plan.len()];
        let mut second = vec![0.0; if moments { self.batch * plan.len() } else { 0 }];
        // Level fields.
        if self.z.len() != levels {
            self.z = (0..levels).map(|n| self.gpu.zeros::<f64>(rows * 2 * self.held(n))).collect();
            self.absz = (0..levels).map(|n| self.gpu.zeros::<f64>(rows * self.held(n))).collect();
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
            for b in 0..self.topo.levels[n].blocks.len() {
                let geom = self.take_geometry(n, b);
                let (na1, nf) = (self.topo.levels[n].blocks[b].a1.len(), self.topo.levels[n].blocks[b].len);
                let t = Instant::now();
                geom.forward(
                    &self.gpu,
                    &self.ker,
                    self.gpu.a(&self.maps[n]),
                    ln,
                    bc,
                    0,
                    na1,
                    self.gpu.a(&self.a1_pos[n][b]),
                    self.gpu.a(&self.z[n]),
                    ln,
                );
                self.gpu.launch1(
                    "modulus_at",
                    na1 * rows,
                    &[
                        self.gpu.a(&self.z[n]),
                        ln.into(),
                        self.gpu.a(&self.a1_pos[n][b]),
                        na1.into(),
                        (rows as i32).into(),
                        self.gpu.a(&self.absz[n]),
                        ln.into(),
                    ],
                );
                let branch = self.branches(n, b, &geom, &which);
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
                            let (a, bo) = self.operands(n, b, c, bi, &branch);
                            items.push(ForwardItem { kind, pad: 0, a, b: bo });
                        }
                    }
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
                self.put_geometry(n, b, geom);
            }
            // The moduli of level n are complete: halo, then the decimation
            // chain to every coarser level.
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
        let levels = self.levels();
        let (bc, l) = (self.batch * self.channels, self.geo.l);
        let rows = bc * l;
        let mut gm: BTreeMap<(usize, usize), CudaSlice<f64>> = BTreeMap::new();
        for d in 0..levels {
            for n in d + 1..levels {
                gm.insert((d, n), self.gpu.zeros::<f64>(rows * self.held(n)));
            }
        }
        let mut mg: Vec<CudaSlice<f64>> = (0..levels).map(|n| self.gpu.zeros::<f64>(bc * self.held(n))).collect();
        for n in (0..levels).rev() {
            let ln = self.held(n);
            let t = Instant::now();
            let chain = self.chain(ex, n, &gm);
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
                let branch = self.branches(n, b, &geom, &which);
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
                        let (a, bo) = self.operands(n, b, &c, bi, &branch);
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
                let t = Instant::now();
                // Second-wavelet cotangents -> modulus cotangents on a1.
                if !gbranch.is_empty() {
                    self.gpu.launch1("fill_i32", ln, &[self.gpu.a(&self.dmap), ln.into(), Arg::I32(-1)]);
                    self.gpu.launch1("dmap_set", na1, &[self.gpu.a(&self.dmap), self.gpu.a(&self.a1_pos[n][b]), na1.into()]);
                }
                for (&d, g) in &gbranch {
                    let slot = gmod.entry(d).or_insert_with(|| self.gpu.zeros::<f64>(rows * na1));
                    geom.adjoint(&self.gpu, &self.ker, self.gpu.a(g), nf, rows, blk.f0, nf, self.gpu.a(&self.dmap), self.gpu.a(slot), na1, true);
                }
                drop(gbranch);
                // Own level: modulus VJP; finer levels: stored-moduli cotangents.
                for (&d, g) in &gmod {
                    if d == n {
                        self.gpu.launch1(
                            "modvjp_add",
                            na1 * rows,
                            &[
                                self.gpu.a(&self.z[n]),
                                ln.into(),
                                0usize.into(),
                                self.gpu.a(&self.a1_pos[n][b]),
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
                // The block's own chain to the coarser levels.
                if let Some(cn) = &chain {
                    let own = self.topo.levels[n].owned_len();
                    let off = (blk.start - self.topo.levels[n].owned.0) as usize;
                    self.gpu.launch1(
                        "modvjp_add",
                        nf * rows,
                        &[
                            self.gpu.a(&self.z[n]),
                            ln.into(),
                            posf_n.into(),
                            NULL,
                            self.gpu.a(cn),
                            own.into(),
                            off.into(),
                            NULL,
                            self.gpu.a(&gz),
                            na1.into(),
                            blk.f0.into(),
                            nf.into(),
                            (rows as i32).into(),
                        ],
                    );
                }
                // Wavelet adjoint on the a1 rows -> map cotangent.
                geom.adjoint(&self.gpu, &self.ker, self.gpu.a(&gz), na1, bc, 0, na1, NULL, self.gpu.a(&mg[n]), ln, true);
                self.tick("gpu_reverse_adjoint", t);
                self.put_geometry(n, b, geom);
            }
            let t = Instant::now();
            for d in 0..n {
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

    /// Cotangent of `|z_n|` on the owned cells of level `n` from the stored
    /// moduli it feeds, `sum_{n2 > n} (D^T)^(n2 - n) gm[n][n2]`, by Horner's
    /// scheme from the coarsest level: `[B C L, owned n]`.
    fn chain(&self, ex: &dyn Exchange, n: usize, gm: &BTreeMap<(usize, usize), CudaSlice<f64>>) -> Option<CudaSlice<f64>> {
        let top = self.levels() - 1;
        if n >= top {
            return None;
        }
        let rows = self.batch * self.channels * self.geo.l;
        let mut cur: Option<CudaSlice<f64>> = None;
        for lev in (n + 1..top).rev() {
            let llev = self.held(lev);
            let mut t = self.gpu.zeros::<f64>(rows * llev);
            let x = cur.as_ref().unwrap_or(&gm[&(n, top)]);
            let p0 = self.topo.levels[lev].own_pos0;
            self.up[lev].apply(&self.gpu, self.gpu.a(x), self.held(lev + 1), rows, self.gpu.a(&t), llev, p0, false);
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
            cur = Some(t);
        }
        let own = self.topo.levels[n].owned_len();
        let c = self.gpu.zeros::<f64>(rows * own);
        let x = cur.as_ref().unwrap_or(&gm[&(n, top)]);
        self.up[n].apply(&self.gpu, self.gpu.a(x), self.held(n + 1), rows, self.gpu.a(&c), own, 0, false);
        Some(c)
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

/// [`Evaluator`] on a device: the same objective as the CPU sweeps.
pub struct GpuEvaluator<'a> {
    pub engine: GpuEngine<'a>,
    plan: Vec<Coefficient>,
    reported: BTreeMap<&'static str, f64>,
}

impl<'a> GpuEvaluator<'a> {
    pub fn new(engine: GpuEngine<'a>, plan: &[Coefficient]) -> Self {
        GpuEvaluator { engine, plan: plan.to_vec(), reported: BTreeMap::new() }
    }
}

impl Evaluator for GpuEvaluator<'_> {
    fn forward(&mut self, ex: &dyn Exchange, x: &[f64], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        self.engine.load(ex, x);
        let plan = std::mem::take(&mut self.plan);
        let out = self.engine.forward(ex, &plan, moments);
        self.plan = plan;
        out
    }

    fn gradient(&mut self, ex: &dyn Exchange, cotangent: &[Complex64]) -> Vec<f64> {
        let plan = std::mem::take(&mut self.plan);
        let mg = self.engine.reverse(ex, &plan, cotangent);
        self.plan = plan;
        let g0 = self.engine.analysis_vjp(ex, mg);
        self.engine.gpu.download(&g0)
    }

    fn take_timers(&mut self) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        for (&k, &v) in &self.engine.timers {
            let before = self.reported.insert(k, v).unwrap_or(0.0);
            out.push((k, v - before));
        }
        out
    }

    fn device_memory(&self) -> Option<(usize, usize)> {
        Some(self.engine.device_memory())
    }
}
