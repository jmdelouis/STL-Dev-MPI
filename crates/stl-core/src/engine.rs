//! Complete ScatCov (mean, var, S1, S2, S3, S4) with patch-local geometry,
//! single process: port of `STL_main/LocalScattering.py` (`_Block`,
//! `LocalEngine`, `apply_local`, `vjp_local`).
//!
//! Forward sweep, levels ascending: each level `n` is cut into blocks `F`;
//! `z_n` is computed on `a1 = F + conv halo` through one partial
//! convolution on `D = a1 + conv halo`; the stored moduli `m[d][n]` of finer
//! levels are read on `a1`; the second wavelets are evaluated on `F`;
//! partial sums are accumulated; then `|z_n|` on `F` is pushed through the
//! decimation pieces into `m[n][n']` of every coarser level.
//!
//! Reverse sweep, levels descending: the block fields are recomputed, the
//! coefficient cotangents applied, the cotangents `gm[d][n]` of the stored
//! moduli accumulated (complete once the coarser levels are done), the
//! block's own chain reversed from `gm[n][n']`, then the modulus VJP and the
//! wavelet adjoint give the map cotangent of level `n`.
//!
//! Layouts (row-major, pixel axis last): maps `[B, C, N_d]`; `z` on `a1`
//! `[B, C, L, |a1|]`; moduli `[B, C, L, *]`; branches
//! `[B, C, L(l1), L(l3), |F|]`.
//!
//! Summation order: each partial sum adds the pixels of a block in id
//! order, blocks in id order, levels ascending.
//!
//! Masks ([`crate::mask`], [`LocalEngine::set_masks`]): the coefficients
//! are averaged over each mask of their native level instead of the whole
//! sky; values, variances and cotangents are then `[B, K, len(plan)]`.

use crate::conv::ConvPlan;
use crate::local::{modulus_vjp, positions, LocalGeometry};
use crate::mask::Masks;
use crate::neighbours::{children, dilate_fine, parents};
use crate::plan::{Coefficient, Family};
use num_complex::Complex64;
use std::collections::BTreeMap;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// `z / sqrt(|z| + eps)`, the cross-channel S1 normalisation.
#[inline]
fn root_normalise(z: Complex64) -> Complex64 {
    z / (z.norm() + 1e-8).sqrt()
}

/// VJP of [`root_normalise`] (`AdjointOps.root_modulus_vjp`).
fn root_modulus_vjp(z: &[Complex64], g: &[Complex64]) -> Vec<Complex64> {
    z.iter()
        .zip(g)
        .map(|(&v, &gi)| {
            let r = v.norm();
            let radial = (gi.conj() * v).re;
            let mag = if r == 0.0 { 1.0 } else { r };
            gi / (r + 1e-8).sqrt() - (0.5 * radial / (r + 1e-8).powf(1.5)) * v / mag
        })
        .collect()
}

/// Model parameters shared by both sweeps.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Number of scales `J`.
    pub j_count: usize,
    /// Level of each scale, `dg(j)`.
    pub j_to_dg: Vec<usize>,
    /// `has_fewer_convolutions` definitions of S3/S4.
    pub fewer: bool,
    /// Overlapping-patch mode with the levels from this one on computed by
    /// one process (see [`Overlap`]); `None`: halo exchanges at every level.
    pub overlap: Option<usize>,
}

impl EngineConfig {
    /// `j_count` scales with `dg(j) = j`, no overlap mode.
    pub fn new(j_count: usize, fewer: bool) -> Self {
        EngineConfig { j_count, j_to_dg: (0..j_count).collect(), fewer, overlap: None }
    }

    /// Number of pyramid levels, `max dg(j) + 1`.
    pub fn levels(&self) -> usize {
        self.j_to_dg.iter().max().map_or(1, |&d| d + 1)
    }
}

/// Pixels of one level held by a process: the whole level, or a sorted
/// set (owned range plus halo).  Fields over a layout are `[R, len]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The whole level of `n` pixels: positions are the NESTED ids.
    Full(usize),
    /// Sorted NESTED ids held (owned range plus halo).
    Sparse(Vec<i64>),
}

impl Layout {
    /// Number of pixels held.
    pub fn len(&self) -> usize {
        match self {
            Layout::Full(n) => *n,
            Layout::Sparse(ids) => ids.len(),
        }
    }

    /// True when no pixel is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Position of the cell `id` (must be held).
    #[inline]
    pub fn pos(&self, id: i64) -> usize {
        match self {
            Layout::Full(n) => {
                debug_assert!((id as usize) < *n);
                id as usize
            }
            Layout::Sparse(ids) => {
                let q = ids.partition_point(|&s| s < id);
                assert!(q < ids.len() && ids[q] == id, "cell id {id} is not held by this process");
                q
            }
        }
    }

    /// Cell id at position `p`.
    #[inline]
    pub fn id(&self, p: usize) -> i64 {
        match self {
            Layout::Full(_) => p as i64,
            Layout::Sparse(ids) => ids[p],
        }
    }
}

/// Overlapping-patch mode.  Levels `< coarse` (fine): each process
/// evaluates its owned blocks (coefficient terms counted) and halo blocks
/// (terms not counted, fields recomputed so that everything its owned
/// terms read is exact locally), and decimates its maps on its map blocks:
/// no exchange during the sweeps.  Levels `>= coarse` are held whole and
/// evaluated by the root process alone; their inputs (stored moduli of the
/// fine levels, the map of level `coarse`) are summed over the processes
/// once, their cotangents handed back once.  An evaluation then needs the
/// level-0 halo of the map, two global sums, one broadcast and the
/// level-0 ghost reduction of the map cotangent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overlap {
    /// First level held whole and evaluated by the root process.
    pub coarse: usize,
    /// This process is the root.
    pub root: bool,
}

/// What one process holds and works on: per level, the owned NESTED range
/// `[start, end)`, the layout of its fields and its blocks (in id order).
/// The pyramid is aligned: the owned range of level `d + 1` is the parents
/// of the owned range of level `d`.
#[derive(Clone, Debug)]
pub struct Partition {
    /// Owned NESTED range `[start, end)` of every level.
    pub owned: Vec<(i64, i64)>,
    /// Layout of the fields of every level.
    pub layouts: Vec<Layout>,
    /// Blocks whose coefficient terms this process counts.
    pub blocks: Vec<Vec<Vec<i64>>>,
    /// Overlap mode: blocks evaluated without counting their terms.
    pub halo_blocks: Vec<Vec<Vec<i64>>>,
    /// Cells where this process decimates the map of each level (and
    /// chains the map cotangent back): the owned blocks, or in overlap
    /// mode the whole region its blocks read.
    pub map_blocks: Vec<Vec<Vec<i64>>>,
    /// Overlapping-patch mode, if any.
    pub overlap: Option<Overlap>,
}

impl Partition {
    /// One process holding every level whole (`LocalGeometry.blocks`).
    pub fn serial(geo: &LocalGeometry, levels: usize) -> Self {
        let blocks: Vec<Vec<Vec<i64>>> = (0..levels).map(|d| geo.blocks(d)).collect();
        Partition {
            owned: (0..levels).map(|d| (0, geo.pixels(d) as i64)).collect(),
            layouts: (0..levels).map(|d| Layout::Full(geo.pixels(d))).collect(),
            halo_blocks: vec![Vec::new(); levels],
            map_blocks: blocks.clone(),
            blocks,
            overlap: None,
        }
    }

    /// First level held whole by the root (`levels` when none, or in the
    /// classic mode).
    pub fn coarse_level(&self) -> usize {
        self.overlap.map_or(self.layouts.len(), |o| o.coarse)
    }

    /// True when `id` is in the owned range of `level`.
    #[inline]
    pub fn owns(&self, level: usize, id: i64) -> bool {
        let (s, e) = self.owned[level];
        s <= id && id < e
    }
}

/// Communication between the processes sharing the sky.  Every process
/// calls the same sequence of operations; `field` is `[rows, len]` over the
/// layout of `level`.
pub trait Exchange {
    /// Owners send their values to the processes holding them as halo.
    fn halo(&self, level: usize, field: &mut [f64], rows: usize);
    /// Contributions accumulated in halo cells are added to the owners'
    /// cells (owner first, then the other processes in rank order) and the
    /// halo cells are reset to zero.
    fn reduce_ghosts(&self, level: usize, field: &mut [f64], rows: usize);
    /// Element-wise sum over the processes, identical on every process.
    fn allreduce(&self, values: &mut [f64]);
    /// True when `halo` and `reduce_ghosts` do nothing (one process), so
    /// that a device backend can skip copying its fields to the host.
    fn trivial(&self) -> bool {
        false
    }

    /// [`halo`](Self::halo) of several fields `(level, field, rows)` at
    /// once: a backend may send them in one message per peer.  Every
    /// process passes the same list (same levels and rows, same order).
    fn halo_many(&self, fields: &mut [(usize, &mut [f64], usize)]) {
        for (level, field, rows) in fields.iter_mut() {
            self.halo(*level, field, *rows);
        }
    }

    /// [`reduce_ghosts`](Self::reduce_ghosts) of several fields at once;
    /// per cell, the additions keep the order of the single-field version.
    fn reduce_ghosts_many(&self, fields: &mut [(usize, &mut [f64], usize)]) {
        for (level, field, rows) in fields.iter_mut() {
            self.reduce_ghosts(*level, field, *rows);
        }
    }
}

/// The single-process exchange: nothing to do.
pub struct Serial;

impl Exchange for Serial {
    fn halo(&self, _: usize, _: &mut [f64], _: usize) {}
    fn reduce_ghosts(&self, _: usize, _: &mut [f64], _: usize) {}
    fn allreduce(&self, _: &mut [f64]) {}
    fn trivial(&self) -> bool {
        true
    }
}

/// A backend evaluating the statistics of a fixed plan and their gradient
/// on one process (a device backend instead of the CPU sweeps of
/// [`LocalEngine`]).  Fields are over the layouts of the process's
/// [`Partition`].
pub trait Evaluator {
    /// Statistics `[B, len(plan)]` of the map whose level-0 field is `x`
    /// (owned cells read), summed over the processes; with `moments`, also
    /// the spatial variance of every coefficient's terms
    /// ([`LocalEngine::forward_moments`]), else an empty vector.
    fn forward(&mut self, ex: &dyn Exchange, x: &[f64], moments: bool) -> (Vec<Complex64>, Vec<f64>);
    /// After [`forward`](Self::forward): the level-0 map cotangent (owned
    /// cells valid) of `Re sum(conj(cotangent) * values)`.
    fn gradient(&mut self, ex: &dyn Exchange, cotangent: &[Complex64]) -> Vec<f64>;
    /// Seconds per phase accumulated since the last call.
    fn take_timers(&mut self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
    /// Device memory peak and total, bytes, if any.
    fn device_memory(&self) -> Option<(usize, usize)> {
        None
    }
}

static SERIAL: Serial = Serial;

/// One forward (and optionally reverse) pass of a plan.
pub struct LocalEngine<'a> {
    /// Geometry parameters and per-domain operators.
    pub geo: &'a LocalGeometry,
    /// Model parameters (scales, levels, S3/S4 definitions).
    pub cfg: &'a EngineConfig,
    /// What this process holds and works on.
    pub part: Partition,
    ex: &'a dyn Exchange,
    /// Batch size `B`.
    pub batch: usize,
    /// Number of channels `C`.
    pub channels: usize,
    /// `maps[d]`, `[B, C, |layout d|]`.
    pub maps: Vec<Vec<f64>>,
    /// Stored decimated moduli `m[d][n] = D^(n-d) |z_d|`, `[B, C, L, |layout n|]`, `d < n`.
    pub m: BTreeMap<(usize, usize), Vec<f64>>,
    gm: BTreeMap<(usize, usize), Vec<f64>>,
    /// Masks of the averages (none: the whole sky).
    pub masks: Option<Masks>,
}

impl<'a> LocalEngine<'a> {
    /// Single-process engine on the map `x` `[B, C, N_0]`; the coarser maps
    /// are its block-wise decimations (`LocalGeometry.analysis_levels`).
    pub fn new(geo: &'a LocalGeometry, cfg: &'a EngineConfig, x: Vec<f64>, batch: usize, channels: usize) -> Self {
        let part = Partition::serial(geo, cfg.levels());
        Self::distributed(geo, cfg, part, &SERIAL, x, batch, channels)
    }

    /// Engine of one process: `x` `[B, C, |layout 0|]` (only the owned
    /// cells are read; the halo is exchanged here).
    pub fn distributed(
        geo: &'a LocalGeometry,
        cfg: &'a EngineConfig,
        part: Partition,
        ex: &'a dyn Exchange,
        mut x: Vec<f64>,
        batch: usize,
        channels: usize,
    ) -> Self {
        let levels = cfg.levels();
        assert_eq!(part.layouts.len(), levels);
        assert_eq!(x.len(), batch * channels * part.layouts[0].len());
        let bc = batch * channels;
        ex.halo(0, &mut x, bc);
        let rows = bc * geo.l;
        let mut m = BTreeMap::new();
        for d in 0..levels {
            for n in d + 1..levels {
                m.insert((d, n), vec![0.0; rows * part.layouts[n].len()]);
            }
        }
        let mut engine = LocalEngine { geo, cfg, part, ex, batch, channels, maps: vec![x], m, gm: BTreeMap::new(), masks: None };
        engine.decimate_levels();
        engine
    }

    /// Loads another map `x` `[B, C, |layout 0|]` into this engine, which
    /// keeps its geometry and partition: the coarser maps are rebuilt and
    /// the stored moduli zeroed (the forward sweep accumulates into them),
    /// as for a new engine on `x`.
    pub fn reload(&mut self, x: &[f64]) {
        assert_eq!(x.len(), self.batch * self.channels * self.part.layouts[0].len());
        self.maps.truncate(1);
        self.maps[0].copy_from_slice(x);
        self.ex.halo(0, &mut self.maps[0], self.batch * self.channels);
        self.decimate_levels();
        for v in self.m.values_mut() {
            v.fill(0.0);
        }
        self.gm.clear();
    }

    /// `maps[1..]`: the decimations of `maps[0]` (halo exchanged).
    fn decimate_levels(&mut self) {
        let bc = self.batch * self.channels;
        for d in 0..self.cfg.levels() - 1 {
            let mut next = self.decimate(d);
            match self.part.overlap {
                None => self.ex.halo(d + 1, &mut next, bc),
                // Each process decimated its owned cells of the first coarse
                // level (zero elsewhere): the sum is the whole level.
                Some(o) if d + 1 == o.coarse => self.ex.allreduce(&mut next),
                Some(_) => {}
            }
            self.maps.push(next);
        }
    }

    /// Number of pyramid levels.
    pub fn levels(&self) -> usize {
        self.maps.len()
    }

    /// Averages over the masks `masks` (on the levels of this engine), or
    /// over the whole sky (`None`).
    pub fn set_masks(&mut self, masks: Option<Masks>) {
        if let Some(m) = &masks {
            assert!(m.counts.iter().all(|c| c.len() == self.levels()), "mask levels");
        }
        self.masks = masks;
    }

    /// Number of averages per coefficient: `K` masks, or 1.
    pub fn mask_count(&self) -> usize {
        self.masks.as_ref().map_or(1, Masks::len)
    }

    /// `field[r, ids]` for every row: `[R, |layout|] -> [R, |ids|]`.
    pub fn gather(&self, level: usize, field: &[f64], ids: &[i64]) -> Vec<f64> {
        let layout = &self.part.layouts[level];
        let pos: Vec<usize> = ids.iter().map(|&i| layout.pos(i)).collect();
        field.chunks(layout.len()).flat_map(|row| pos.iter().map(move |&p| row[p])).collect()
    }

    /// `field[r, ids] += values[r, :]`.
    fn scatter_add(&self, level: usize, field: &mut [f64], ids: &[i64], values: &[f64]) {
        let layout = &self.part.layouts[level];
        let (n, m) = (layout.len(), ids.len());
        let pos: Vec<usize> = ids.iter().map(|&i| layout.pos(i)).collect();
        for (r, row) in field.chunks_mut(n).enumerate() {
            for (i, &p) in pos.iter().enumerate() {
                row[p] += values[r * m + i];
            }
        }
    }

    /// Owned cells of a field, `[R, end - start]`.
    pub fn owned_values(&self, level: usize, field: &[f64]) -> Vec<f64> {
        let (s, e) = self.part.owned[level];
        let ids: Vec<i64> = (s..e).collect();
        self.gather(level, field, &ids)
    }

    /// `maps[dg] -> maps[dg + 1]` on the owned cells, by blocks of owned
    /// coarse cells: each row is exact on the domain of its children plus
    /// the doubled decimation halo, the same rows (to the bit) as
    /// `LocalGeometry.decimate` computes from the fine blocks.
    fn decimate(&self, dg: usize) -> Vec<f64> {
        let x = &self.maps[dg];
        let rows = self.batch * self.channels;
        let mut out = vec![0.0; rows * self.part.layouts[dg + 1].len()];
        for coarse in &self.part.map_blocks[dg + 1] {
            let domain = decimation_domain(self.geo, dg, coarse);
            let down = self.geo.down(dg, &domain);
            let y = down.apply(&self.gather(dg, x, &domain));
            let picked = gather_cols(&y, down.out_ids.len(), &positions(&down.out_ids, coarse));
            let layout = &self.part.layouts[dg + 1];
            let nl = layout.len();
            for r in 0..rows {
                for (i, &c) in coarse.iter().enumerate() {
                    out[r * nl + layout.pos(c)] = picked[r * coarse.len() + i];
                }
            }
        }
        out
    }

    /// Transpose of [`decimate`](Self::decimate) added into `out` (level
    /// `dg`) on the cells of `fine_blocks`; `g` (level `dg + 1`) must hold
    /// every coarse cell they read.
    fn decimate_adjoint_add(&self, g: &[f64], dg: usize, out: &mut [f64], fine_blocks: &[Vec<i64>]) {
        let rings = 2 * self.geo.down_rings(dg);
        for block in fine_blocks {
            let domain = dilate_fine(block, self.geo.nside(dg), rings);
            let down = self.geo.down(dg, &domain);
            let x = down.adjoint(&self.gather(dg + 1, g, &down.out_ids));
            let picked = gather_cols(&x, domain.len(), &positions(&domain, block));
            self.scatter_add(dg, out, block, &picked);
        }
    }

    /// Supports of the block's decimation chain, one per level from `n`.
    fn supports(&self, n: usize, block: &[i64]) -> Vec<Vec<i64>> {
        supports(self.geo, self.levels(), n, block)
    }

    fn groups(plan: &[Coefficient]) -> BTreeMap<usize, Vec<usize>> {
        let mut g: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, c) in plan.iter().enumerate() {
            g.entry(c.native_dg).or_default().push(i);
        }
        g
    }

    /// Coefficient values `[B, len(plan)]`, `[B, K, len(plan)]` with masks
    /// (summed over the processes); fills the stored moduli.
    pub fn forward(&mut self, plan: &[Coefficient]) -> Vec<Complex64> {
        self.forward_impl(plan, false).0
    }

    /// [`forward`](Self::forward) plus, for every coefficient `S = <f_p>`
    /// (a pixel mean at its native level), the spatial variance of the
    /// summed terms, `V = <|f_p|^2> - |S|^2` `[B, len(plan)]`.  Under
    /// ergodicity `V / N` (N pixels of the native level) estimates the
    /// variance of the coefficient.
    pub fn forward_moments(&mut self, plan: &[Coefficient]) -> (Vec<Complex64>, Vec<f64>) {
        let (values, second) = self.forward_impl(plan, true);
        let var = values.iter().zip(&second).map(|(s, m2)| (m2 - s.norm_sqr()).max(0.0)).collect();
        (values, var)
    }

    fn forward_impl(&mut self, plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let groups = Self::groups(plan);
        let (k_count, np) = (self.mask_count(), plan.len());
        let mut values = vec![ZERO; self.batch * k_count * np];
        let mut second = vec![0.0; if moments { self.batch * k_count * np } else { 0 }];
        let rows = self.batch * self.channels * self.geo.l;
        let coarse = self.part.coarse_level();
        let masks = self.masks.take();
        for n in 0..self.levels() {
            let counted = self.part.blocks[n].iter().map(|b| (b.clone(), true));
            let halo = self.part.halo_blocks[n].iter().map(|b| (b.clone(), false));
            let work: Vec<(Vec<i64>, bool)> = counted.chain(halo).collect();
            for (block, counted) in work {
                let mut fields = Block::new(self, n, &block);
                let picks: Vec<Vec<usize>> = match &masks {
                    Some(m) if counted => (0..k_count).map(|k| m.pick(k, n, &block)).collect(),
                    _ => Vec::new(),
                };
                if counted {
                    for &i in groups.get(&n).map(Vec::as_slice).unwrap_or(&[]) {
                        let (a, b) = fields.operands(self, &plan[i]);
                        let Some(m) = &masks else {
                            let p = fields.partial(self, &plan[i], &a, &b);
                            for (bi, v) in p.into_iter().enumerate() {
                                values[bi * np + i] += v;
                            }
                            if moments {
                                for (bi, v) in fields.partial_second(self, &plan[i], &a, &b).into_iter().enumerate() {
                                    second[bi * np + i] += v;
                                }
                            }
                            continue;
                        };
                        let nf = fields.nf();
                        for (k, pos) in picks.iter().enumerate() {
                            if pos.is_empty() {
                                continue;
                            }
                            let (ak, bk) = (pick_cols(&a, nf, pos), pick_cols(&b, nf, pos));
                            let total = m.counts[k][n] as f64;
                            for (bi, v) in term_sums(&plan[i], &ak, &bk, self.batch, pos.len(), total).into_iter().enumerate() {
                                values[(bi * k_count + k) * np + i] += v;
                            }
                            if moments {
                                for (bi, v) in term_second_sums(&plan[i], &ak, &bk, self.batch, pos.len(), total).into_iter().enumerate() {
                                    second[(bi * k_count + k) * np + i] += v;
                                }
                            }
                        }
                    }
                }
                // Push |z_n| on F to every coarser level (a halo block only
                // to the fine levels: the coarse ones are summed over the
                // processes and its cells are counted by their owner).
                let top = if counted { self.levels() } else { coarse };
                let chain = self.supports(n, &block);
                let mut a: Vec<f64> = fields.z_on_block().iter().map(|v| v.norm()).collect();
                for n2 in n + 1..top {
                    let (next, _) = self.geo.piece_forward(&a, &chain[n2 - n - 1], n2 - 1);
                    a = next;
                    let mut target = std::mem::take(self.m.get_mut(&(n, n2)).unwrap());
                    self.scatter_add(n2, &mut target, &chain[n2 - n], &a);
                    self.m.insert((n, n2), target);
                }
            }
            let ex = self.ex;
            match self.part.overlap {
                // The moduli pushed from level n are complete: owners collect
                // them, then hand them out as halo to the coarser levels'
                // readers (all coarser levels in one exchange each way).
                None => {
                    let mut fields: Vec<(usize, &mut [f64], usize)> =
                        self.m.range_mut((n, n + 1)..(n + 1, 0)).map(|(&(_, n2), v)| (n2, v.as_mut_slice(), rows)).collect();
                    if !fields.is_empty() {
                        ex.reduce_ghosts_many(&mut fields);
                        ex.halo_many(&mut fields);
                    }
                }
                // Fine levels done: the coarse stored moduli of the fine
                // levels, summed over the processes in one global sum.
                Some(o) if n + 1 == o.coarse && o.coarse < self.levels() => {
                    let keys: Vec<(usize, usize)> = self.m.keys().copied().filter(|&(d, n2)| d < o.coarse && n2 >= o.coarse).collect();
                    let mut flat: Vec<f64> = keys.iter().flat_map(|k| self.m[k].iter().copied()).collect();
                    ex.allreduce(&mut flat);
                    let mut at = 0;
                    for k in keys {
                        let field = self.m.get_mut(&k).unwrap();
                        let len = field.len();
                        field.copy_from_slice(&flat[at..at + len]);
                        at += len;
                    }
                }
                Some(_) => {}
            }
        }
        self.masks = masks;
        let mut flat: Vec<f64> = values.iter().flat_map(|v| [v.re, v.im]).collect();
        self.ex.allreduce(&mut flat);
        if moments {
            self.ex.allreduce(&mut second);
        }
        (flat.chunks(2).map(|p| Complex64::new(p[0], p[1])).collect(), second)
    }

    /// Map cotangents of every level (owned cells valid, halo zero) for the
    /// coefficient cotangent `gradient` `[B, len(plan)]` (`[B, K, len(plan)]`
    /// with masks) (of
    /// `Re sum(conj(gradient) * values)`); requires a previous
    /// [`forward`](Self::forward).  Coefficients whose cotangent is zero for
    /// every batch are skipped.
    pub fn reverse(&mut self, plan: &[Coefficient], gradient: &[Complex64]) -> Vec<Vec<f64>> {
        let (k_count, np) = (self.mask_count(), plan.len());
        assert_eq!(gradient.len(), self.batch * k_count * np);
        let active: Vec<bool> = (0..np).map(|i| (0..self.batch * k_count).any(|r| gradient[r * np + i] != ZERO)).collect();
        let masks = self.masks.take();
        let groups = Self::groups(plan);
        let (bc, rows) = (self.batch * self.channels, self.batch * self.channels * self.geo.l);
        self.gm = self.m.iter().map(|(&k, v)| (k, vec![0.0; v.len()])).collect();
        let mut maps_gradient: Vec<Vec<f64>> = self.maps.iter().map(|x| vec![0.0; x.len()]).collect();
        let coarse = self.part.coarse_level();
        for n in (0..self.levels()).rev() {
            let counted = self.part.blocks[n].iter().map(|b| (b.clone(), true));
            let halo = self.part.halo_blocks[n].iter().map(|b| (b.clone(), false));
            let work: Vec<(Vec<i64>, bool)> = counted.chain(halo).collect();
            for (block, counted) in work {
                let mut fields = Block::new(self, n, &block);
                let picks: Vec<Vec<usize>> = match &masks {
                    Some(m) if counted => (0..k_count).map(|k| m.pick(k, n, &block)).collect(),
                    _ => Vec::new(),
                };
                if counted {
                    for &i in groups.get(&n).map(Vec::as_slice).unwrap_or(&[]) {
                        if !active[i] {
                            continue;
                        }
                        let (a, b) = fields.operands(self, &plan[i]);
                        let Some(m) = &masks else {
                            let g: Vec<Complex64> = (0..self.batch).map(|b| gradient[b * np + i]).collect();
                            fields.reverse_coefficient(self, &plan[i], &a, &b, &g, &mut maps_gradient);
                            continue;
                        };
                        // Operand cotangents of every mask, added on the block.
                        let nf = fields.nf();
                        let (mut ga, mut gb) = (vec![ZERO; self.batch * nf], Vec::new());
                        for (k, pos) in picks.iter().enumerate() {
                            let g: Vec<Complex64> = (0..self.batch).map(|b| gradient[(b * k_count + k) * np + i]).collect();
                            if pos.is_empty() || g.iter().all(|&v| v == ZERO) {
                                continue;
                            }
                            let (ak, bk) = (pick_cols(&a, nf, pos), pick_cols(&b, nf, pos));
                            let (gak, gbk) = operand_cotangents(&plan[i], &ak, &bk, &g, self.batch, pos.len(), m.counts[k][n] as f64);
                            add_cols(&mut ga, nf, pos, &gak);
                            if !gbk.is_empty() {
                                gb.resize(self.batch * nf, ZERO);
                                add_cols(&mut gb, nf, pos, &gbk);
                            }
                        }
                        if !b.is_empty() && gb.is_empty() {
                            gb = vec![ZERO; self.batch * nf];
                        }
                        fields.apply_cotangents(self, &plan[i], &ga, &gb, &mut maps_gradient);
                    }
                }
                let top = if counted { self.levels() } else { coarse };
                fields.finish_reverse(self, &mut maps_gradient, top);
            }
            let ex = self.ex;
            match self.part.overlap {
                // gm[d][n] is complete once level n is done: collect (with
                // the map cotangent of level n, one exchange), then hand out
                // as halo to the finer levels' chains (one exchange).
                None => {
                    let mut fields: Vec<(usize, &mut [f64], usize)> =
                        self.gm.iter_mut().filter(|(&(_, k), _)| k == n).map(|(_, v)| (n, v.as_mut_slice(), rows)).collect();
                    fields.push((n, maps_gradient[n].as_mut_slice(), bc));
                    ex.reduce_ghosts_many(&mut fields);
                    fields.pop();
                    if !fields.is_empty() {
                        ex.halo_many(&mut fields);
                    }
                }
                // Coarse levels done (on the root): chain their map
                // cotangents down to the first coarse level, then hand the
                // cotangents the fine levels need to every process (one
                // global sum, zero contributions from the other processes).
                Some(o) if n == o.coarse => {
                    if o.root {
                        let levels = self.levels();
                        for d in (o.coarse + 1..levels).rev() {
                            let all = self.geo.blocks(d - 1);
                            let (fine, coarse_part) = maps_gradient.split_at_mut(d);
                            self.decimate_adjoint_add(&coarse_part[0], d - 1, &mut fine[d - 1], &all);
                        }
                    }
                    let keys: Vec<(usize, usize)> = self.gm.keys().copied().filter(|&(d, n2)| d < o.coarse && n2 >= o.coarse).collect();
                    let mut flat: Vec<f64> = keys.iter().flat_map(|k| self.gm[k].iter().copied()).collect();
                    flat.extend_from_slice(&maps_gradient[o.coarse]);
                    ex.allreduce(&mut flat);
                    let mut at = 0;
                    for k in keys {
                        let field = self.gm.get_mut(&k).unwrap();
                        let len = field.len();
                        field.copy_from_slice(&flat[at..at + len]);
                        at += len;
                    }
                    // The first coarse level was decimated by the owners of
                    // its cells: each keeps the cotangent of its own cells.
                    let (s, e) = self.part.owned[o.coarse];
                    let layout = &self.part.layouts[o.coarse];
                    let np = layout.len();
                    let mg = &mut maps_gradient[o.coarse];
                    for r in 0..bc {
                        for p in 0..np {
                            let id = layout.id(p);
                            mg[r * np + p] = if s <= id && id < e { flat[at + r * np + p] } else { 0.0 };
                        }
                    }
                }
                Some(_) => {}
            }
        }
        self.gm.clear();
        self.masks = masks;
        maps_gradient
    }

    /// Chain the map cotangents of all levels back to level 0 through the
    /// transposed decimations, then the band cotangents of the Laplacian
    /// pyramid (`LocalGeometry.analysis_vjp`).  Returns `(map0, bands)` over
    /// the layouts, owned cells valid.
    pub fn analysis_vjp(&self, maps_gradient: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
        assert!(self.part.overlap.is_none(), "band cotangents are not available in the overlap mode");
        let bc = self.batch * self.channels;
        let g0 = self.map_cotangent(maps_gradient);
        let mut bands = vec![g0.clone()];
        // Up-sampling adjoint: sum of the four children.  A parent may be
        // owned by another rank (coarse levels shared by more ranks than
        // cells): its contributions are reduced to the owner.
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
            self.ex.reduce_ghosts(d + 1, &mut coarse, bc);
            bands.push(coarse);
        }
        (g0, bands)
    }

    /// Cotangent of the level-0 map only (the first part of
    /// [`analysis_vjp`](Self::analysis_vjp), without the band cotangents
    /// and their exchanges): what a synthesis with the map as variable needs.
    pub fn map_cotangent(&self, mut maps_gradient: Vec<Vec<f64>>) -> Vec<f64> {
        let bc = self.batch * self.channels;
        match self.part.overlap {
            None => {
                for d in (1..self.levels()).rev() {
                    self.ex.halo(d, &mut maps_gradient[d], bc);
                    let (fine, coarse) = maps_gradient.split_at_mut(d);
                    self.decimate_adjoint_add(&coarse[0], d - 1, &mut fine[d - 1], &self.part.blocks[d - 1]);
                }
            }
            // Local partial cotangents chained over the map regions (the
            // coarse levels were chained by the root in `reverse`), then
            // the halo contributions of level 0 summed on their owners.
            Some(o) => {
                for d in (1..o.coarse.min(self.levels() - 1) + 1).rev() {
                    let (fine, coarse) = maps_gradient.split_at_mut(d);
                    self.decimate_adjoint_add(&coarse[0], d - 1, &mut fine[d - 1], &self.part.map_blocks[d - 1]);
                }
                self.ex.reduce_ghosts(0, &mut maps_gradient[0], bc);
            }
        }
        maps_gradient.swap_remove(0)
    }
}

/// Fine cells read by the decimation of the coarse cells `coarse` (level
/// `dg + 1`): their children plus the doubled decimation halo.
pub fn decimation_domain(geo: &LocalGeometry, dg: usize, coarse: &[i64]) -> Vec<i64> {
    dilate_fine(&children(coarse), geo.nside(dg), 2 * geo.down_rings(dg))
}

/// Supports of the decimation chain of a block of level `n`, one per level
/// from `n` to `levels - 1`.
pub fn supports(geo: &LocalGeometry, levels: usize, n: usize, block: &[i64]) -> Vec<Vec<i64>> {
    let mut chain = vec![block.to_vec()];
    for level in n..levels - 1 {
        let s = parents(&dilate_fine(chain.last().unwrap(), geo.nside(level), geo.down_rings(level)));
        chain.push(s);
    }
    chain
}

/// Domains of a block of level `n`: `(a1, domain)` = block + one and two
/// convolution halos.
pub fn block_domains(geo: &LocalGeometry, n: usize, block: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let nside = geo.nside(n);
    let a1 = dilate_fine(block, nside, geo.conv_rings);
    let domain = dilate_fine(&a1, nside, geo.conv_rings);
    (a1, domain)
}

/// `x[r, pos]` for every row of `[R, n]`.
fn gather_cols(x: &[f64], n: usize, pos: &[usize]) -> Vec<f64> {
    x.chunks(n).flat_map(|row| pos.iter().map(move |&p| row[p])).collect()
}

/// Real or complex operand `[B, |F|]`.
type Operand = Vec<Complex64>;

/// Fields of one block `F` of level `n`.
struct Block {
    n: usize,
    block: Vec<i64>,
    a1: Vec<i64>,
    domain: Vec<i64>,
    /// Convolution from the domain to the rows `a1`.
    conv: ConvPlan,
    rows_a1: Vec<usize>,
    f_in_a1: Vec<usize>,
    /// `[B, C, L, |a1|]`.
    z: Vec<Complex64>,
    modulus: BTreeMap<usize, Vec<f64>>,
    branch: BTreeMap<usize, Vec<Complex64>>,
    gz: Option<Vec<Complex64>>,
    /// Insertion order kept so that accumulations follow the reference.
    gbranch: Vec<(usize, Vec<Complex64>)>,
    gmod: Vec<(usize, Vec<f64>)>,
}

impl Block {
    fn new(engine: &LocalEngine, n: usize, block: &[i64]) -> Self {
        let geo = engine.geo;
        let (a1, domain) = block_domains(geo, n, block);
        let conv = geo.conv_plan(n, &domain, &a1);
        let rows_a1 = positions(&domain, &a1);
        let f_in_a1 = positions(&a1, block);
        let x = engine.gather(n, &engine.maps[n], &domain);
        let rows = engine.batch * engine.channels;
        let y = conv.forward(&geo.kernel, &x, rows);
        let z = join(&y, rows, geo.l, a1.len(), None);
        Block {
            n,
            block: block.to_vec(),
            a1,
            domain,
            conv,
            rows_a1,
            f_in_a1,
            z,
            modulus: BTreeMap::new(),
            branch: BTreeMap::new(),
            gz: None,
            gbranch: Vec::new(),
            gmod: Vec::new(),
        }
    }

    fn na1(&self) -> usize {
        self.a1.len()
    }

    fn nf(&self) -> usize {
        self.block.len()
    }

    /// `z` restricted to `F`: `[B, C, L, |F|]`.
    fn z_on_block(&self) -> Vec<Complex64> {
        let na1 = self.na1();
        self.z.chunks(na1).flat_map(|row| self.f_in_a1.iter().map(move |&p| row[p])).collect()
    }

    /// Decimated modulus of level `d <= n` on `a1`: `[B, C, L, |a1|]`.
    fn modulus(&mut self, engine: &LocalEngine, d: usize) -> &Vec<f64> {
        if !self.modulus.contains_key(&d) {
            let v = if d == self.n {
                self.z.iter().map(|v| v.norm()).collect()
            } else {
                engine.gather(self.n, &engine.m[&(d, self.n)], &self.a1)
            };
            self.modulus.insert(d, v);
        }
        &self.modulus[&d]
    }

    /// Second wavelet of `modulus(d)` on `F`: `[B, C, L(l1), L(l3), |F|]`.
    fn branch(&mut self, engine: &LocalEngine, d: usize) -> &Vec<Complex64> {
        if !self.branch.contains_key(&d) {
            let geo = engine.geo;
            let (kd, na1) = (self.domain.len(), self.na1());
            let rows = engine.batch * engine.channels * geo.l;
            let rows_a1 = self.rows_a1.clone();
            let m = self.modulus(engine, d);
            let mut padded = vec![0.0; rows * kd];
            for r in 0..rows {
                for (i, &p) in rows_a1.iter().enumerate() {
                    padded[r * kd + p] = m[r * na1 + i];
                }
            }
            // Second wavelet on F only.
            let y = self.conv.forward_rows(&geo.kernel, &padded, rows, Some(&self.f_in_a1));
            let v = join(&y, rows, geo.l, self.nf(), None);
            self.branch.insert(d, v);
        }
        &self.branch[&d]
    }

    /// `z[:, channel, angle]` on `F`: `[B, |F|]`.
    fn z_core(&self, engine: &LocalEngine, channel: usize, angle: usize) -> Operand {
        let (l, na1) = (engine.geo.l, self.na1());
        (0..engine.batch)
            .flat_map(|b| {
                let row = &self.z[((b * engine.channels + channel) * l + angle) * na1..][..na1];
                self.f_in_a1.iter().map(move |&p| row[p])
            })
            .collect()
    }

    /// `maps[level][:, channel]` on `F`.
    fn map_core(&self, engine: &LocalEngine, level: usize, channel: usize) -> Operand {
        let layout = &engine.part.layouts[level];
        let np = layout.len();
        (0..engine.batch)
            .flat_map(|b| {
                let row = &engine.maps[level][(b * engine.channels + channel) * np..][..np];
                self.block.iter().map(move |&i| Complex64::new(row[layout.pos(i)], 0.0))
            })
            .collect()
    }

    fn branch_core(&mut self, engine: &LocalEngine, d: usize, channel: usize, l1: usize, l3: usize) -> Operand {
        let (l, nf, batch, channels) = (engine.geo.l, self.nf(), engine.batch, engine.channels);
        let br = self.branch(engine, d);
        (0..batch)
            .flat_map(|b| br[((((b * channels + channel) * l + l1) * l + l3) * nf)..][..nf].iter().copied())
            .collect()
    }

    fn modulus_core(&mut self, engine: &LocalEngine, d: usize, channel: usize, l1: usize) -> Operand {
        let (l, na1, batch, channels) = (engine.geo.l, self.na1(), engine.batch, engine.channels);
        let f_in_a1 = self.f_in_a1.clone();
        let m = self.modulus(engine, d);
        (0..batch)
            .flat_map(|b| {
                let row = &m[((b * channels + channel) * l + l1) * na1..][..na1];
                f_in_a1.iter().map(move |&p| Complex64::new(row[p], 0.0)).collect::<Vec<_>>()
            })
            .collect()
    }

    fn operands(&mut self, engine: &LocalEngine, c: &Coefficient) -> (Operand, Operand) {
        let dg = &engine.cfg.j_to_dg;
        let none = Vec::new;
        match c.family {
            Family::Mean | Family::Var => (self.map_core(engine, 0, c.c1), none()),
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                (self.z_core(engine, c.c1, l1), self.z_core(engine, c.c2, l1))
            }
            Family::S3 => {
                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                if !engine.cfg.fewer {
                    (self.branch_core(engine, d1, c.c1, l1, l3), self.z_core(engine, c.c2, l3))
                } else {
                    (self.map_core(engine, self.n, c.c1), self.branch_core(engine, d1, c.c2, l1, l3))
                }
            }
            Family::S4 => {
                let (d1, d2) = (dg[c.j1.unwrap()], dg[c.j2.unwrap()]);
                let (l1, l2, l3) = (c.l1.unwrap(), c.l2.unwrap(), c.l3.unwrap());
                let a = if engine.cfg.fewer {
                    self.modulus_core(engine, d1, c.c1, l1)
                } else {
                    self.branch_core(engine, d1, c.c1, l1, l3)
                };
                (a, self.branch_core(engine, d2, c.c2, l2, l3))
            }
        }
    }

    /// Partial sum `[B]` of the coefficient over the block.
    fn partial(&self, engine: &LocalEngine, c: &Coefficient, a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
        term_sums(c, a, b, engine.batch, self.nf(), engine.geo.pixels(self.n) as f64)
    }

    /// Partial sum `[B]` of `|f_p|^2 / N` over the block, `f_p` the terms
    /// whose mean is the coefficient (see [`partial`](Self::partial)).
    fn partial_second(&self, engine: &LocalEngine, c: &Coefficient, a: &[Complex64], b: &[Complex64]) -> Vec<f64> {
        term_second_sums(c, a, b, engine.batch, self.nf(), engine.geo.pixels(self.n) as f64)
    }

    // -- reverse ----------------------------------------------------------------

    fn add_z(&mut self, engine: &LocalEngine, channel: usize, angle: usize, g: &[Complex64]) {
        let (l, na1, nf) = (engine.geo.l, self.na1(), self.nf());
        let gz = self.gz.get_or_insert_with(|| vec![ZERO; self.z.len()]);
        for b in 0..engine.batch {
            let row = &mut gz[((b * engine.channels + channel) * l + angle) * na1..][..na1];
            for (i, &p) in self.f_in_a1.iter().enumerate() {
                row[p] += g[b * nf + i];
            }
        }
    }

    fn add_branch(&mut self, engine: &LocalEngine, d: usize, channel: usize, l1: usize, l3: usize, g: &[Complex64]) {
        let size = self.branch(engine, d).len();
        let (l, nf) = (engine.geo.l, self.nf());
        let at = match self.gbranch.iter().position(|(k, _)| *k == d) {
            Some(i) => i,
            None => {
                self.gbranch.push((d, vec![ZERO; size]));
                self.gbranch.len() - 1
            }
        };
        let gb = &mut self.gbranch[at].1;
        for b in 0..engine.batch {
            let row = &mut gb[(((b * engine.channels + channel) * l + l1) * l + l3) * nf..][..nf];
            row.iter_mut().zip(&g[b * nf..(b + 1) * nf]).for_each(|(r, &v)| *r += v);
        }
    }

    fn gmod_slot(&mut self, d: usize, size: usize) -> &mut Vec<f64> {
        let at = match self.gmod.iter().position(|(k, _)| *k == d) {
            Some(i) => i,
            None => {
                self.gmod.push((d, vec![0.0; size]));
                self.gmod.len() - 1
            }
        };
        &mut self.gmod[at].1
    }

    fn add_modulus_core(&mut self, engine: &LocalEngine, d: usize, channel: usize, l1: usize, g: &[Complex64]) {
        let size = self.modulus(engine, d).len();
        let (l, na1, nf) = (engine.geo.l, self.na1(), self.nf());
        let f_in_a1 = self.f_in_a1.clone();
        let gm = self.gmod_slot(d, size);
        for b in 0..engine.batch {
            let row = &mut gm[((b * engine.channels + channel) * l + l1) * na1..][..na1];
            for (i, &p) in f_in_a1.iter().enumerate() {
                row[p] += g[b * nf + i].re;
            }
        }
    }

    fn add_map(&self, engine: &LocalEngine, out: &mut [f64], level: usize, channel: usize, g: &[f64]) {
        let layout = &engine.part.layouts[level];
        let (np, nf) = (layout.len(), self.nf());
        for b in 0..engine.batch {
            let row = &mut out[(b * engine.channels + channel) * np..][..np];
            for (i, &id) in self.block.iter().enumerate() {
                row[layout.pos(id)] += g[b * nf + i];
            }
        }
    }

    fn reverse_coefficient(
        &mut self,
        engine: &LocalEngine,
        c: &Coefficient,
        a: &[Complex64],
        b: &[Complex64],
        g: &[Complex64],
        maps_gradient: &mut [Vec<f64>],
    ) {
        let (ga, gb) = operand_cotangents(c, a, b, g, engine.batch, self.nf(), engine.geo.pixels(self.n) as f64);
        self.apply_cotangents(engine, c, &ga, &gb, maps_gradient);
    }

    /// Adds the operand cotangents `ga`, `gb` `[B, |F|]` of the coefficient
    /// `c` ([`operand_cotangents`]) to the block's field cotangents.
    fn apply_cotangents(&mut self, engine: &LocalEngine, c: &Coefficient, ga: &[Complex64], gb: &[Complex64], maps_gradient: &mut [Vec<f64>]) {
        let dg = &engine.cfg.j_to_dg;
        match c.family {
            Family::Mean | Family::Var => {
                let re: Vec<f64> = ga.iter().map(|v| v.re).collect();
                self.add_map(engine, &mut maps_gradient[0], 0, c.c1, &re);
            }
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                self.add_z(engine, c.c1, l1, ga);
                if c.c1 != c.c2 {
                    self.add_z(engine, c.c2, l1, gb);
                }
            }
            Family::S3 | Family::S4 => {
                let d1 = dg[c.j1.unwrap()];
                let (l1, l3) = (c.l1.unwrap(), c.l3.unwrap());
                if c.family == Family::S3 {
                    if engine.cfg.fewer {
                        let re: Vec<f64> = ga.iter().map(|v| v.re).collect();
                        self.add_map(engine, &mut maps_gradient[self.n], self.n, c.c1, &re);
                        self.add_branch(engine, d1, c.c2, l1, l3, gb);
                    } else {
                        self.add_branch(engine, d1, c.c1, l1, l3, ga);
                        self.add_z(engine, c.c2, l3, gb);
                    }
                } else {
                    if engine.cfg.fewer {
                        self.add_modulus_core(engine, d1, c.c1, l1, ga);
                    } else {
                        self.add_branch(engine, d1, c.c1, l1, l3, ga);
                    }
                    self.add_branch(engine, dg[c.j2.unwrap()], c.c2, c.l2.unwrap(), l3, gb);
                }
            }
        }
    }

    /// Reverse of the block's fields; its own decimation chain is reversed
    /// for the targets of the levels below `top`.
    fn finish_reverse(&mut self, engine: &mut LocalEngine, maps_gradient: &mut [Vec<f64>], top: usize) {
        let geo = engine.geo;
        let (n, l, kd, na1, nf) = (self.n, geo.l, self.domain.len(), self.na1(), self.nf());
        let bc = engine.batch * engine.channels;
        // Second-wavelet cotangents (on F) -> modulus cotangents on a1.
        for (d, g) in std::mem::take(&mut self.gbranch) {
            let rows = bc * l;
            let packed = split(&g, rows, l, nf, None);
            let gx = self.conv.adjoint_rows(&geo.kernel, &packed, rows, Some(&self.f_in_a1));
            let gm: Vec<f64> = gx.chunks(kd).flat_map(|row| self.rows_a1.iter().map(move |&p| row[p])).collect();
            let slot = self.gmod_slot(d, gm.len());
            slot.iter_mut().zip(&gm).for_each(|(s, v)| *s += v);
        }
        // Modulus cotangents: own level -> modulus VJP; finer levels -> stored.
        for (d, gm) in std::mem::take(&mut self.gmod) {
            if d == n {
                let v = modulus_vjp(&self.z, &gm);
                let gz = self.gz.get_or_insert_with(|| vec![ZERO; self.z.len()]);
                gz.iter_mut().zip(&v).for_each(|(a, b)| *a += b);
            } else {
                let mut target = std::mem::take(engine.gm.get_mut(&(d, n)).unwrap());
                engine.scatter_add(n, &mut target, &self.a1, &gm);
                engine.gm.insert((d, n), target);
            }
        }
        // The block's own chain to coarser levels.
        let chain = engine.supports(n, &self.block);
        let mut g_fine: Option<Vec<f64>> = None;
        for n2 in n + 1..top {
            let mut g = engine.gather(n2, &engine.gm[&(n, n2)], &chain[n2 - n]);
            for level in (n..n2).rev() {
                g = geo.piece_adjoint(&g, &chain[level - n], &chain[level - n + 1], level);
            }
            g_fine = Some(match g_fine {
                None => g,
                Some(mut acc) => {
                    acc.iter_mut().zip(&g).for_each(|(a, b)| *a += b);
                    acc
                }
            });
        }
        if let Some(gf) = g_fine {
            let zf = self.z_on_block();
            let v = modulus_vjp(&zf, &gf);
            let gz = self.gz.get_or_insert_with(|| vec![ZERO; self.z.len()]);
            for r in 0..bc * l {
                for (i, &p) in self.f_in_a1.iter().enumerate() {
                    gz[r * na1 + p] += v[r * nf + i];
                }
            }
        }
        let Some(gz) = self.gz.take() else { return };
        // Wavelet adjoint on the a1 rows -> map cotangent on the domain.
        let packed = split(&gz, bc, l, na1, None);
        let gx = self.conv.adjoint(&geo.kernel, &packed, bc);
        debug_assert_eq!(gx.len(), bc * kd);
        engine.scatter_add(n, &mut maps_gradient[n], &self.domain, &gx);
    }
}

/// Columns `pos` of the rows of `nf` values of `x` (empty stays empty).
fn pick_cols(x: &[Complex64], nf: usize, pos: &[usize]) -> Vec<Complex64> {
    x.chunks(nf).flat_map(|row| pos.iter().map(move |&p| row[p])).collect()
}

/// `out[r, pos[j]] += v[r, j]`, rows of `nf` values in `out`.
fn add_cols(out: &mut [Complex64], nf: usize, pos: &[usize], v: &[Complex64]) {
    for (row, src) in out.chunks_mut(nf).zip(v.chunks(pos.len())) {
        for (&p, &s) in pos.iter().zip(src) {
            row[p] += s;
        }
    }
}

/// Sums `[B]` of the terms of the coefficient `c` over `nf` pixels,
/// divided by `n_total`: the operands `a`, `b` are `[B, nf]`
/// ([`Block::partial`], [`crate::disc`]).
pub(crate) fn term_sums(c: &Coefficient, a: &[Complex64], b: &[Complex64], batch: usize, nf: usize, n_total: f64) -> Vec<Complex64> {
    (0..batch)
        .map(|bi| {
            let a = &a[bi * nf..][..nf];
            let sum: Complex64 = match c.family {
                Family::Mean => a.iter().sum(),
                Family::Var => a.iter().map(|v| Complex64::new(v.norm_sqr_exact(), 0.0)).sum(),
                Family::S2 if c.c1 == c.c2 => a.iter().map(|v| Complex64::new(v.norm_sqr_exact(), 0.0)).sum(),
                Family::S1 if c.c1 == c.c2 => a.iter().map(|v| Complex64::new(v.norm(), 0.0)).sum(),
                Family::S1 => {
                    let b = &b[bi * nf..][..nf];
                    a.iter().zip(b).map(|(&x, &y)| root_normalise(x) * root_normalise(y).conj()).sum()
                }
                _ => {
                    let b = &b[bi * nf..][..nf];
                    a.iter().zip(b).map(|(&x, &y)| x * y.conj()).sum()
                }
            };
            sum / n_total
        })
        .collect()
}

/// Sums `[B]` of `|f_p|^2 / n_total`, `f_p` the terms whose mean is the
/// coefficient (see [`term_sums`]).
pub(crate) fn term_second_sums(c: &Coefficient, a: &[Complex64], b: &[Complex64], batch: usize, nf: usize, n_total: f64) -> Vec<f64> {
    (0..batch)
        .map(|bi| {
            let a = &a[bi * nf..][..nf];
            let sum: f64 = match c.family {
                Family::Mean => a.iter().map(|v| v.norm_sqr()).sum(),
                Family::Var => a.iter().map(|v| v.norm_sqr_exact().powi(2)).sum(),
                Family::S2 if c.c1 == c.c2 => a.iter().map(|v| v.norm_sqr_exact().powi(2)).sum(),
                Family::S1 if c.c1 == c.c2 => a.iter().map(|v| v.norm_sqr()).sum(),
                Family::S1 => {
                    let b = &b[bi * nf..][..nf];
                    a.iter().zip(b).map(|(&x, &y)| (root_normalise(x) * root_normalise(y).conj()).norm_sqr()).sum()
                }
                _ => {
                    let b = &b[bi * nf..][..nf];
                    a.iter().zip(b).map(|(&x, &y)| (x * y.conj()).norm_sqr()).sum()
                }
            };
            sum / n_total
        })
        .collect()
}

/// Cotangents `(ga, gb)` `[B, nf]` of the operands `a`, `b` of the
/// coefficient `c` for its cotangent `g` `[B]` (see [`term_sums`]).  `ga`
/// is the cotangent of the map (mean, var: real parts), of `z` (S1, S2;
/// through the modulus for an auto S1) or of the first operand (S3, S4);
/// `gb` that of `b` (empty for mean, var and the auto S1, S2).
pub(crate) fn operand_cotangents(
    c: &Coefficient,
    a: &[Complex64],
    b: &[Complex64],
    g: &[Complex64],
    batch: usize,
    nf: usize,
    n_total: f64,
) -> (Vec<Complex64>, Vec<Complex64>) {
    // Per-pixel scaling of a cotangent: ga[b, i] = f(g[b], i).
    let per = |f: &dyn Fn(Complex64, usize) -> Complex64| -> Vec<Complex64> { (0..batch * nf).map(|k| f(g[k / nf], k)).collect() };
    match c.family {
        Family::Mean => (per(&|gi, _| Complex64::new(gi.re / n_total, 0.0)), Vec::new()),
        Family::Var => (per(&|gi, k| Complex64::new(2.0 * gi.re * a[k].re / n_total, 0.0)), Vec::new()),
        Family::S1 | Family::S2 if c.c1 == c.c2 => {
            let ga = if c.family == Family::S1 {
                let scale: Vec<f64> = (0..batch * nf).map(|k| g[k / nf].re / n_total).collect();
                modulus_vjp(a, &scale)
            } else {
                per(&|gi, k| 2.0 * gi.re * a[k] / n_total)
            };
            (ga, Vec::new())
        }
        Family::S1 => {
            let ga = per(&|gi, k| gi * root_normalise(b[k]) / n_total);
            let gb = per(&|gi, k| gi.conj() * root_normalise(a[k]) / n_total);
            (root_modulus_vjp(a, &ga), root_modulus_vjp(b, &gb))
        }
        _ => (per(&|gi, k| gi * b[k] / n_total), per(&|gi, k| gi.conj() * a[k] / n_total)),
    }
}

/// Convolution output `[R, 2L, k]` -> complex `[R, L, |pick|]` at the
/// output rows `pick` (all by default) (`S3Multiscale._join`).
pub(crate) fn join(y: &[f64], rows: usize, l: usize, k: usize, pick: Option<&[usize]>) -> Vec<Complex64> {
    let all: Vec<usize>;
    let pick = match pick {
        Some(p) => p,
        None => {
            all = (0..k).collect();
            &all
        }
    };
    let mut z = Vec::with_capacity(rows * l * pick.len());
    for r in 0..rows {
        for o in 0..l {
            let re = &y[(r * 2 * l + 2 * o) * k..][..k];
            let im = &y[(r * 2 * l + 2 * o + 1) * k..][..k];
            z.extend(pick.iter().map(|&p| Complex64::new(re[p], im[p])));
        }
    }
    z
}

/// Inverse of [`join`]: complex `[R, L, m]` -> `[R, 2L, k]` with the values
/// at rows `at` (`0..m` by default, then `k = m`), zero elsewhere.
pub(crate) fn split(g: &[Complex64], rows: usize, l: usize, m: usize, at: Option<(&[usize], usize)>) -> Vec<f64> {
    let k = at.map_or(m, |(_, k)| k);
    let mut out = vec![0.0; rows * 2 * l * k];
    for r in 0..rows {
        for o in 0..l {
            let src = &g[(r * l + o) * m..][..m];
            for (i, v) in src.iter().enumerate() {
                let p = at.map_or(i, |(pos, _)| pos[i]);
                out[(r * 2 * l + 2 * o) * k + p] = v.re;
                out[(r * 2 * l + 2 * o + 1) * k + p] = v.im;
            }
        }
    }
    out
}

trait NormSqrExact {
    fn norm_sqr_exact(&self) -> f64;
}

impl NormSqrExact for Complex64 {
    /// `abs(z)^2` as torch computes `a.abs().square()`.
    #[inline]
    fn norm_sqr_exact(&self) -> f64 {
        let r = self.norm();
        r * r
    }
}
