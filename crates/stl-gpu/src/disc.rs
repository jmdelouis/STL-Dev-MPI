//! Disc-patch statistics on the device: the model of
//! `stl_core::disc::DiscModel` (support, restricted smooth decimation,
//! wavelet bound to the support, averages on the core) for a batch of
//! patches, with its exact VJP.
//!
//! The patches of a batch are grouped by disc: the members of a group
//! share the operators of their disc and run as the rows of one launch
//! (`[members, C, ...]`, pixel axis last).  The operators of a disc live on
//! the device ([`DiscOps`]): the convolution geometry of every level built
//! by the device kernels on the support (`bind_support` renormalisation,
//! [`DeviceGeometry`]), once for the support rows and once for the core
//! rows (second wavelets, which are only read on the core), and the
//! decimations of the host [`DiscSupport`] (forward and transposed CSR, so
//! that the adjoint is a gather in the order of the CPU).  They are kept in
//! a cache, least recently used out first: before a disc is built, discs
//! are evicted until the device has `reserve` bytes free beyond it (the
//! free memory is read then, so that other users of the device, e.g. a
//! network on the same GPU, are accounted for), and the cache never holds
//! more than the optional `budget` bytes.  The discs of the current batch
//! are never evicted.
//!
//! The terms read fields restricted to the core of their level (one buffer
//! per group and slot, see `disc.cu`); their sums run in one launch per
//! level for every (coefficient, patch), their cotangents in one launch per
//! level for every (operand row, patch, pixel), contributions in plan
//! order.

use crate::device::{Arg, Gpu, NULL};
use crate::engine::{F_ABS, F_MEAN, F_PROD, F_S1X, F_SQ, R_GA, R_GB, R_MEAN, R_S1SAME, R_S1X_A, R_S1X_B, R_S2SAME, R_VAR};
use crate::geometry::{DeviceGeometry, Grid, Spec};
use cudarc::driver::{CudaSlice, DeviceRepr};
use num_complex::Complex64;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;
use stl_core::decimation::Csr;
use stl_core::disc::{DiscSupport, BIND_EPS};
use stl_core::engine::EngineConfig;
use stl_core::local::LocalGeometry;
use stl_core::plan::{Coefficient, Family};

/// Identity of a disc (e.g. the bits of its centre, radius and core).
pub type DiscKey = [u64; 5];

/// Where the patches are read from or their cotangents written to: host
/// memory, or a device address of this GPU's context (e.g. a CUDA tensor
/// of the same device, its producer's stream synchronised by the caller).
#[derive(Clone, Copy, Debug)]
pub enum Patches<'a> {
    /// Host values.
    Host(&'a [f64]),
    /// Device address of the values (`f64`, contiguous).
    Device(u64),
}

/// Threads per block of `disc_terms` / `disc_terms_rev` (`DISC_T`).
const T: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DiscOp {
    slot: i32,
    row: i32,
    cplx: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DiscTerm {
    kind: i32,
    index: i32,
    a: DiscOp,
    b: DiscOp,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DiscPatch {
    group: i32,
    member: i32,
    out: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DiscTarget {
    t: DiscOp,
    start: u32,
    end: u32,
}

unsafe impl DeviceRepr for DiscOp {}
unsafe impl DeviceRepr for DiscTerm {}
unsafe impl DeviceRepr for DiscPatch {}
unsafe impl DeviceRepr for DiscTarget {}

const NO_OP: DiscOp = DiscOp { slot: -1, row: 0, cplx: 0, pad: 0 };

/// Sparse rows on the device (`u32` columns).
struct DevCsr {
    indptr: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    vals: CudaSlice<f64>,
    n_rows: usize,
}

impl DevCsr {
    fn new(gpu: &Gpu, c: &Csr) -> Self {
        assert!(c.indices.len() < u32::MAX as usize);
        let indptr: Vec<u32> = c.indptr.iter().map(|&v| v as u32).collect();
        let cols: Vec<u32> = c.indices.iter().map(|&v| v as u32).collect();
        DevCsr { indptr: gpu.upload(&indptr), cols: gpu.upload(&cols), vals: gpu.upload(&c.values), n_rows: c.n_rows }
    }

    fn bytes(&self) -> usize {
        self.indptr.len() * 4 + self.cols.len() * 4 + self.vals.len() * 8
    }

    /// `y[r y_stride + i] (+)= sum_q vals x[r x_stride + cols]`, `r < rows`
    /// (row sums in column order, as `Csr::apply`).
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, gpu: &Gpu, x: Arg, x_stride: usize, rows: usize, y: Arg, y_stride: usize, accumulate: bool) {
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
                0usize.into(),
                (accumulate as i32).into(),
            ],
        );
    }
}

/// The device operators of one disc.
struct DiscOps {
    /// Support sizes `N_n`.
    n: Vec<usize>,
    /// Core sizes.
    nc: Vec<usize>,
    /// Wavelet on the support rows of every level.
    conv: Vec<DeviceGeometry>,
    /// Wavelet on the core rows (None: the core is the whole support).
    conv_core: Vec<Option<DeviceGeometry>>,
    /// Core positions in the support (None: the whole support).
    core_pos: Vec<Option<CudaSlice<u32>>>,
    /// Decimation `n -> n + 1` and its transpose.
    down: Vec<DevCsr>,
    up: Vec<DevCsr>,
    bytes: usize,
    last_use: u64,
}

impl DiscOps {
    fn new(gpu: &Gpu, geo: &LocalGeometry, s: &DiscSupport) -> Self {
        let levels = s.levels();
        let (mut conv, mut conv_core, mut core_pos) = (Vec::new(), Vec::new(), Vec::new());
        let mut bytes = 0;
        for n in 0..levels {
            let spec = Spec { nside: geo.nside(n), kernel_sz: geo.kernel_sz, g: geo.l, gauge: geo.gauge };
            let g = Grid::new(gpu, &s.ids[n], None);
            let k = s.ids[n].len();
            let geom = DeviceGeometry::build_eps(gpu, &g, &g.ids, k, &spec, false, BIND_EPS);
            bytes += geom.bytes();
            if s.core[n].len() < k {
                let ids: Vec<i64> = s.core[n].iter().map(|&p| s.ids[n][p]).collect();
                let rows = gpu.upload(&ids);
                let gc = DeviceGeometry::build_eps(gpu, &g, &rows, ids.len(), &spec, false, BIND_EPS);
                let pos: Vec<u32> = s.core[n].iter().map(|&p| p as u32).collect();
                bytes += gc.bytes() + 4 * pos.len();
                conv_core.push(Some(gc));
                core_pos.push(Some(gpu.upload(&pos)));
            } else {
                conv_core.push(None);
                core_pos.push(None);
            }
            conv.push(geom);
        }
        let down: Vec<DevCsr> = s.down.iter().map(|d| DevCsr::new(gpu, &d.forward)).collect();
        let up: Vec<DevCsr> = s.down.iter().map(|d| DevCsr::new(gpu, &d.transposed)).collect();
        bytes += down.iter().chain(&up).map(DevCsr::bytes).sum::<usize>();
        DiscOps {
            n: s.ids.iter().map(Vec::len).collect(),
            nc: s.core.iter().map(Vec::len).collect(),
            conv,
            conv_core,
            core_pos,
            down,
            up,
            bytes,
            last_use: 0,
        }
    }

    fn core_conv(&self, n: usize) -> &DeviceGeometry {
        self.conv_core[n].as_ref().unwrap_or(&self.conv[n])
    }

    fn core_arg(&self, gpu: &Gpu, n: usize) -> Arg {
        self.core_pos[n].as_ref().map_or(NULL, |p| gpu.a(p))
    }
}

/// Forward fields of one group kept for the reverse.
struct GroupState {
    key: DiscKey,
    /// Patch indices of the members.
    members: Vec<usize>,
    /// `z_n` `[m C, L, 2, N_n]`.
    z: Vec<CudaSlice<f64>>,
    /// Operand fields on the core, `[level][slot]`.
    slots: Vec<Vec<Option<CudaSlice<f64>>>>,
}

struct State {
    groups: Vec<GroupState>,
    /// Level-0 sizes of the patches (pixels, one channel).
    sizes: Vec<usize>,
}

/// Disc-patch statistics on one GPU; see the module documentation.
pub struct GpuDiscEngine {
    gpu: Arc<Gpu>,
    geo: LocalGeometry,
    cfg: EngineConfig,
    channels: usize,
    ker: CudaSlice<f64>,
    cache: HashMap<DiscKey, DiscOps>,
    budget: Option<usize>,
    reserve: usize,
    clock: u64,
    /// Geometry cache hits and misses (discs of the batches).
    pub hits: u64,
    /// See `hits`.
    pub misses: u64,
    state: Option<State>,
    timers: BTreeMap<&'static str, f64>,
}

impl GpuDiscEngine {
    /// Engine of `channels` channels; the device operators of the discs are
    /// cached while the device keeps `reserve` bytes free, within `budget`
    /// bytes if given (those of the current batch always).
    pub fn new(gpu: Arc<Gpu>, geo: &LocalGeometry, cfg: &EngineConfig, channels: usize, budget: Option<usize>, reserve: usize) -> Self {
        assert!(channels >= 1);
        let ker = gpu.upload(&geo.kernel.w);
        GpuDiscEngine {
            gpu,
            geo: geo.clone(),
            cfg: cfg.clone(),
            channels,
            ker,
            cache: HashMap::new(),
            budget,
            reserve,
            clock: 0,
            hits: 0,
            misses: 0,
            state: None,
            timers: BTreeMap::new(),
        }
    }

    /// The device.
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// Discs cached on the device, their bytes, the budget and the reserve.
    pub fn cache_info(&self) -> (usize, usize, Option<usize>, usize) {
        (self.cache.len(), self.cache.values().map(|o| o.bytes).sum(), self.budget, self.reserve)
    }

    /// Accumulated wall-clock seconds by phase.
    pub fn timers(&self) -> &BTreeMap<&'static str, f64> {
        &self.timers
    }

    fn tick(&mut self, name: &'static str, t: Instant) {
        *self.timers.entry(name).or_default() += t.elapsed().as_secs_f64();
    }

    fn n_slots(&self) -> usize {
        2 + 2 * self.cfg.levels()
    }

    /// Rows per member of every slot: maps on the core (`C`, real), `z` on
    /// the core (`C L`, complex), stored moduli of level `d` (`C L`, real),
    /// second wavelets of level `d` (`C L L`, complex).
    fn slot_rows(&self) -> Vec<i32> {
        let (c, l, levels) = (self.channels, self.geo.l, self.cfg.levels());
        let mut r = vec![c as i32, (c * l) as i32];
        r.extend(std::iter::repeat((c * l) as i32).take(levels));
        r.extend(std::iter::repeat((c * l * l) as i32).take(levels));
        r
    }

    fn mod_slot(&self, d: usize) -> i32 {
        (2 + d) as i32
    }

    fn branch_slot(&self, d: usize) -> i32 {
        (2 + self.cfg.levels() + d) as i32
    }

    /// The operands of a coefficient (as `DiscModel::operands`).
    fn operands(&self, c: &Coefficient) -> (DiscOp, DiscOp) {
        let l = self.geo.l;
        let dg = &self.cfg.j_to_dg;
        let op = |slot: i32, row: usize, cplx: bool| DiscOp { slot, row: row as i32, cplx: cplx as i32, pad: 0 };
        let map = |ch: usize| op(0, ch, false);
        let z = |ch: usize, a: usize| op(1, ch * l + a, true);
        let br = |d: usize, ch: usize, l1: usize, l3: usize| op(self.branch_slot(d), (ch * l + l1) * l + l3, true);
        match c.family {
            Family::Mean | Family::Var => (map(c.c1), NO_OP),
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                (z(c.c1, l1), z(c.c2, l1))
            }
            Family::S3 => {
                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                if !self.cfg.fewer {
                    (br(d1, c.c1, l1, l3), z(c.c2, l3))
                } else {
                    (map(c.c1), br(d1, c.c2, l1, l3))
                }
            }
            Family::S4 => {
                let (d1, d2) = (dg[c.j1.unwrap()], dg[c.j2.unwrap()]);
                let (l1, l2, l3) = (c.l1.unwrap(), c.l2.unwrap(), c.l3.unwrap());
                let a = if self.cfg.fewer { op(self.mod_slot(d1), c.c1 * l + l1, false) } else { br(d1, c.c1, l1, l3) };
                (a, br(d2, c.c2, l2, l3))
            }
        }
    }

    /// Second wavelets `(d, n)` and core moduli `(d, n)` read by the plan.
    fn reads(&self, plan: &[Coefficient]) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let dg = &self.cfg.j_to_dg;
        let (mut br, mut md) = (Vec::new(), Vec::new());
        for c in plan {
            let n = c.native_dg;
            match c.family {
                Family::S3 => br.push((dg[c.j1.unwrap()], n)),
                Family::S4 => {
                    if self.cfg.fewer {
                        md.push((dg[c.j1.unwrap()], n));
                    } else {
                        br.push((dg[c.j1.unwrap()], n));
                    }
                    br.push((dg[c.j2.unwrap()], n));
                }
                _ => {}
            }
        }
        for v in [&mut br, &mut md] {
            v.sort_unstable();
            v.dedup();
        }
        (br, md)
    }

    /// Evicts the least recently used disc not in the current batch; false
    /// if there is none.
    fn evict_one(&mut self) -> bool {
        let old = self.cache.iter().filter(|(_, o)| o.last_use != self.clock).min_by_key(|(_, o)| o.last_use).map(|(k, _)| *k);
        old.map(|k| self.cache.remove(&k)).is_some()
    }

    /// Makes the operators of `discs` resident (building the missing ones,
    /// after evicting least recently used discs until the device keeps
    /// `reserve` bytes free beyond the new one), then evicts beyond `budget`.
    fn prepare(&mut self, discs: &[(DiscKey, &DiscSupport)]) {
        let t = Instant::now();
        self.clock += 1;
        let mut missing: Vec<(DiscKey, &DiscSupport)> = Vec::new();
        for &(k, s) in discs {
            if let Some(o) = self.cache.get_mut(&k) {
                if o.last_use != self.clock {
                    self.hits += 1;
                }
                o.last_use = self.clock;
            } else if !missing.iter().any(|(m, _)| *m == k) {
                assert_eq!(s.levels(), self.cfg.levels(), "disc support levels");
                missing.push((k, s));
            }
        }
        for (k, s) in missing {
            self.misses += 1;
            if !self.cache.is_empty() {
                let est = self.cache.values().map(|o| o.bytes).sum::<usize>() / self.cache.len();
                loop {
                    // Freed buffers return to the device at a synchronisation.
                    self.gpu.sync();
                    if self.gpu.memory().0 >= self.reserve + est || !self.evict_one() {
                        break;
                    }
                }
            }
            let mut o = DiscOps::new(&self.gpu, &self.geo, s);
            o.last_use = self.clock;
            self.cache.insert(k, o);
        }
        if let Some(budget) = self.budget {
            while self.cache.values().map(|o| o.bytes).sum::<usize>() > budget && self.evict_one() {}
        }
        self.tick("geometry", t);
    }

    fn copy_rows(&self, src: &CudaSlice<f64>, s_stride: usize, spos: Arg, dst: &CudaSlice<f64>, d_stride: usize, dpos: Arg, n: usize, rows: usize, add: bool) {
        self.gpu.launch1(
            if add { "gather_add" } else { "gather_copy" },
            n * rows,
            &[self.gpu.a(src), s_stride.into(), 0usize.into(), spos, self.gpu.a(dst), d_stride.into(), 0usize.into(), dpos, n.into(), (rows as i32).into()],
        );
    }

    /// Statistics `[P, len(plan)]` of the patches `x` (patch after patch,
    /// `[C, |ids_0|]` each) on the discs `discs` (one per patch, the
    /// support of `key`), with the spatial variances (`moments`, else
    /// empty); keeps the state of [`reverse`](Self::reverse).
    pub fn forward(&mut self, discs: &[(DiscKey, &DiscSupport)], x: &[f64], plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        assert_eq!(x.len(), self.channels * discs.iter().map(|(_, s)| s.ids[0].len()).sum::<usize>(), "patch shape mismatch");
        self.forward_from(discs, Patches::Host(x), plan, moments)
    }

    /// [`forward`](Self::forward) reading the patches from host or device
    /// memory.
    pub fn forward_from(&mut self, discs: &[(DiscKey, &DiscSupport)], x: Patches, plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>) {
        let started = Instant::now();
        self.state = None;
        let p_count = discs.len();
        assert!(p_count > 0 && p_count < 65536, "1 to 65535 patches per batch");
        let sizes: Vec<usize> = discs.iter().map(|(_, s)| s.ids[0].len()).collect();
        let c = self.channels;
        self.prepare(discs);
        let mut offsets = vec![0usize];
        for s in &sizes {
            offsets.push(offsets.last().unwrap() + c * s);
        }
        // Groups by disc, in order of first appearance.
        let mut order: Vec<DiscKey> = Vec::new();
        let mut members: HashMap<DiscKey, Vec<usize>> = HashMap::new();
        for (i, (k, _)) in discs.iter().enumerate() {
            members.entry(*k).or_insert_with(|| {
                order.push(*k);
                Vec::new()
            });
            members.get_mut(k).unwrap().push(i);
        }
        let (l, levels, n_slots) = (self.geo.l, self.cfg.levels(), self.n_slots());
        let (pairs, modc) = self.reads(plan);
        let moduli = !pairs.is_empty() || !modc.is_empty();
        let gpu = self.gpu.clone();
        let t = Instant::now();
        let mut groups = Vec::with_capacity(order.len());
        for key in order {
            let mem = members.remove(&key).unwrap();
            let ops = &self.cache[&key];
            let r0 = mem.len() * c;
            let x0 = match x {
                Patches::Host(x) => {
                    let mut host = Vec::with_capacity(r0 * ops.n[0]);
                    for &p in &mem {
                        host.extend_from_slice(&x[offsets[p]..offsets[p + 1]]);
                    }
                    gpu.upload(&host)
                }
                Patches::Device(ptr) => {
                    let y = gpu.zeros::<f64>(r0 * ops.n[0]);
                    for (m, &p) in mem.iter().enumerate() {
                        let len = offsets[p + 1] - offsets[p];
                        let src = Arg::Ptr(ptr + 8 * offsets[p] as u64);
                        gpu.launch1("gather_copy", len, &[src, 0usize.into(), 0usize.into(), NULL, gpu.a(&y), 0usize.into(), (m * len).into(), NULL, len.into(), 1i32.into()]);
                    }
                    y
                }
            };
            let mut maps = vec![x0];
            for d in 0..levels - 1 {
                let y = gpu.zeros::<f64>(r0 * ops.n[d + 1]);
                ops.down[d].apply(&gpu, gpu.a(&maps[d]), ops.n[d], r0, gpu.a(&y), ops.n[d + 1], false);
                maps.push(y);
            }
            let mut mods: BTreeMap<(usize, usize), CudaSlice<f64>> = BTreeMap::new();
            let mut z = Vec::with_capacity(levels);
            let mut slots: Vec<Vec<Option<CudaSlice<f64>>>> = Vec::with_capacity(levels);
            for n in 0..levels {
                let (nn, nc) = (ops.n[n], ops.nc[n]);
                let zn = gpu.zeros::<f64>(r0 * l * 2 * nn);
                ops.conv[n].forward(&gpu, &self.ker, gpu.a(&maps[n]), nn, r0, 0, nn, NULL, gpu.a(&zn), nn);
                if moduli {
                    let a = gpu.zeros::<f64>(r0 * l * nn);
                    gpu.launch1("modulus_at", nn * r0 * l, &[gpu.a(&zn), nn.into(), NULL, nn.into(), ((r0 * l) as i32).into(), gpu.a(&a), nn.into(), NULL]);
                    for d in 0..n {
                        let y = gpu.zeros::<f64>(r0 * l * nn);
                        ops.down[n - 1].apply(&gpu, gpu.a(&mods[&(d, n - 1)]), ops.n[n - 1], r0 * l, gpu.a(&y), nn, false);
                        mods.insert((d, n), y);
                    }
                    mods.insert((n, n), a);
                    // Moduli of level n - 1 are no longer read.
                    if n > 0 {
                        for d in 0..n {
                            mods.remove(&(d, n - 1));
                        }
                    }
                }
                let core = ops.core_arg(&gpu, n);
                let mut s: Vec<Option<CudaSlice<f64>>> = (0..n_slots).map(|_| None).collect();
                let mc = gpu.zeros::<f64>(r0 * nc);
                self.copy_rows(&maps[n], nn, core, &mc, nc, NULL, nc, r0, false);
                s[0] = Some(mc);
                let zc = gpu.zeros::<f64>(r0 * l * 2 * nc);
                self.copy_rows(&zn, nn, core, &zc, nc, NULL, nc, r0 * l * 2, false);
                s[1] = Some(zc);
                for &(d, _) in modc.iter().filter(|p| p.1 == n) {
                    let m = gpu.zeros::<f64>(r0 * l * nc);
                    self.copy_rows(&mods[&(d, n)], nn, core, &m, nc, NULL, nc, r0 * l, false);
                    s[self.mod_slot(d) as usize] = Some(m);
                }
                for &(d, _) in pairs.iter().filter(|p| p.1 == n) {
                    let b = gpu.zeros::<f64>(r0 * l * l * 2 * nc);
                    ops.core_conv(n).forward(&gpu, &self.ker, gpu.a(&mods[&(d, n)]), nn, r0 * l, 0, nc, NULL, gpu.a(&b), nc);
                    s[self.branch_slot(d) as usize] = Some(b);
                }
                slots.push(s);
                z.push(zn);
            }
            groups.push(GroupState { key, members: mem, z, slots });
        }
        self.tick("forward_fields", t);
        // Term sums, one launch per level.
        let t = Instant::now();
        let np = plan.len();
        let rows = gpu.upload(&self.slot_rows());
        let patches = gpu.upload(&self.patch_list(&groups));
        let out = gpu.zeros::<f64>(p_count * np * 3);
        for n in 0..levels {
            let terms: Vec<DiscTerm> = plan
                .iter()
                .enumerate()
                .filter(|(_, c)| c.native_dg == n)
                .map(|(i, c)| {
                    let kind = match c.family {
                        Family::Mean => F_MEAN,
                        Family::Var => F_SQ,
                        Family::S1 if c.c1 == c.c2 => F_ABS,
                        Family::S2 if c.c1 == c.c2 => F_SQ,
                        Family::S1 => F_S1X,
                        _ => F_PROD,
                    };
                    let (a, b) = self.operands(c);
                    DiscTerm { kind, index: i as i32, a, b }
                })
                .collect();
            if terms.is_empty() {
                continue;
            }
            let (base, ncs) = self.level_tables(&groups, n);
            let terms_d = gpu.upload(&terms);
            gpu.launch(
                "disc_terms",
                (terms.len() as u32, p_count as u32),
                T as u32,
                &[gpu.a(&terms_d), gpu.a(&patches), gpu.a(&base), gpu.a(&rows), (n_slots as i32).into(), gpu.a(&ncs), np.into(), (moments as i32).into(), gpu.a(&out)],
            );
        }
        let h = gpu.download(&out);
        self.tick("forward_terms", t);
        let mut values = vec![Complex64::new(0.0, 0.0); p_count * np];
        let mut var = vec![0.0; if moments { p_count * np } else { 0 }];
        for (p, (_, s)) in discs.iter().enumerate() {
            for (i, coeff) in plan.iter().enumerate() {
                let nc = s.core[coeff.native_dg].len() as f64;
                let o = &h[(p * np + i) * 3..][..3];
                let v = Complex64::new(o[0], o[1]) / nc;
                values[p * np + i] = v;
                if moments {
                    var[p * np + i] = (o[2] / nc - v.norm_sqr()).max(0.0);
                }
            }
        }
        self.state = Some(State { groups, sizes });
        self.tick("forward", started);
        (values, var)
    }

    fn patch_list(&self, groups: &[GroupState]) -> Vec<DiscPatch> {
        let mut v = Vec::new();
        for (g, gs) in groups.iter().enumerate() {
            for (m, &p) in gs.members.iter().enumerate() {
                v.push(DiscPatch { group: g as i32, member: m as i32, out: p as i32, pad: 0 });
            }
        }
        v
    }

    /// Slot addresses `[groups, slots]` of the forward fields and core sizes `[groups]` of level `n`.
    fn level_tables(&self, groups: &[GroupState], n: usize) -> (CudaSlice<u64>, CudaSlice<i64>) {
        let mut base = Vec::new();
        let mut ncs = Vec::new();
        for gs in groups {
            for s in &gs.slots[n] {
                base.push(s.as_ref().map_or(0, |b| self.gpu.ptr(b)));
            }
            ncs.push(self.cache[&gs.key].nc[n] as i64);
        }
        (self.gpu.upload(&base), self.gpu.upload(&ncs))
    }

    /// Patch cotangents (the layout of the input of the forward) of
    /// `Re sum(conj(cotangent) * values)`, `cotangent` `[P, len(plan)]`,
    /// after [`forward`](Self::forward) of the same plan.
    pub fn reverse(&mut self, plan: &[Coefficient], cotangent: &[Complex64]) -> Vec<f64> {
        let state = self.state.as_ref().expect("no forward state");
        let mut out = vec![0.0; self.channels * state.sizes.iter().sum::<usize>()];
        self.reverse_into(plan, cotangent, Patches::Host(&[]), Some(&mut out));
        out
    }

    /// [`reverse`](Self::reverse) writing the patch cotangents to the
    /// device address `out` (the layout of the forward input); synchronises
    /// the stream.
    pub fn reverse_device(&mut self, plan: &[Coefficient], cotangent: &[Complex64], out: u64) {
        self.reverse_into(plan, cotangent, Patches::Device(out), None);
        self.gpu.sync();
    }

    fn reverse_into(&mut self, plan: &[Coefficient], cotangent: &[Complex64], dest: Patches, host: Option<&mut Vec<f64>>) {
        let started = Instant::now();
        let state = self.state.take().expect("no forward state");
        self.reverse_with(&state, plan, cotangent, dest, host);
        self.state = Some(state);
        self.tick("reverse", started);
    }

    fn reverse_with(&mut self, state: &State, plan: &[Coefficient], cotangent: &[Complex64], dest: Patches, mut host: Option<&mut Vec<f64>>) {
        let p_count = state.sizes.len();
        let np = plan.len();
        assert_eq!(cotangent.len(), p_count * np, "cotangent shape mismatch");
        let gpu = self.gpu.clone();
        let (c, l, levels, n_slots) = (self.channels, self.geo.l, self.cfg.levels(), self.n_slots());
        let t = Instant::now();
        let flat: Vec<f64> = cotangent.iter().flat_map(|v| [v.re, v.im]).collect();
        let cot = gpu.upload(&flat);
        let rows = gpu.upload(&self.slot_rows());
        let patches = gpu.upload(&self.patch_list(&state.groups));
        // Gradient slots, the shape of the forward slots.
        let grads: Vec<Vec<Vec<Option<CudaSlice<f64>>>>> = state
            .groups
            .iter()
            .map(|gs| gs.slots.iter().map(|lv| lv.iter().map(|s| s.as_ref().map(|b| gpu.zeros::<f64>(b.len()))).collect()).collect())
            .collect();
        for n in 0..levels {
            // Targets and their contributions in plan order.
            let mut targets: Vec<(DiscOp, Vec<DiscTerm>)> = Vec::new();
            let mut index: HashMap<DiscOp, usize> = HashMap::new();
            for (i, co) in plan.iter().enumerate() {
                if co.native_dg != n {
                    continue;
                }
                let (a, b) = self.operands(co);
                let mut push = |t: DiscOp, kind: i32| {
                    let at = *index.entry(t).or_insert_with(|| {
                        targets.push((t, Vec::new()));
                        targets.len() - 1
                    });
                    targets[at].1.push(DiscTerm { kind, index: i as i32, a, b });
                };
                match co.family {
                    Family::Mean => push(a, R_MEAN),
                    Family::Var => push(a, R_VAR),
                    Family::S1 | Family::S2 if co.c1 == co.c2 => push(a, if co.family == Family::S1 { R_S1SAME } else { R_S2SAME }),
                    Family::S1 => {
                        push(a, R_S1X_A);
                        push(b, R_S1X_B);
                    }
                    _ => {
                        push(a, R_GA);
                        push(b, R_GB);
                    }
                }
            }
            if targets.is_empty() {
                continue;
            }
            let (mut flat_t, mut flat_c) = (Vec::new(), Vec::new());
            for (op, cs) in &targets {
                let start = flat_c.len() as u32;
                flat_c.extend_from_slice(cs);
                flat_t.push(DiscTarget { t: *op, start, end: flat_c.len() as u32 });
            }
            let (base, ncs) = self.level_tables(&state.groups, n);
            let mut gb = Vec::new();
            for g in &grads {
                for s in &g[n] {
                    gb.push(s.as_ref().map_or(0, |b| gpu.ptr(b)));
                }
            }
            let gbase = gpu.upload(&gb);
            let ncmax = state.groups.iter().map(|gs| self.cache[&gs.key].nc[n]).max().unwrap();
            let (dt, dc) = (gpu.upload(&flat_t), gpu.upload(&flat_c));
            gpu.launch3(
                "disc_terms_rev",
                (ncmax.div_ceil(T) as u32, flat_t.len() as u32, p_count as u32),
                T as u32,
                &[gpu.a(&dt), gpu.a(&dc), gpu.a(&patches), gpu.a(&base), gpu.a(&gbase), gpu.a(&rows), (n_slots as i32).into(), gpu.a(&ncs), gpu.a(&cot), np.into()],
            );
        }
        self.tick("reverse_terms", t);
        let t = Instant::now();
        let mut offsets = vec![0usize];
        for s in &state.sizes {
            offsets.push(offsets.last().unwrap() + c * s);
        }
        for (gs, gr) in state.groups.iter().zip(&grads) {
            let ops = &self.cache[&gs.key];
            let r0 = gs.members.len() * c;
            let gmaps: Vec<CudaSlice<f64>> = (0..levels).map(|n| gpu.zeros::<f64>(r0 * ops.n[n])).collect();
            let mut gmod: BTreeMap<(usize, usize), CudaSlice<f64>> = BTreeMap::new();
            for n in (0..levels).rev() {
                let (nn, nc) = (ops.n[n], ops.nc[n]);
                let core = ops.core_arg(&gpu, n);
                let gz = gpu.zeros::<f64>(r0 * l * 2 * nn);
                let slots = &gr[n];
                if let Some(g) = &slots[1] {
                    self.copy_rows(g, nc, NULL, &gz, nn, core, nc, r0 * l * 2, true);
                }
                if let Some(g) = &slots[0] {
                    self.copy_rows(g, nc, NULL, &gmaps[n], nn, core, nc, r0, true);
                }
                for d in 0..levels {
                    if let Some(g) = &slots[self.mod_slot(d) as usize] {
                        let dst = gmod.entry((d, n)).or_insert_with(|| gpu.zeros::<f64>(r0 * l * nn));
                        self.copy_rows(g, nc, NULL, dst, nn, core, nc, r0 * l, true);
                    }
                }
                // Second wavelets on the core -> moduli of level n.
                for d in 0..levels {
                    if let Some(g) = &slots[self.branch_slot(d) as usize] {
                        let dst = gmod.entry((d, n)).or_insert_with(|| gpu.zeros::<f64>(r0 * l * nn));
                        ops.core_conv(n).adjoint(&gpu, &self.ker, gpu.a(g), nc, r0 * l, 0, nc, NULL, gpu.a(dst), nn, true);
                    }
                }
                // Stored moduli of finer levels: one decimation back.
                for d in 0..n {
                    if let Some(g) = gmod.remove(&(d, n)) {
                        let dst = gmod.entry((d, n - 1)).or_insert_with(|| gpu.zeros::<f64>(r0 * l * ops.n[n - 1]));
                        ops.up[n - 1].apply(&gpu, gpu.a(&g), nn, r0 * l, gpu.a(dst), ops.n[n - 1], true);
                    }
                }
                if let Some(g) = gmod.remove(&(n, n)) {
                    gpu.launch1(
                        "modvjp_add",
                        nn * r0 * l,
                        &[
                            gpu.a(&gs.z[n]),
                            nn.into(),
                            0usize.into(),
                            NULL,
                            gpu.a(&g),
                            nn.into(),
                            0usize.into(),
                            NULL,
                            gpu.a(&gz),
                            nn.into(),
                            0usize.into(),
                            nn.into(),
                            ((r0 * l) as i32).into(),
                        ],
                    );
                }
                ops.conv[n].adjoint(&gpu, &self.ker, gpu.a(&gz), nn, r0, 0, nn, NULL, gpu.a(&gmaps[n]), nn, true);
            }
            for n in (1..levels).rev() {
                ops.up[n - 1].apply(&gpu, gpu.a(&gmaps[n]), ops.n[n], r0, gpu.a(&gmaps[n - 1]), ops.n[n - 1], true);
            }
            let n0 = ops.n[0];
            match (dest, host.as_deref_mut()) {
                (Patches::Device(ptr), _) => {
                    for (m, &p) in gs.members.iter().enumerate() {
                        let len = c * n0;
                        let dst = Arg::Ptr(ptr + 8 * offsets[p] as u64);
                        gpu.launch1("gather_copy", len, &[gpu.a(&gmaps[0]), 0usize.into(), (m * len).into(), NULL, dst, 0usize.into(), 0usize.into(), NULL, len.into(), 1i32.into()]);
                    }
                }
                (Patches::Host(_), Some(out)) => {
                    let h = gpu.download(&gmaps[0]);
                    for (m, &p) in gs.members.iter().enumerate() {
                        out[offsets[p]..offsets[p + 1]].copy_from_slice(&h[m * c * n0..(m + 1) * c * n0]);
                    }
                }
                (Patches::Host(_), None) => unreachable!("host cotangents without a destination"),
            }
        }
        self.tick("reverse_chain", t);
    }
}
