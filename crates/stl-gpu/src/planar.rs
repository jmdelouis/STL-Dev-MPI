//! The planar block-wise engine (`stl_core::planar::PlanarEngine`) on one
//! CUDA device, for a batch of images at once (the images are planes of
//! every kernel, with the channels and orientations).
//!
//! Same sweeps, windows and pieces as the CPU engine: tiles of each level,
//! `z_n` on `a1 = F + 2` straight from the level map (periodic indices in
//! the kernel), stored moduli `m[d][n]` read on `a1`, second wavelets on
//! `F`, partial sums by `coef_partial` (the kernel of the HEALPix engine),
//! pieces of `|z_n| 1_F` pushed to the coarser levels; in reverse,
//! `coef_reverse`, the adjoints of the second and first wavelets, the
//! transposed pieces pulled from `gm[n][n']` and the map cotangents chained
//! back to level 0.  All kernels are gathers (`planar.cu`): the results do
//! not depend on the launch configuration.
//!
//! Two uses of the tile size: one tile per level (`tile >= H, W`) keeps
//! whole-level fields on the device (fastest), small tiles bound the device
//! memory by the maps, the stored moduli and one tile's fields.

use crate::device::{Arg, Gpu};
use crate::engine::{Contrib, ForwardItem, Operand, Target, CHUNK, F_MEAN, F_PROD, F_S1X, F_SQ, NONE, R_GA, R_GB, R_MEAN, R_S1X_A, R_S1X_B, R_VAR};
use cudarc::driver::CudaSlice;
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use stl_core::plan::{Coefficient, Family};
use stl_core::planar::grid::Window;
use stl_core::planar::kernels::{wavelet, L, SMOOTH};
use stl_core::planar::PlanarEngine;

type C64 = Complex64;
const F64: u64 = 8;
/// Convolution halo.
const HC: usize = 2;

/// Statistics of a batch on the device, for [`GpuPlanar::reverse`].
pub struct GpuPyramid {
    /// `maps[n]`: `[B C, H_n, W_n]`.
    pub maps: Vec<CudaSlice<f64>>,
    /// `m[d][n - d - 1]`: `[B C L, H_n, W_n]`.
    pub m: Vec<Vec<CudaSlice<f64>>>,
}

/// Operand of a statistic on a tile (`stl_core::planar::engine`).
#[derive(Clone, Copy, Debug)]
enum Op {
    X(usize),
    Z(usize, usize),
    M(usize, usize, usize),
    B(usize, usize, usize, usize),
}

/// Fields of one tile of level `n` for the whole batch.
struct Tile {
    n: usize,
    f: Window,
    a1: Window,
    /// `[2, P L, |a1|]` (re then im).
    z: CudaSlice<f64>,
    /// Restrictions to `F`.
    x: CudaSlice<f64>,
    zf: CudaSlice<f64>,
    mf: Vec<CudaSlice<f64>>,
    /// Second wavelets on `F`: `[2, P L L, |F|]`.
    br: Vec<CudaSlice<f64>>,
}

/// Cotangents on `F` of the tile operands.
struct Grads {
    x: CudaSlice<f64>,
    z: CudaSlice<f64>,
    m: Vec<CudaSlice<f64>>,
    br: Vec<CudaSlice<f64>>,
}

/// The planar engine on one device: the sweeps of `cpu` for `batch` images
/// at once.
pub struct GpuPlanar {
    /// The device.
    pub gpu: Arc<Gpu>,
    /// The CPU engine of the same operator (plan, tiles, level sizes).
    pub cpu: PlanarEngine,
    /// Images evaluated together (`B`).
    pub batch: usize,
    ranges: Vec<Range<usize>>,
    wre: CudaSlice<f64>,
    wim: CudaSlice<f64>,
    smooth: CudaSlice<f64>,
    /// Largest device memory in use seen during the sweeps (all processes), bytes.
    pub peak_used: AtomicUsize,
}

impl GpuPlanar {
    /// The engine of `cpu` on `gpu` for `batch` images: uploads the wavelet and
    /// smoothing kernels and the coefficient ranges of every native level.
    pub fn new(gpu: Arc<Gpu>, cpu: PlanarEngine, batch: usize) -> Self {
        let plan = cpu.plan();
        let ranges = (0..cpu.levels())
            .map(|n| plan.partition_point(|c| c.native_dg < n)..plan.partition_point(|c| c.native_dg <= n))
            .collect();
        let w = wavelet();
        let wre: Vec<f64> = w.iter().flat_map(|k| k.iter().map(|v| v.re)).collect();
        let wim: Vec<f64> = w.iter().flat_map(|k| k.iter().map(|v| v.im)).collect();
        let s: Vec<f64> = SMOOTH.iter().flatten().copied().collect();
        GpuPlanar { wre: gpu.upload(&wre), wim: gpu.upload(&wim), smooth: gpu.upload(&s), gpu, cpu, batch, ranges, peak_used: AtomicUsize::new(0) }
    }

    /// The coefficient plan (that of `cpu`).
    pub fn plan(&self) -> &[Coefficient] {
        self.cpu.plan()
    }

    fn planes(&self) -> usize {
        self.batch * self.cpu.channels
    }

    fn note_memory(&self) {
        let (free, total) = self.gpu.memory();
        self.peak_used.fetch_max(total - free, Ordering::Relaxed);
    }

    fn p(&self, s: &CudaSlice<f64>, off: usize) -> u64 {
        self.gpu.ptr(s) + off as u64 * F64
    }

    fn read(&self, src: &CudaSlice<f64>, planes: usize, win: &Window) -> CudaSlice<f64> {
        let out = self.gpu.zeros::<f64>(planes * win.len());
        let (h, w) = win.level();
        let g = &self.gpu;
        g.launch1(
            "pl_read",
            planes * win.len(),
            &[g.a(src), h.into(), w.into(), planes.into(), win.rows.start.into(), win.cols.start.into(), win.rows.len.into(), win.cols.len.into(), g.a(&out)],
        );
        out
    }

    fn accumulate(&self, dst: &CudaSlice<f64>, planes: usize, win: &Window, src: &CudaSlice<f64>) {
        let (h, w) = win.level();
        let n = planes * win.rows.len.min(h) * win.cols.len.min(w);
        let g = &self.gpu;
        g.launch1(
            "pl_accum",
            n,
            &[g.a(dst), h.into(), w.into(), planes.into(), win.rows.start.into(), win.cols.start.into(), win.rows.len.into(), win.cols.len.into(), g.a(src)],
        );
    }

    fn crop(&self, src: &CudaSlice<f64>, planes: usize, outer: &Window) -> CudaSlice<f64> {
        let (sh, sw) = (outer.rows.len, outer.cols.len);
        let out = self.gpu.zeros::<f64>(planes * (sh - 2 * HC) * (sw - 2 * HC));
        let g = &self.gpu;
        g.launch1("pl_crop", out.len(), &[g.a(src), planes.into(), sh.into(), sw.into(), HC.into(), g.a(&out)]);
        out
    }

    fn embed_add(&self, src: &CudaSlice<f64>, planes: usize, inner: &Window, dst: &CudaSlice<f64>) {
        let g = &self.gpu;
        g.launch1("pl_embed_add", src.len(), &[g.a(src), planes.into(), inner.rows.len.into(), inner.cols.len.into(), HC.into(), g.a(dst)]);
    }

    fn add(&self, dst: &CudaSlice<f64>, src: &CudaSlice<f64>) {
        assert_eq!(dst.len(), src.len());
        let g = &self.gpu;
        g.launch1("pl_add", dst.len(), &[g.a(dst), g.a(src), dst.len().into()]);
    }

    /// `[2, planes L, oh, ow]` wavelet outputs of `planes` real planes of
    /// `sh x sw` from the origin `(r0, c0)` (periodic indices).
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, src: &CudaSlice<f64>, sh: usize, sw: usize, r0: i64, c0: i64, planes: usize, oh: usize, ow: usize) -> CudaSlice<f64> {
        let n = planes * L * oh * ow;
        let out = self.gpu.zeros::<f64>(2 * n);
        let g = &self.gpu;
        g.launch1(
            "pl_conv",
            planes * oh * ow,
            &[g.a(src), sh.into(), sw.into(), r0.into(), c0.into(), planes.into(), g.a(&self.wre), g.a(&self.wim), (L as i32).into(), oh.into(), ow.into(), g.a(&out), n.into()],
        );
        out
    }

    /// Adjoint of [`Self::conv`] on a local source: `dst` `[planes, (oh + 4) (ow + 4)]`.
    fn conv_adjoint(&self, gz: &CudaSlice<f64>, planes: usize, oh: usize, ow: usize, dst: &CudaSlice<f64>, accumulate: bool) {
        let g = &self.gpu;
        let im = planes * L * oh * ow;
        g.launch1(
            "pl_conv_adj",
            planes * (oh + 2 * HC) * (ow + 2 * HC),
            &[g.a(gz), im.into(), planes.into(), (L as i32).into(), oh.into(), ow.into(), g.a(&self.wre), g.a(&self.wim), g.a(dst), (accumulate as i32).into()],
        );
    }

    fn window_args(fine: &Window, coarse: &Window) -> [Arg; 10] {
        [
            fine.rows.start.into(),
            fine.rows.len.into(),
            fine.rows.size.into(),
            fine.cols.start.into(),
            fine.cols.len.into(),
            fine.cols.size.into(),
            coarse.rows.start.into(),
            coarse.rows.len.into(),
            coarse.cols.start.into(),
            coarse.cols.len.into(),
        ]
    }

    /// Decimation of a field on `fine` (zero elsewhere) onto `fine.coarse()`.
    fn decimate(&self, x: &CudaSlice<f64>, planes: usize, fine: &Window) -> (Window, CudaSlice<f64>) {
        let coarse = fine.coarse();
        let y = self.gpu.zeros::<f64>(planes * coarse.len());
        let g = &self.gpu;
        let mut args = vec![g.a(x), planes.into()];
        args.extend(Self::window_args(fine, &coarse));
        args.extend([g.a(&self.smooth), g.a(&y)]);
        g.launch1("pl_dec", y.len(), &args);
        (coarse, y)
    }

    fn decimate_adjoint(&self, gy: &CudaSlice<f64>, planes: usize, fine: &Window) -> CudaSlice<f64> {
        let coarse = fine.coarse();
        let x = self.gpu.zeros::<f64>(planes * fine.len());
        let g = &self.gpu;
        let mut args = vec![g.a(gy), planes.into()];
        args.extend(Self::window_args(fine, &coarse));
        args.extend([g.a(&self.smooth), g.a(&x)]);
        g.launch1("pl_dec_adj", x.len(), &args);
        x
    }

    /// `maps[n] = D^n x` on the device, `x` `[B, C, H, W]`.
    pub fn maps(&self, x: &CudaSlice<f64>) -> Vec<CudaSlice<f64>> {
        let p = self.planes();
        let mut maps = vec![self.gpu.copy(x)];
        for n in 1..self.cpu.levels() {
            let (h, w) = self.cpu.size(n - 1);
            let y = self.decimate(&maps[n - 1], p, &Window::full(h, w)).1;
            maps.push(y);
        }
        maps
    }

    fn tile(&self, n: usize, f: &Window, map: &CudaSlice<f64>, m: &[Vec<CudaSlice<f64>>]) -> Tile {
        let (p, pl) = (self.planes(), self.planes() * L);
        let (h, w) = self.cpu.size(n);
        let a1 = f.grow(HC);
        let (ah, aw) = (a1.rows.len, a1.cols.len);
        let z = self.conv(map, h, w, a1.rows.start, a1.cols.start, p, ah, aw);
        let mut ma: Vec<CudaSlice<f64>> = (0..n).map(|d| self.read(&m[d][n - d - 1], pl, &a1)).collect();
        let own = self.gpu.zeros::<f64>(pl * a1.len());
        let g = &self.gpu;
        g.launch1("pl_abs", own.len(), &[g.a(&z), own.len().into(), own.len().into(), g.a(&own)]);
        ma.push(own);
        let br = ma.iter().map(|md| self.conv(md, ah, aw, HC as i64, HC as i64, pl, f.rows.len, f.cols.len)).collect();
        Tile {
            n,
            f: *f,
            a1,
            x: self.read(map, p, f),
            zf: self.crop(&z, 2 * pl, &a1),
            mf: ma.iter().map(|v| self.crop(v, pl, &a1)).collect(),
            z,
            br,
        }
    }

    /// Operands of a coefficient (CPU `PlanarEngine::operands`, S1 cross as one item).
    fn operands(&self, c: &Coefficient) -> (i32, Op, Op) {
        let n = c.native_dg;
        let u = |o: Option<usize>| o.unwrap();
        let fewer = self.cpu.fewer;
        match c.family {
            Family::Mean => (F_MEAN, Op::X(c.c1), Op::X(c.c1)),
            Family::Var => (F_SQ, Op::X(c.c1), Op::X(c.c1)),
            Family::S1 if c.c1 == c.c2 => (F_MEAN, Op::M(n, c.c1, u(c.l1)), Op::M(n, c.c1, u(c.l1))),
            Family::S1 => (F_S1X, Op::Z(c.c1, u(c.l1)), Op::Z(c.c2, u(c.l1))),
            Family::S2 if c.c1 == c.c2 => (F_SQ, Op::M(n, c.c1, u(c.l1)), Op::M(n, c.c1, u(c.l1))),
            Family::S2 => (F_PROD, Op::Z(c.c1, u(c.l1)), Op::Z(c.c2, u(c.l1))),
            Family::S3 if fewer => (F_PROD, Op::X(c.c1), Op::B(u(c.j1), c.c2, u(c.l1), u(c.l3))),
            Family::S3 => (F_PROD, Op::B(u(c.j1), c.c1, u(c.l1), u(c.l3)), Op::Z(c.c2, u(c.l3))),
            Family::S4 if fewer => (F_PROD, Op::M(u(c.j1), c.c1, u(c.l1)), Op::B(u(c.j2), c.c2, u(c.l2), u(c.l3))),
            Family::S4 => (F_PROD, Op::B(u(c.j1), c.c1, u(c.l1), u(c.l3)), Op::B(u(c.j2), c.c2, u(c.l2), u(c.l3))),
        }
    }

    /// Operand of image `b` on the tile fields (`grads`: on their cotangents).
    fn operand(&self, t: &Tile, grads: Option<&Grads>, b: usize, op: Op) -> Operand {
        let (ch, fl) = (self.cpu.channels, t.f.len());
        let pl = self.planes() * L;
        match (op, grads) {
            (Op::X(c), None) => Operand { ptr: self.p(&t.x, (b * ch + c) * fl), im_off: -1 },
            (Op::X(c), Some(g)) => Operand { ptr: self.p(&g.x, (b * ch + c) * fl), im_off: -1 },
            (Op::Z(c, l), None) => Operand { ptr: self.p(&t.zf, ((b * ch + c) * L + l) * fl), im_off: (pl * fl) as i64 },
            (Op::Z(c, l), Some(g)) => Operand { ptr: self.p(&g.z, ((b * ch + c) * L + l) * fl), im_off: (pl * fl) as i64 },
            (Op::M(d, c, l), None) => Operand { ptr: self.p(&t.mf[d], ((b * ch + c) * L + l) * fl), im_off: -1 },
            (Op::M(d, c, l), Some(g)) => Operand { ptr: self.p(&g.m[d], ((b * ch + c) * L + l) * fl), im_off: -1 },
            (Op::B(d, c, l1, l3), None) => Operand { ptr: self.p(&t.br[d], (((b * ch + c) * L + l1) * L + l3) * fl), im_off: (pl * L * fl) as i64 },
            (Op::B(d, c, l1, l3), Some(g)) => Operand { ptr: self.p(&g.br[d], (((b * ch + c) * L + l1) * L + l3) * fl), im_off: (pl * L * fl) as i64 },
        }
    }

    /// `(sum re, sum im)` of the items over `n` pixels (chunk sums added in order).
    fn partial_sums(&self, items: &[ForwardItem], n: usize) -> Vec<(f64, f64)> {
        let n_chunks = n.div_ceil(CHUNK);
        let mut out = Vec::with_capacity(items.len());
        for part in items.chunks(65535) {
            let dev = self.gpu.upload(part);
            let buf = self.gpu.zeros::<f64>(part.len() * n_chunks * 3);
            self.gpu.launch("coef_partial", (n_chunks as u32, part.len() as u32), 256, &[self.gpu.a(&dev), n.into(), CHUNK.into(), 0i32.into(), self.gpu.a(&buf)]);
            let h = self.gpu.download(&buf);
            for k in 0..part.len() {
                let (mut re, mut im) = (0.0, 0.0);
                for c in 0..n_chunks {
                    re += h[(k * n_chunks + c) * 3];
                    im += h[(k * n_chunks + c) * 3 + 1];
                }
                out.push((re, im));
            }
        }
        out
    }

    /// Statistics `[B, len(plan)]` of the batch `x` `[B, C, H, W]` on the
    /// device, and the pyramid for the reverse sweep.
    pub fn forward(&self, x: &CudaSlice<f64>) -> (Vec<C64>, GpuPyramid) {
        let levels = self.cpu.levels();
        let pl = self.planes() * L;
        let np = self.plan().len();
        let maps = self.maps(x);
        let m: Vec<Vec<CudaSlice<f64>>> = (0..levels).map(|d| (d + 1..levels).map(|n| self.gpu.zeros::<f64>(pl * self.pixels(n))).collect()).collect();
        let mut stats = vec![C64::default(); self.batch * np];
        for n in 0..levels {
            let norm = self.pixels(n) as f64;
            for f in self.cpu.tiles(n) {
                let t = self.tile(n, &f, &maps[n], &m);
                let mut items = Vec::new();
                for b in 0..self.batch {
                    for i in self.ranges[n].clone() {
                        let (kind, a, bb) = self.operands(&self.plan()[i]);
                        items.push(ForwardItem { kind, pad: 0, a: self.operand(&t, None, b, a), b: if kind == F_MEAN || kind == F_SQ { NONE } else { self.operand(&t, None, b, bb) }, mask: 0 });
                    }
                }
                let sums = self.partial_sums(&items, f.len());
                let mut k = 0;
                for b in 0..self.batch {
                    for i in self.ranges[n].clone() {
                        stats[b * np + i] += C64::new(sums[k].0, sums[k].1) / norm;
                        k += 1;
                    }
                }
                // Push |z_n| on F to the coarser levels.
                let mut win = f;
                let mut piece: Option<CudaSlice<f64>> = None;
                for k in n + 1..levels {
                    let (next, y) = self.decimate(piece.as_ref().unwrap_or(&t.mf[n]), pl, &win);
                    self.accumulate(&m[n][k - n - 1], pl, &next, &y);
                    win = next;
                    piece = Some(y);
                }
                self.note_memory();
            }
        }
        (stats, GpuPyramid { maps, m })
    }

    fn pixels(&self, n: usize) -> usize {
        let (h, w) = self.cpu.size(n);
        h * w
    }

    /// Cotangent on `F` of `|z_n|` from `gm[n][*]` (complete).
    fn pull(&self, t: &Tile, gm_n: &[CudaSlice<f64>]) -> Option<CudaSlice<f64>> {
        let (n, top) = (t.n, self.cpu.levels() - 1);
        if n == top {
            return None;
        }
        let pl = self.planes() * L;
        let mut wins = vec![t.f];
        for _ in n..top {
            let next = wins.last().unwrap().coarse();
            wins.push(next);
        }
        let mut g = self.read(&gm_n[top - n - 1], pl, &wins[top - n]);
        for k in (n..top).rev() {
            let back = self.decimate_adjoint(&g, pl, &wins[k - n]);
            g = if k > n {
                let own = self.read(&gm_n[k - n - 1], pl, &wins[k - n]);
                self.add(&own, &back);
                own
            } else {
                back
            };
        }
        Some(g)
    }

    /// Map cotangents `[B, C, H, W]` (on the device) of `Re sum conj(cot) S`,
    /// `cot` `[B, len(plan)]`.
    pub fn reverse(&self, pyr: &GpuPyramid, cot: &[C64]) -> CudaSlice<f64> {
        let levels = self.cpu.levels();
        let (p, pl) = (self.planes(), self.planes() * L);
        let np = self.plan().len();
        assert_eq!(cot.len(), self.batch * np);
        let gm: Vec<Vec<CudaSlice<f64>>> = (0..levels).map(|d| (d + 1..levels).map(|n| self.gpu.zeros::<f64>(pl * self.pixels(n))).collect()).collect();
        let gmaps: Vec<CudaSlice<f64>> = (0..levels).map(|n| self.gpu.zeros::<f64>(p * self.pixels(n))).collect();
        for n in (0..levels).rev() {
            let norm = self.pixels(n) as f64;
            for f in self.cpu.tiles(n) {
                let t = self.tile(n, &f, &pyr.maps[n], &pyr.m);
                let fl = f.len();
                let gr = Grads {
                    x: self.gpu.zeros::<f64>(p * fl),
                    z: self.gpu.zeros::<f64>(2 * pl * fl),
                    m: (0..=n).map(|_| self.gpu.zeros::<f64>(pl * fl)).collect(),
                    br: (0..=n).map(|_| self.gpu.zeros::<f64>(2 * pl * L * fl)).collect(),
                };
                self.coefficient_cotangents(&t, &gr, cot, norm);
                let (fh, fw) = (f.rows.len, f.cols.len);
                let (ah, aw) = (t.a1.rows.len, t.a1.cols.len);
                let mut gmod = None;
                for d in 0..=n {
                    let ga = self.gpu.zeros::<f64>(pl * t.a1.len());
                    self.embed_add(&gr.m[d], pl, &f, &ga);
                    self.conv_adjoint(&gr.br[d], pl, fh, fw, &ga, true);
                    if d < n {
                        self.accumulate(&gm[d][n - d - 1], pl, &t.a1, &ga);
                    } else {
                        gmod = Some(ga);
                    }
                }
                let gmod = gmod.unwrap();
                if let Some(pull) = self.pull(&t, &gm[n]) {
                    self.embed_add(&pull, pl, &f, &gmod);
                }
                let gza = self.gpu.zeros::<f64>(2 * pl * t.a1.len());
                self.embed_add(&gr.z, 2 * pl, &f, &gza);
                let na = pl * t.a1.len();
                let g = &self.gpu;
                g.launch1("pl_modvjp", na, &[g.a(&t.z), na.into(), g.a(&gmod), na.into(), g.a(&gza), na.into()]);
                let dom = t.a1.grow(HC);
                let gxd = self.gpu.zeros::<f64>(p * dom.len());
                self.conv_adjoint(&gza, p, ah, aw, &gxd, false);
                self.accumulate(&gmaps[n], p, &dom, &gxd);
                self.accumulate(&gmaps[n], p, &f, &gr.x);
                self.note_memory();
            }
        }
        let mut gmaps = gmaps;
        let mut g = gmaps.pop().unwrap();
        for n in (0..levels - 1).rev() {
            let (h, w) = self.cpu.size(n);
            let back = self.decimate_adjoint(&g, p, &Window::full(h, w));
            g = gmaps.pop().unwrap();
            self.add(&g, &back);
        }
        g
    }

    fn coefficient_cotangents(&self, t: &Tile, gr: &Grads, cot: &[C64], norm: f64) {
        let np = self.plan().len();
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut targets: Vec<(Operand, Vec<Contrib>)> = Vec::new();
        let mut push = |target: Operand, kind: i32, a: Operand, b: Operand, g: C64| {
            let k = *index.entry(target.ptr).or_insert_with(|| {
                targets.push((target, Vec::new()));
                targets.len() - 1
            });
            targets[k].1.push(Contrib { kind, pad: 0, a, b, g_re: g.re, g_im: g.im, inv_n: 1.0 / norm, mask: 0 });
        };
        for b in 0..self.batch {
            for i in self.ranges[t.n].clone() {
                let g = cot[b * np + i];
                if g == C64::default() {
                    continue;
                }
                let (kind, oa, ob) = self.operands(&self.plan()[i]);
                let a = self.operand(t, None, b, oa);
                let bv = self.operand(t, None, b, ob);
                let ga = self.operand(t, Some(gr), b, oa);
                let gb = self.operand(t, Some(gr), b, ob);
                match kind {
                    F_MEAN => push(ga, R_MEAN, a, NONE, g),
                    F_SQ => push(ga, R_VAR, a, NONE, g),
                    F_S1X => {
                        push(ga, R_S1X_A, a, bv, g);
                        push(gb, R_S1X_B, a, bv, g);
                    }
                    _ => {
                        push(ga, R_GA, a, bv, g);
                        push(gb, R_GB, a, bv, g);
                    }
                }
            }
        }
        if targets.is_empty() {
            return;
        }
        let mut flat_t = Vec::with_capacity(targets.len());
        let mut flat_c = Vec::new();
        for (op, cs) in &targets {
            let start = flat_c.len() as u32;
            flat_c.extend_from_slice(cs);
            flat_t.push(Target { ptr: op.ptr, im_off: op.im_off, start, end: flat_c.len() as u32 });
        }
        let fl = t.f.len();
        let dc = self.gpu.upload(&flat_c);
        for part in flat_t.chunks(65535) {
            let dt = self.gpu.upload(part);
            self.gpu.launch("coef_reverse", (fl.div_ceil(256) as u32, part.len() as u32), 256, &[self.gpu.a(&dt), self.gpu.a(&dc), fl.into()]);
        }
    }

    /// Host convenience: statistics of `x` `[B, C, H, W]`.
    pub fn statistics(&self, x: &[f64]) -> Vec<C64> {
        self.forward(&self.gpu.upload(x)).0
    }

    /// Host convenience: statistics and the map cotangent of `Re sum conj(cot) S`.
    pub fn gradient(&self, x: &[f64], cot: &[C64]) -> (Vec<C64>, Vec<f64>) {
        let (s, pyr) = self.forward(&self.gpu.upload(x));
        let g = self.reverse(&pyr, cot);
        (s, self.gpu.download(&g))
    }
}

// ---------------------------------------------------------------------------
// Fourier terms on the device: spectrum in Gaussian rings and Minkowski
// functionals (stl_core::planar::spectrum and the Minkowski block of stl-cli).
// ---------------------------------------------------------------------------

use crate::cufft::{cufft, CuFft, Handle, FORWARD, INVERSE};
use crate::harmonic::pixel_sum;
use std::sync::Mutex;
use stl_core::planar::spectrum::{GaussianRings, CUT};

/// Unnormalised complex 2D FFT of `P` interleaved planes `H x W`: rows,
/// transposition, columns, transposition (cuFFT plans per plane count).
pub struct GpuFft2 {
    gpu: Arc<Gpu>,
    /// Rows `H` of a plane.
    pub h: usize,
    /// Columns `W` of a plane.
    pub w: usize,
    fft: &'static CuFft,
    plans: Mutex<HashMap<usize, (Handle, Handle, CudaSlice<u8>)>>,
}

impl GpuFft2 {
    /// The transform of `H x W` planes; fails if cuFFT does not load.  Plans are
    /// made at the first use of each plane count.
    pub fn new(gpu: Arc<Gpu>, h: usize, w: usize) -> Result<Self, String> {
        Ok(GpuFft2 { gpu, h, w, fft: cufft()?, plans: Mutex::new(HashMap::new()) })
    }

    fn plans(&self, planes: usize) -> (Handle, Handle) {
        let mut map = self.plans.lock().unwrap();
        let e = map.entry(planes).or_insert_with(|| {
            let stream = self.gpu.stream.cu_stream() as *mut std::ffi::c_void;
            let (rows, br) = self.fft.plan_z2z(self.w, planes * self.h, self.w, stream).unwrap_or_else(|e| panic!("{e}"));
            let (cols, bc) = self.fft.plan_z2z(self.h, planes * self.w, self.h, stream).unwrap_or_else(|e| panic!("{e}"));
            let work = self.gpu.zeros::<u8>(br.max(bc).max(16));
            let wp = self.gpu.ptr(&work);
            unsafe {
                self.fft.set_work_area(rows, wp).unwrap_or_else(|e| panic!("{e}"));
                self.fft.set_work_area(cols, wp).unwrap_or_else(|e| panic!("{e}"));
            }
            (rows, cols, work)
        });
        (e.0, e.1)
    }

    /// In place, `x` interleaved `[planes, H, W]`.
    pub fn run(&self, x: &CudaSlice<f64>, planes: usize, inverse: bool) {
        let (rows, cols) = self.plans(planes);
        let dir = if inverse { INVERSE } else { FORWARD };
        let (h, w) = (self.h, self.w);
        let g = &self.gpu;
        let t = g.zeros::<f64>(x.len());
        unsafe { self.fft.exec_z2z(rows, g.ptr(x), dir) }.unwrap_or_else(|e| panic!("{e}"));
        g.launch1("pl_transpose", planes * h * w, &[g.a(x), h.into(), w.into(), planes.into(), g.a(&t)]);
        unsafe { self.fft.exec_z2z(cols, g.ptr(&t), dir) }.unwrap_or_else(|e| panic!("{e}"));
        g.launch1("pl_transpose", planes * h * w, &[g.a(&t), w.into(), h.into(), planes.into(), g.a(x)]);
    }

    /// `(x, 0)` interleaved for `n` real values at `x`.
    fn complex(&self, x: Arg, n: usize) -> CudaSlice<f64> {
        let out = self.gpu.zeros::<f64>(2 * n);
        self.gpu.launch1("pl_to_complex", n, &[x, n.into(), self.gpu.a(&out)]);
        out
    }
}

impl Drop for GpuFft2 {
    fn drop(&mut self) {
        for (_, (r, c, _)) in self.plans.lock().unwrap().drain() {
            self.fft.destroy(r);
            self.fft.destroy(c);
        }
    }
}

/// What the cotangent of the terms needs (device buffers).
pub struct TermsState {
    /// Orthonormal transforms of the planes, interleaved `[P, H, W]`.
    xs: Option<CudaSlice<f64>>,
    /// Standardised planes `[P, N]` and derivative maps `[P, 5, N]`.
    u: Option<CudaSlice<f64>>,
    d: Option<CudaSlice<f64>>,
    floor: Vec<f64>,
    sigma: Vec<f64>,
}

/// Spectrum and Minkowski functionals of a batch of `batch` images of
/// `channels` channels on the device.
pub struct GpuPlanarTerms {
    gpu: Arc<Gpu>,
    /// Images per batch (`B`).
    pub batch: usize,
    /// Channels per image (`C`).
    pub channels: usize,
    fft: GpuFft2,
    /// Ring tables: log centres, log widths, log f per pixel; bin norms on the host.
    rings: Option<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>, Vec<f64>)>,
    /// Minkowski thresholds, their count and smoothing (`None`: no functionals).
    mk: Option<(CudaSlice<f64>, usize, f64)>,
}

impl GpuPlanarTerms {
    /// The terms of `H x W` images: the spectrum on the Gaussian `rings` and the
    /// Minkowski functionals at the thresholds and smoothing `minkowski`, each if
    /// given; fails if cuFFT does not load.
    pub fn new(gpu: Arc<Gpu>, h: usize, w: usize, channels: usize, batch: usize, rings: Option<&GaussianRings>, minkowski: Option<(&[f64], f64)>) -> Result<Self, String> {
        let rings = rings.map(|r| {
            let (lc, ls, lr, norm) = r.tables();
            (gpu.upload(lc), gpu.upload(ls), gpu.upload(lr), norm.to_vec())
        });
        let mk = minkowski.map(|(nu, eps)| (gpu.upload(nu), nu.len(), eps));
        Ok(GpuPlanarTerms { fft: GpuFft2::new(gpu.clone(), h, w)?, gpu, batch, channels, rings, mk })
    }

    fn npix(&self) -> usize {
        self.fft.h * self.fft.w
    }

    /// Plane indices `(a, b)` of every image and channel pair `c1 <= c2`.
    fn pair_planes(&self) -> Vec<i32> {
        let c = self.channels;
        let pairs: Vec<(usize, usize)> = (0..c).flat_map(|a| (a..c).map(move |b| (a, b))).collect();
        (0..self.batch).flat_map(|b| pairs.iter().flat_map(move |&(p, q)| [(b * c + p) as i32, (b * c + q) as i32])).collect()
    }

    /// Spectra `[B, pairs x bins]`, Minkowski values `[B, C x 3K]` of the
    /// batch `x` `[B, C, H, W]` and the state of the cotangent.
    pub fn forward(&self, x: &CudaSlice<f64>) -> (Vec<C64>, Vec<f64>, TermsState) {
        let g = &self.gpu;
        let n = self.npix();
        let planes = self.batch * self.channels;
        let mut st = TermsState { xs: None, u: None, d: None, floor: Vec::new(), sigma: Vec::new() };
        let mut ps = Vec::new();
        if let Some((lc, ls, lr, norm)) = &self.rings {
            let xs = self.fft.complex(g.a(x), planes * n);
            self.fft.run(&xs, planes, false);
            g.launch1("pl_cscale", planes * n, &[g.a(&xs), (1.0 / (n as f64).sqrt()).into(), (planes * n).into()]);
            let idx = self.pair_planes();
            let q = idx.len() / 2;
            let nb = norm.len();
            let didx = g.upload(&idx);
            let chunks = n.div_ceil(CHUNK);
            let out = g.zeros::<f64>(q * nb * chunks * 2);
            g.launch3("pl_rings", (chunks as u32, nb as u32, q as u32), 256, &[g.a(&xs), g.a(&didx), g.a(lr), g.a(lc), g.a(ls), CUT.into(), n.into(), CHUNK.into(), g.a(&out)]);
            let h = g.download(&out);
            for qq in 0..q {
                for k in 0..nb {
                    let (mut re, mut im) = (0.0, 0.0);
                    for c in 0..chunks {
                        re += h[((qq * nb + k) * chunks + c) * 2];
                        im += h[((qq * nb + k) * chunks + c) * 2 + 1];
                    }
                    ps.push(C64::new(re, im) / norm[k]);
                }
            }
            st.xs = Some(xs);
        }
        let mut mf = Vec::new();
        if let Some((nu, k, eps)) = &self.mk {
            let k = *k;
            let u = g.zeros::<f64>(planes * n);
            for p in 0..planes {
                let at = g.at(x, p * n);
                let mean = pixel_sum(g, at, at, n, 0.0, 0) / n as f64;
                let sigma = (pixel_sum(g, at, at, n, mean, 1) / n as f64 + 1e-12).sqrt();
                g.launch1("mk_standard", n, &[at, n.into(), mean.into(), sigma.into(), g.at(&u, p * n)]);
                st.sigma.push(sigma);
            }
            let uc = self.fft.complex(g.a(&u), planes * n);
            self.fft.run(&uc, planes, false);
            let dc = g.zeros::<f64>(2 * 5 * planes * n);
            g.launch1("pl_spec_mult", planes * n, &[g.a(&uc), self.fft.h.into(), self.fft.w.into(), planes.into(), 0i32.into(), 5i32.into(), 0i32.into(), g.a(&dc)]);
            self.fft.run(&dc, 5 * planes, true);
            let d = g.zeros::<f64>(5 * planes * n);
            g.launch1("pl_real_part", 5 * planes * n, &[g.a(&dc), (1.0 / n as f64).into(), (5 * planes * n).into(), g.a(&d), 0i32.into()]);
            let tau = 2.0 * std::f64::consts::PI;
            let cn = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * eps);
            let (mag, curv) = (g.zeros::<f64>(n), g.zeros::<f64>(n));
            let chunks = n.div_ceil(CHUNK);
            for p in 0..planes {
                let dp = p * 5 * n;
                let floor = 1e-4 * pixel_sum(g, g.at(&d, dp), g.at(&d, dp + n), n, 0.0, 2) / n as f64 + 1e-12;
                g.launch1("mk_geom", n, &[g.at(&d, dp), n.into(), floor.into(), g.a(&mag), g.a(&curv)]);
                let out = g.zeros::<f64>(chunks * k * 3);
                g.launch("mk_forward", (chunks as u32, k as u32), 256, &[g.at(&u, p * n), g.a(&mag), g.a(&curv), n.into(), CHUNK.into(), g.a(nu), (*eps).into(), cn.into(), tau.into(), g.a(&out)]);
                let h = g.download(&out);
                let mut v = vec![0.0; 3 * k];
                for c in 0..chunks {
                    for t in 0..k {
                        for f in 0..3 {
                            v[f * k + t] += h[(c * k + t) * 3 + f];
                        }
                    }
                }
                mf.extend(v.iter().map(|a| a / n as f64));
                st.floor.push(floor);
            }
            st.u = Some(u);
            st.d = Some(d);
        }
        (ps, mf, st)
    }

    /// Adds into `out` `[B, C, H, W]` the map cotangent of the spectra
    /// (`gps`) and the Minkowski values (`gmf`).
    pub fn vjp(&self, st: &TermsState, gps: &[C64], gmf: &[f64], out: &CudaSlice<f64>) {
        let g = &self.gpu;
        let n = self.npix();
        let planes = self.batch * self.channels;
        if let (Some((lc, ls, lr, norm)), Some(xs)) = (&self.rings, &st.xs) {
            let nb = norm.len();
            let idx = self.pair_planes();
            let q = idx.len() / 2;
            let gs: Vec<f64> = gps.chunks(nb).flat_map(|c| c.iter().zip(norm).flat_map(|(v, m)| [v.re / m, v.im / m])).collect();
            assert_eq!(gs.len(), 2 * q * nb);
            let (didx, dg) = (g.upload(&idx), g.upload(&gs));
            let gx = g.zeros::<f64>(2 * planes * n);
            g.launch1(
                "pl_rings_vjp",
                planes * n,
                &[g.a(xs), g.a(&didx), (q as i32).into(), g.a(&dg), (nb as i32).into(), g.a(lr), g.a(lc), g.a(ls), CUT.into(), n.into(), planes.into(), g.a(&gx)],
            );
            self.fft.run(&gx, planes, true);
            g.launch1("pl_real_part", planes * n, &[g.a(&gx), (1.0 / (n as f64).sqrt()).into(), (planes * n).into(), g.a(out), 1i32.into()]);
        }
        if let (Some((nu, k, eps)), Some(u), Some(d)) = (&self.mk, &st.u, &st.d) {
            let k = *k;
            let tau = 2.0 * std::f64::consts::PI;
            let cn = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * eps);
            let gu = g.zeros::<f64>(planes * n);
            let gd = g.zeros::<f64>(5 * planes * n);
            for p in 0..planes {
                let c = g.upload(&gmf[p * 3 * k..][..3 * k]);
                g.launch1(
                    "mk_vjp",
                    n,
                    &[g.at(u, p * n), g.at(d, p * 5 * n), n.into(), st.floor[p].into(), g.a(nu), (k as i32).into(), (*eps).into(), cn.into(), tau.into(), g.a(&c), g.at(&gu, p * n), g.at(&gd, p * 5 * n)],
                );
            }
            let gdc = self.fft.complex(g.a(&gd), 5 * planes * n);
            self.fft.run(&gdc, 5 * planes, false);
            let back = g.zeros::<f64>(2 * planes * n);
            g.launch1("pl_spec_mult_sum", planes * n, &[g.a(&gdc), self.fft.h.into(), self.fft.w.into(), planes.into(), g.a(&back)]);
            self.fft.run(&back, planes, true);
            g.launch1("pl_real_part", planes * n, &[g.a(&back), (1.0 / n as f64).into(), (planes * n).into(), g.a(&gu), 1i32.into()]);
            let fin = g.zeros::<f64>(n);
            for p in 0..planes {
                let (gp, up) = (g.at(&gu, p * n), g.at(u, p * n));
                let mg = pixel_sum(g, gp, gp, n, 0.0, 0) / n as f64;
                let mu = pixel_sum(g, gp, up, n, 0.0, 3) / n as f64;
                g.launch1("mk_final", n, &[gp, up, n.into(), mg.into(), mu.into(), st.sigma[p].into(), g.a(&fin)]);
                g.launch1("pl_add", n, &[g.at(out, p * n), g.a(&fin), n.into()]);
            }
        }
    }
}
