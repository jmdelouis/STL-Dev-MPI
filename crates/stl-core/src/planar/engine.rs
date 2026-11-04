//! Complete planar ScatCov (mean, var, S1..S4) with square tiles on a
//! periodic grid: the block-wise engine of `LocalScattering.LocalEngine`
//! and `_Block` on the operators of `WaveletOperator2Dkernel_torch`
//! (`pbc=True`, `dg(j) = j`), single image.
//!
//! Forward sweep, levels ascending: each level `n` is cut into tiles `F`;
//! `z_n` is computed on `a1 = F + 2` from `maps[n]` on `F + 4`; the stored
//! moduli `m[d][n]` of the finer levels are read on `a1`; the second
//! wavelets are evaluated on `F`; partial sums are accumulated; then
//! `|z_n|` on `F` is pushed through the decimation pieces into `m[n][n']`
//! of every coarser level.
//!
//! Reverse sweep, levels descending: the tile fields are recomputed, the
//! coefficient cotangents applied, the cotangents `gm[d][n]` of the stored
//! moduli accumulated (complete once the coarser levels are done), the
//! tile's own chain reversed from `gm[n][n']`, then the modulus VJP and the
//! wavelet adjoint give the cotangent of `maps[n]`; finally the map
//! cotangents of all levels are chained back to level 0 through the
//! transposed decimations.
//!
//! Statistics (STL-Dev `ST_Operator.apply`, `norm="vanilla"`, all means
//! over the whole level since nothing is cropped with periodic
//! boundaries), for `n = j3`, `d1 = j1`, `d2 = j2`:
//!
//! ```text
//! mean(c) = <x_c>,   var(c) = <x_c^2>
//! S1 = <|z|>  (c1 = c2),   <rn(z_c1) conj rn(z_c2)>,  rn(z) = z / sqrt(|z| + 1e-8)
//! S2 = <|z|^2> (c1 = c2),  <z_c1 conj z_c2>
//! S3 = <(psi_l3 * m[d1][n]_{c1,l1}) conj z_{c2,l3}>
//!      fewer: <maps[n]_c1 conj (psi_l3 * m[d1][n]_{c2,l1})>
//! S4 = <(psi_l3 * m[d1][n]_{c1,l1}) conj (psi_l3 * m[d2][n]_{c2,l2})>
//!      fewer: <m[d1][n]_{c1,l1} conj (psi_l3 * m[d2][n]_{c2,l2})>
//! ```
//!
//! with `m[n][n] = |z_n|`.  Coefficients are in the order of
//! [`crate::plan::build_plan`].  Cotangents of complex statistics follow
//! `Re sum conj(g) dS`.
//!
//! Layouts (row-major, pixels last): maps `[C, H_n, W_n]`; stored moduli
//! `[C, L, H_n, W_n]`; tile fields `[C, L, (L,) window]`.  Summation order:
//! each partial sum adds the pixels of a tile row-major and divides by the
//! level size, tiles in row-major order, independent of the threads.

use super::grid::{self, conv_valid, conv_valid_adjoint, crop, decimate, decimate_adjoint, embed, Span, Window};
use super::kernels::{wavelet, K, L};
use crate::plan::{build_plan, Coefficient, Family};
use num_complex::Complex64;
use std::ops::Range;

type C64 = Complex64;
const ZERO: C64 = C64::new(0.0, 0.0);
/// Convolution halo, `K // 2`.
const HC: usize = K / 2;

#[inline]
fn root_normalise(z: C64) -> C64 {
    z / (z.norm() + 1e-8).sqrt()
}

/// VJP of [`root_normalise`] (`AdjointOps.root_modulus_vjp`).
fn root_modulus_vjp(z: C64, g: C64) -> C64 {
    let r = z.norm();
    let radial = (g.conj() * z).re;
    let mag = if r == 0.0 { 1.0 } else { r };
    g / (r + 1e-8).sqrt() - (0.5 * radial / (r + 1e-8).powf(1.5)) * z / mag
}

/// Planar operator: image size, channels, scales and tiling.
#[derive(Clone, Debug)]
pub struct PlanarEngine {
    /// Image height at level 0, pixels.
    pub h: usize,
    /// Image width at level 0, pixels.
    pub w: usize,
    /// Number of channels `C`.
    pub channels: usize,
    /// Number of scales `J` (levels `0 .. J`).
    pub j_count: usize,
    /// `has_fewer_convolutions` definitions of S3/S4.
    pub fewer: bool,
    /// Tile side in pixels of every level (clipped to the level).
    pub tile: usize,
    /// Host threads over the tiles of a level.
    pub threads: usize,
    plan: Vec<Coefficient>,
    ranges: Vec<Range<usize>>,
    wav: Vec<[C64; K * K]>,
}

/// What the reverse sweep needs of the forward one: the map pyramid and
/// the stored moduli `m[d][n]` (`m[d][n - d - 1]`).
pub struct Pyramid {
    /// `maps[n] = D^n x`, `[C, H_n, W_n]`.
    pub maps: Vec<Vec<f64>>,
    /// Stored moduli `m[d][n - d - 1]` (`d < n`), `[C, L, H_n, W_n]`.
    pub m: Vec<Vec<Vec<f64>>>,
}

impl Pyramid {
    /// Values held, in `f64` (maps and stored moduli).
    pub fn len(&self) -> usize {
        self.maps.iter().map(Vec::len).sum::<usize>() + self.m.iter().flatten().map(Vec::len).sum::<usize>()
    }

    /// True when nothing is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Operand of a statistic on a tile.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// `maps[n]` of a channel.
    X(usize),
    /// `z_n[c, l]`.
    Z(usize, usize),
    /// `rn(z_n[c, l])`.
    R(usize, usize),
    /// `m[d][n][c, l]` (`|z_n|` for `d = n`).
    M(usize, usize, usize),
    /// `psi_l3 * m[d][n][c, l1]`: `(d, c, l1, l3)`.
    B(usize, usize, usize, usize),
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Mean,
    Square,
    Cov,
}

enum Plane<'a> {
    Re(&'a [f64]),
    Cx(&'a [C64]),
}

/// Fields of one tile `F` of level `n`.
struct Tile {
    n: usize,
    f: Window,
    a1: Window,
    /// `z_n` on `a1`, `[C, L, |a1|]`.
    z: Vec<C64>,
    /// Restrictions to `F` (`mf[d]`: `m[d][n]`, `d <= n`).
    x: Vec<f64>,
    zf: Vec<C64>,
    rf: Vec<C64>,
    mf: Vec<Vec<f64>>,
    /// Second wavelets on `F`, `[C, L(l1), L(l3), |F|]`.
    br: Vec<Vec<C64>>,
}

/// Cotangents on `F` of the tile operands.
struct Grads {
    /// Pixels of the tile.
    fl: usize,
    x: Vec<f64>,
    z: Vec<C64>,
    r: Vec<C64>,
    m: Vec<Vec<f64>>,
    br: Vec<Vec<C64>>,
}

/// What a tile adds to the global state in the forward sweep.
struct ForwardOut {
    sums: Vec<C64>,
    /// `(level n', window, |z_n| piece)` for `m[n][n']`.
    pieces: Vec<(usize, Window, Vec<f64>)>,
}

/// What a tile adds in the reverse sweep.
struct ReverseOut {
    /// `(d, a1, values)` for `gm[d][n]`, `d < n`.
    gm: Vec<(usize, Window, Vec<f64>)>,
    /// Cotangent of `maps[n]` on `F + 4` and on `F`.
    dom: (Window, Vec<f64>),
    own: (Window, Vec<f64>),
}

impl PlanarEngine {
    /// `h x w` images of `channels` channels, `j_count` scales (levels
    /// `0 .. j_count`), all channel pairs.
    pub fn new(h: usize, w: usize, channels: usize, j_count: usize, fewer: bool, tile: usize) -> Self {
        assert!(j_count >= 1 && channels >= 1 && tile >= 1);
        let div = 1usize << (j_count - 1);
        assert!(h % div == 0 && w % div == 0, "{h} x {w} images cannot be decimated {} times", j_count - 1);
        let j_to_dg: Vec<usize> = (0..j_count).collect();
        let plan = build_plan(j_count, L, &j_to_dg, channels, None, &Family::ALL);
        let ranges = (0..j_count)
            .map(|n| {
                let s = plan.partition_point(|c| c.native_dg < n);
                let e = plan.partition_point(|c| c.native_dg <= n);
                s..e
            })
            .collect();
        PlanarEngine { h, w, channels, j_count, fewer, tile, threads: 1, plan, ranges, wav: wavelet() }
    }

    /// Number of host threads over the tiles of a level (at least 1).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Coefficients in plan order (sorted by native level).
    pub fn plan(&self) -> &[Coefficient] {
        &self.plan
    }

    /// Number of pyramid levels, `J`.
    pub fn levels(&self) -> usize {
        self.j_count
    }

    /// `(H_n, W_n)`.
    pub fn size(&self, n: usize) -> (usize, usize) {
        (self.h >> n, self.w >> n)
    }

    fn pixels(&self, n: usize) -> usize {
        let (h, w) = self.size(n);
        h * w
    }

    /// Tiles of level `n`, row-major.
    pub fn tiles(&self, n: usize) -> Vec<Window> {
        let (h, w) = self.size(n);
        let (th, tw) = (self.tile.min(h), self.tile.min(w));
        let mut out = Vec::new();
        for r in (0..h).step_by(th) {
            for c in (0..w).step_by(tw) {
                out.push(Window { rows: Span::new(r as i64, th.min(h - r), h), cols: Span::new(c as i64, tw.min(w - c), w) });
            }
        }
        out
    }

    /// `maps[n] = D^n x` for every level.
    pub fn maps(&self, x: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(x.len(), self.channels * self.h * self.w, "map size");
        let mut maps = vec![x.to_vec()];
        for n in 1..self.levels() {
            let (h, w) = self.size(n - 1);
            let next = grid::decimate_level(&maps[n - 1], h, w, self.channels);
            maps.push(next);
        }
        maps
    }

    /// Statistics in plan order.
    pub fn statistics(&self, x: &[f64]) -> Vec<C64> {
        self.forward(x).0
    }

    /// Statistics and the pyramid for [`PlanarEngine::reverse`].
    pub fn forward(&self, x: &[f64]) -> (Vec<C64>, Pyramid) {
        let levels = self.levels();
        let (c, ll) = (self.channels, L);
        let maps = self.maps(x);
        let mut m: Vec<Vec<Vec<f64>>> = (0..levels).map(|d| (d + 1..levels).map(|n| vec![0.0; c * ll * self.pixels(n)]).collect()).collect();
        let mut stats = vec![ZERO; self.plan.len()];
        for n in 0..levels {
            let tiles = self.tiles(n);
            let range = self.ranges[n].clone();
            // Tiles read m[d][n] (d < n), the sink adds into m[n][*].
            let (finer, rest) = m.split_at_mut(n);
            let own = &mut rest[0];
            self.run(
                &tiles,
                |f| {
                    let t = self.tile_fields(n, f, &maps[n], finer);
                    let sums = range.clone().map(|i| t.partial(self, &self.plan[i])).collect();
                    ForwardOut { sums, pieces: self.push(&t) }
                },
                |out: ForwardOut| {
                    for (k, v) in range.clone().zip(out.sums) {
                        stats[k] += v;
                    }
                    for (np, win, piece) in out.pieces {
                        grid::accumulate(&mut own[np - n - 1], &win, &piece, c * ll);
                    }
                },
            );
        }
        (stats, Pyramid { maps, m })
    }

    /// Map cotangent of `Re sum conj(cot) S`.
    pub fn reverse(&self, pyr: &Pyramid, cot: &[C64]) -> Vec<f64> {
        assert_eq!(cot.len(), self.plan.len());
        let levels = self.levels();
        let (c, ll) = (self.channels, L);
        let mut gm: Vec<Vec<Vec<f64>>> = (0..levels).map(|d| (d + 1..levels).map(|n| vec![0.0; c * ll * self.pixels(n)]).collect()).collect();
        let mut gmaps: Vec<Vec<f64>> = (0..levels).map(|n| vec![0.0; c * self.pixels(n)]).collect();
        for n in (0..levels).rev() {
            let tiles = self.tiles(n);
            // Tiles read gm[n][*] (complete), the sink adds into gm[d][n] (d < n).
            let (finer, rest) = gm.split_at_mut(n);
            let coarse = &rest[0];
            let gmap = &mut gmaps[n];
            self.run(
                &tiles,
                |f| {
                    let t = self.tile_fields(n, f, &pyr.maps[n], &pyr.m);
                    let mut g = t.grads(c, n);
                    for i in self.ranges[n].clone() {
                        t.cotangent(self, &self.plan[i], cot[i], &mut g);
                    }
                    self.finish_reverse(&t, g, coarse)
                },
                |out: ReverseOut| {
                    for (d, win, v) in out.gm {
                        grid::accumulate(&mut finer[d][n - d - 1], &win, &v, c * ll);
                    }
                    grid::accumulate(gmap, &out.dom.0, &out.dom.1, c);
                    grid::accumulate(gmap, &out.own.0, &out.own.1, c);
                },
            );
        }
        let mut g = gmaps.pop().unwrap();
        for n in (0..levels - 1).rev() {
            let (h, w) = self.size(n);
            let back = grid::decimate_level_adjoint(&g, h, w, c);
            g = gmaps.pop().unwrap();
            g.iter_mut().zip(back).for_each(|(a, b)| *a += b);
        }
        g
    }

    /// `f` over the tiles by groups of `threads`, each group's results
    /// handed to `sink` in tile order before the next group starts (the
    /// transients stay bounded by one group of tiles).
    fn run<T: Send>(&self, tiles: &[Window], f: impl Fn(&Window) -> T + Sync, mut sink: impl FnMut(T)) {
        for group in tiles.chunks(self.threads) {
            if group.len() == 1 {
                sink(f(&group[0]));
                continue;
            }
            let f = &f;
            let part: Vec<T> = std::thread::scope(|s| {
                let hs: Vec<_> = group.iter().map(|t| s.spawn(move || f(t))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            part.into_iter().for_each(&mut sink);
        }
    }

    fn tile_fields(&self, n: usize, f: &Window, map: &[f64], m: &[Vec<Vec<f64>>]) -> Tile {
        let (c, ll) = (self.channels, L);
        let a1 = f.grow(HC);
        let dom = a1.grow(HC);
        let (ah, aw) = (a1.rows.len, a1.cols.len);
        let (fh, fw) = (f.rows.len, f.cols.len);
        let xd = grid::read(map, &dom, c);
        let mut z = Vec::with_capacity(c * ll * a1.len());
        for ch in 0..c {
            let plane = &xd[ch * dom.len()..][..dom.len()];
            for l in 0..ll {
                z.extend(conv_valid(plane, ah, aw, &self.wav[l]));
            }
        }
        let mut ma: Vec<Vec<f64>> = (0..n).map(|d| grid::read(&m[d][n - d - 1], &a1, c * ll)).collect();
        ma.push(z.iter().map(|v| v.norm()).collect());
        let mut br = Vec::with_capacity(n + 1);
        for md in &ma {
            let mut b = Vec::with_capacity(c * ll * ll * f.len());
            for p in 0..c * ll {
                let plane = &md[p * a1.len()..][..a1.len()];
                for l3 in 0..ll {
                    b.extend(conv_valid(plane, fh, fw, &self.wav[l3]));
                }
            }
            br.push(b);
        }
        let zf = crop(&z, &a1, HC, c * ll);
        let rf = if c > 1 { zf.iter().map(|&v| root_normalise(v)).collect() } else { Vec::new() };
        Tile {
            n,
            f: *f,
            a1,
            x: crop(&xd, &dom, 2 * HC, c),
            mf: ma.iter().map(|v| crop(v, &a1, HC, c * ll)).collect(),
            zf,
            rf,
            z,
            br,
        }
    }

    /// Pieces `D^(n' - n) (|z_n| 1_F)` for every coarser level `n'`.
    fn push(&self, t: &Tile) -> Vec<(usize, Window, Vec<f64>)> {
        let planes = self.channels * L;
        let mut out = Vec::new();
        let mut win = t.f;
        let mut piece = t.mf[t.n].clone();
        for np in t.n + 1..self.levels() {
            let (next, v) = decimate(&piece, &win, planes);
            out.push((np, next, v.clone()));
            win = next;
            piece = v;
        }
        out
    }

    /// Cotangent on `F` of `|z_n|` from the stored cotangents `gm[n][n']`
    /// of the coarser levels (complete; `gm` is `gm[n]`): the transposed
    /// pieces, Horner-wise.
    fn pull(&self, t: &Tile, gm: &[Vec<f64>]) -> Option<Vec<f64>> {
        let n = t.n;
        let top = self.levels() - 1;
        if n == top {
            return None;
        }
        let planes = self.channels * L;
        let mut wins = vec![t.f];
        for _ in n..top {
            let next = wins.last().unwrap().coarse();
            wins.push(next);
        }
        let mut g = grid::read(&gm[top - n - 1], &wins[top - n], planes);
        for k in (n..top).rev() {
            let back = decimate_adjoint(&g, &wins[k - n], planes);
            g = if k > n {
                let mut own = grid::read(&gm[k - n - 1], &wins[k - n], planes);
                own.iter_mut().zip(back).for_each(|(a, b)| *a += b);
                own
            } else {
                back
            };
        }
        Some(g)
    }

    /// `gm`: the stored cotangents `gm[n][n']` of the tile's level.
    fn finish_reverse(&self, t: &Tile, g: Grads, gm: &[Vec<f64>]) -> ReverseOut {
        let (c, ll, n) = (self.channels, L, t.n);
        let (fh, fw) = (t.f.rows.len, t.f.cols.len);
        let mut gz = g.z;
        if c > 1 {
            for ((gv, &zv), &rv) in gz.iter_mut().zip(&t.zf).zip(&g.r) {
                *gv += root_modulus_vjp(zv, rv);
            }
        }
        let mut out_gm = Vec::new();
        let mut gmod_a1 = Vec::new();
        for d in 0..=n {
            let mut ga = embed(&g.m[d], &t.f, HC, c * ll);
            let alen = t.a1.len();
            for p in 0..c * ll {
                let dst = &mut ga[p * alen..][..alen];
                for l3 in 0..ll {
                    conv_valid_adjoint(&g.br[d][(p * ll + l3) * t.f.len()..][..t.f.len()], fh, fw, &self.wav[l3], dst);
                }
            }
            if d < n {
                out_gm.push((d, t.a1, ga));
            } else {
                gmod_a1 = ga;
            }
        }
        if let Some(pull) = self.pull(t, gm) {
            let e = embed(&pull, &t.f, HC, c * ll);
            gmod_a1.iter_mut().zip(e).for_each(|(a, b)| *a += b);
        }
        let mut gza = embed(&gz, &t.f, HC, c * ll);
        for ((gv, &zv), &gmv) in gza.iter_mut().zip(&t.z).zip(&gmod_a1) {
            let mag = zv.norm();
            *gv += gmv * zv / if mag == 0.0 { 1.0 } else { mag };
        }
        let dom = t.a1.grow(HC);
        let (ah, aw) = (t.a1.rows.len, t.a1.cols.len);
        let mut gx = vec![0.0; c * dom.len()];
        for ch in 0..c {
            let dst = &mut gx[ch * dom.len()..][..dom.len()];
            for l in 0..ll {
                conv_valid_adjoint(&gza[(ch * ll + l) * t.a1.len()..][..t.a1.len()], ah, aw, &self.wav[l], dst);
            }
        }
        ReverseOut { gm: out_gm, dom: (dom, gx), own: (t.f, g.x) }
    }
}

impl Tile {
    fn plane(&self, op: Op) -> Plane<'_> {
        let fl = self.f.len();
        let ll = L;
        match op {
            Op::X(c) => Plane::Re(&self.x[c * fl..][..fl]),
            Op::Z(c, l) => Plane::Cx(&self.zf[(c * ll + l) * fl..][..fl]),
            Op::R(c, l) => Plane::Cx(&self.rf[(c * ll + l) * fl..][..fl]),
            Op::M(d, c, l) => Plane::Re(&self.mf[d][(c * ll + l) * fl..][..fl]),
            Op::B(d, c, l1, l3) => Plane::Cx(&self.br[d][((c * ll + l1) * ll + l3) * fl..][..fl]),
        }
    }

    fn grads(&self, c: usize, n: usize) -> Grads {
        let fl = self.f.len();
        Grads {
            fl,
            x: vec![0.0; c * fl],
            z: vec![ZERO; c * L * fl],
            r: if c > 1 { vec![ZERO; c * L * fl] } else { Vec::new() },
            m: (0..=n).map(|_| vec![0.0; c * L * fl]).collect(),
            br: (0..=n).map(|_| vec![ZERO; c * L * L * fl]).collect(),
        }
    }

    fn partial(&self, e: &PlanarEngine, coef: &Coefficient) -> C64 {
        let (kind, a, b) = e.operands(coef);
        let norm = e.pixels(self.n) as f64;
        let s = match (kind, self.plane(a), self.plane(b)) {
            (Kind::Mean, Plane::Re(a), _) => C64::new(a.iter().sum(), 0.0),
            (Kind::Square, Plane::Re(a), _) => C64::new(a.iter().map(|v| v * v).sum(), 0.0),
            (Kind::Cov, Plane::Re(a), Plane::Cx(b)) => a.iter().zip(b).fold(ZERO, |s, (x, y)| s + x * y.conj()),
            (Kind::Cov, Plane::Cx(a), Plane::Cx(b)) => a.iter().zip(b).fold(ZERO, |s, (x, y)| s + x * y.conj()),
            (Kind::Cov, Plane::Cx(a), Plane::Re(b)) => a.iter().zip(b).fold(ZERO, |s, (x, y)| s + x * y),
            _ => unreachable!("operands of {coef:?}"),
        };
        s / norm
    }

    fn cotangent(&self, e: &PlanarEngine, coef: &Coefficient, g: C64, acc: &mut Grads) {
        let (kind, a, b) = e.operands(coef);
        let g = g / e.pixels(self.n) as f64;
        match kind {
            Kind::Mean => add_re(acc, a, |_| g.re),
            Kind::Square => {
                let Plane::Re(x) = self.plane(a) else { unreachable!() };
                add_re(acc, a, |p| 2.0 * g.re * x[p]);
            }
            Kind::Cov => {
                let (pa, pb) = (self.plane(a), self.plane(b));
                let bv = |p: usize| match &pb {
                    Plane::Re(v) => C64::new(v[p], 0.0),
                    Plane::Cx(v) => v[p],
                };
                let av = |p: usize| match &pa {
                    Plane::Re(v) => C64::new(v[p], 0.0),
                    Plane::Cx(v) => v[p],
                };
                match pa {
                    Plane::Re(_) => add_re(acc, a, |p| (g * bv(p)).re),
                    Plane::Cx(_) => add_cx(acc, a, |p| g * bv(p)),
                }
                match pb {
                    Plane::Re(_) => add_re(acc, b, |p| (g.conj() * av(p)).re),
                    Plane::Cx(_) => add_cx(acc, b, |p| g.conj() * av(p)),
                }
            }
        }
    }
}

fn add_re(acc: &mut Grads, op: Op, f: impl Fn(usize) -> f64) {
    let fl = acc.fl;
    let dst = match op {
        Op::X(c) => &mut acc.x[c * fl..][..fl],
        Op::M(d, c, l) => &mut acc.m[d][(c * L + l) * fl..][..fl],
        _ => unreachable!("{op:?} is complex"),
    };
    dst.iter_mut().enumerate().for_each(|(p, v)| *v += f(p));
}

fn add_cx(acc: &mut Grads, op: Op, f: impl Fn(usize) -> C64) {
    let fl = acc.fl;
    let dst = match op {
        Op::Z(c, l) => &mut acc.z[(c * L + l) * fl..][..fl],
        Op::R(c, l) => &mut acc.r[(c * L + l) * fl..][..fl],
        Op::B(d, c, l1, l3) => &mut acc.br[d][((c * L + l1) * L + l3) * fl..][..fl],
        _ => unreachable!("{op:?} is real"),
    };
    dst.iter_mut().enumerate().for_each(|(p, v)| *v += f(p));
}

impl PlanarEngine {
    /// Kind and operands of a coefficient at its native level `n = j3`
    /// (`j1` for S1/S2), scale `j` at level `j`.
    fn operands(&self, c: &Coefficient) -> (Kind, Op, Op) {
        let n = c.native_dg;
        let l = |o: Option<usize>| o.unwrap();
        match c.family {
            Family::Mean => (Kind::Mean, Op::X(c.c1), Op::X(c.c1)),
            Family::Var => (Kind::Square, Op::X(c.c1), Op::X(c.c1)),
            Family::S1 if c.c1 == c.c2 => (Kind::Mean, Op::M(n, c.c1, l(c.l1)), Op::M(n, c.c1, l(c.l1))),
            Family::S1 => (Kind::Cov, Op::R(c.c1, l(c.l1)), Op::R(c.c2, l(c.l1))),
            Family::S2 if c.c1 == c.c2 => (Kind::Square, Op::M(n, c.c1, l(c.l1)), Op::M(n, c.c1, l(c.l1))),
            Family::S2 => (Kind::Cov, Op::Z(c.c1, l(c.l1)), Op::Z(c.c2, l(c.l1))),
            Family::S3 if self.fewer => (Kind::Cov, Op::X(c.c1), Op::B(l(c.j1), c.c2, l(c.l1), l(c.l3))),
            Family::S3 => (Kind::Cov, Op::B(l(c.j1), c.c1, l(c.l1), l(c.l3)), Op::Z(c.c2, l(c.l3))),
            Family::S4 if self.fewer => (Kind::Cov, Op::M(l(c.j1), c.c1, l(c.l1)), Op::B(l(c.j2), c.c2, l(c.l2), l(c.l3))),
            Family::S4 => (Kind::Cov, Op::B(l(c.j1), c.c1, l(c.l1), l(c.l3)), Op::B(l(c.j2), c.c2, l(c.l2), l(c.l3))),
        }
    }
}
