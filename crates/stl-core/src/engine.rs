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

use crate::conv::{convolution_adjoint, convolution_forward, ConvGeometry};
use crate::local::{modulus_vjp, positions, LocalGeometry};
use crate::neighbours::{dilate_fine, parents};
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
}

impl EngineConfig {
    pub fn new(j_count: usize, fewer: bool) -> Self {
        EngineConfig { j_count, j_to_dg: (0..j_count).collect(), fewer }
    }

    /// Number of pyramid levels, `max dg(j) + 1`.
    pub fn levels(&self) -> usize {
        self.j_to_dg.iter().max().map_or(1, |&d| d + 1)
    }
}

/// One forward (and optionally reverse) pass of a plan.
pub struct LocalEngine<'a> {
    pub geo: &'a LocalGeometry,
    pub cfg: &'a EngineConfig,
    pub batch: usize,
    pub channels: usize,
    /// `maps[d]`, `[B, C, N_d]`.
    pub maps: Vec<Vec<f64>>,
    /// Stored decimated moduli `m[d][n] = D^(n-d) |z_d|`, `[B, C, L, N_n]`, `d < n`.
    pub m: BTreeMap<(usize, usize), Vec<f64>>,
    gm: BTreeMap<(usize, usize), Vec<f64>>,
}

impl<'a> LocalEngine<'a> {
    /// Engine on the map `x` `[B, C, N_0]`; the coarser maps are its
    /// block-wise decimations (`LocalGeometry.analysis_levels`).
    pub fn new(geo: &'a LocalGeometry, cfg: &'a EngineConfig, x: Vec<f64>, batch: usize, channels: usize) -> Self {
        assert_eq!(x.len(), batch * channels * geo.pixels(0));
        let levels = cfg.levels();
        let mut maps = vec![x];
        for d in 0..levels - 1 {
            let next = geo.decimate(&maps[d], d);
            maps.push(next);
        }
        let rows = batch * channels * geo.l;
        let mut m = BTreeMap::new();
        for d in 0..levels {
            for n in d + 1..levels {
                m.insert((d, n), vec![0.0; rows * geo.pixels(n)]);
            }
        }
        LocalEngine { geo, cfg, batch, channels, maps, m, gm: BTreeMap::new() }
    }

    pub fn levels(&self) -> usize {
        self.maps.len()
    }

    /// Supports of the block's decimation chain, one per level from `n`.
    fn supports(&self, n: usize, block: &[i64]) -> Vec<Vec<i64>> {
        let mut chain = vec![block.to_vec()];
        for level in n..self.levels() - 1 {
            let s = parents(&dilate_fine(chain.last().unwrap(), self.geo.nside(level), self.geo.down_rings(level)));
            chain.push(s);
        }
        chain
    }

    fn groups(plan: &[Coefficient]) -> BTreeMap<usize, Vec<usize>> {
        let mut g: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, c) in plan.iter().enumerate() {
            g.entry(c.native_dg).or_default().push(i);
        }
        g
    }

    /// Coefficient values `[B, len(plan)]`; fills the stored moduli.
    pub fn forward(&mut self, plan: &[Coefficient]) -> Vec<Complex64> {
        let groups = Self::groups(plan);
        let mut values = vec![ZERO; self.batch * plan.len()];
        let rows = self.batch * self.channels * self.geo.l;
        for n in 0..self.levels() {
            for block in self.geo.blocks(n) {
                let mut fields = Block::new(self, n, &block);
                for &i in groups.get(&n).map(Vec::as_slice).unwrap_or(&[]) {
                    let (a, b) = fields.operands(self, &plan[i]);
                    let p = fields.partial(self, &plan[i], &a, &b);
                    for (bi, v) in p.into_iter().enumerate() {
                        values[bi * plan.len() + i] += v;
                    }
                }
                // Push |z_n| on F to every coarser level.
                let chain = self.supports(n, &block);
                let mut a: Vec<f64> = fields.z_on_block().iter().map(|v| v.norm()).collect();
                for n2 in n + 1..self.levels() {
                    let (next, _) = self.geo.piece_forward(&a, &chain[n2 - n - 1], n2 - 1);
                    a = next;
                    let support = &chain[n2 - n];
                    let target = self.m.get_mut(&(n, n2)).unwrap();
                    let np = self.geo.pixels(n2);
                    let ms = support.len();
                    for r in 0..rows {
                        for (i, &id) in support.iter().enumerate() {
                            target[r * np + id as usize] += a[r * ms + i];
                        }
                    }
                }
            }
        }
        values
    }

    /// Map cotangents of every level for the coefficient cotangent `gradient`
    /// `[B, len(plan)]` (of `Re sum(conj(gradient) * values)`); requires a
    /// previous [`forward`](Self::forward).  Coefficients whose cotangent is
    /// zero for every batch are skipped.
    pub fn reverse(&mut self, plan: &[Coefficient], gradient: &[Complex64]) -> Vec<Vec<f64>> {
        assert_eq!(gradient.len(), self.batch * plan.len());
        let active: Vec<bool> =
            (0..plan.len()).map(|i| (0..self.batch).any(|b| gradient[b * plan.len() + i] != ZERO)).collect();
        let groups = Self::groups(plan);
        self.gm = self.m.iter().map(|(&k, v)| (k, vec![0.0; v.len()])).collect();
        let mut maps_gradient: Vec<Vec<f64>> = self.maps.iter().map(|x| vec![0.0; x.len()]).collect();
        for n in (0..self.levels()).rev() {
            for block in self.geo.blocks(n) {
                let mut fields = Block::new(self, n, &block);
                for &i in groups.get(&n).map(Vec::as_slice).unwrap_or(&[]) {
                    if !active[i] {
                        continue;
                    }
                    let g: Vec<Complex64> = (0..self.batch).map(|b| gradient[b * plan.len() + i]).collect();
                    let (a, b) = fields.operands(self, &plan[i]);
                    fields.reverse_coefficient(self, &plan[i], &a, &b, &g, &mut maps_gradient);
                }
                fields.finish_reverse(self, &mut maps_gradient);
            }
        }
        self.gm.clear();
        maps_gradient
    }

    /// Chain the map cotangents of all levels back to level 0 through the
    /// transposed decimations, then the band cotangents of the Laplacian
    /// pyramid (`LocalGeometry.analysis_vjp`).  Returns `(map0, bands)`.
    pub fn analysis_vjp(&self, mut maps_gradient: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
        for d in (1..self.levels()).rev() {
            let g = self.geo.decimate_adjoint(&maps_gradient[d], d - 1);
            maps_gradient[d - 1].iter_mut().zip(&g).for_each(|(a, b)| *a += b);
        }
        let g0 = maps_gradient.swap_remove(0);
        let mut bands = vec![g0.clone()];
        for d in 0..self.levels() - 1 {
            let fine = bands.last().unwrap();
            let (nf, nc) = (self.geo.pixels(d), self.geo.pixels(d + 1));
            let rows = fine.len() / nf;
            let mut coarse = vec![0.0; rows * nc];
            for r in 0..rows {
                for p in 0..nf {
                    coarse[r * nc + p / 4] += fine[r * nf + p];
                }
            }
            bands.push(coarse);
        }
        (g0, bands)
    }
}

/// Real or complex operand `[B, |F|]`.
type Operand = Vec<Complex64>;

/// Fields of one block `F` of level `n`.
struct Block {
    n: usize,
    block: Vec<i64>,
    a1: Vec<i64>,
    domain: Vec<i64>,
    conv: ConvGeometry,
    rows_a1: Vec<usize>,
    rows_f: Vec<usize>,
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
        let nside = geo.nside(n);
        let a1 = dilate_fine(block, nside, geo.conv_rings);
        let domain = dilate_fine(&a1, nside, geo.conv_rings);
        let conv = geo.conv(n, &domain);
        let rows_a1 = positions(&domain, &a1);
        let rows_f = positions(&domain, block);
        let f_in_a1 = positions(&a1, block);
        let np = geo.pixels(n);
        let x: Vec<f64> =
            engine.maps[n].chunks(np).flat_map(|row| domain.iter().map(move |&i| row[i as usize])).collect();
        let rows = engine.batch * engine.channels;
        let y = convolution_forward(&conv, &geo.kernel, &x, rows);
        let z = join(&y, rows, geo.l, domain.len(), &rows_a1);
        Block {
            n,
            block: block.to_vec(),
            a1,
            domain,
            conv,
            rows_a1,
            rows_f,
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
                let np = engine.geo.pixels(self.n);
                let a1 = &self.a1;
                engine.m[&(d, self.n)].chunks(np).flat_map(|row| a1.iter().map(move |&i| row[i as usize])).collect()
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
            let y = convolution_forward(&self.conv, &geo.kernel, &padded, rows);
            let v = join(&y, rows, geo.l, kd, &self.rows_f);
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
        let np = engine.geo.pixels(level);
        (0..engine.batch)
            .flat_map(|b| {
                let row = &engine.maps[level][(b * engine.channels + channel) * np..][..np];
                self.block.iter().map(move |&i| Complex64::new(row[i as usize], 0.0))
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
        let n_total = engine.geo.pixels(self.n) as f64;
        let nf = self.nf();
        (0..engine.batch)
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
        let (np, nf) = (engine.geo.pixels(level), self.nf());
        for b in 0..engine.batch {
            let row = &mut out[(b * engine.channels + channel) * np..][..np];
            for (i, &id) in self.block.iter().enumerate() {
                row[id as usize] += g[b * nf + i];
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
        let n_total = engine.geo.pixels(self.n) as f64;
        let nf = self.nf();
        let dg = &engine.cfg.j_to_dg;
        // Per-pixel scaling of a cotangent: ga[b, i] = f(g[b], i).
        let per = |f: &dyn Fn(Complex64, usize) -> Complex64| -> Vec<Complex64> {
            (0..engine.batch * nf).map(|k| f(g[k / nf], k)).collect()
        };
        match c.family {
            Family::Mean => {
                let ga: Vec<f64> = (0..engine.batch * nf).map(|k| g[k / nf].re / n_total).collect();
                self.add_map(engine, &mut maps_gradient[0], 0, c.c1, &ga);
            }
            Family::Var => {
                let ga: Vec<f64> = (0..engine.batch * nf).map(|k| 2.0 * g[k / nf].re * a[k].re / n_total).collect();
                self.add_map(engine, &mut maps_gradient[0], 0, c.c1, &ga);
            }
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                if c.c1 == c.c2 {
                    let ga = if c.family == Family::S1 {
                        let scale: Vec<f64> = (0..engine.batch * nf).map(|k| g[k / nf].re / n_total).collect();
                        modulus_vjp(a, &scale)
                    } else {
                        per(&|gi, k| 2.0 * gi.re * a[k] / n_total)
                    };
                    self.add_z(engine, c.c1, l1, &ga);
                } else {
                    let (ga, gb) = if c.family == Family::S1 {
                        let ga = per(&|gi, k| gi * root_normalise(b[k]) / n_total);
                        let gb = per(&|gi, k| gi.conj() * root_normalise(a[k]) / n_total);
                        (root_modulus_vjp(a, &ga), root_modulus_vjp(b, &gb))
                    } else {
                        (per(&|gi, k| gi * b[k] / n_total), per(&|gi, k| gi.conj() * a[k] / n_total))
                    };
                    self.add_z(engine, c.c1, l1, &ga);
                    self.add_z(engine, c.c2, l1, &gb);
                }
            }
            Family::S3 | Family::S4 => {
                let ga = per(&|gi, k| gi * b[k] / n_total);
                let gb = per(&|gi, k| gi.conj() * a[k] / n_total);
                let d1 = dg[c.j1.unwrap()];
                let (l1, l3) = (c.l1.unwrap(), c.l3.unwrap());
                if c.family == Family::S3 {
                    if engine.cfg.fewer {
                        let re: Vec<f64> = ga.iter().map(|v| v.re).collect();
                        self.add_map(engine, &mut maps_gradient[self.n], self.n, c.c1, &re);
                        self.add_branch(engine, d1, c.c2, l1, l3, &gb);
                    } else {
                        self.add_branch(engine, d1, c.c1, l1, l3, &ga);
                        self.add_z(engine, c.c2, l3, &gb);
                    }
                } else {
                    if engine.cfg.fewer {
                        self.add_modulus_core(engine, d1, c.c1, l1, &ga);
                    } else {
                        self.add_branch(engine, d1, c.c1, l1, l3, &ga);
                    }
                    self.add_branch(engine, dg[c.j2.unwrap()], c.c2, c.l2.unwrap(), l3, &gb);
                }
            }
        }
    }

    fn finish_reverse(&mut self, engine: &mut LocalEngine, maps_gradient: &mut [Vec<f64>]) {
        let geo = engine.geo;
        let (n, l, kd, na1, nf) = (self.n, geo.l, self.domain.len(), self.na1(), self.nf());
        let bc = engine.batch * engine.channels;
        // Second-wavelet cotangents -> modulus cotangents on a1.
        for (d, g) in std::mem::take(&mut self.gbranch) {
            let rows = bc * l;
            let padded = split(&g, rows, l, nf, kd, &self.rows_f);
            let gx = convolution_adjoint(&self.conv, &geo.kernel, &padded, rows);
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
                let np = geo.pixels(n);
                let target = engine.gm.get_mut(&(d, n)).unwrap();
                for r in 0..bc * l {
                    for (i, &id) in self.a1.iter().enumerate() {
                        target[r * np + id as usize] += gm[r * na1 + i];
                    }
                }
            }
        }
        // The block's own chain to coarser levels.
        let chain = engine.supports(n, &self.block);
        let mut g_fine: Option<Vec<f64>> = None;
        for n2 in n + 1..engine.levels() {
            let np = geo.pixels(n2);
            let support = &chain[n2 - n];
            let mut g: Vec<f64> = engine.gm[&(n, n2)]
                .chunks(np)
                .flat_map(|row| support.iter().map(move |&i| row[i as usize]))
                .collect();
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
        // Wavelet adjoint on a1 rows -> map cotangent on the domain.
        let padded = split(&gz, bc, l, na1, kd, &self.rows_a1);
        let gx = convolution_adjoint(&self.conv, &geo.kernel, &padded, bc);
        let np = geo.pixels(n);
        for r in 0..bc {
            for (i, &id) in self.domain.iter().enumerate() {
                maps_gradient[n][r * np + id as usize] += gx[r * kd + i];
            }
        }
    }
}

/// Convolution output `[R, 2L, kd]` -> complex `[R, L, |pick|]` at the
/// domain positions `pick` (`S3Multiscale._join`).
fn join(y: &[f64], rows: usize, l: usize, kd: usize, pick: &[usize]) -> Vec<Complex64> {
    let mut z = Vec::with_capacity(rows * l * pick.len());
    for r in 0..rows {
        for o in 0..l {
            let re = &y[(r * 2 * l + 2 * o) * kd..][..kd];
            let im = &y[(r * 2 * l + 2 * o + 1) * kd..][..kd];
            z.extend(pick.iter().map(|&p| Complex64::new(re[p], im[p])));
        }
    }
    z
}

/// Inverse of [`join`]: complex `[R, L, m]` -> `[R, 2L, kd]`, zero outside `at`.
fn split(g: &[Complex64], rows: usize, l: usize, m: usize, kd: usize, at: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; rows * 2 * l * kd];
    for r in 0..rows {
        for o in 0..l {
            let src = &g[(r * l + o) * m..][..m];
            for (i, &p) in at.iter().enumerate() {
                out[(r * 2 * l + 2 * o) * kd + p] = src[i].re;
                out[(r * 2 * l + 2 * o + 1) * kd + p] = src[i].im;
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
