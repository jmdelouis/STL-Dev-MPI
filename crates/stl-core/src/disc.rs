//! Statistics on a disc patch: the complete ScatCov of a map known only on
//! the pixels of a spherical disc, with its exact VJP.
//!
//! Support: level 0 holds every NESTED pixel whose centre lies within the
//! angular distance `R` of the disc centre ([`disc_pixels`]); level `d + 1`
//! holds the parents of level `d` (a parent is kept when at least one of
//! its children is).  The pyramid is the smooth decimation restricted to
//! the support ([`LocalGeometry::down`] on the support: the Gaussian
//! weights of the fine pixels present, l1-renormalised over them), the
//! wavelet is bound to the support of its level with renormalised bilinear
//! weights (`bind_support`, [`crate::conv::ConvGeometry`]).  Nothing
//! outside the disc is read, so the whole patch is one domain: no blocks
//! and no halos.  At `R = 180` degrees every level is the full sky and the
//! operators are the full-sky ones, so the statistics are those of
//! [`crate::engine::LocalEngine`] (to the summation order).
//!
//! Averages: with a core radius `r` the coefficients are pixel means over
//! the core of their native level, `core_0` the support pixels whose centre
//! lies within `r` and `core_{d+1}` the parents of `core_d`; without one,
//! over the whole support of the level.  The halo `R - r` then absorbs the
//! edge effects of the convolutions.  Formulas and cotangents of the terms
//! are those of the full-sky engine ([`crate::engine`]).
//!
//! Layouts (row-major, pixel axis last, one patch, `C` channels): maps
//! `[C, N_d]`; `z_n` `[C, L, N_n]`; moduli `[C, L, N_n]`; second wavelets
//! on the core `[C, L(l1), L(l3), |core_n|]`.

use crate::conv::ConvPlan;
use crate::engine::{join, operand_cotangents, split, term_second_sums, term_sums, EngineConfig};
use crate::healpix::{npix, pix2ang_nest};
use crate::local::{modulus_vjp, positions, Down, LocalGeometry};
use crate::neighbours::parents;
use crate::plan::{Coefficient, Family};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Unit vector of the NESTED pixel centre `pix` at `nside`.
pub fn pixel_vector(pix: i64, nside: i64) -> [f64; 3] {
    let (theta, phi) = pix2ang_nest(pix, nside);
    let s = theta.sin();
    [s * phi.cos(), s * phi.sin(), theta.cos()]
}

/// Angle between two unit vectors (`atan2(|u x v|, u . v)`, accurate at
/// every separation).
#[inline]
fn angle(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
}

/// `v / |v|` (a nonzero vector).
pub fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    assert!(n > 0.0 && n.is_finite(), "disc centre must be a nonzero finite vector");
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Sorted NESTED pixels of `nside` whose centre lies within the angle
/// `radius` (radians) of the unit vector `centre`; every pixel when
/// `radius >= pi`.  Hierarchical descent from the 12 base pixels, keeping
/// the cells whose centre is within `radius` plus a bound of their radius.
pub fn disc_pixels(nside: i64, centre: [f64; 3], radius: f64) -> Vec<i64> {
    assert!(radius > 0.0, "disc radius must be positive");
    if radius >= PI {
        return (0..npix(nside)).collect();
    }
    let order = nside.trailing_zeros();
    let mut cells: Vec<i64> = (0..12).collect();
    for k in 0..=order {
        let ns = 1i64 << k;
        // Bound of the angular radius of a cell of nside `ns` (the HEALPix
        // maximum is 0.84 rad at nside 1 and below 0.84 / ns beyond).
        let slack = if k == order { 0.0 } else { 1.5 / ns as f64 };
        let keep: Vec<i64> = cells.into_iter().filter(|&p| angle(&centre, &pixel_vector(p, ns)) <= radius + slack).collect();
        cells = if k == order { keep } else { keep.iter().flat_map(|&p| (0..4).map(move |c| 4 * p + c)).collect() };
    }
    cells.sort_unstable();
    cells
}

/// Operators of one disc patch, level by level.
pub struct DiscGeometry {
    /// Support of every level, sorted NESTED ids.
    pub ids: Vec<Vec<i64>>,
    /// Positions in `ids[n]` of the pixels averaged at level `n`.
    pub core: Vec<Vec<usize>>,
    /// Wavelet of level `n` on its support.
    pub conv: Vec<ConvPlan>,
    /// Decimation `d -> d + 1` restricted to the support (`down[d].out_ids == ids[d + 1]`).
    pub down: Vec<Down>,
}

impl DiscGeometry {
    /// Disc of the angle `radius` (radians) around `centre` (any nonzero
    /// vector), statistics averaged within `core` (radians) if given.
    pub fn new(geo: &LocalGeometry, levels: usize, centre: [f64; 3], radius: f64, core: Option<f64>) -> Self {
        let centre = unit(centre);
        let ids0 = disc_pixels(geo.nside0, centre, radius);
        let core0 = match core {
            Some(r) if r < radius => {
                assert!(r > 0.0, "core radius must be positive");
                let c: Vec<i64> = ids0.iter().copied().filter(|&p| angle(&centre, &pixel_vector(p, geo.nside0)) <= r).collect();
                assert!(!c.is_empty(), "the core holds no pixel centre");
                Some(c)
            }
            _ => None,
        };
        Self::from_support(geo, levels, ids0, core0)
    }

    /// Patch on the level-0 support `ids0` (sorted NESTED), averaged on
    /// `core0` (sorted subset) or on the whole support.
    pub fn from_support(geo: &LocalGeometry, levels: usize, ids0: Vec<i64>, core0: Option<Vec<i64>>) -> Self {
        assert!(!ids0.is_empty(), "empty disc support");
        assert!(ids0.windows(2).all(|w| w[0] < w[1]), "support must be sorted and unique");
        assert!(levels >= 1 && (geo.nside0 >> (levels - 1)) >= 1, "too many levels for nside");
        let mut ids = vec![ids0];
        let mut down = Vec::new();
        for d in 0..levels - 1 {
            let dn = geo.down(d, &ids[d]);
            ids.push(dn.out_ids.clone());
            down.push(dn);
        }
        let mut core = Vec::with_capacity(levels);
        let mut c = core0;
        for n in 0..levels {
            core.push(match &c {
                Some(cells) => positions(&ids[n], cells),
                None => (0..ids[n].len()).collect(),
            });
            c = c.map(|cells| parents(&cells));
        }
        let conv = (0..levels).map(|n| geo.conv_plan(n, &ids[n], &ids[n])).collect();
        DiscGeometry { ids, core, conv, down }
    }

    /// Number of pyramid levels.
    pub fn levels(&self) -> usize {
        self.ids.len()
    }

    /// Number of pixels averaged at level `n`.
    pub fn core_pixels(&self, n: usize) -> usize {
        self.core[n].len()
    }
}

/// Forward fields of one patch kept for the reverse.
pub struct DiscState {
    /// `maps[d]` `[C, N_d]`.
    maps: Vec<Vec<f64>>,
    /// `z_n` `[C, L, N_n]`.
    z: Vec<Vec<Complex64>>,
    /// `mods[(d, n)] = D^(n-d) |z_d|` `[C, L, N_n]`, `d <= n`.
    mods: BTreeMap<(usize, usize), Vec<f64>>,
    /// Second wavelets of `mods[(d, n)]` on the core `[C, L, L, |core_n|]`.
    branch: BTreeMap<(usize, usize), Vec<Complex64>>,
}

/// The model of the disc statistics: wavelet, decimation, scales and the
/// S3/S4 definitions (`dg(j)` levels, `fewer`), `C` channels.
pub struct DiscModel<'a> {
    /// Geometry parameters (kernel, orientations, decimation).
    pub geo: &'a LocalGeometry,
    /// Scales and S3/S4 definitions.
    pub cfg: &'a EngineConfig,
    /// Number of channels `C`.
    pub channels: usize,
}

impl DiscModel<'_> {
    fn branch_pairs(&self, plan: &[Coefficient]) -> Vec<(usize, usize)> {
        let dg = &self.cfg.j_to_dg;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for c in plan {
            let n = c.native_dg;
            match c.family {
                Family::S3 => pairs.push((dg[c.j1.unwrap()], n)),
                Family::S4 => {
                    if !self.cfg.fewer {
                        pairs.push((dg[c.j1.unwrap()], n));
                    }
                    pairs.push((dg[c.j2.unwrap()], n));
                }
                _ => {}
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Statistics `[len(plan)]` of the patch `x` `[C, |ids_0|]`, with the
    /// spatial variances of the terms (`moments`, else empty) and the
    /// state of the reverse.
    pub fn forward(&self, g: &DiscGeometry, x: &[f64], plan: &[Coefficient], moments: bool) -> (Vec<Complex64>, Vec<f64>, DiscState) {
        let (c, l, levels) = (self.channels, self.geo.l, g.levels());
        assert_eq!(levels, self.cfg.levels(), "disc geometry levels");
        assert_eq!(x.len(), c * g.ids[0].len(), "patch shape mismatch");
        let mut maps = vec![x.to_vec()];
        for d in 0..levels - 1 {
            let next = g.down[d].apply(&maps[d]);
            maps.push(next);
        }
        let mut z = Vec::with_capacity(levels);
        let mut mods = BTreeMap::new();
        for n in 0..levels {
            let nn = g.ids[n].len();
            let y = g.conv[n].forward(&self.geo.kernel, &maps[n], c);
            let zn = join(&y, c, l, nn, None);
            mods.insert((n, n), zn.iter().map(|v| v.norm()).collect::<Vec<f64>>());
            for d in 0..n {
                let m = g.down[n - 1].apply(&mods[&(d, n - 1)]);
                mods.insert((d, n), m);
            }
            z.push(zn);
        }
        let mut branch = BTreeMap::new();
        for (d, n) in self.branch_pairs(plan) {
            let rows = c * l;
            let y = g.conv[n].forward_rows(&self.geo.kernel, &mods[&(d, n)], rows, Some(&g.core[n]));
            branch.insert((d, n), join(&y, rows, l, g.core[n].len(), None));
        }
        let state = DiscState { maps, z, mods, branch };
        let mut values = Vec::with_capacity(plan.len());
        let mut var = Vec::with_capacity(if moments { plan.len() } else { 0 });
        for coeff in plan {
            let n = coeff.native_dg;
            let nf = g.core[n].len();
            let (a, b) = self.operands(g, &state, coeff);
            let s = term_sums(coeff, &a, &b, 1, nf, nf as f64)[0];
            if moments {
                let m2 = term_second_sums(coeff, &a, &b, 1, nf, nf as f64)[0];
                var.push((m2 - s.norm_sqr()).max(0.0));
            }
            values.push(s);
        }
        (values, var, state)
    }

    /// Operands `[|core_n|]` of a coefficient (as `Block::operands`).
    fn operands(&self, g: &DiscGeometry, s: &DiscState, c: &Coefficient) -> (Vec<Complex64>, Vec<Complex64>) {
        let (l, n) = (self.geo.l, c.native_dg);
        let dg = &self.cfg.j_to_dg;
        let core = &g.core[n];
        let nn = g.ids[n].len();
        let nc = core.len();
        let z_at = |ch: usize, angle: usize| -> Vec<Complex64> {
            let row = &s.z[n][(ch * l + angle) * nn..][..nn];
            core.iter().map(|&p| row[p]).collect()
        };
        let map_at = |level: usize, ch: usize| -> Vec<Complex64> {
            let m = g.ids[level].len();
            let row = &s.maps[level][ch * m..][..m];
            g.core[level].iter().map(|&p| Complex64::new(row[p], 0.0)).collect()
        };
        let mod_at = |d: usize, ch: usize, l1: usize| -> Vec<Complex64> {
            let row = &s.mods[&(d, n)][(ch * l + l1) * nn..][..nn];
            core.iter().map(|&p| Complex64::new(row[p], 0.0)).collect()
        };
        let branch_at = |d: usize, ch: usize, l1: usize, l3: usize| -> Vec<Complex64> {
            s.branch[&(d, n)][((ch * l + l1) * l + l3) * nc..][..nc].to_vec()
        };
        match c.family {
            Family::Mean | Family::Var => (map_at(0, c.c1), Vec::new()),
            Family::S1 | Family::S2 => {
                let l1 = c.l1.unwrap();
                (z_at(c.c1, l1), z_at(c.c2, l1))
            }
            Family::S3 => {
                let (d1, l1, l3) = (dg[c.j1.unwrap()], c.l1.unwrap(), c.l3.unwrap());
                if !self.cfg.fewer {
                    (branch_at(d1, c.c1, l1, l3), z_at(c.c2, l3))
                } else {
                    (map_at(n, c.c1), branch_at(d1, c.c2, l1, l3))
                }
            }
            Family::S4 => {
                let (d1, d2) = (dg[c.j1.unwrap()], dg[c.j2.unwrap()]);
                let (l1, l2, l3) = (c.l1.unwrap(), c.l2.unwrap(), c.l3.unwrap());
                let a = if self.cfg.fewer { mod_at(d1, c.c1, l1) } else { branch_at(d1, c.c1, l1, l3) };
                (a, branch_at(d2, c.c2, l2, l3))
            }
        }
    }

    /// Patch cotangent `[C, |ids_0|]` of `Re sum(conj(cotangent) * values)`
    /// after [`forward`](Self::forward) of the same plan.
    pub fn reverse(&self, g: &DiscGeometry, s: &DiscState, plan: &[Coefficient], cotangent: &[Complex64]) -> Vec<f64> {
        assert_eq!(cotangent.len(), plan.len(), "cotangent shape mismatch");
        let (c, l, levels) = (self.channels, self.geo.l, g.levels());
        let dg = &self.cfg.j_to_dg;
        let len = |n: usize| g.ids[n].len();
        let mut gmaps: Vec<Vec<f64>> = (0..levels).map(|n| vec![0.0; c * len(n)]).collect();
        let mut gz: Vec<Vec<Complex64>> = (0..levels).map(|n| vec![ZERO; c * l * len(n)]).collect();
        let mut gbranch: BTreeMap<(usize, usize), Vec<Complex64>> = BTreeMap::new();
        let mut gmod: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        for n in (0..levels).rev() {
            let (nn, core) = (len(n), &g.core[n]);
            let nc = core.len();
            for (i, coeff) in plan.iter().enumerate() {
                if coeff.native_dg != n || cotangent[i] == ZERO {
                    continue;
                }
                let (a, b) = self.operands(g, s, coeff);
                let (ga, gb) = operand_cotangents(coeff, &a, &b, &cotangent[i..i + 1], 1, nc, nc as f64);
                let add_z = |gz: &mut Vec<Vec<Complex64>>, ch: usize, angle: usize, v: &[Complex64]| {
                    let row = &mut gz[n][(ch * l + angle) * nn..][..nn];
                    core.iter().zip(v).for_each(|(&p, &x)| row[p] += x);
                };
                let add_branch = |gb: &mut BTreeMap<(usize, usize), Vec<Complex64>>, d: usize, ch: usize, l1: usize, l3: usize, v: &[Complex64]| {
                    let slot = gb.entry((d, n)).or_insert_with(|| vec![ZERO; c * l * l * nc]);
                    slot[((ch * l + l1) * l + l3) * nc..][..nc].iter_mut().zip(v).for_each(|(r, &x)| *r += x);
                };
                let add_map = |gm: &mut Vec<Vec<f64>>, level: usize, ch: usize, v: &[Complex64]| {
                    let m = len(level);
                    let row = &mut gm[level][ch * m..][..m];
                    g.core[level].iter().zip(v).for_each(|(&p, x)| row[p] += x.re);
                };
                match coeff.family {
                    Family::Mean | Family::Var => add_map(&mut gmaps, 0, coeff.c1, &ga),
                    Family::S1 | Family::S2 => {
                        let l1 = coeff.l1.unwrap();
                        add_z(&mut gz, coeff.c1, l1, &ga);
                        if coeff.c1 != coeff.c2 {
                            add_z(&mut gz, coeff.c2, l1, &gb);
                        }
                    }
                    Family::S3 => {
                        let (d1, l1, l3) = (dg[coeff.j1.unwrap()], coeff.l1.unwrap(), coeff.l3.unwrap());
                        if self.cfg.fewer {
                            add_map(&mut gmaps, n, coeff.c1, &ga);
                            add_branch(&mut gbranch, d1, coeff.c2, l1, l3, &gb);
                        } else {
                            add_branch(&mut gbranch, d1, coeff.c1, l1, l3, &ga);
                            add_z(&mut gz, coeff.c2, l3, &gb);
                        }
                    }
                    Family::S4 => {
                        let (d1, l1, l3) = (dg[coeff.j1.unwrap()], coeff.l1.unwrap(), coeff.l3.unwrap());
                        if self.cfg.fewer {
                            let slot = gmod.entry((d1, n)).or_insert_with(|| vec![0.0; c * l * nn]);
                            let row = &mut slot[(coeff.c1 * l + l1) * nn..][..nn];
                            core.iter().zip(&ga).for_each(|(&p, x)| row[p] += x.re);
                        } else {
                            add_branch(&mut gbranch, d1, coeff.c1, l1, l3, &ga);
                        }
                        add_branch(&mut gbranch, dg[coeff.j2.unwrap()], coeff.c2, coeff.l2.unwrap(), l3, &gb);
                    }
                }
            }
            // Second wavelets on the core -> moduli of level n.
            let keys: Vec<(usize, usize)> = gbranch.keys().copied().filter(|&(_, k)| k == n).collect();
            for key in keys {
                let gb = gbranch.remove(&key).unwrap();
                let rows = c * l;
                let packed = split(&gb, rows, l, nc, None);
                let gx = g.conv[n].adjoint_rows(&self.geo.kernel, &packed, rows, Some(core));
                let slot = gmod.entry(key).or_insert_with(|| vec![0.0; rows * nn]);
                slot.iter_mut().zip(&gx).for_each(|(a, b)| *a += b);
            }
            // Stored moduli of finer levels: one decimation back.
            for d in 0..n {
                if let Some(gm) = gmod.remove(&(d, n)) {
                    let back = g.down[n - 1].adjoint(&gm);
                    let slot = gmod.entry((d, n - 1)).or_insert_with(|| vec![0.0; back.len()]);
                    slot.iter_mut().zip(&back).for_each(|(a, b)| *a += b);
                }
            }
            if let Some(gm) = gmod.remove(&(n, n)) {
                let v = modulus_vjp(&s.z[n], &gm);
                gz[n].iter_mut().zip(&v).for_each(|(a, b)| *a += b);
            }
            if gz[n].iter().any(|v| *v != ZERO) {
                let packed = split(&gz[n], c, l, nn, None);
                let gx = g.conv[n].adjoint(&self.geo.kernel, &packed, c);
                gmaps[n].iter_mut().zip(&gx).for_each(|(a, b)| *a += b);
            }
        }
        for n in (1..levels).rev() {
            let back = g.down[n - 1].adjoint(&gmaps[n]);
            gmaps[n - 1].iter_mut().zip(&back).for_each(|(a, b)| *a += b);
        }
        gmaps.swap_remove(0)
    }
}
