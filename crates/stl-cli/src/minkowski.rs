//! Minkowski functionals per level of the 4-children pyramid (`stl-synth
//! --minkowski-weight`).  At every level n of the term, the map x_n (the
//! means of the 4^n children of each cell, as in `mf_levels.py`) has the
//! soft functionals of Schmalzing & Gorski (1998) at the thresholds
//! `t_i = mu_n + nu_i sigma_n` of the target (nu = -2 ... 5 by 0.2):
//!
//! ```text
//! V0 = <s(u)>,   V1 = <s'(u) |grad x|> / (4 h),
//! V2 = <s'(u) (2 x_1 x_2 x_12 - x_1^2 x_22 - x_2^2 x_11) / (|grad x|^2 + eps)> / (2 pi h),
//! ```
//!
//! with `u = (x - t_i) / h`, s the logistic function (`delta(x - t)` becomes
//! `s'(u) / h`), `h = 0.2 sigma_n` and `eps = 0.01 <|grad x|^2>` of the
//! target.  The derivatives are those of the least-squares quadratic through
//! the cell and its 8 (or 7) neighbours, in gnomonic coordinates of the
//! tangent plane at the cell, where the Christoffel symbols vanish: the
//! covariant gradient and Hessian, and V1, V2 do not depend on the frame.
//!
//! The term is `w / B sum_m mean_{n, k, i} ((V_k,i(map_m) - V_k,i(target)) / sigma_k,i)^2`,
//! sigma the standard deviation of the functional over the realizations
//! (`stl-minkowski --sigma-out`, same estimator), the thresholds where sigma
//! is below 5 % of its maximum left out as in `mf_levels.py`.  The ranks of a
//! group gather each level (a few MB), work on their own cells plus one ring
//! of neighbours, and sum the functionals: no other exchange.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::Path;
use stl_core::healpix::pix2ang_nest;
use stl_core::neighbours::kth_neighbourhood;
use stl_mpi::Comm;

/// Thresholds in units of the target's std at each level (`mf_levels.py`).
pub fn default_nu() -> Vec<f64> {
    (0..36).map(|i| -2.0 + 0.2 * i as f64).collect()
}

/// Smoothing `h` of the thresholds and regularisation `eps` of V2, relative
/// to the target's std and mean squared gradient at each level.
pub const DEFAULT_H: f64 = 0.2;
pub const DEFAULT_EPS: f64 = 0.01;

/// Standard deviations of the functionals over realizations, per level
/// (`stl-minkowski --sigma-out`), with the estimator's parameters.
pub struct SigmaFile {
    pub nu: Vec<f64>,
    pub h: f64,
    pub eps: f64,
    /// nside -> `[V0, V1, V2]` concatenated (3 x nu.len()).
    pub sigma: BTreeMap<usize, Vec<f64>>,
}

impl SigmaFile {
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let bad = |what: &str| format!("{}: {what}", path.display());
        let nums = |v: &serde_json::Value| -> Option<Vec<f64>> { v.as_array()?.iter().map(|x| x.as_f64()).collect() };
        let nu = nums(&v["nu"]).ok_or(bad("no 'nu'"))?;
        let h = v["h"].as_f64().ok_or(bad("no 'h'"))?;
        let eps = v["eps"].as_f64().ok_or(bad("no 'eps'"))?;
        let mut sigma = BTreeMap::new();
        for (k, lv) in v["levels"].as_object().ok_or(bad("no 'levels'"))? {
            let nside: usize = k.parse().map_err(|_| bad("level keys are nside values"))?;
            let rows = lv["sigma"].as_array().ok_or(bad("level without 'sigma'"))?;
            let s: Vec<f64> = rows.iter().map(nums).collect::<Option<Vec<_>>>().ok_or(bad("bad 'sigma'"))?.concat();
            if s.len() != 3 * nu.len() {
                return Err(bad(&format!("level {nside}: {} values of sigma, expected 3 x {}", s.len(), nu.len())));
            }
            sigma.insert(nside, s);
        }
        Ok(SigmaFile { nu, h, eps, sigma })
    }
}

/// Least-squares derivatives `x_1, x_2, x_11, x_12, x_22` of the cell `p` of
/// a level (frame `e_theta, e_phi`): the cells of the stencil (the cell first,
/// padded with it at zero weight) and their weights.
fn stencil(p: i64, nside: i64) -> ([u32; 9], [[f64; 9]; 5]) {
    let ids = kth_neighbourhood(p, nside, 1);
    let unit = |q: i64| {
        let (t, f) = pix2ang_nest(q, nside);
        [t.sin() * f.cos(), t.sin() * f.sin(), t.cos()]
    };
    let (t, f) = pix2ang_nest(p, nside);
    let c = unit(p);
    let et = [t.cos() * f.cos(), t.cos() * f.sin(), -t.sin()];
    let ep = [-f.sin(), f.cos(), 0.0];
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // Coordinates in units of the cell size, so that the normal equations are O(1).
    let delta = (4.0 * PI / (12 * nside * nside) as f64).sqrt();
    let m = ids.len().min(9);
    let mut a = [[0.0; 6]; 9];
    for (j, &q) in ids.iter().take(m).enumerate() {
        let v = unit(q);
        let (x, y) = (dot(&v, &et) / dot(&v, &c) / delta, dot(&v, &ep) / dot(&v, &c) / delta);
        a[j] = [1.0, x, y, 0.5 * x * x, x * y, 0.5 * y * y];
    }
    // Normal equations [A^T A | A^T], Gauss-Jordan with partial pivoting.
    let mut aug = [[0.0; 15]; 6];
    for r in 0..6 {
        for k in 0..6 {
            aug[r][k] = (0..m).map(|j| a[j][r] * a[j][k]).sum();
        }
        for j in 0..m {
            aug[r][6 + j] = a[j][r];
        }
    }
    for col in 0..6 {
        let piv = (col..6).max_by(|&i, &k| aug[i][col].abs().total_cmp(&aug[k][col].abs())).unwrap();
        aug.swap(col, piv);
        let d = aug[col][col];
        aug[col].iter_mut().for_each(|v| *v /= d);
        for r in 0..6 {
            if r != col {
                let f = aug[r][col];
                if f != 0.0 {
                    let pivot_row = aug[col];
                    aug[r].iter_mut().zip(pivot_row.iter()).for_each(|(v, p)| *v -= f * p);
                }
            }
        }
    }
    let mut nb = [p as u32; 9];
    for (j, &q) in ids.iter().take(m).enumerate() {
        nb[j] = q as u32;
    }
    let mut w = [[0.0; 9]; 5];
    for k in 0..5 {
        let scale = if k < 2 { delta } else { delta * delta };
        for j in 0..m {
            w[k][j] = aug[k + 1][6 + j] / scale;
        }
    }
    (nb, w)
}

/// Logistic `s(u_i)` and `s'(u_i)`, `u_i = (x - t0 - i dt) / h`, for every
/// threshold: one exponential, then ratios `exp(dt / h)` from the threshold
/// nearest to x (no overflow in the direction where s saturates).
fn logistic(x: f64, t0: f64, dt: f64, h: f64, r: f64, out: &mut [(f64, f64)]) {
    let nt = out.len();
    let k = (((x - t0) / dt).round().max(0.0) as usize).min(nt - 1);
    let e0 = (-(x - t0 - k as f64 * dt) / h).exp();
    let set = |e: f64| {
        if e.is_finite() {
            let s = 1.0 / (1.0 + e);
            (s, e * s * s)
        } else {
            (0.0, 0.0)
        }
    };
    let mut e = e0;
    for o in out[k..].iter_mut() {
        *o = set(e);
        e *= r;
    }
    let mut e = e0 / r;
    for o in out[..k].iter_mut().rev() {
        *o = set(e);
        e /= r;
    }
}

/// One level of the term on this rank.
struct Level {
    /// Level (0: the map's nside), its nside and number of cells.
    n: usize,
    nside: usize,
    cells: usize,
    /// Owned range of cells; then the work set: owned cells and the halo
    /// (neighbours owned by other ranks), their stencils.
    own: (usize, usize),
    work: Vec<u32>,
    nb: Vec<[u32; 9]>,
    w: Vec<[[f64; 9]; 5]>,
    /// First threshold and step, smoothing, regularisations of |grad x| (V1) and of V2.
    t0: f64,
    dt: f64,
    h: f64,
    eps1: f64,
    eps: f64,
    /// `[V0, V1, V2]` of the target and the inverse of sigma (0: left out).
    target: Vec<f64>,
    inv_sigma: Vec<f64>,
}

/// Per cell of the work set: the five derivatives and `|grad x|`, `Q`.
struct Local {
    d: [f64; 5],
    g: f64,
    q: f64,
    den: f64,
}

impl Level {
    fn new(n: usize, npix: usize, s: usize, e: usize) -> Self {
        let nside = ((npix / 12) as f64).sqrt().round() as usize >> n;
        let cells = npix >> (2 * n);
        let own = (s >> (2 * n), e >> (2 * n));
        let ns = nside as i64;
        let mut work: Vec<u32> = (own.0..own.1).map(|p| p as u32).collect();
        let mut nb = Vec::with_capacity(work.len());
        let mut w = Vec::with_capacity(work.len());
        let mut halo: Vec<u32> = Vec::new();
        for p in own.0..own.1 {
            let (b, wt) = stencil(p as i64, ns);
            halo.extend(b.iter().filter(|&&q| !(own.0..own.1).contains(&(q as usize))));
            nb.push(b);
            w.push(wt);
        }
        halo.sort_unstable();
        halo.dedup();
        for &p in &halo {
            let (b, wt) = stencil(p as i64, ns);
            nb.push(b);
            w.push(wt);
        }
        work.extend(halo);
        Level { n, nside, cells, own, work, nb, w, t0: 0.0, dt: 1.0, h: 1.0, eps1: 0.0, eps: 0.0, target: Vec::new(), inv_sigma: Vec::new() }
    }

    fn owned(&self) -> usize {
        self.own.1 - self.own.0
    }

    fn local(&self, i: usize, y: &[f64]) -> Local {
        let mut d = [0.0; 5];
        for (k, dk) in d.iter_mut().enumerate() {
            *dk = self.nb[i].iter().zip(&self.w[i][k]).map(|(&q, wt)| wt * y[q as usize]).sum();
        }
        let g2 = d[0] * d[0] + d[1] * d[1];
        let den = g2 + self.eps;
        let num = 2.0 * d[0] * d[1] * d[3] - d[0] * d[0] * d[4] - d[1] * d[1] * d[2];
        Local { d, g: (g2 + self.eps1).sqrt(), q: num / den, den }
    }

    /// `[V0, V1, V2]` sums over the owned cells of the full level `y` (to be
    /// added over the ranks, then divided by `norms`).
    fn sums(&self, y: &[f64], nt: usize) -> Vec<f64> {
        let r = (self.dt / self.h).exp();
        let mut out = vec![0.0; 3 * nt];
        let mut sp = vec![(0.0, 0.0); nt];
        for i in 0..self.owned() {
            let l = self.local(i, y);
            logistic(y[self.work[i] as usize], self.t0, self.dt, self.h, r, &mut sp);
            for (t, &(s, ds)) in sp.iter().enumerate() {
                out[t] += s;
                out[nt + t] += ds * l.g;
                out[2 * nt + t] += ds * l.q;
            }
        }
        out
    }

    /// Normalisations of the three sums: `1 / N`, `1 / (4 N h)`, `1 / (2 pi N h)`.
    fn norms(&self) -> [f64; 3] {
        let nc = self.cells as f64;
        [1.0 / nc, 1.0 / (4.0 * nc * self.h), 1.0 / (2.0 * PI * nc * self.h)]
    }

    /// Gradient on the owned cells of `sum_k,i c_k,i V_k,i(y)`, `c` the cotangents of the functionals.
    fn vjp(&self, y: &[f64], c: &[f64], nt: usize) -> Vec<f64> {
        let r = (self.dt / self.h).exp();
        let nm = self.norms();
        let a: Vec<f64> = (0..3 * nt).map(|k| c[k] * nm[k / nt]).collect();
        let mut grad = vec![0.0; self.owned()];
        let mut sp = vec![(0.0, 0.0); nt];
        for (i, &p) in self.work.iter().enumerate() {
            let l = self.local(i, y);
            logistic(y[p as usize], self.t0, self.dt, self.h, r, &mut sp);
            // s'' = s' (1 - 2 s); d/dx of s(u) is s'(u) / h.
            let (mut direct, mut ag, mut aq) = (0.0, 0.0, 0.0);
            for (t, &(s, ds)) in sp.iter().enumerate() {
                direct += (a[t] * ds + (a[nt + t] * l.g + a[2 * nt + t] * l.q) * ds * (1.0 - 2.0 * s)) / self.h;
                ag += a[nt + t] * ds;
                aq += a[2 * nt + t] * ds;
            }
            if i < self.owned() {
                grad[i] += direct;
            }
            let [gx, gy, hxx, hxy, hyy] = l.d;
            let (q, den) = (l.q, l.den);
            let cot = [
                ag * gx / l.g + aq * ((2.0 * gy * hxy - 2.0 * gx * hyy) / den - 2.0 * gx * q / den),
                ag * gy / l.g + aq * ((2.0 * gx * hxy - 2.0 * gy * hxx) / den - 2.0 * gy * q / den),
                aq * (-gy * gy / den),
                aq * (2.0 * gx * gy / den),
                aq * (-gx * gx / den),
            ];
            for (j, &qn) in self.nb[i].iter().enumerate() {
                let qn = qn as usize;
                if (self.own.0..self.own.1).contains(&qn) {
                    grad[qn - self.own.0] += (0..5).map(|k| self.w[i][k][j] * cot[k]).sum::<f64>();
                }
            }
        }
        grad
    }
}

pub struct Minkowski {
    pub weight: f64,
    pub nu: Vec<f64>,
    pub h: f64,
    pub eps: f64,
    /// Where sigma comes from (a file, or 1 % of the target's maximum).
    pub sigma_source: String,
    levels: Vec<Level>,
    /// Number of thresholds kept over the levels and functionals.
    terms: usize,
    members: usize,
}

impl Minkowski {
    /// `target`: owned slice `[s, e)` of the target (the field the term acts
    /// on); `levels` aligned on every rank's range; `comm`: the ranks sharing
    /// the sky.  Without `sigma`, sigma is 1 % of the target's maximum of
    /// each functional over the thresholds (a crude scale).
    #[allow(clippy::too_many_arguments)]
    pub fn new<C: Comm>(
        weight: f64,
        target: &[f64],
        s: usize,
        npix: usize,
        members: usize,
        levels: &[usize],
        sigma: Option<(&SigmaFile, &str)>,
        comm: &C,
    ) -> Result<Self, String> {
        let e = s + target.len();
        let (nu, h, eps) = match sigma {
            Some((f, _)) => (f.nu.clone(), f.h, f.eps),
            None => (default_nu(), DEFAULT_H, DEFAULT_EPS),
        };
        let nt = nu.len();
        if nt < 2 || nu.windows(2).any(|p| ((p[1] - p[0]) - (nu[1] - nu[0])).abs() > 1e-9) {
            return Err("Minkowski thresholds must be equally spaced".into());
        }
        let mut mk = Minkowski {
            weight,
            nu: nu.clone(),
            h,
            eps,
            sigma_source: sigma.map_or("1 % of the target's maximum".into(), |(_, p)| p.to_string()),
            levels: levels.iter().map(|&n| Level::new(n, npix, s, e)).collect(),
            terms: 0,
            members,
        };
        let ys = mk.gather(target, target.len(), comm);
        for (lv, y) in mk.levels.iter_mut().zip(&ys) {
            // Mean, std and mean squared gradient of the target at this level.
            let mut m = [0.0; 3];
            for i in 0..lv.owned() {
                let v = y[lv.work[i] as usize];
                let l = lv.local(i, y);
                m[0] += v;
                m[1] += v * v;
                m[2] += l.d[0] * l.d[0] + l.d[1] * l.d[1];
            }
            comm.allreduce_sum(&mut m);
            let nc = lv.cells as f64;
            let (mean, std, grad2) = (m[0] / nc, (m[1] / nc - (m[0] / nc).powi(2)).max(0.0).sqrt(), m[2] / nc);
            lv.t0 = mean + nu[0] * std;
            lv.dt = (nu[1] - nu[0]) * std;
            lv.h = h * std;
            lv.eps = eps * grad2;
            lv.eps1 = 1e-8 * grad2;
            let mut v = lv.sums(y, nt);
            comm.allreduce_sum(&mut v);
            let nm = lv.norms();
            v.iter_mut().enumerate().for_each(|(k, x)| *x *= nm[k / nt]);
            lv.target = v;
            let sig: Vec<f64> = match sigma {
                Some((f, path)) => f.sigma.get(&lv.nside).cloned().ok_or(format!("{path}: no sigma for nside {}", lv.nside))?,
                None => (0..3)
                    .flat_map(|k| {
                        let mx = lv.target[k * nt..(k + 1) * nt].iter().fold(0.0f64, |a, b| a.max(b.abs()));
                        std::iter::repeat(0.01 * mx).take(nt)
                    })
                    .collect(),
            };
            // Thresholds where sigma is below 5 % of its maximum are left out.
            lv.inv_sigma = vec![0.0; 3 * nt];
            for k in 0..3 {
                let row = &sig[k * nt..(k + 1) * nt];
                let mx = row.iter().cloned().filter(|v| v.is_finite()).fold(0.0, f64::max);
                for t in 0..nt {
                    if row[t].is_finite() && row[t] > 0.05 * mx {
                        lv.inv_sigma[k * nt + t] = 1.0 / row[t];
                    }
                }
            }
        }
        mk.terms = mk.levels.iter().map(|lv| lv.inv_sigma.iter().filter(|&&v| v > 0.0).count()).sum();
        if mk.terms == 0 {
            return Err("Minkowski term: no threshold kept".into());
        }
        Ok(mk)
    }

    /// nside of the levels of the term.
    pub fn levels_nside(&self) -> Vec<usize> {
        self.levels.iter().map(|lv| lv.nside).collect()
    }

    /// Every row of `x` (rows of `len` owned values at level 0) at every
    /// level, gathered over the group: `[level][row * cells + cell]`.
    fn gather<C: Comm>(&self, x: &[f64], len: usize, comm: &C) -> Vec<Vec<f64>> {
        self.levels
            .iter()
            .map(|lv| {
                let k = 1usize << (2 * lv.n);
                let mut full = vec![0.0; x.len() / len * lv.cells];
                for (row, f) in x.chunks(len).zip(full.chunks_mut(lv.cells)) {
                    for (o, c) in f[lv.own.0..lv.own.1].iter_mut().zip(row.chunks(k)) {
                        *o = c.iter().sum::<f64>() / k as f64;
                    }
                }
                comm.allreduce_sum(&mut full);
                full
            })
            .collect()
    }

    /// The functionals of every row, `[row][level, V0..V2, threshold]`.
    pub fn functionals<C: Comm>(&self, x: &[f64], len: usize, comm: &C) -> Vec<Vec<f64>> {
        self.values(&self.gather(x, len, comm), x.len() / len, comm)
    }

    /// The functionals of the gathered levels `ys` of `rows` rows.
    fn values<C: Comm>(&self, ys: &[Vec<f64>], rows: usize, comm: &C) -> Vec<Vec<f64>> {
        let nt = self.nu.len();
        let mut v: Vec<f64> = (0..rows)
            .flat_map(|row| self.levels.iter().zip(ys).flat_map(move |(lv, y)| lv.sums(&y[row * lv.cells..(row + 1) * lv.cells], nt)))
            .collect();
        comm.allreduce_sum(&mut v);
        let norms: Vec<f64> = self.levels.iter().flat_map(|lv| lv.norms().into_iter().flat_map(move |m| std::iter::repeat(m).take(nt))).collect();
        v.chunks(norms.len()).map(|r| r.iter().zip(&norms).map(|(x, m)| x * m).collect()).collect()
    }

    /// The term and its gradient on the owned slice `x` of this group's
    /// members; `comm` shares the sky, `cross` links the groups.
    pub fn loss_and_grad<C: Comm>(&self, x: &[f64], len: usize, comm: &C, cross: Option<&C>) -> (f64, Vec<f64>) {
        let nt = self.nu.len();
        let ys = self.gather(x, len, comm);
        let vals = self.values(&ys, x.len() / len, comm);
        let w = self.weight / self.members as f64 / self.terms as f64;
        let mut loss = 0.0;
        let mut grad = vec![0.0; x.len()];
        for (row, (v, g)) in vals.iter().zip(grad.chunks_mut(len)).enumerate() {
            for (l, (lv, y)) in self.levels.iter().zip(&ys).enumerate() {
                let c: Vec<f64> = (0..3 * nt)
                    .map(|k| {
                        let z = (v[l * 3 * nt + k] - lv.target[k]) * lv.inv_sigma[k];
                        loss += w * z * z;
                        w * 2.0 * z * lv.inv_sigma[k]
                    })
                    .collect();
                let gl = lv.vjp(&y[row * lv.cells..(row + 1) * lv.cells], &c, nt);
                // x_n is the mean of the 4^n children.
                let k = 1usize << (2 * lv.n);
                for (i, gi) in g.iter_mut().enumerate() {
                    *gi += gl[i / k] / k as f64;
                }
            }
        }
        let mut total = [loss];
        if let Some(c) = cross {
            c.allreduce_sum(&mut total);
        }
        (total[0], grad)
    }

    /// Deviations from the target in units of sigma, over all the members
    /// (`first`: index of this group's first member), per level and
    /// functional: median and maximum of |z| over members and thresholds,
    /// rms over the thresholds of the members' median z (`mf_levels.py`).
    pub fn report<C: Comm>(&self, x: &[f64], len: usize, first: usize, comm: &C, cross: Option<&C>) -> serde_json::Value {
        let nt = self.nu.len();
        let per = 3 * nt * self.levels.len();
        let mut all = vec![0.0; self.members * per];
        for (row, v) in self.functionals(x, len, comm).into_iter().enumerate() {
            all[(first + row) * per..(first + row + 1) * per].copy_from_slice(&v);
        }
        if let Some(c) = cross {
            c.allreduce_sum(&mut all);
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let n = v.len();
            if n == 0 { f64::NAN } else if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
        };
        let mut out = serde_json::Map::new();
        for (l, lv) in self.levels.iter().enumerate() {
            let mut fun = serde_json::Map::new();
            for k in 0..3 {
                let kept: Vec<usize> = (0..nt).filter(|&t| lv.inv_sigma[k * nt + t] > 0.0).collect();
                let z = |m: usize, t: usize| (all[m * per + l * 3 * nt + k * nt + t] - lv.target[k * nt + t]) * lv.inv_sigma[k * nt + t];
                let abs: Vec<f64> = (0..self.members).flat_map(|m| kept.iter().map(move |&t| z(m, t).abs())).collect();
                let sys = kept.iter().map(|&t| median((0..self.members).map(|m| z(m, t)).collect()).powi(2)).sum::<f64>() / kept.len().max(1) as f64;
                fun.insert(
                    format!("V{k}"),
                    serde_json::json!({
                        "median_abs_z": median(abs.clone()),
                        "max_abs_z": abs.iter().cloned().fold(0.0, f64::max),
                        "rms_of_median_z": sys.sqrt(),
                        "thresholds_kept": kept.len(),
                    }),
                );
            }
            out.insert(lv.nside.to_string(), serde_json::Value::Object(fun));
        }
        serde_json::Value::Object(out)
    }

    /// The target's functionals, `[level, V0..V2, threshold]`.
    pub fn target_values(&self) -> Vec<f64> {
        self.levels.iter().flat_map(|lv| lv.target.iter().copied()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stl_mpi::threads::run_ranks;

    /// Heavy-tailed smooth field on the sphere: a few low multipoles of the
    /// pixel direction, exponentiated, and narrow peaks.
    fn field(nside: usize, seed: f64) -> Vec<f64> {
        let npix = 12 * nside * nside;
        (0..npix)
            .map(|p| {
                let (t, f) = pix2ang_nest(p as i64, nside as i64);
                let (x, y, z) = (t.sin() * f.cos(), t.sin() * f.sin(), t.cos());
                let g = (3.0 * x + seed).sin() + 0.7 * (4.0 * y * z - seed).cos() + 0.5 * (5.0 * z + 2.0 * x * y + seed).sin();
                let peak = (-((x - 0.3).powi(2) + (y + 0.5 * seed.cos()).powi(2) + (z - 0.4).powi(2)) * 40.0).exp();
                (0.8 * g).exp() + 3.0 * peak
            })
            .collect()
    }

    /// x = n.a on the sphere: the excursion sets are caps, and x is uniform
    /// on [-1, 1] with |grad x| = sqrt(1 - x^2) and Q = x, so that
    /// V0 = (1 - t) / 2, V1 = sqrt(1 - t^2) / 8, V2 = t / (4 pi) for hard
    /// thresholds (Schmalzing & Gorski normalisation), here convolved with the
    /// logistic kernel by quadrature.  Checks the stencil, the frame
    /// invariance and the normalisations.
    #[test]
    fn caps_of_a_linear_field() {
        let nside = 128usize;
        let npix = 12 * nside * nside;
        let a = [0.36, -0.48, 0.8];
        let map: Vec<f64> = (0..npix)
            .map(|p| {
                let (t, f) = pix2ang_nest(p as i64, nside as i64);
                a[0] * t.sin() * f.cos() + a[1] * t.sin() * f.sin() + a[2] * t.cos()
            })
            .collect();
        // std of n.a is 1 / sqrt(3): nu -1.5 ... 1.5 are t = -0.87 ... 0.87.
        let nu: Vec<f64> = (0..16).map(|i| -1.5 + 0.2 * i as f64).collect();
        let sig = SigmaFile { nu: nu.clone(), h: 0.03, eps: 1e-10, sigma: BTreeMap::from([(nside, vec![1.0; 3 * nu.len()])]) };
        run_ranks(1, |comm| {
            let mk = Minkowski::new(1.0, &map, 0, npix, 1, &[0], Some((&sig, "unit")), comm).unwrap();
            let v = mk.target_values();
            let lv = &mk.levels[0];
            let nt = nu.len();
            let q = 200_000;
            for i in 0..nt {
                let t = lv.t0 + i as f64 * lv.dt;
                let mut exact = [0.0; 3];
                for j in 0..q {
                    let x = -1.0 + 2.0 * (j as f64 + 0.5) / q as f64;
                    let e = (-(x - t) / lv.h).exp();
                    let (s, ds) = (1.0 / (1.0 + e), e / (1.0 + e).powi(2) / lv.h);
                    let w = 1.0 / q as f64;
                    exact[0] += w * s;
                    exact[1] += w * ds * (1.0 - x * x).sqrt() / 4.0;
                    exact[2] += w * ds * x / (2.0 * PI);
                }
                for k in 0..3 {
                    let err = (v[k * nt + i] - exact[k]).abs() / exact[1];
                    assert!(err < 1e-3, "V{k} at t = {t:.3}: {} vs {} (err {err:.1e})", v[k * nt + i], exact[k]);
                }
            }
        });
    }

    /// Gradient of the term against central differences, 2 members on 2
    /// ranks (halo cells across the ranks), levels 0 to 2 of nside 32.
    #[test]
    fn gradient_matches_finite_differences() {
        let nside = 32usize;
        let npix = 12 * nside * nside;
        let target = field(nside, 0.3);
        let maps = [field(nside, 1.1), field(nside, 2.7)];
        run_ranks(2, |comm| {
            let (s, e) = (comm.rank() * npix / 2, (comm.rank() + 1) * npix / 2);
            let mk = Minkowski::new(0.7, &target[s..e], s, npix, 2, &[0, 1, 2], None, comm).unwrap();
            let len = e - s;
            let x: Vec<f64> = maps.iter().flat_map(|m| m[s..e].iter().copied()).collect();
            let (l0, g) = mk.loss_and_grad(&x, len, comm, None);
            assert!(l0 > 0.0);
            let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for &k in &[0, 5, 37, len - 1, len + 9, len + 2001, 2 * len - 3] {
                let h = 1e-5;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                if comm.rank() == 0 {
                    xp[k] += h;
                    xm[k] -= h;
                }
                let (lp, _) = mk.loss_and_grad(&xp, len, comm, None);
                let (lm, _) = mk.loss_and_grad(&xm, len, comm, None);
                if comm.rank() == 0 {
                    let fd = (lp - lm) / (2.0 * h);
                    let rel = (fd - g[k]).abs() / g[k].abs().max(1e-3 * gmax);
                    assert!(rel <= 1e-5, "pixel {k}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
                }
            }
        });
    }

    /// Same loss, gradient and functionals on 1 and 4 ranks.
    #[test]
    fn ranks_agree() {
        let nside = 32usize;
        let npix = 12 * nside * nside;
        let target = field(nside, 0.3);
        let maps = [field(nside, 1.1), field(nside, 2.7)];
        let run = |ranks: usize| {
            run_ranks(ranks, |comm| {
                let (s, e) = (comm.rank() * npix / ranks, (comm.rank() + 1) * npix / ranks);
                let mk = Minkowski::new(0.7, &target[s..e], s, npix, 2, &[0, 1, 2], None, comm).unwrap();
                let x: Vec<f64> = maps.iter().flat_map(|m| m[s..e].iter().copied()).collect();
                let (l, g) = mk.loss_and_grad(&x, e - s, comm, None);
                (l, g, mk.functionals(&x, e - s, comm), mk.target_values())
            })
        };
        let one = run(1).remove(0);
        let four = run(4);
        let len = npix / 4;
        for (r, (l, g, v, t)) in four.iter().enumerate() {
            assert!((l - one.0).abs() <= 1e-12 * one.0, "rank {r}: loss {l} vs {}", one.0);
            for (a, b) in t.iter().zip(&one.3).chain(v.concat().iter().zip(&one.2.concat())) {
                assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "rank {r}: functional {a} vs {b}");
            }
            let scale = one.1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for row in 0..2 {
                for i in 0..len {
                    let (a, b) = (g[row * len + i], one.1[row * npix + r * len + i]);
                    assert!((a - b).abs() <= 1e-12 * scale, "rank {r} row {row} pixel {i}: {a} vs {b}");
                }
            }
        }
    }
}
