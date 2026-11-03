//! Losses on the batch mean and the batch variance of the members'
//! features (`stl_port/losses.py`, `sc_variance.py`, `sc_normalization.py`).
//!
//! Features are complex (real ones have a zero imaginary part); with `B`
//! members, `mu = mean_b x_b`, `v = sum_b |x_b - mu|^2 / (B - 1)`, and the
//! cotangent of member b is `a / B + 2 c (x_b - mu) / (B - 1)` for the
//! cotangents `a` of `mu` and `c` of `v`.  Cotangents of complex values
//! follow `Re sum conj(c) dx`.

use num_complex::Complex64;
use stl_core::plan::{Coefficient, Family};

type C64 = Complex64;

/// `f32` epsilon and smallest normal: the fallbacks of the source's float32
/// statistics.
pub const SOURCE_EPS: f64 = f32::EPSILON as f64;
pub const SOURCE_TINY: f64 = f32::MIN_POSITIVE as f64;
/// Dimensionless floor of the SC variance ratio (v2).
pub const RATIO_FLOOR: f64 = 1e-12;

/// Lower middle order statistic (`torch.median`).
pub fn torch_median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(v.len() - 1) / 2]
}

/// Mean and unbiased variance over the members of `x` `[B, n]`.
pub fn moments(x: &[C64], n: usize) -> (Vec<C64>, Vec<f64>) {
    let b = x.len() / n;
    assert!(b >= 2, "population losses need at least two members");
    let mut mu = vec![C64::default(); n];
    for row in x.chunks(n) {
        mu.iter_mut().zip(row).for_each(|(m, v)| *m += v);
    }
    mu.iter_mut().for_each(|m| *m /= b as f64);
    let mut var = vec![0.0; n];
    for row in x.chunks(n) {
        var.iter_mut().zip(row.iter().zip(&mu)).for_each(|(s, (v, m))| *s += (v - m).norm_sqr());
    }
    var.iter_mut().for_each(|s| *s /= (b - 1) as f64);
    (mu, var)
}

/// Member cotangents from those of the mean and the variance.
pub fn moments_vjp(x: &[C64], mu: &[C64], dm: &[C64], dv: &[f64]) -> Vec<C64> {
    let n = mu.len();
    let b = (x.len() / n) as f64;
    x.chunks(n).flat_map(|row| (0..n).map(move |i| dm[i] / b + 2.0 * dv[i] * (row[i] - mu[i]) / (b - 1.0))).collect()
}

/// `mean(max(log(v / t), log 1e-12)^2)` and its cotangent on `v` (zero below
/// the floor); every reference variance must be finite and positive.
pub fn variance_ratio_loss(v: &[f64], t: &[f64]) -> Result<(f64, Vec<f64>), String> {
    if let Some(i) = t.iter().position(|x| !x.is_finite() || *x <= 0.0) {
        return Err(format!("SC variance ratio: reference variance {i} is not finite and positive ({})", t[i]));
    }
    if v.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("SC variance ratio: non-finite or negative member variance".into());
    }
    let floor = RATIO_FLOOR.ln();
    let n = v.len() as f64;
    let mut loss = 0.0;
    let cot = v
        .iter()
        .zip(t)
        .map(|(&v, &t)| {
            let lr = if v > 0.0 { v.ln() - t.ln() } else { f64::NEG_INFINITY };
            let d = lr.max(floor);
            loss += d * d;
            if lr >= floor && v > 0.0 {
                (2.0 / n) * d / v
            } else {
                0.0
            }
        })
        .collect();
    Ok((loss / n, cot))
}

/// The native SC coordinates of the source (`SCNormalization`): the `var`
/// slot becomes the standard error of the map mean,
/// `sqrt(max(<x^2> - <x>^2, 0) / N_0)`, the `mean` slot its real part;
/// sigmas `sqrt(V / N_level)` from the spatial variances `V` of the terms
/// (the `var` slot takes the sigma of the mean).
pub struct ScCoordinates {
    mean: usize,
    second: usize,
    npix: f64,
    native_pixels: Vec<f64>,
}

impl ScCoordinates {
    pub fn new(plan: &[Coefficient], nside: usize) -> Result<Self, String> {
        let find = |f: Family| {
            let v: Vec<usize> = plan.iter().enumerate().filter(|(_, c)| c.family == f).map(|(i, _)| i).collect();
            if v.len() == 1 {
                Ok(v[0])
            } else {
                Err(format!("the plan needs exactly one {} coefficient", f.name()))
            }
        };
        let npix = (12 * nside * nside) as f64;
        let native_pixels = plan.iter().map(|c| npix / 4f64.powi(c.native_dg as i32)).collect();
        Ok(ScCoordinates { mean: find(Family::Mean)?, second: find(Family::Var)?, npix, native_pixels })
    }

    fn s0(&self, row: &[C64]) -> Result<(f64, f64), String> {
        let (m, q) = (row[self.mean].re, row[self.second].re);
        let v = q - m * m;
        let tol = 64.0 * f64::EPSILON * (q.abs() + m * m).max(f64::MIN_POSITIVE);
        if v < -tol {
            return Err("native second moment implies a negative map variance".into());
        }
        Ok((m, (v.max(0.0) / self.npix).sqrt()))
    }

    /// Features of raw statistics `[B, n]`.
    pub fn forward(&self, raw: &[C64], n: usize) -> Result<Vec<C64>, String> {
        let mut out = raw.to_vec();
        for row in out.chunks_mut(n) {
            let (m, s) = self.s0(row)?;
            row[self.mean] = C64::new(m, 0.0);
            row[self.second] = C64::new(s, 0.0);
        }
        Ok(out)
    }

    /// Raw cotangents from feature cotangents.
    pub fn vjp(&self, raw: &[C64], cot: &[C64], n: usize) -> Result<Vec<C64>, String> {
        let mut out = cot.to_vec();
        for (row, c) in raw.chunks(n).zip(out.chunks_mut(n)) {
            let (m, s) = self.s0(row)?;
            let dq = if s > 0.0 { c[self.second].re / (2.0 * self.npix * s) } else { 0.0 };
            c[self.mean] = C64::new(c[self.mean].re - 2.0 * m * dq, 0.0);
            c[self.second] = C64::new(dq, 0.0);
        }
        Ok(out)
    }

    /// Within-map sigmas of one map's coefficients from their spatial variances.
    pub fn sigma(&self, spatial: &[f64]) -> Vec<f64> {
        let mut s: Vec<f64> = spatial.iter().zip(&self.native_pixels).map(|(v, n)| (v / n).sqrt()).collect();
        s[self.second] = s[self.mean];
        s
    }
}

/// Physical sigma equalisation per family on the last orientation axis
/// (`l1` for S1 / S2, `l3` for S3 / S4), `equalize_sigma`.
pub fn equalize_sigma(sigma: &[f64], plan: &[Coefficient]) -> Result<Vec<f64>, String> {
    let mut out = sigma.to_vec();
    for fam in [Family::S1, Family::S2, Family::S3, Family::S4] {
        let ii: Vec<usize> = (0..plan.len()).filter(|&i| plan[i].family == fam).collect();
        if ii.is_empty() {
            continue;
        }
        let vals: Vec<f64> = ii.iter().map(|&i| sigma[i]).collect();
        let med = torch_median(&vals);
        let mut s: Vec<f64> = vals.iter().map(|v| v.clamp(med * 1e-6, med * 1e6)).collect();
        let label = |i: usize| if matches!(fam, Family::S1 | Family::S2) { plan[i].l1 } else { plan[i].l3 };
        let labels: Vec<usize> = ii.iter().map(|&i| label(i).ok_or("physical sigma equalisation: coefficient without orientation")).collect::<Result<_, _>>()?;
        let mut distinct = labels.clone();
        distinct.sort();
        distinct.dedup();
        let counts: Vec<usize> = distinct.iter().map(|d| labels.iter().filter(|l| *l == d).count()).collect();
        if counts.iter().any(|&c| c != counts[0]) {
            return Err("physical sigma equalisation needs complete orientation tuples".into());
        }
        let mut scale: Vec<f64> = distinct
            .iter()
            .map(|d| {
                let v: Vec<f64> = s.iter().zip(&labels).filter(|(_, l)| *l == d).map(|(v, _)| *v).collect();
                v.iter().sum::<f64>() / v.len() as f64
            })
            .collect();
        let ms = torch_median(&scale) + 1e-12;
        scale.iter_mut().for_each(|v| *v /= ms);
        for (v, l) in s.iter_mut().zip(&labels) {
            let j = distinct.iter().position(|d| d == l).unwrap();
            *v /= scale[j] + 1e-12;
        }
        for (k, &i) in ii.iter().enumerate() {
            out[i] = s[k].max(1e-12);
        }
    }
    Ok(out)
}

/// Value of a block: total, mean part, variance part.
#[derive(Clone, Copy, Debug, Default)]
pub struct Block {
    pub loss: f64,
    pub mean: f64,
    pub variance: f64,
}

/// SC block on the features `x` `[B, n]` (`sc_loss`): Gaussian, one
/// coefficient-count mean of `|mu - t|^2 / sigma^2` over the whole plan;
/// physical, the sum of the four S1..S4 family means with equalised sigmas;
/// plus the variance ratio over the same coefficients.
pub fn sc_loss(
    x: &[C64],
    target: &[C64],
    target_var: &[f64],
    sigma: &[f64],
    plan: &[Coefficient],
    physical: bool,
) -> Result<(Block, Vec<C64>), String> {
    let n = plan.len();
    let mask: Vec<bool> = plan.iter().map(|c| !physical || matches!(c.family, Family::S1 | Family::S2 | Family::S3 | Family::S4)).collect();
    let sig = if physical { equalize_sigma(sigma, plan)? } else { sigma.to_vec() };
    let sig: Vec<f64> = sig.into_iter().map(|s| s.max(1e-12)).collect();
    let (mu, var) = moments(x, n);
    let mut dm = vec![C64::default(); n];
    let mut dv = vec![0.0; n];
    let groups: Vec<Vec<usize>> = if physical {
        [Family::S1, Family::S2, Family::S3, Family::S4].iter().map(|f| (0..n).filter(|&i| plan[i].family == *f).collect()).collect()
    } else {
        vec![(0..n).collect()]
    };
    let mut mean_loss = 0.0;
    for g in &groups {
        if g.is_empty() {
            return Err("the physical SC block needs every S1..S4 family".into());
        }
        let count = g.len() as f64;
        let mut s = 0.0;
        for &i in g {
            let r = mu[i] - target[i];
            s += r.norm_sqr() / (sig[i] * sig[i]);
            dm[i] = 2.0 * r / (count * sig[i] * sig[i]);
        }
        mean_loss += s / count;
    }
    let sel: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    let (vl, vc) = variance_ratio_loss(&sel.iter().map(|&i| var[i]).collect::<Vec<_>>(), &sel.iter().map(|&i| target_var[i]).collect::<Vec<_>>())?;
    for (k, &i) in sel.iter().enumerate() {
        dv[i] = vc[k];
    }
    Ok((Block { loss: mean_loss + vl, mean: mean_loss, variance: vl }, moments_vjp(x, &mu, &dm, &dv)))
}

fn real(x: &[f64]) -> Vec<C64> {
    x.iter().map(|&v| C64::new(v, 0.0)).collect()
}

fn re(c: Vec<C64>) -> Vec<f64> {
    c.into_iter().map(|v| v.re).collect()
}

/// C_l block (`power_loss`): `10 mean (log mu - log t)^2` and
/// `2.5 mean log((v + e) / (v_t + e))^2` over the bands of centre in
/// `[20, cutoff]`, `e = 0.01 max(v_t, 1e-30)`.
pub fn power_loss(x: &[f64], target: &[f64], target_var: &[f64], centers: &[f64], cutoff: f64) -> Result<(Block, Vec<f64>), String> {
    let mask: Vec<bool> = centers.iter().map(|&c| c >= 20.0 && c <= cutoff).collect();
    if !mask.iter().any(|&m| m) {
        return Err("C_l variance: no band between 20 and the cutoff".into());
    }
    power_loss_masked(x, target, target_var, &mask)
}

/// [`power_loss`] with the bands of the variance term given by `mask`.
pub fn power_loss_masked(x: &[f64], target: &[f64], target_var: &[f64], mask: &[bool]) -> Result<(Block, Vec<f64>), String> {
    let n = target.len();
    let xc = real(x);
    let (mu, var) = moments(&xc, n);
    let count = mask.iter().filter(|&&m| m).count();
    if count == 0 {
        return Err("power variance: no band selected".into());
    }
    let eps = 1e-12 * torch_median(&target.iter().map(|v| v.abs()).collect::<Vec<_>>()).max(1e-30);
    let mut dm = vec![C64::default(); n];
    let mut dv = vec![0.0; n];
    let mut mean_loss = 0.0;
    for i in 0..n {
        let safe = mu[i].re.max(eps);
        let d = safe.ln() - target[i].max(eps).ln();
        mean_loss += d * d;
        dm[i] = C64::new(if mu[i].re >= eps { 20.0 * d / (n as f64 * safe) } else { 0.0 }, 0.0);
    }
    mean_loss *= 10.0 / n as f64;
    let mut vl = 0.0;
    for i in (0..n).filter(|&i| mask[i]) {
        let e = 0.01 * target_var[i].max(1e-30);
        let r = ((var[i] + e) / (target_var[i] + e)).ln();
        vl += r * r;
        dv[i] = 5.0 * r / (count as f64 * (var[i] + e));
    }
    vl *= 2.5 / count as f64;
    Ok((Block { loss: mean_loss + vl, mean: mean_loss, variance: vl }, re(moments_vjp(&xc, &mu, &dm, &dv))))
}

/// PDF block (`pdf_loss`).
pub fn pdf_loss(x: &[f64], target: &[f64], target_var: &[f64]) -> (Block, Vec<f64>) {
    let n = target.len();
    let xc = real(x);
    let (mu, var) = moments(&xc, n);
    let tmax = target.iter().fold(0f64, |m, &v| m.max(v.max(0.0)));
    let mask: Vec<bool> = target.iter().map(|&t| t.is_finite() && t > tmax * 5e-6).collect();
    if !mask.iter().any(|&m| m) {
        return (Block::default(), vec![0.0; x.len()]);
    }
    let std: Vec<f64> = target_var.iter().map(|v| v.sqrt()).collect();
    let pos: Vec<f64> = (0..n).filter(|&i| mask[i] && std[i].is_finite() && std[i] > 0.0).map(|i| std[i]).collect();
    let sfloor = if pos.is_empty() { SOURCE_EPS } else { 0.05 * torch_median(&pos) }.max(SOURCE_TINY);
    let mut dm = vec![C64::default(); n];
    let mut dv = vec![0.0; n];
    let count = mask.iter().filter(|&&m| m).count() as f64;
    let mut mean_loss = 0.0;
    for i in (0..n).filter(|&i| mask[i]) {
        let den = std[i].max(sfloor);
        let d = (mu[i].re - target[i]) / den;
        mean_loss += d * d;
        dm[i] = C64::new(2.0 * d / (count * den), 0.0);
    }
    mean_loss /= count;
    let posv: Vec<f64> = (0..n).filter(|&i| mask[i] && target_var[i].is_finite() && target_var[i] > 0.0).map(|i| target_var[i]).collect();
    let vfloor = if posv.is_empty() { SOURCE_EPS } else { 1e-4 * torch_median(&posv) }.max(SOURCE_TINY);
    let vmask: Vec<usize> = (0..n).filter(|&i| mask[i] && target_var[i].is_finite() && target_var[i] > vfloor).collect();
    let mut vl = 0.0;
    if !vmask.is_empty() {
        let e = (0.01 * vfloor).max(SOURCE_TINY);
        let c = vmask.len() as f64;
        for &i in &vmask {
            let r = ((var[i] + e) / (target_var[i] + e)).ln();
            vl += r * r;
            dv[i] = 2.0 * r / (c * (var[i] + e));
        }
        vl /= c;
    }
    (Block { loss: mean_loss + vl, mean: mean_loss, variance: vl }, re(moments_vjp(&xc, &mu, &dm, &dv)))
}

/// Minkowski block (`mf_loss`) on `x` `[B, 3, k]`: per functional, mean of
/// `(mu - t)^2 / v_t` (weight 1) and `10 mean log ratio^2`.
pub fn mf_loss(x: &[f64], target: &[f64], target_var: &[f64]) -> (Block, Vec<f64>) {
    let n = target.len();
    let k = n / 3;
    let xc = real(x);
    let (mu, var) = moments(&xc, n);
    let mut dm = vec![C64::default(); n];
    let mut dv = vec![0.0; n];
    let (mut ml, mut vl) = (0.0, 0.0);
    for c in 0..3 {
        let r = c * k..(c + 1) * k;
        let floor = (1e-4 * target_var[r.clone()].iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v))).max(SOURCE_TINY);
        let idx: Vec<usize> = r.filter(|&i| target_var[i].is_finite() && target_var[i] > floor).collect();
        if idx.is_empty() {
            continue;
        }
        let count = idx.len() as f64;
        let e = (0.01 * floor).max(SOURCE_TINY);
        let (mut a, mut b) = (0.0, 0.0);
        for &i in &idx {
            let d = mu[i].re - target[i];
            a += d * d / target_var[i];
            dm[i] = C64::new(2.0 * d / (count * target_var[i]), 0.0);
            let lr = ((var[i] + e) / (target_var[i] + e)).ln();
            b += lr * lr;
            dv[i] = 20.0 * lr / (count * (var[i] + e));
        }
        ml += a / count;
        vl += 10.0 * b / count;
    }
    (Block { loss: ml + vl, mean: ml, variance: vl }, re(moments_vjp(&xc, &mu, &dm, &dv)))
}
