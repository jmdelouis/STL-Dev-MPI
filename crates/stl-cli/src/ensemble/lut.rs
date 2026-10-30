//! Pooled look-up table between the physical map kappa and the Gaussian
//! variable g (`stl_port/lut.py`): knots at the quantiles of every pixel of
//! the reference maps on a tail-rich probability grid, physical knots
//! rounded to `f32` as the source, linear interpolation in Gaussian
//! coordinates with linear extrapolation beyond the end knots.

/// `erf^-1(x)` to double precision: M. Giles' approximation (2010), then
/// Newton steps on `erf` (`|x| <= 1/2`) or on `erfc` of `1 - |x|` (exact
/// there) in the tails.
pub fn erfinv(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    let mut w = -((1.0 - x) * (1.0 + x)).ln();
    let p = if w < 5.0 {
        w -= 2.5;
        [3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
            .iter()
            .fold(2.81022636e-08, |p, c| c + p * w)
    } else {
        w = w.sqrt() - 3.0;
        [0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]
            .iter()
            .fold(-0.000200214257, |p, c| c + p * w)
    };
    let y = p * x;
    let k = 2.0 / std::f64::consts::PI.sqrt();
    if x.abs() <= 0.5 {
        let mut y = y;
        for _ in 0..4 {
            y -= (libm::erf(y) - x) / (k * (-y * y).exp());
        }
        return y;
    }
    // Tails (the approximation is single precision and degrades beyond
    // |x| = 1 - 2^-24): Newton on ln erfc(z) = ln(1 - |x|), z = |y|.
    let lt = (1.0 - x.abs()).ln();
    let mut z = y.abs();
    if !z.is_finite() || z <= 0.0 {
        z = (-lt).sqrt();
    }
    for _ in 0..60 {
        let e = libm::erfc(z);
        let step = (e.ln() - lt) / (-k * (-z * z).exp() / e);
        z -= step;
        if step.abs() <= 1e-16 * z {
            break;
        }
    }
    z * x.signum()
}

const PCLIP: f64 = 1e-15;

/// Normal quantile of `p` as the source computes it: `sqrt(2) erfinv(2 p - 1)`
/// (p clipped to `[1e-15, 1 - 1e-15]`), rounded to `f32`.
pub fn source_ppf(p: f64) -> f64 {
    let p = p.clamp(PCLIP, 1.0 - PCLIP);
    ((2f64.sqrt() * erfinv(2.0 * p - 1.0)) as f32) as f64
}

/// numpy `linspace(start, stop, num, endpoint)`.
pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Vec<f64> {
    let div = if endpoint { num.saturating_sub(1) } else { num };
    let step = (stop - start) / div.max(1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    if endpoint && num > 1 {
        v[num - 1] = stop;
    }
    v
}

/// The probability grid of the knots (`probability_grid`): 25 % of the knots
/// logarithmic in each tail down to `0.5 / (count + 1)`, the rest linear.
pub fn probability_grid(nbins: usize, count: usize) -> Vec<f64> {
    let pmin = (0.5 / (count as f64 + 1.0)).max(PCLIP);
    let pmax = 1.0 - pmin;
    let tail_p = (1e-3f64).max(pmin * 10.0).min(0.49);
    let n_tail = 4usize.max((0.25 * nbins as f64).floor() as usize);
    let n_core = nbins as i64 - 2 * n_tail as i64;
    let mut p = if n_core < 8 {
        linspace(pmin, pmax, nbins, true)
    } else {
        let left: Vec<f64> = linspace(pmin.log10(), tail_p.log10(), n_tail, false).into_iter().map(|e| 10f64.powf(e)).collect();
        let core = linspace(tail_p, 1.0 - tail_p, n_core as usize, true);
        let mut all: Vec<f64> = left.iter().copied().chain(core).chain(left.iter().rev().map(|v| 1.0 - v)).collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all.dedup();
        let kept: Vec<f64> = all.into_iter().filter(|&v| v >= pmin && v <= pmax).collect();
        if kept.len() == nbins {
            kept
        } else {
            linspace(pmin, pmax, nbins, true)
        }
    };
    for v in p.iter_mut() {
        *v = v.clamp(PCLIP, 1.0 - PCLIP);
    }
    for i in 1..p.len() {
        if p[i] <= p[i - 1] {
            p[i] = libm::nextafter(p[i - 1], 1.0);
        }
    }
    p
}

/// numpy `quantile(sorted, q, method="linear")` of sorted values.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = n as f64 * q + (1.0 + q * -1.0) - 1.0;
    let prev = virtual_index.floor();
    let gamma = virtual_index - prev;
    let clip = |i: f64| (i.max(0.0) as usize).min(n - 1);
    let (a, b) = (sorted[clip(prev)], sorted[clip(prev + 1.0)]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// Index `i` of the interval `[knots[i], knots[i+1]]` of `x` (right-sided
/// search, clipped to the end intervals: linear extrapolation).
fn interval(x: f64, knots: &[f64]) -> usize {
    knots.partition_point(|&k| k <= x).saturating_sub(1).min(knots.len() - 2)
}

#[derive(Clone, Debug)]
pub struct Lut {
    pub probabilities: Vec<f64>,
    /// Physical knots (values of `f32`).
    pub physical: Vec<f64>,
    /// Gaussian knots `source_ppf(probabilities)`.
    pub gaussian: Vec<f64>,
}

impl Lut {
    pub fn new(probabilities: Vec<f64>, physical: Vec<f64>) -> Result<Self, String> {
        let gaussian: Vec<f64> = probabilities.iter().map(|&p| source_ppf(p)).collect();
        if probabilities.len() < 2 || physical.len() != probabilities.len() {
            return Err("LUT: matching knot vectors of length >= 2 required".into());
        }
        for v in [&probabilities, &physical, &gaussian] {
            if v.iter().any(|x| !x.is_finite()) || v.windows(2).any(|w| w[1] <= w[0]) {
                return Err("LUT knots must be finite and strictly increasing".into());
            }
        }
        Ok(Lut { probabilities, physical, gaussian })
    }

    /// Knots at the quantiles of the pooled reference values (each rounded
    /// to `f32` first, as the source), `nbins` knots.
    pub fn from_values(mut pool: Vec<f64>, nbins: usize) -> Result<Self, String> {
        if pool.iter().any(|v| !v.is_finite()) {
            return Err("LUT: non-finite reference pixel".into());
        }
        pool.iter_mut().for_each(|v| *v = (*v as f32) as f64);
        pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = probability_grid(nbins, pool.len());
        let mut x: Vec<f32> = p.iter().map(|&q| quantile_sorted(&pool, q) as f32).collect();
        let eps = (((x[x.len() - 1] - x[0]).abs() as f64) * 1e-7).max(1e-12) as f32;
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                x[i] = x[i - 1] + eps;
            }
        }
        Lut::new(p, x.into_iter().map(|v| v as f64).collect())
    }

    /// The Gaussian value of a physical value (`gaussianize`, `f32` values).
    pub fn gaussianize(&self, x: f64) -> f64 {
        let i = interval(x, &self.physical);
        let slope = (self.probabilities[i + 1] - self.probabilities[i]) / (self.physical[i + 1] - self.physical[i]);
        source_ppf(self.probabilities[i] + (x - self.physical[i]) * slope)
    }

    /// `f(g)` and `f'(g)` (the inverse map, before the monopole removal).
    pub fn inverse(&self, g: f64) -> (f64, f64) {
        let i = interval(g, &self.gaussian);
        let slope = (self.physical[i + 1] - self.physical[i]) / (self.gaussian[i + 1] - self.gaussian[i]);
        (self.physical[i] + (g - self.gaussian[i]) * slope, slope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `erf(erfinv(x)) = x` to a few ulp, in the tails on `1 - |x|`.
    #[test]
    fn erfinv_inverts_erf() {
        for &x in &[-1.0 + 2e-15, -0.999999, -0.9, -0.5, -0.1, 0.0, 1e-12, 0.3, 0.5, 0.75, 0.999, 1.0 - 4e-15] {
            let y = erfinv(x);
            if x.abs() <= 0.5 {
                assert!((libm::erf(y) - x).abs() <= 4e-16, "erf(erfinv({x}))");
            } else {
                let t = 1.0 - f64::abs(x);
                assert!((libm::erfc(y.abs()) - t).abs() <= 1e-14 * t, "erfc(|erfinv({x})|)");
            }
        }
    }
}
