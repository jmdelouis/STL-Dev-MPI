//! L-BFGS with the strong Wolfe line search, a line-by-line port of
//! `torch.optim.LBFGS` (torch 2.14, `line_search_fn="strong_wolfe"`) as
//! `STL_main.LossOptim.PyramidLBFGS` drives it, so that loss trajectories
//! can be compared with STL-Dev.
//!
//! The variable is distributed: every process holds a slice of `x`, and
//! every reduction (dot products, norms) goes through a [`Reduce`] so that
//! all processes take identical decisions.  `p += alpha * u` is a fused
//! multiply-add as in torch's vectorised `add_(u, alpha=...)`.

/// Global reductions over the processes sharing the variable.
pub trait Reduce {
    /// Sum of the local values, identical on every process.
    fn sum(&self, local: f64) -> f64;
    /// Maximum of the local values, identical on every process.
    fn max(&self, local: f64) -> f64;
}

/// One process holding the whole variable.
pub struct Local;

impl Reduce for Local {
    fn sum(&self, local: f64) -> f64 {
        local
    }
    fn max(&self, local: f64) -> f64 {
        local
    }
}

/// `torch.optim.LBFGS` settings (`PyramidLBFGS` defaults).
#[derive(Clone, Debug)]
pub struct LbfgsOptions {
    pub lr: f64,
    pub max_iter: usize,
    /// Default `max_iter * 5 / 4`.
    pub max_eval: Option<usize>,
    pub tolerance_grad: f64,
    pub tolerance_change: f64,
    pub history_size: usize,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        LbfgsOptions {
            lr: 1.0,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-12,
            tolerance_change: 1e-15,
            history_size: 10,
        }
    }
}

fn dot(r: &dyn Reduce, a: &[f64], b: &[f64]) -> f64 {
    r.sum(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

fn abs_max(r: &dyn Reduce, a: &[f64]) -> f64 {
    r.max(a.iter().fold(0f64, |m, v| m.max(v.abs())))
}

fn abs_sum(r: &dyn Reduce, a: &[f64]) -> f64 {
    r.sum(a.iter().map(|v| v.abs()).sum())
}

/// `p += alpha * u` (torch `add_(u, alpha=alpha)`, fused).
fn axpy(p: &mut [f64], alpha: f64, u: &[f64]) {
    p.iter_mut().zip(u).for_each(|(p, &u)| *p = u.mul_add(alpha, *p));
}

/// `_cubic_interpolate` of torch (polyinterp.lua): minimiser of the cubic
/// through `(x1, f1, g1)` and `(x2, f2, g2)`, clamped to `bounds`.
pub fn cubic_interpolate(x1: f64, f1: f64, g1: f64, x2: f64, f2: f64, g2: f64, bounds: Option<(f64, f64)>) -> f64 {
    let (xmin, xmax) = bounds.unwrap_or(if x1 <= x2 { (x1, x2) } else { (x2, x1) });
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;
    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(xmin).min(xmax)
    } else {
        (xmin + xmax) / 2.0
    }
}

/// Python's `min(a, b)` / `max(a, b)`: the first argument on ties.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Objective: loss (identical on every process) and local gradient slice.
pub type Objective<'f> = dyn FnMut(&[f64]) -> (f64, Vec<f64>) + 'f;

/// Objective along the line: `phi(t) = f(x + t d)` and its gradient.
struct Line<'a, 'f> {
    x: &'a [f64],
    d: &'a [f64],
    f: &'a mut Objective<'f>,
    evals: usize,
}

impl Line<'_, '_> {
    fn eval(&mut self, t: f64) -> (f64, Vec<f64>) {
        let mut p = self.x.to_vec();
        axpy(&mut p, t, self.d);
        self.evals += 1;
        (self.f)(&p)
    }
}

/// `_strong_wolfe` of torch (lswolfe.lua) with its default `c1 = 1e-4`,
/// `c2 = 0.9`, `tolerance_change = 1e-9`.  Returns `(f, g, t, evaluations)`.
#[allow(clippy::too_many_arguments)]
fn strong_wolfe(
    r: &dyn Reduce,
    line: &mut Line,
    mut t: f64,
    f: f64,
    g: &[f64],
    gtd: f64,
    max_ls: usize,
) -> (f64, Vec<f64>, f64) {
    let (c1, c2, tolerance_change) = (1e-4, 0.9, 1e-9);
    let d_norm = abs_max(r, line.d);
    let (mut f_new, mut g_new) = line.eval(t);
    let mut gtd_new = dot(r, &g_new, line.d);

    let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0.0, f, g.to_vec(), gtd);
    let mut done = false;
    let mut ls_iter = 0;
    let mut bracket: Vec<f64> = Vec::new();
    let mut bracket_f: Vec<f64> = Vec::new();
    let mut bracket_g: Vec<Vec<f64>> = Vec::new();
    let mut bracket_gtd: Vec<f64> = Vec::new();
    while ls_iter < max_ls {
        if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev) {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }
        if gtd_new.abs() <= -c2 * gtd {
            bracket = vec![t];
            bracket_f = vec![f_new];
            bracket_g = vec![g_new.clone()];
            done = true;
            break;
        }
        if gtd_new >= 0.0 {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let tmp = t;
        t = cubic_interpolate(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, Some((min_step, max_step)));
        t_prev = tmp;
        f_prev = f_new;
        g_prev = g_new.clone();
        gtd_prev = gtd_new;
        (f_new, g_new) = line.eval(t);
        gtd_new = dot(r, &g_new, line.d);
        ls_iter += 1;
    }
    if ls_iter == max_ls {
        bracket = vec![0.0, t];
        bracket_f = vec![f, f_new];
        bracket_g = vec![g.to_vec(), g_new.clone()];
    }

    // Zoom phase.
    let mut insuf_progress = false;
    let (mut low, mut high) = if bracket_f[0] <= *bracket_f.last().unwrap() { (0, 1) } else { (1, 0) };
    while !done && ls_iter < max_ls {
        if (bracket[1] - bracket[0]).abs() * d_norm < tolerance_change {
            break;
        }
        t = cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0], bracket[1], bracket_f[1], bracket_gtd[1], None);
        let (bmax, bmin) = (py_max(bracket[0], bracket[1]), py_min(bracket[0], bracket[1]));
        let eps = 0.1 * (bmax - bmin);
        if py_min(bmax - t, t - bmin) < eps {
            if insuf_progress || t >= bmax || t <= bmin {
                t = if (t - bmax).abs() < (t - bmin).abs() { bmax - eps } else { bmin + eps };
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }
        (f_new, g_new) = line.eval(t);
        gtd_new = dot(r, &g_new, line.d);
        ls_iter += 1;
        if f_new > (f + c1 * t * gtd) || f_new >= bracket_f[low] {
            bracket[high] = t;
            bracket_f[high] = f_new;
            bracket_g[high] = g_new.clone();
            bracket_gtd[high] = gtd_new;
            (low, high) = if bracket_f[0] <= bracket_f[1] { (0, 1) } else { (1, 0) };
        } else {
            if gtd_new.abs() <= -c2 * gtd {
                done = true;
            } else if gtd_new * (bracket[high] - bracket[low]) >= 0.0 {
                bracket[high] = bracket[low];
                bracket_f[high] = bracket_f[low];
                bracket_g[high] = bracket_g[low].clone();
                bracket_gtd[high] = bracket_gtd[low];
            }
            bracket[low] = t;
            bracket_f[low] = f_new;
            bracket_g[low] = g_new.clone();
            bracket_gtd[low] = gtd_new;
        }
    }
    let low = low.min(bracket.len() - 1);
    (bracket_f[low], std::mem::take(&mut bracket_g[low]), bracket[low])
}

/// One `LBFGS.step(closure)` on the local slice `x` (updated in place):
/// `f(x)` returns the loss (identical on every process) and the local slice
/// of its gradient.  Returns the number of evaluations.
pub fn minimize(x: &mut [f64], opts: &LbfgsOptions, r: &dyn Reduce, f: &mut Objective) -> usize {
    let max_eval = opts.max_eval.unwrap_or(opts.max_iter * 5 / 4);
    let (mut loss, mut flat_grad) = f(x);
    let mut current_evals = 1;
    if abs_max(r, &flat_grad) <= opts.tolerance_grad {
        return current_evals;
    }
    let mut d: Vec<f64> = Vec::new();
    let mut t = 0.0;
    let mut old_dirs: Vec<Vec<f64>> = Vec::new();
    let mut old_stps: Vec<Vec<f64>> = Vec::new();
    let mut ro: Vec<f64> = Vec::new();
    let mut h_diag = 1.0;
    let mut prev_flat_grad: Vec<f64> = Vec::new();
    let mut al = vec![0.0; opts.history_size];
    let mut n_iter = 0;
    while n_iter < opts.max_iter {
        n_iter += 1;
        if n_iter == 1 {
            d = flat_grad.iter().map(|v| -v).collect();
        } else {
            let y: Vec<f64> = flat_grad.iter().zip(&prev_flat_grad).map(|(a, b)| a - b).collect();
            let s: Vec<f64> = d.iter().map(|v| v * t).collect();
            let ys = dot(r, &y, &s);
            if ys > 1e-10 {
                if old_dirs.len() == opts.history_size {
                    old_dirs.remove(0);
                    old_stps.remove(0);
                    ro.remove(0);
                }
                h_diag = ys / dot(r, &y, &y);
                old_dirs.push(y);
                old_stps.push(s);
                ro.push(1.0 / ys);
            }
            let num_old = old_dirs.len();
            let mut q: Vec<f64> = flat_grad.iter().map(|v| -v).collect();
            for i in (0..num_old).rev() {
                al[i] = dot(r, &old_stps[i], &q) * ro[i];
                axpy(&mut q, -al[i], &old_dirs[i]);
            }
            d = q.iter().map(|v| v * h_diag).collect();
            for i in 0..num_old {
                let be_i = dot(r, &old_dirs[i], &d) * ro[i];
                axpy(&mut d, al[i] - be_i, &old_stps[i]);
            }
        }
        prev_flat_grad.clone_from(&flat_grad);
        let prev_loss = loss;

        t = if n_iter == 1 { py_min(1.0, 1.0 / abs_sum(r, &flat_grad)) * opts.lr } else { opts.lr };
        let gtd = dot(r, &flat_grad, &d);
        if gtd > -opts.tolerance_change {
            break;
        }
        let x_init = x.to_vec();
        let mut line = Line { x: &x_init, d: &d, f, evals: 0 };
        let (l, g, step) = strong_wolfe(r, &mut line, t, loss, &flat_grad, gtd, max_eval - current_evals);
        let ls_func_evals = line.evals;
        (loss, flat_grad, t) = (l, g, step);
        x.copy_from_slice(&x_init);
        axpy(x, t, &d);
        let opt_cond = abs_max(r, &flat_grad) <= opts.tolerance_grad;
        current_evals += ls_func_evals;

        if n_iter == opts.max_iter || current_evals >= max_eval || opt_cond {
            break;
        }
        if r.max(d.iter().fold(0f64, |m, v| m.max((v * t).abs()))) <= opts.tolerance_change {
            break;
        }
        if (loss - prev_loss).abs() < opts.tolerance_change {
            break;
        }
    }
    current_evals
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock from torch's own LBFGS test: converges to (1, 1).
    #[test]
    fn rosenbrock() {
        let mut x = vec![-1.5, 2.0];
        let opts = LbfgsOptions { max_iter: 100, tolerance_grad: 1e-12, tolerance_change: 1e-15, ..Default::default() };
        let mut f = |p: &[f64]| {
            let (a, b) = (p[0], p[1]);
            let loss = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            (loss, vec![-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)])
        };
        minimize(&mut x, &opts, &Local, &mut f);
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6, "{x:?}");
    }
}
