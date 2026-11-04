//! Anti-aliased decimation `D` (one level) on a partial grid: port of
//! `TorchGeometry.smooth_decimation_csr` with the candidate lists of
//! `LocalGeometry.torch_down` (`HealPixDown("smooth")`, sphere).

use crate::healpix::{centre_lonlat, npix, order_of};
use crate::neighbours::{kth_neighbourhood, parents};
use std::f64::consts::PI;

/// `math.degrees` / `math.radians` of CPython (`x * 180/pi`, `x * pi/180`).
#[inline]
fn py_degrees(x: f64) -> f64 {
    x * (180.0 / PI)
}

#[inline]
fn py_radians(x: f64) -> f64 {
    x * (PI / 180.0)
}

/// Row normalisation of the decimation weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightNorm {
    /// Rows sum to one (l1 norm).
    L1,
    /// Rows have unit Euclidean norm (l2 norm).
    L2,
}

/// Parameters of the smooth decimation (defaults: radius of 3 fine pixels,
/// `sigma = radius / 2`, l1 rows).
#[derive(Clone, Copy, Debug)]
pub struct DecimationParams {
    /// Cone radius in degrees; `None`: 3 fine pixels.
    pub radius_deg: Option<f64>,
    /// Gaussian width in degrees; `None`: `radius / 2`.
    pub sigma_deg: Option<f64>,
    /// Row normalisation.
    pub norm: WeightNorm,
}

impl Default for DecimationParams {
    fn default() -> Self {
        DecimationParams { radius_deg: None, sigma_deg: None, norm: WeightNorm::L1 }
    }
}

/// Compressed sparse rows, `n_rows x n_cols`.
#[derive(Clone, Debug, PartialEq)]
pub struct Csr {
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// Row pointers, `n_rows + 1` entries: row `r` is `indptr[r]..indptr[r + 1]`.
    pub indptr: Vec<usize>,
    /// Column of each stored entry.
    pub indices: Vec<usize>,
    /// Value of each stored entry.
    pub values: Vec<f64>,
}

impl Csr {
    /// Columns and values of row `r`.
    pub fn row(&self, r: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.indptr[r], self.indptr[r + 1]);
        (&self.indices[a..b], &self.values[a..b])
    }

    /// Transpose, columns of each row in increasing order.
    pub fn transpose(&self) -> Csr {
        let mut count = vec![0usize; self.n_cols + 1];
        for &c in &self.indices {
            count[c + 1] += 1;
        }
        for i in 0..self.n_cols {
            count[i + 1] += count[i];
        }
        let indptr = count.clone();
        let mut next = count;
        let mut indices = vec![0usize; self.indices.len()];
        let mut values = vec![0f64; self.values.len()];
        for r in 0..self.n_rows {
            let (cols, vals) = self.row(r);
            for (&c, &v) in cols.iter().zip(vals) {
                indices[next[c]] = r;
                values[next[c]] = v;
                next[c] += 1;
            }
        }
        Csr { n_rows: self.n_cols, n_cols: self.n_rows, indptr, indices, values }
    }

    /// `y = A x`, summing each row in column order.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n_cols);
        (0..self.n_rows)
            .map(|r| {
                let (cols, vals) = self.row(r);
                cols.iter().zip(vals).fold(0.0, |s, (&c, &v)| s + v * x[c])
            })
            .collect()
    }
}

/// Candidate fine pixels of every coarse pixel (`LocalGeometry.torch_down`):
/// children of its 2-ring coarse neighbourhood, or every fine pixel at the
/// smallest levels.  Returns one list per coarse pixel.
pub fn candidates(coarse: &[i64], nside: i64) -> Vec<Vec<i64>> {
    let cn = nside / 2;
    if cn >= 4 {
        coarse
            .iter()
            .map(|&c| kth_neighbourhood(c, cn, 2).iter().flat_map(|&v| (0..4).map(move |i| 4 * v + i)).collect())
            .collect()
    } else {
        let all: Vec<i64> = (0..npix(nside)).collect();
        coarse.iter().map(|_| all.clone()).collect()
    }
}

#[inline]
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = 0.5 * (lat2 - lat1);
    let dlon = 0.5 * (lon2 - lon1);
    let (sa, so) = (dlat.sin(), dlon.sin());
    let a = (sa * sa + lat1.cos() * lat2.cos() * (so * so)).clamp(0.0, 1.0);
    2.0 * a.sqrt().asin()
}

/// Smooth decimation of the fine grid `ids` (NESTED, sorted, unique, at
/// `nside`) to its parents.  Returns the sorted coarse ids and the forward
/// matrix `[N_out, N_in]` (columns index `ids`); fine pixels outside `ids`
/// are unavailable, as in the partial operator.
pub fn smooth_decimation(ids: &[i64], nside: i64, params: &DecimationParams) -> (Vec<i64>, Csr) {
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "cell ids must be sorted and unique");
    order_of(nside);
    let pix_deg = py_degrees((4.0 * PI / npix(nside) as f64).sqrt());
    let radius = py_radians(params.radius_deg.unwrap_or(3.0 * pix_deg));
    let sigma = match params.sigma_deg {
        None => radius / 2.0,
        Some(s) => py_radians(s),
    };
    let coarse = parents(ids);
    let cand = candidates(&coarse, nside);
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (r, &c) in coarse.iter().enumerate() {
        let mut cols: Vec<usize> = cand[r]
            .iter()
            .filter(|&&v| v >= 0)
            .filter_map(|&v| {
                let q = ids.partition_point(|&s| s < v);
                (q < ids.len() && ids[q] == v).then_some(q)
            })
            .collect();
        cols.sort_unstable();
        cols.dedup();
        let fine: Vec<i64> = cols.iter().map(|&q| ids[q]).collect();
        for (q, v) in cols.into_iter().zip(row_weights(c, &fine, nside, radius, sigma, params.norm)) {
            if v != 0.0 {
                indices.push(q);
                values.push(v);
            }
        }
        indptr.push(indices.len());
    }
    let n_rows = coarse.len();
    (coarse, Csr { n_rows, n_cols: ids.len(), indptr, indices, values })
}

/// Normalised weights of the coarse cell `c` over the available fine cells
/// `fine` (sorted), zeros included.
fn row_weights(c: i64, fine: &[i64], nside: i64, radius: f64, sigma: f64, norm: WeightNorm) -> Vec<f64> {
    let (lon_c, lat_c) = centre_lonlat(c, nside / 2);
    let mut w: Vec<f64> = fine
        .iter()
        .map(|&f| {
            let (lon_f, lat_f) = centre_lonlat(f, nside);
            let gamma = haversine(lat_c, lon_c, lat_f, lon_f);
            let t = gamma / sigma;
            if gamma > radius {
                0.0
            } else {
                (-0.5 * (t * t)).exp()
            }
        })
        .collect();
    let mut row_sum = w.iter().fold(0.0, |s, &v| s + v);
    if row_sum <= 0.0 && !w.is_empty() {
        // Every candidate cut: uniform over the available pixels (library fallback).
        let u = 1.0 / w.len() as f64;
        w.iter_mut().for_each(|v| *v = u);
        row_sum = w.iter().fold(0.0, |s, &v| s + v);
    }
    let scale = match norm {
        WeightNorm::L1 => row_sum.max(1e-300),
        WeightNorm::L2 => w.iter().fold(0.0, |s, &v| s + v * v).sqrt().max(1e-300),
    };
    w.iter().map(|&v| v / scale).collect()
}

/// Row of the coarse cell `c` (at `nside / 2`) of the full-sky decimation
/// from `nside`: its fine cells (sorted) and nonzero weights.  It is the
/// row [`smooth_decimation`] builds on any grid holding all its candidates,
/// to the bit.
pub fn decimation_row(c: i64, nside: i64, params: &DecimationParams) -> (Vec<i64>, Vec<f64>) {
    let pix_deg = py_degrees((4.0 * PI / npix(nside) as f64).sqrt());
    let radius = py_radians(params.radius_deg.unwrap_or(3.0 * pix_deg));
    let sigma = match params.sigma_deg {
        None => radius / 2.0,
        Some(s) => py_radians(s),
    };
    let mut fine: Vec<i64> = candidates(&[c], nside).swap_remove(0).into_iter().filter(|&v| v >= 0).collect();
    fine.sort_unstable();
    fine.dedup();
    let w = row_weights(c, &fine, nside, radius, sigma, params.norm);
    fine.into_iter().zip(w).filter(|(_, v)| *v != 0.0).unzip()
}
