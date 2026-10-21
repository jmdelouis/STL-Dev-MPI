//! Distributed evaluation of an STL-Dev `stats` fixture and comparison with
//! it: shared by the thread tests and the `stl-mpi-check` binary.

use crate::comm::Comm;
use crate::decomposition::{build_partition_for, Decomposition};
use crate::halo::HaloExchange;
use num_complex::Complex64;
use std::path::{Path, PathBuf};
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::stencil::Gauge;

/// One fixture case directory (`fixtures/stats/<case>`).
pub struct Fixture {
    pub dir: PathBuf,
    pub meta: serde_json::Value,
}

impl Fixture {
    pub fn open(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref().to_path_buf();
        let text = std::fs::read_to_string(dir.join("meta.json"))
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Fixture { meta: serde_json::from_str(&text).unwrap(), dir }
    }

    pub fn name(&self) -> &str {
        self.meta["name"].as_str().unwrap()
    }

    pub fn int(&self, key: &str) -> i64 {
        self.meta[key].as_i64().unwrap_or_else(|| panic!("{}: no integer {key}", self.name()))
    }

    pub fn f64(&self, array: &str) -> Vec<f64> {
        let path = self.dir.join(format!("{array}.npy"));
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap().into_vec::<f64>().unwrap()
    }

    pub fn complex(&self, name: &str) -> Vec<Complex64> {
        let (re, im) = (self.f64(&format!("{name}_re")), self.f64(&format!("{name}_im")));
        re.iter().zip(&im).map(|(&a, &b)| Complex64::new(a, b)).collect()
    }

    pub fn geometry(&self) -> LocalGeometry {
        LocalGeometry::new(
            self.int("nside"),
            self.int("kernel_sz") as usize,
            self.int("L") as usize,
            Gauge::parse(self.meta["gauge"].as_str().unwrap()).unwrap(),
            self.int("block_pixels") as usize,
        )
    }

    pub fn config(&self) -> EngineConfig {
        let mut cfg = EngineConfig::new(self.int("J") as usize, self.meta["fewer"].as_bool().unwrap());
        cfg.j_to_dg = self.meta["j_to_dg"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        cfg
    }

    pub fn plan(&self) -> Vec<Coefficient> {
        let cfg = self.config();
        build_plan(cfg.j_count, self.int("L") as usize, &cfg.j_to_dg, self.int("channels") as usize, None, &Family::ALL)
    }
}

/// What one rank computed: statistics (identical on every rank) and the
/// owned slices `[B, C, end - start]` of the gradients.
#[derive(Clone, Debug)]
pub struct RankResult {
    pub values: Vec<Complex64>,
    pub owned: Vec<(i64, i64)>,
    pub maps_gradient: Vec<Vec<f64>>,
    pub bands: Vec<Vec<f64>>,
    /// Cells held per level (owned + halo) and halo cells per level.
    pub held: Vec<usize>,
    pub halo: Vec<usize>,
}

/// Forward and reverse sweeps of the fixture on the rank of `comm`.  Halo
/// cells of the input map are left at zero: they must come from the owners.
pub fn run_rank<C: Comm>(fx: &Fixture, comm: &C) -> RankResult {
    run_rank_unit(fx, comm, None)
}

/// [`run_rank`] with the level-0 ranges aligned on `unit` instead of the
/// default (to test ranges that do not align with the coarse levels).
pub fn run_rank_unit<C: Comm>(fx: &Fixture, comm: &C, unit: Option<i64>) -> RankResult {
    let geo = fx.geometry();
    let cfg = fx.config();
    let plan = fx.plan();
    let (batch, channels) = (fx.int("batch") as usize, fx.int("channels") as usize);
    let map0 = fx.f64("map0");
    let gradient = fx.complex("gradient");
    run_problem(&geo, &cfg, &plan, &map0, &gradient, batch, channels, comm, unit)
}

/// Forward and reverse sweeps of any problem on the rank of `comm`
/// (`map0` full sky `[B, C, N_0]`, read on the owned cells only).
#[allow(clippy::too_many_arguments)]
pub fn run_problem<C: Comm>(
    geo: &LocalGeometry,
    cfg: &EngineConfig,
    plan: &[Coefficient],
    map0: &[f64],
    gradient: &[Complex64],
    batch: usize,
    channels: usize,
    comm: &C,
    unit: Option<i64>,
) -> RankResult {
    let dec = match unit {
        Some(u) => Decomposition::with_unit(geo, cfg.levels(), comm.size(), u),
        None => Decomposition::new(geo, cfg.levels(), comm.size()),
    };
    let part = build_partition_for(geo, cfg, &dec, comm.rank());
    let ex = HaloExchange::new(comm, &dec, &part);
    let (n0, layout0) = (geo.pixels(0), &part.layouts[0]);
    let mut x = vec![0.0; batch * channels * layout0.len()];
    for r in 0..batch * channels {
        for p in 0..layout0.len() {
            let id = layout0.id(p);
            if part.owns(0, id) {
                x[r * layout0.len() + p] = map0[r * n0 + id as usize];
            }
        }
    }
    let held = part.layouts.iter().map(|l| l.len()).collect();
    let halo = ex.halo_cells();
    let mut engine = LocalEngine::distributed(geo, cfg, part, &ex, x, batch, channels);
    let values = engine.forward(plan);
    let mg = engine.reverse(plan, gradient);
    let maps_gradient: Vec<Vec<f64>> = mg.iter().enumerate().map(|(d, g)| engine.owned_values(d, g)).collect();
    let (_, bands) = engine.analysis_vjp(mg);
    let bands = bands.iter().enumerate().map(|(d, g)| engine.owned_values(d, g)).collect();
    RankResult { values, owned: engine.part.owned.clone(), maps_gradient, bands, held, halo }
}

/// Overlap mode (`cfg.overlap` set): statistics, owned level-0 range and
/// owned slice `[B, C, end - start]` of the map cotangent (the band and
/// per-level cotangents are not formed in this mode).
#[allow(clippy::too_many_arguments)]
pub fn run_problem_overlap<C: Comm>(
    geo: &LocalGeometry,
    cfg: &EngineConfig,
    plan: &[Coefficient],
    map0: &[f64],
    gradient: &[Complex64],
    batch: usize,
    channels: usize,
    comm: &C,
) -> (Vec<Complex64>, (i64, i64), Vec<f64>) {
    assert!(cfg.overlap.is_some());
    let dec = Decomposition::new(geo, cfg.levels(), comm.size());
    let part = build_partition_for(geo, cfg, &dec, comm.rank());
    let ex = HaloExchange::new(comm, &dec, &part);
    let (n0, layout0) = (geo.pixels(0), &part.layouts[0]);
    let mut x = vec![0.0; batch * channels * layout0.len()];
    for r in 0..batch * channels {
        for p in 0..layout0.len() {
            let id = layout0.id(p);
            if part.owns(0, id) {
                x[r * layout0.len() + p] = map0[r * n0 + id as usize];
            }
        }
    }
    let owned0 = part.owned[0];
    let mut engine = LocalEngine::distributed(geo, cfg, part, &ex, x, batch, channels);
    let values = engine.forward(plan);
    let mg = engine.reverse(plan, gradient);
    let g0 = engine.map_cotangent(mg);
    (values, owned0, engine.owned_values(0, &g0))
}

/// Collective: every rank's result on rank 0 (`None` elsewhere).
pub fn gather_to_root<C: Comm>(comm: &C, res: &RankResult, rows: usize) -> Option<Vec<RankResult>> {
    let size = comm.size();
    let owned: Vec<i64> = res.owned.iter().flat_map(|&(s, e)| [s, e]).collect();
    let fields: Vec<f64> = res.maps_gradient.iter().chain(&res.bands).flatten().copied().collect();
    let meta: Vec<i64> = res.held.iter().chain(&res.halo).map(|&v| v as i64).collect();
    let to_root = |v: Vec<i64>| (0..size).map(|p| if p == 0 { v.clone() } else { Vec::new() }).collect();
    let owned_all = comm.alltoallv_i64(to_root(owned));
    let meta_all = comm.alltoallv_i64(to_root(meta));
    let fields_all = comm.alltoallv_f64((0..size).map(|p| if p == 0 { fields.clone() } else { Vec::new() }).collect());
    if comm.rank() != 0 {
        return None;
    }
    let levels = res.owned.len();
    Some(
        (0..size)
            .map(|p| {
                let owned: Vec<(i64, i64)> = owned_all[p].chunks(2).map(|c| (c[0], c[1])).collect();
                let mut at = 0;
                let mut take = |d: usize| {
                    let len = rows * (owned[d].1 - owned[d].0) as usize;
                    let v = fields_all[p][at..at + len].to_vec();
                    at += len;
                    v
                };
                let maps_gradient = (0..levels).map(&mut take).collect();
                let bands = (0..levels).map(&mut take).collect();
                let m = &meta_all[p];
                RankResult {
                    values: res.values.clone(),
                    owned,
                    maps_gradient,
                    bands,
                    held: m[..levels].iter().map(|&v| v as usize).collect(),
                    halo: m[levels..].iter().map(|&v| v as usize).collect(),
                }
            })
            .collect(),
    )
}

/// Flat dump of an assembled result (statistics re/im, map cotangents of
/// every level, band cotangents), for comparisons across runs.
pub fn to_flat(a: &Assembled) -> Vec<f64> {
    flat(&a.values).into_iter().chain(a.maps_gradient.iter().chain(&a.bands).flatten().copied()).collect()
}

pub fn from_flat(v: &[f64], like: &Assembled) -> Assembled {
    let nv = like.values.len();
    let values = v[..2 * nv].chunks(2).map(|c| Complex64::new(c[0], c[1])).collect();
    let mut at = 2 * nv;
    let mut take = |f: &Vec<f64>| {
        let out = v[at..at + f.len()].to_vec();
        at += f.len();
        out
    };
    let maps_gradient = like.maps_gradient.iter().map(&mut take).collect();
    let bands = like.bands.iter().map(&mut take).collect();
    Assembled { values, maps_gradient, bands }
}

/// Full-sky gradients assembled from the ranks' owned slices.
#[derive(Clone, Debug)]
pub struct Assembled {
    pub values: Vec<Complex64>,
    pub maps_gradient: Vec<Vec<f64>>,
    pub bands: Vec<Vec<f64>>,
}

pub fn assemble(results: &[RankResult], rows: usize) -> Assembled {
    let levels = results[0].owned.len();
    for r in results {
        assert_eq!(r.values, results[0].values, "statistics differ between ranks");
    }
    let field = |pick: &dyn Fn(&RankResult) -> &Vec<Vec<f64>>, d: usize| -> Vec<f64> {
        let n = results.iter().map(|r| r.owned[d].1).max().unwrap() as usize;
        let mut out = vec![0.0; rows * n];
        for res in results {
            let (s, e) = res.owned[d];
            let len = (e - s) as usize;
            let part = &pick(res)[d];
            for row in 0..rows {
                out[row * n + s as usize..][..len].copy_from_slice(&part[row * len..][..len]);
            }
        }
        out
    };
    Assembled {
        values: results[0].values.clone(),
        maps_gradient: (0..levels).map(|d| field(&|r| &r.maps_gradient, d)).collect(),
        bands: (0..levels).map(|d| field(&|r| &r.bands, d)).collect(),
    }
}

/// Worst error of `a` against `b` relative to `atol + rtol |b|`.
pub fn close_ratio(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs() / (atol + rtol * y.abs())))
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn flat(v: &[Complex64]) -> Vec<f64> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// Comparison of an assembled result with the fixture (CLAUDE.md
/// validation 2 and 3) and, optionally, with another assembled result
/// (validation 4: summation-order differences only, below 1e-13).
#[derive(Clone, Debug)]
pub struct Report {
    /// Worst ratio to the statistics tolerance (1e-11 + 1e-10 relative).
    pub stats_ratio: f64,
    /// Worst ratio to the gradient tolerance (1e-10 + 1e-9 relative), all
    /// levels and bands.
    pub gradient_ratio: f64,
    /// Largest absolute difference to the other result, statistics and gradients.
    pub other_diff: Option<f64>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.stats_ratio <= 1.0 && self.gradient_ratio <= 1.0 && self.other_diff.is_none_or(|d| d <= 1e-13)
    }
}

pub fn compare(fx: &Fixture, a: &Assembled, other: Option<&Assembled>) -> Report {
    let stats_ratio = close_ratio(&flat(&a.values), &flat(&fx.complex("values")), 1e-11, 1e-10);
    let mut gradient_ratio = 0f64;
    for (d, g) in a.maps_gradient.iter().enumerate() {
        gradient_ratio = gradient_ratio.max(close_ratio(g, &fx.f64(&format!("maps_gradient{d}")), 1e-10, 1e-9));
    }
    for (d, g) in a.bands.iter().enumerate() {
        gradient_ratio = gradient_ratio.max(close_ratio(g, &fx.f64(&format!("band_gradient{d}")), 1e-10, 1e-9));
    }
    let other_diff = other.map(|o| {
        let mut m = max_abs_diff(&flat(&a.values), &flat(&o.values));
        for (x, y) in a.maps_gradient.iter().zip(&o.maps_gradient).chain(a.bands.iter().zip(&o.bands)) {
            m = m.max(max_abs_diff(x, y));
        }
        m
    });
    Report { stats_ratio, gradient_ratio, other_diff }
}
