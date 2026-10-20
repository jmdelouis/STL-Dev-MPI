//! The device sweeps against STL-Dev's `LocalScattering` (`apply_local` /
//! `vjp_local`, fixtures `fixtures/stats`) with the tolerances of the CPU
//! tests, and against the CPU engine on the same inputs.

mod common;

use common::{cases, max_abs_diff, Case};
use num_complex::Complex64;
use std::sync::Arc;
use stl_core::engine::{EngineConfig, LocalEngine, Partition, Serial};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::stencil::Gauge;
use stl_gpu::{Gpu, GpuEngine, GpuOptions};

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::new(0) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            eprintln!("no device, test skipped: {e}");
            None
        }
    }
}

fn complex(case: &Case, name: &str) -> Vec<Complex64> {
    let (re, im) = (case.f64(&format!("{name}_re")), case.f64(&format!("{name}_im")));
    re.iter().zip(&im).map(|(&a, &b)| Complex64::new(a, b)).collect()
}

fn close_ratio(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs() / (atol + rtol * y.abs())))
}

struct Setup {
    geo: LocalGeometry,
    cfg: EngineConfig,
    batch: usize,
    channels: usize,
    plan: Vec<Coefficient>,
}

fn setup(case: &Case) -> Setup {
    let l = case.int("L") as usize;
    let j = case.int("J") as usize;
    let geo = LocalGeometry::new(
        case.int("nside"),
        case.int("kernel_sz") as usize,
        l,
        Gauge::parse(case.str("gauge")).unwrap(),
        case.int("block_pixels") as usize,
    );
    let mut cfg = EngineConfig::new(j, case.meta["fewer"].as_bool().unwrap());
    cfg.j_to_dg = case.meta["j_to_dg"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let channels = case.int("channels") as usize;
    let plan = build_plan(j, l, &cfg.j_to_dg, channels, None, &Family::ALL);
    Setup { geo, cfg, batch: case.int("batch") as usize, channels, plan }
}

fn flat(v: &[Complex64]) -> Vec<f64> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// Statistics, stored moduli, map cotangents and band gradients against the
/// fixtures; `block_rows` small enough to give several device blocks per
/// level on some cases.
#[test]
fn statistics_and_gradients() {
    let Some(gpu) = gpu() else { return };
    for (case, block_rows) in cases("stats", "stats").into_iter().zip([0usize, 100, 1000, 0, 37, 0, 500, 0].into_iter().cycle()) {
        let s = setup(&case);
        let part = Partition::serial(&s.geo, s.cfg.levels());
        let opts = GpuOptions { block_rows: if block_rows == 0 { 1 << 20 } else { block_rows }, ..Default::default() };
        let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &opts);
        e.load(&Serial, &case.f64("map0"));
        for d in 0..e.levels() {
            let err = max_abs_diff(&e.map_host(d), &case.f64(&format!("map{d}")));
            assert!(err <= 1e-12, "{}: map {d} {err:e}", case.name());
        }
        let (values, _) = e.forward(&Serial, &s.plan, false);
        let reference = complex(&case, "values");
        let ratio = close_ratio(&flat(&values), &flat(&reference), 1e-11, 1e-10);
        let err = max_abs_diff(&flat(&values), &flat(&reference));
        for d in 0..e.levels() {
            for n in d + 1..e.levels() {
                let m = e.m_host(d, n);
                let er = max_abs_diff(&m, &case.f64(&format!("m{d}_{n}")));
                assert!(er <= 1e-12, "{}: stored moduli m[{d}][{n}] {er:e}", case.name());
            }
        }
        let mg = e.reverse(&Serial, &s.plan, &complex(&case, "gradient"));
        let mut gerr = 0f64;
        for (d, g) in mg.iter().enumerate() {
            let g = gpu.download(g);
            let r = close_ratio(&g, &case.f64(&format!("maps_gradient{d}")), 1e-10, 1e-9);
            gerr = gerr.max(max_abs_diff(&g, &case.f64(&format!("maps_gradient{d}"))));
            assert!(r <= 1.0, "{}: maps gradient {d} ({r} x tolerance)", case.name());
        }
        let g0 = e.analysis_vjp(&Serial, mg);
        let bands = e.bands(&Serial, gpu.download(&g0));
        for (d, g) in bands.iter().enumerate() {
            let r = close_ratio(g, &case.f64(&format!("band_gradient{d}")), 1e-10, 1e-9);
            assert!(r <= 1.0, "{}: band gradient {d} ({r} x tolerance)", case.name());
        }
        eprintln!(
            "{}: {} coefficients, {:?} device blocks, statistics max error {err:.2e} ({ratio:.2e} x tol), gradient max error {gerr:.2e}",
            case.name(),
            s.plan.len(),
            e.blocks_per_level(),
        );
        assert!(ratio <= 1.0, "{}: statistics ({ratio} x tolerance)", case.name());
    }
}

/// The geometry cache only saves work: every policy and budget gives the
/// same statistics and gradients, to the bit, over two evaluations.
#[test]
fn geometry_cache_policies_agree() {
    use stl_gpu::GeometryCache;
    let Some(gpu) = gpu() else { return };
    let case = cases("stats", "stats").into_iter().find(|c| c.name().contains("n32_J4_L4_k5_C1")).unwrap();
    let s = setup(&case);
    let x = case.f64("map0");
    let g = complex(&case, "gradient");
    let run = |cache: GeometryCache, cache_bytes: usize| {
        let part = Partition::serial(&s.geo, s.cfg.levels());
        let opts = GpuOptions { block_rows: 1500, cache, cache_bytes, ..Default::default() };
        let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &opts);
        let mut out = Vec::new();
        for _ in 0..2 {
            e.load(&Serial, &x);
            let (v, _) = e.forward(&Serial, &s.plan, false);
            let mg = e.reverse(&Serial, &s.plan, &g);
            out.push((v, gpu.download(&e.analysis_vjp(&Serial, mg))));
        }
        out
    };
    let reference = run(GeometryCache::None, 0);
    assert_eq!(reference[0], reference[1]);
    for (cache, bytes) in [
        (GeometryCache::Full, 1 << 34),
        (GeometryCache::Full, 3 << 20),
        (GeometryCache::Transposition, 1 << 34),
        (GeometryCache::Transposition, 2 << 20),
    ] {
        assert_eq!(run(cache, bytes), reference, "{cache:?} with {bytes} bytes");
    }
}

/// Device against CPU engine on the same inputs: statistics, their spatial
/// variance and the level-0 gradient agree to rounding.
#[test]
fn device_matches_cpu_engine() {
    let Some(gpu) = gpu() else { return };
    for case in cases("stats", "stats") {
        let s = setup(&case);
        let x = case.f64("map0");
        let g = complex(&case, "gradient");
        let mut cpu = LocalEngine::new(&s.geo, &s.cfg, x.clone(), s.batch, s.channels);
        let (v_cpu, var_cpu) = cpu.forward_moments(&s.plan);
        let mg = cpu.reverse(&s.plan, &g);
        let (g_cpu, _) = cpu.analysis_vjp(mg);

        let part = Partition::serial(&s.geo, s.cfg.levels());
        let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &GpuOptions::default());
        e.load(&Serial, &x);
        let (v_gpu, var_gpu) = e.forward(&Serial, &s.plan, true);
        let mg = e.reverse(&Serial, &s.plan, &g);
        let g_gpu = gpu.download(&e.analysis_vjp(&Serial, mg));
        let scale = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
        let ev = max_abs_diff(&flat(&v_cpu), &flat(&v_gpu)) / scale(&flat(&v_cpu));
        let evar = max_abs_diff(&var_cpu, &var_gpu) / scale(&var_cpu);
        let eg = max_abs_diff(&g_cpu, &g_gpu) / scale(&g_cpu);
        eprintln!("{}: statistics {ev:.1e}, variances {evar:.1e}, gradient {eg:.1e} (relative to max)", case.name());
        assert!(ev <= 1e-13 && evar <= 1e-12 && eg <= 1e-12, "{}", case.name());
    }
}
