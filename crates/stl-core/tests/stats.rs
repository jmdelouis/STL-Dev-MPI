//! Milestone 3: single-process level sweeps against STL-Dev's
//! `LocalScattering` (`apply_local` / `vjp_local`, fixtures `fixtures/stats`).

mod common;

use common::{cases, max_abs_diff, Case};
use num_complex::Complex64;
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::stencil::Gauge;

fn reference_plan(case: &Case) -> Vec<Coefficient> {
    let text = std::fs::read_to_string(case.dir.join("plan.json")).unwrap();
    let rows: serde_json::Value = serde_json::from_str(&text).unwrap();
    let opt = |v: &serde_json::Value| v.as_u64().map(|x| x as usize);
    rows.as_array()
        .unwrap()
        .iter()
        .map(|r| Coefficient {
            family: Family::parse(r[0].as_str().unwrap()).unwrap(),
            native_dg: r[1].as_u64().unwrap() as usize,
            c1: r[2].as_u64().unwrap() as usize,
            c2: r[3].as_u64().unwrap() as usize,
            j1: opt(&r[4]),
            j2: opt(&r[5]),
            j3: opt(&r[6]),
            l1: opt(&r[7]),
            l2: opt(&r[8]),
            l3: opt(&r[9]),
        })
        .collect()
}

fn complex(case: &Case, name: &str) -> Vec<Complex64> {
    let (re, im) = (case.f64(&format!("{name}_re")), case.f64(&format!("{name}_im")));
    re.iter().zip(&im).map(|(&a, &b)| Complex64::new(a, b)).collect()
}

/// `|a - b| <= atol + rtol |b|` everywhere; returns the worst ratio to the bound.
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
    assert_eq!(cfg.levels() as i64, case.int("levels"));
    let channels = case.int("channels") as usize;
    let plan = build_plan(j, l, &cfg.j_to_dg, channels, None, &Family::ALL);
    assert_eq!(plan, reference_plan(case), "{}: plan", case.name());
    Setup { geo, cfg, batch: case.int("batch") as usize, channels, plan }
}

/// The map gradient is the derivative of `Re sum(conj(g) * S(x))`:
/// central finite difference along a random direction, independently of
/// the reference.
#[test]
fn gradient_matches_finite_difference() {
    for case in cases("stats", "stats").into_iter().filter(|c| c.int("nside") == 16 && c.int("block_pixels") == 64) {
        let s = setup(&case);
        let x = case.f64("map0");
        let g = complex(&case, "gradient");
        let pairing = |x: Vec<f64>| -> f64 {
            let mut e = LocalEngine::new(&s.geo, &s.cfg, x, s.batch, s.channels);
            e.forward(&s.plan).iter().zip(&g).map(|(v, gi)| (gi.conj() * v).re).sum()
        };
        let mut engine = LocalEngine::new(&s.geo, &s.cfg, x.clone(), s.batch, s.channels);
        engine.forward(&s.plan);
        let maps_gradient = engine.reverse(&s.plan, &g);
        let (grad, _) = engine.analysis_vjp(maps_gradient);
        // Deterministic pseudo-random direction.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let v: Vec<f64> = (0..x.len())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let eps = 1e-5;
        let shifted = |sign: f64| x.iter().zip(&v).map(|(a, b)| a + sign * eps * b).collect::<Vec<_>>();
        let fd = (pairing(shifted(1.0)) - pairing(shifted(-1.0))) / (2.0 * eps);
        let exact: f64 = grad.iter().zip(&v).map(|(a, b)| a * b).sum();
        let rel = (fd - exact).abs() / exact.abs();
        eprintln!("{}: directional derivative {exact:.12e}, finite difference {fd:.12e}, rel {rel:.1e}", case.name());
        assert!(rel <= 1e-7, "{}: {exact} vs {fd}", case.name());
    }
}

/// The geometry rebuilt by sub-blocks of rows gives the same statistics
/// (forward rows are computed identically) and gradients up to the order
/// in which the sub-blocks' adjoints are added.
#[test]
fn geometry_by_sub_blocks() {
    for case in cases("stats", "stats").into_iter().filter(|c| c.int("nside") == 16 && c.int("block_pixels") == 256) {
        let s = setup(&case);
        let run = |chunk: Option<usize>| {
            let mut geo = s.geo.clone();
            geo.conv_chunk = chunk;
            let mut e = LocalEngine::new(&geo, &s.cfg, case.f64("map0"), s.batch, s.channels);
            let values = e.forward(&s.plan);
            let mg = e.reverse(&s.plan, &complex(&case, "gradient"));
            (values, e.analysis_vjp(mg).0)
        };
        let (v_whole, g_whole) = run(None);
        let (v_chunk, g_chunk) = run(Some(37));
        assert_eq!(v_whole, v_chunk, "{}: statistics", case.name());
        let scale = g_whole.iter().fold(0f64, |m, v| m.max(v.abs()));
        let err = max_abs_diff(&g_whole, &g_chunk);
        eprintln!("{}: sub-block gradient difference {err:.1e} (scale {scale:.1e})", case.name());
        assert!(err <= 1e-14 * scale, "{}: gradient {err:e}", case.name());
    }
}

/// Second moments: the values are those of `forward` (bit for bit), and
/// for `mean` and `var` the spatial variance is the variance of `x` and of
/// `x^2` over the map.
#[test]
fn coefficient_spatial_variance() {
    let case = cases("stats", "stats").into_iter().find(|c| c.name().contains("C2_B2_full_b64")).unwrap();
    let s = setup(&case);
    let x = case.f64("map0");
    let mut e = LocalEngine::new(&s.geo, &s.cfg, x.clone(), s.batch, s.channels);
    let values = e.forward(&s.plan);
    let mut e = LocalEngine::new(&s.geo, &s.cfg, x.clone(), s.batch, s.channels);
    let (values2, var) = e.forward_moments(&s.plan);
    assert_eq!(values, values2);
    let n = s.geo.pixels(0);
    for (i, c) in s.plan.iter().enumerate() {
        if !matches!(c.family, Family::Mean | Family::Var) {
            continue;
        }
        for b in 0..s.batch {
            let row = &x[(b * s.channels + c.c1) * n..][..n];
            let f: Vec<f64> = row.iter().map(|v| if c.family == Family::Mean { *v } else { v * v }).collect();
            let m = f.iter().sum::<f64>() / n as f64;
            let expect = f.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
            let got = var[b * s.plan.len() + i];
            assert!((got - expect).abs() <= 1e-12 * expect, "{:?} batch {b}: {got} vs {expect}", c.family);
        }
    }
    assert!(var.iter().all(|v| *v >= 0.0));
}

#[test]
fn statistics_and_gradients() {
    for case in cases("stats", "stats") {
        let s = setup(&case);
        let mut engine = LocalEngine::new(&s.geo, &s.cfg, case.f64("map0"), s.batch, s.channels);
        for (d, x) in engine.maps.iter().enumerate() {
            let err = max_abs_diff(x, &case.f64(&format!("map{d}")));
            assert!(err <= 1e-12, "{}: map {d} {err:e}", case.name());
        }

        // Statistics: 1e-11 absolute, 1e-10 relative (CLAUDE.md, validation 2).
        let values = engine.forward(&s.plan);
        let reference = complex(&case, "values");
        let flat = |v: &[Complex64]| v.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<_>>();
        let ratio = close_ratio(&flat(&values), &flat(&reference), 1e-11, 1e-10);
        let err = max_abs_diff(&flat(&values), &flat(&reference));
        for (&(d, n), m) in &engine.m {
            let e = max_abs_diff(m, &case.f64(&format!("m{d}_{n}")));
            assert!(e <= 1e-12, "{}: stored moduli m[{d}][{n}] {e:e}", case.name());
        }

        // Gradients: 1e-10 (validation 3), per level and after analysis_vjp.
        let maps_gradient = engine.reverse(&s.plan, &complex(&case, "gradient"));
        let mut gerr = 0f64;
        for (d, g) in maps_gradient.iter().enumerate() {
            let r = close_ratio(g, &case.f64(&format!("maps_gradient{d}")), 1e-10, 1e-9);
            gerr = gerr.max(max_abs_diff(g, &case.f64(&format!("maps_gradient{d}"))));
            assert!(r <= 1.0, "{}: maps gradient {d} ({r} x tolerance)", case.name());
        }
        let (_, bands) = engine.analysis_vjp(maps_gradient);
        for (d, g) in bands.iter().enumerate() {
            let r = close_ratio(g, &case.f64(&format!("band_gradient{d}")), 1e-10, 1e-9);
            assert!(r <= 1.0, "{}: band gradient {d} ({r} x tolerance)", case.name());
        }
        let scale = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
        let gscale = (0..engine.levels()).fold(0f64, |m, d| m.max(scale(&case.f64(&format!("maps_gradient{d}")))));
        eprintln!(
            "{}: {} coefficients, statistics max error {err:.2e} (max |S| {:.2e}, {ratio:.3} x tol), \
             gradient max error {gerr:.2e} (max |g| {gscale:.2e})",
            case.name(),
            s.plan.len(),
            scale(&flat(&reference)),
        );
        assert!(ratio <= 1.0, "{}: statistics ({ratio} x tolerance)", case.name());
    }
}
