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
use stl_gpu::{ConvForm, Gpu, GpuEngine, GpuOptions};

/// Both forms of the convolution geometry.
const CONVS: [ConvForm; 2] = [ConvForm::Stencils, ConvForm::Dia];

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
/// fixtures, with both geometries; `block_rows` small enough to give
/// several device blocks per level on some cases.
#[test]
fn statistics_and_gradients() {
    let Some(gpu) = gpu() else { return };
    for conv in CONVS {
        for (case, block_rows) in cases("stats", "stats").into_iter().zip([0usize, 100, 1000, 0, 37, 0, 500, 0].into_iter().cycle()) {
            let s = setup(&case);
            let part = Partition::serial(&s.geo, s.cfg.levels());
            let opts = GpuOptions { block_rows: if block_rows == 0 { 1 << 20 } else { block_rows }, conv, ..Default::default() };
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
                "{} ({}): {} coefficients, {:?} device blocks, statistics max error {err:.2e} ({ratio:.2e} x tol), gradient max error {gerr:.2e}",
                case.name(),
                conv.name(),
                s.plan.len(),
                e.blocks_per_level(),
            );
            assert!(ratio <= 1.0, "{}: statistics ({ratio} x tolerance)", case.name());
        }
    }
}

/// The geometry cache only saves work: every policy and budget, on the
/// device and on the host, gives the same statistics and gradients, to the
/// bit, over three evaluations (the third one reads the host copies made by
/// the first two), with either geometry.
#[test]
fn geometry_cache_policies_agree() {
    use stl_gpu::GeometryCache;
    let Some(gpu) = gpu() else { return };
    let case = cases("stats", "stats").into_iter().find(|c| c.name().contains("n32_J4_L4_k5_C1")).unwrap();
    let s = setup(&case);
    let x = case.f64("map0");
    let g = complex(&case, "gradient");
    let run = |conv: ConvForm, cache: GeometryCache, cache_bytes: usize, host_cache_bytes: usize| {
        let part = Partition::serial(&s.geo, s.cfg.levels());
        // A budget ending in 1: calibrate the host copies (keep or drop them).
        let host_calibrate = host_cache_bytes % 2 == 1;
        let opts = GpuOptions { block_rows: 1500, conv, cache, cache_bytes, host_cache_bytes, host_calibrate, ..Default::default() };
        let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &opts);
        let mut out = Vec::new();
        for _ in 0..3 {
            e.load(&Serial, &x);
            let (v, _) = e.forward(&Serial, &s.plan, false);
            let mg = e.reverse(&Serial, &s.plan, &g);
            out.push((v, gpu.download(&e.analysis_vjp(&Serial, mg))));
        }
        let (budget, used) = e.geometry_host();
        assert!(used <= budget, "{cache:?}: host copies {used} over the budget {budget}");
        let uploads = e.timers.contains_key("gpu_geometry_upload");
        (out, used, uploads)
    };
    for conv in CONVS {
        let (reference, _, _) = run(conv, GeometryCache::None, 0, 0);
        assert_eq!(reference[0], reference[1]);
        assert_eq!(reference[0], reference[2]);
        for (cache, bytes, host) in [
            (GeometryCache::Full, 1 << 34, 0),
            (GeometryCache::Full, 3 << 20, 0),
            (GeometryCache::Transposition, 1 << 34, 0),
            (GeometryCache::Transposition, 2 << 20, 0),
            (GeometryCache::Transposition, 64 << 10, 0),
            (GeometryCache::Full, 3 << 20, 1 << 34),
            (GeometryCache::Transposition, 2 << 20, 1 << 34),
            (GeometryCache::Transposition, 2 << 20, 1 << 20),
            (GeometryCache::Host, 1 << 34, 1 << 34),
            (GeometryCache::Host, 1 << 34, 1 << 20),
            (GeometryCache::Host, 1 << 34, 0),
            (GeometryCache::Host, 1 << 34, (1 << 34) + 1),
            (GeometryCache::Transposition, 2 << 20, (1 << 34) + 1),
        ] {
            let (out, used, uploads) = run(conv, cache, bytes, host);
            assert_eq!(out, reference, "{conv:?}: {cache:?} with {bytes} bytes, {host} on the host");
            if cache == GeometryCache::Host && host == 1 << 34 {
                assert!(used > 0 && uploads, "{conv:?}: Host policy made no host copy");
            }
        }
    }
}

/// Wavelet fields recomputed per block in the reverse sweep give, to the
/// bit, what the kept fields give: statistics, variances, stored moduli and
/// map cotangents of every level, over two evaluations, with either
/// geometry and with one or several device blocks per level; also for a
/// plan without S3 / S4.
#[test]
fn recomputed_fields_match_kept_fields() {
    use stl_gpu::FieldStore;
    let Some(gpu) = gpu() else { return };
    for conv in CONVS {
        for (case, block_rows) in cases("stats", "stats").into_iter().zip([37usize, 1 << 20, 500].into_iter().cycle()) {
            let s = setup(&case);
            let x = case.f64("map0");
            let g = complex(&case, "gradient");
            let short = build_plan(case.int("J") as usize, s.geo.l, &s.cfg.j_to_dg, s.channels, None, &Family::ALL[..4]);
            let g_short: Vec<Complex64> =
                (0..s.batch).flat_map(|b| short.iter().map(|c| g[b * s.plan.len() + s.plan.iter().position(|p| p == c).unwrap()]).collect::<Vec<_>>()).collect();
            let run = |fields: FieldStore| {
                let part = Partition::serial(&s.geo, s.cfg.levels());
                let opts = GpuOptions { block_rows, conv, fields, ..Default::default() };
                let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &opts);
                assert_eq!(e.field_store(), fields);
                let mut out = Vec::new();
                for (plan, g) in [(&s.plan, &g), (&short, &g_short), (&s.plan, &g)] {
                    e.load(&Serial, &x);
                    let (v, var) = e.forward(&Serial, plan, true);
                    let m: Vec<Vec<f64>> = e.m.keys().map(|&(d, n)| e.m_host(d, n)).collect();
                    let mg: Vec<Vec<f64>> = e.reverse(&Serial, plan, g).iter().map(|f| gpu.download(f)).collect();
                    out.push((flat(&v), var, m, mg));
                }
                (out, e.blocks_per_level())
            };
            let ((kept, blocks), (recomputed, _)) = (run(FieldStore::Keep), run(FieldStore::Recompute));
            eprintln!("{} ({}): {blocks:?} device blocks", case.name(), conv.name());
            assert!(kept == recomputed, "{} ({}): recomputed fields differ from kept fields", case.name(), conv.name());
        }
    }
}

/// The DIA geometry against the bound stencils: the same operator, so the
/// same statistics, variances and gradients to the summation orders.
#[test]
fn dia_matches_stencils() {
    let Some(gpu) = gpu() else { return };
    for case in cases("stats", "stats") {
        let s = setup(&case);
        let x = case.f64("map0");
        let g = complex(&case, "gradient");
        let run = |conv: ConvForm| {
            let part = Partition::serial(&s.geo, s.cfg.levels());
            let opts = GpuOptions { block_rows: 1500, conv, ..Default::default() };
            let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &opts);
            assert_eq!(e.conv_form(), conv);
            e.load(&Serial, &x);
            let (v, var) = e.forward(&Serial, &s.plan, true);
            let mg = e.reverse(&Serial, &s.plan, &g);
            (flat(&v), var, gpu.download(&e.analysis_vjp(&Serial, mg)))
        };
        let (a, b) = (run(ConvForm::Stencils), run(ConvForm::Dia));
        let rel = |p: &[f64], q: &[f64]| max_abs_diff(p, q) / p.iter().fold(0f64, |m, v| m.max(v.abs()));
        let (ev, evar, eg) = (rel(&a.0, &b.0), rel(&a.1, &b.1), rel(&a.2, &b.2));
        eprintln!("{}: DIA vs stencils: statistics {ev:.1e}, variances {evar:.1e}, gradient {eg:.1e} (relative to max)", case.name());
        assert!(ev <= 1e-14 && evar <= 1e-13 && eg <= 1e-13, "{}", case.name());
    }
}

/// Device against CPU engine on the same inputs: statistics, their spatial
/// variance and the level-0 gradient agree to rounding.
#[test]
fn device_matches_cpu_engine() {
    let Some(gpu) = gpu() else { return };
    for conv in CONVS {
        for case in cases("stats", "stats") {
            let s = setup(&case);
            let x = case.f64("map0");
            let g = complex(&case, "gradient");
            let mut cpu = LocalEngine::new(&s.geo, &s.cfg, x.clone(), s.batch, s.channels);
            let (v_cpu, var_cpu) = cpu.forward_moments(&s.plan);
            let mg = cpu.reverse(&s.plan, &g);
            let (g_cpu, _) = cpu.analysis_vjp(mg);

            let part = Partition::serial(&s.geo, s.cfg.levels());
            let mut e = GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, s.batch, s.channels, &GpuOptions { conv, ..Default::default() });
            e.load(&Serial, &x);
            let (v_gpu, var_gpu) = e.forward(&Serial, &s.plan, true);
            let mg = e.reverse(&Serial, &s.plan, &g);
            let g_gpu = gpu.download(&e.analysis_vjp(&Serial, mg));
            let scale = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
            let ev = max_abs_diff(&flat(&v_cpu), &flat(&v_gpu)) / scale(&flat(&v_cpu));
            let evar = max_abs_diff(&var_cpu, &var_gpu) / scale(&var_cpu);
            let eg = max_abs_diff(&g_cpu, &g_gpu) / scale(&g_cpu);
            eprintln!("{} ({}): statistics {ev:.1e}, variances {evar:.1e}, gradient {eg:.1e} (relative to max)", case.name(), conv.name());
            assert!(ev <= 1e-13 && evar <= 1e-12 && eg <= 1e-12, "{}", case.name());
        }
    }
}

/// One engine shared by several plans and batches (target statistics at
/// batch 1, the synthesis at batch 2, the mean / var / S1 / S2 statistics of
/// a transformed map, as `stl-synth --kappa-table`) gives, to the bit, what
/// separate engines give; the plan without S3 / S4 (no moduli computed)
/// gives the values and gradient of the full plan restricted to it.
#[test]
fn shared_engine_matches_separate_engines() {
    use stl_core::engine::Evaluator;
    use stl_gpu::GpuEvaluator;
    let Some(gpu) = gpu() else { return };
    let case = cases("stats", "stats").into_iter().find(|c| c.name().contains("n32_J4_L4_k5_C1")).unwrap();
    let s = setup(&case);
    let kplan = build_plan(case.int("J") as usize, s.geo.l, &s.cfg.j_to_dg, s.channels, None, &Family::ALL[..4]);
    let n0 = s.channels * s.geo.pixels(0);
    let single = case.f64("map0")[..n0].to_vec();
    let kappa = |x: &[f64]| x.iter().map(|v| (1.5 * v).exp() - 1.0).collect::<Vec<f64>>();
    let pair = |a: &[f64], b: &[f64]| [a, b].concat();
    let maps = [pair(&single, &single.iter().map(|v| v.tanh()).collect::<Vec<_>>()), pair(&single.iter().map(|v| -0.5 * v).collect::<Vec<_>>(), &single)];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut cotangent = |n: usize| -> Vec<Complex64> {
        (0..n)
            .map(|_| {
                let mut u = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                };
                Complex64::new(u(), u())
            })
            .collect()
    };
    let (g_full, g_kappa) = (cotangent(2 * s.plan.len()), cotangent(2 * kplan.len()));
    let opts = GpuOptions { block_rows: 1500, ..Default::default() };
    let engine = |batch: usize| {
        let part = Partition::serial(&s.geo, s.cfg.levels());
        GpuEngine::new(gpu.clone(), &s.geo, &s.cfg, part, batch, s.channels, &opts)
    };
    type Out = (Vec<Complex64>, Vec<f64>, Vec<f64>);
    let eval = |ev: &mut GpuEvaluator, x: &[f64], g: &[Complex64]| -> Out {
        let (v, var) = ev.forward(&Serial, x, true);
        (v, var, ev.gradient(&Serial, g))
    };
    let probes = |p: &mut GpuEvaluator, k: &mut GpuEvaluator| (p.forward(&Serial, &single, true), k.forward(&Serial, &kappa(&single), true));
    let run = |main: &mut GpuEvaluator, k: &mut GpuEvaluator| -> Vec<Out> {
        let mut out = Vec::new();
        for x in &maps {
            out.push(eval(main, x, &g_full));
            out.push(eval(k, &kappa(x), &g_kappa));
        }
        out
    };
    let separate = {
        let probe = probes(&mut GpuEvaluator::new(engine(1), &s.plan), &mut GpuEvaluator::new(engine(1), &kplan));
        (probe, run(&mut GpuEvaluator::new(engine(2), &s.plan), &mut GpuEvaluator::new(engine(2), &kplan)))
    };
    let shared = {
        let main = GpuEvaluator::new(engine(2), &s.plan);
        let probe = probes(&mut main.share(&s.plan, 1), &mut main.share(&kplan, 1));
        (probe, run(&mut main.share(&s.plan, 2), &mut main.share(&kplan, 2)))
    };
    assert!(shared == separate, "shared engine differs from separate engines");
    // The kappa plan against the full plan restricted to it (cotangent zero elsewhere).
    let at: Vec<usize> = kplan.iter().map(|c| s.plan.iter().position(|p| p == c).unwrap()).collect();
    let mut g_embedded = vec![Complex64::new(0.0, 0.0); 2 * s.plan.len()];
    for b in 0..2 {
        for (k, &i) in at.iter().enumerate() {
            g_embedded[b * s.plan.len() + i] = g_kappa[b * kplan.len() + k];
        }
    }
    let x = kappa(&maps[0]);
    let (vf, varf, gf) = eval(&mut GpuEvaluator::new(engine(2), &s.plan), &x, &g_embedded);
    let (vk, vark, gk) = &separate.1[1];
    for b in 0..2 {
        for (k, &i) in at.iter().enumerate() {
            assert!(vk[b * kplan.len() + k] == vf[b * s.plan.len() + i], "value of {:?}", kplan[k]);
            assert!(vark[b * kplan.len() + k] == varf[b * s.plan.len() + i], "variance of {:?}", kplan[k]);
        }
    }
    assert!(*gk == gf, "gradient of the plan without moduli differs from the full plan's");
}
