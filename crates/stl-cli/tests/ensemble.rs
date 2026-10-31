//! The Rust port of P. Campeti's ensemble objective against his Python
//! (`fixtures/ensemble`, `tools/dump_ensemble_reference.py`): the SHT
//! against DUCC, the LUT, the calibration targets, the five blocks and the
//! gradient; and the gradient against finite differences.

use num_complex::Complex64;
use std::path::PathBuf;
use stl_cli::ensemble::calibration::{calibrate, Calibration, Targets};
use stl_cli::ensemble::cli::setup;
use stl_cli::ensemble::lut::Lut;
use stl_cli::ensemble::objective::EnsembleObjective;
use stl_core::sht::Sht;
use stl_mpi::synthesis::Objective;

fn dir(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ensemble").join(case)
}

fn f(case: &str, name: &str) -> Vec<f64> {
    stl_mpi::io::read_npy(&dir(case).join(format!("{name}.npy"))).unwrap()
}

fn c(case: &str, name: &str) -> Vec<Complex64> {
    f(case, &format!("{name}_re")).into_iter().zip(f(case, &format!("{name}_im"))).map(|(a, b)| Complex64::new(a, b)).collect()
}

fn meta(case: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir(case).join("meta.json")).unwrap()).unwrap()
}

/// Largest difference relative to the largest reference value.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / s
}

fn relc(a: &[Complex64], b: &[Complex64]) -> f64 {
    let s = b.iter().fold(0f64, |m, v| m.max(v.norm())).max(1e-300);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).norm())) / s
}

#[test]
fn sht_matches_ducc() {
    let case = "sht_n16";
    let sht = Sht::new(16, 47);
    let x = f(case, "x");
    let alm = sht.analysis(&x);
    let e = [
        ("analysis", relc(&alm, &c(case, "alm"))),
        ("power", rel(&sht.power(&alm), &f(case, "cl"))),
        ("synthesis", rel(&sht.synthesis(&c(case, "acot")), &f(case, "synthesis"))),
        ("analysis_vjp", rel(&sht.analysis_vjp(&c(case, "acot")), &f(case, "analysis_vjp"))),
    ];
    let ref_alm = c(case, "alm");
    let d = sht.derivatives(&ref_alm);
    let cots: Vec<Vec<f64>> = (0..5).map(|k| f(case, &format!("cot{k}"))).collect();
    let dv = sht.derivatives_vjp([&cots[0], &cots[1], &cots[2], &cots[3], &cots[4]]);
    let mut worst = 0f64;
    for (name, err) in e.iter().copied().chain((0..5).map(|k| ("derivative", rel(&d[k], &f(case, &format!("deriv{k}")))))).chain([("derivative_vjp", relc(&dv, &c(case, "deriv_vjp")))]) {
        eprintln!("{name}: {err:e}");
        worst = worst.max(err);
    }
    assert!(worst < 1e-12, "SHT against DUCC: {worst:e}");
}

/// The calibration of the fixture's references against the Python one.
#[test]
fn calibration_matches_python() {
    let case = "objective_n16";
    let m = meta(case);
    let nside = m["nside"].as_u64().unwrap() as usize;
    let npix = 12 * nside * nside;
    let refs: Vec<Vec<f64>> = f(case, "references").chunks(npix).map(|r| r.to_vec()).collect();
    let lut = Lut::from_values(refs.iter().flatten().copied().collect(), 8192).unwrap();
    assert_eq!(lut.probabilities, f(case, "probabilities"), "probability grid");
    assert_eq!(lut.physical, f(case, "physical_knots"), "physical knots");
    let gk = f(case, "gaussian_knots");
    let gerr = lut.gaussian.iter().zip(&gk).filter(|(a, b)| a != b).count();
    assert!(gerr <= 2, "{gerr} Gaussian knots differ (f32 rounding of erfinv)");
    let ids: Vec<usize> = f(case, "bootstrap_ids").iter().map(|&v| v as usize).collect();
    let scales = m["scales"].as_u64().unwrap() as usize;
    let (geo, cfg, plan) = setup(nside, scales, 256);
    let out = stl_mpi::threads::run_ranks(1, |comm| {
        let dummy = vec![Complex64::new(0.0, 0.0); plan.len()];
        let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
        calibrate(&refs, vec![], ids.clone(), &plan, scales, 8192, |x| probe.statistics_moments(&probe.slice(x))).unwrap()
    });
    let cal = &out[0];
    let t = &cal.targets;
    let g = |k: &str| f(case, k);
    let checks = [
        ("gsc_mean", relc(&t.gsc_mean, &c(case, "gsc_mean"))),
        ("gsc_var", rel(&t.gsc_var, &g("gsc_var"))),
        ("gsc_sigma", rel(&t.gsc_sigma, &g("gsc_sigma"))),
        ("psc_mean", relc(&t.psc_mean, &c(case, "psc_mean"))),
        ("psc_var", rel(&t.psc_var, &g("psc_var"))),
        ("psc_sigma", rel(&t.psc_sigma, &g("psc_sigma"))),
        ("power_mean", rel(&t.power_mean, &g("power_mean"))),
        ("power_var", rel(&t.power_var, &g("power_var"))),
        ("pdf_mean", rel(&t.pdf_mean, &g("pdf_mean"))),
        ("pdf_var", rel(&t.pdf_var, &g("pdf_var"))),
        ("mf_mean", rel(&t.mf_mean, &g("mf_mean"))),
        ("mf_var", rel(&t.mf_var, &g("mf_var"))),
        ("pdf_support", rel(&t.pdf_support, &g("pdf_support"))),
        ("initialization_cl", rel(&t.initialization_cl, &g("initialization_cl"))),
    ];
    for (k, e) in checks {
        eprintln!("{k}: {e:e}");
        assert!(e < 1e-9, "{k}: {e:e}");
    }
}

/// The calibration of the fixture (the Python targets).
fn fixture_calibration() -> Calibration {
    let case = "objective_n16";
    let g = |k: &str| f(case, k);
    let m = meta(case);
    let sup = g("pdf_support");
    Calibration {
        nside: m["nside"].as_u64().unwrap() as usize,
        scales: m["scales"].as_u64().unwrap() as usize,
        orientations: 4,
        kernel: 5,
        plan_len: g("gsc_var").len(),
        references: vec![],
        bootstrap_ids: g("bootstrap_ids").iter().map(|&v| v as usize).collect(),
        lut: Lut::new(g("probabilities"), g("physical_knots")).unwrap(),
        targets: Targets {
            gsc_mean: c(case, "gsc_mean"),
            gsc_var: g("gsc_var"),
            gsc_sigma: g("gsc_sigma"),
            psc_mean: c(case, "psc_mean"),
            psc_var: g("psc_var"),
            psc_sigma: g("psc_sigma"),
            power_mean: g("power_mean"),
            power_var: g("power_var"),
            pdf_mean: g("pdf_mean"),
            pdf_var: g("pdf_var"),
            mf_mean: g("mf_mean"),
            mf_var: g("mf_var"),
            pdf_support: [sup[0], sup[1]],
            initialization_cl: g("initialization_cl"),
            initialization_reference_mean: 0.0,
        },
    }
}

/// Loss, blocks and gradient of the 10 members against the Python objective;
/// the gradient against central differences along a random direction.
#[test]
fn objective_matches_python() {
    let case = "objective_n16";
    let m = meta(case);
    let cal = fixture_calibration();
    let members = m["members"].as_u64().unwrap() as usize;
    let (geo, cfg, plan) = setup(cal.nside, cal.scales, 256);
    let x = f(case, "members");
    let out = stl_mpi::threads::run_ranks(1, |comm| {
        let dummy = vec![Complex64::new(0.0, 0.0); members * plan.len()];
        let og = Objective::new(&geo, &cfg, &plan, &dummy, members, 1, comm);
        let ok = Objective::with_partition(&geo, &cfg, &plan, &dummy, members, 1, comm, og.dec.clone(), og.part.clone());
        let obj = EnsembleObjective::new(&og, &ok, None, &cal, &plan, members, 0, m["ps_variance_upper"].as_f64().unwrap()).unwrap();
        let (k, _) = obj.physical(&x);
        let (loss, grad) = obj.loss_and_grad(&x);
        let blocks = *obj.last.borrow();
        // Central differences along u.
        let u: Vec<f64> = (0..x.len()).map(|i| stl_cli::pixel_normal(5, i as i64)).collect();
        let h = 1e-6;
        let at = |t: f64| obj.loss_and_grad(&x.iter().zip(&u).map(|(a, b)| a + t * b).collect::<Vec<_>>()).0;
        let fd = (at(h) - at(-h)) / (2.0 * h);
        let an: f64 = grad.iter().zip(&u).map(|(a, b)| a * b).sum();
        (loss, blocks, grad, k, fd, an)
    });
    let (loss, blocks, grad, kappa, fd, an) = &out[0];
    eprintln!("kappa: {:e}", rel(kappa, &f(case, "kappa")));
    let py = &m["blocks"];
    for (b, name) in blocks.iter().zip(stl_cli::ensemble::objective::BLOCKS) {
        let r = py[name].as_f64().unwrap();
        eprintln!("{name}: rust {:.15e} python {r:.15e} ({:e})", b.loss, (b.loss - r).abs() / r);
        assert!((b.loss - r).abs() <= 1e-9 * r, "{name}");
    }
    let r = m["loss"].as_f64().unwrap();
    assert!((loss - r).abs() <= 1e-9 * r, "loss {loss} vs {r}");
    let eg = rel(grad, &f(case, "gradient"));
    eprintln!("gradient: {eg:e}; directional derivative {an:.12e}, finite differences {fd:.12e} ({:e})", (an - fd).abs() / fd.abs());
    assert!(eg < 1e-8, "gradient {eg:e}");
    // The method's gradient is not the derivative of its loss everywhere:
    // the Minkowski curvature floor is held fixed in the cotangent (as the
    // source's detached floor) and the CIC PDF has kinks at the bin centres,
    // so finite differences only check the order of magnitude (the Python
    // gradient, equal to this one above, differs from them the same way).
    assert!((an - fd).abs() <= 1e-2 * fd.abs(), "finite differences");
}

/// Wall time of the host parts of one member at nside 128 (`--ignored`).
#[test]
#[ignore]
fn host_timings_nside128() {
    use stl_cli::ensemble::fields::{band_powers, lp_windows, Minkowski};
    use std::time::Instant;
    let nside = 128;
    let sht = Sht::new(nside, 3 * nside - 1);
    let x: Vec<f64> = (0..sht.npix).map(|i| stl_cli::pixel_normal(3, i as i64)).collect();
    let t = Instant::now();
    let alm = sht.analysis(&x);
    eprintln!("analysis {:.4} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let d = sht.derivatives(&alm);
    eprintln!("derivatives {:.4} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let g = sht.derivatives_vjp([&d[0], &d[1], &d[2], &d[3], &d[4]]);
    eprintln!("derivatives_vjp {:.4} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let _ = sht.analysis_vjp(&g);
    eprintln!("analysis_vjp {:.4} s", t.elapsed().as_secs_f64());
    let (w, _) = lp_windows(3 * nside - 1);
    let t = Instant::now();
    let _ = band_powers(&sht, &w, &x);
    eprintln!("band powers {:.4} s", t.elapsed().as_secs_f64());
    let mk = Minkowski::default();
    let t = Instant::now();
    let (v, s) = mk.forward(&sht, &x);
    eprintln!("minkowski forward {:.4} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let _ = mk.vjp(&sht, &s, &v);
    eprintln!("minkowski vjp {:.4} s", t.elapsed().as_secs_f64());
}
