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

#[cfg(feature = "gpu")]
fn device() -> Option<std::sync::Arc<stl_gpu::Gpu>> {
    match stl_gpu::Gpu::new(0) {
        Ok(g) => Some(std::sync::Arc::new(g)),
        Err(e) => {
            eprintln!("no device, test skipped: {e}");
            None
        }
    }
}

/// The harmonic terms on the device (`--harmonic gpu`) against the host:
/// the five blocks and the gradient of the fixture objective.
#[cfg(feature = "gpu")]
#[test]
fn device_terms_match_the_host() {
    let Some(gpu) = device() else { return };
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
        let mut obj = EnsembleObjective::new(&og, &ok, None, &cal, &plan, members, 0, m["ps_variance_upper"].as_f64().unwrap()).unwrap();
        let (_, host) = obj.loss_and_grad(&x);
        let hb = *obj.last.borrow();
        obj.use_device_terms(gpu.clone()).unwrap();
        let (_, dev) = obj.loss_and_grad(&x);
        let db = *obj.last.borrow();
        (hb, host, db, dev)
    });
    let (hb, host, db, dev) = &out[0];
    for ((h, d), name) in hb.iter().zip(db).zip(stl_cli::ensemble::objective::BLOCKS) {
        let e = (h.loss - d.loss).abs() / h.loss;
        eprintln!("{name}: host {:.15e} device {:.15e} ({e:.2e})", h.loss, d.loss);
        assert!(e < 1e-12, "{name}");
    }
    let e = rel(dev, host);
    eprintln!("gradient: {e:.2e}");
    assert!(e < 1e-12, "gradient {e:e}");
}

/// Wall time of the harmonic terms of one member on the device, as
/// `host_profile` (`--ignored`, `--features gpu`): the transforms one by
/// one, then the forward and reverse passes of the three terms.
#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn device_profile() {
    use std::time::Instant;
    use stl_cli::ensemble::fields::{lp_windows, Cic, Minkowski};
    use stl_cli::ensemble::gpu_terms::GpuTerms;
    use stl_gpu::harmonic::GpuSht;
    let Some(gpu) = device() else { return };
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let nside = env("STL_PROFILE_NSIDE").unwrap_or(128);
    let reps = env("STL_PROFILE_REPS").unwrap_or(3);
    let best = |f: &mut dyn FnMut()| {
        (0..reps)
            .map(|_| {
                gpu.sync();
                let t = Instant::now();
                f();
                gpu.sync();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let t = Instant::now();
    let sht = Sht::new(nside, 3 * nside - 1);
    let dev = GpuSht::new(&gpu, &sht).unwrap();
    let setup = t.elapsed().as_secs_f64();
    let x: Vec<f64> = (0..sht.npix).map(|i| (0.5 * stl_cli::pixel_normal(3, i as i64)).exp() - 1.0).collect();
    let dx = gpu.upload(&x);
    let alm = dev.analysis(&gpu, &dx);
    let d = dev.derivatives(&gpu, &alm);
    let analysis = best(&mut || drop(dev.analysis(&gpu, &dx)));
    let synthesis = best(&mut || drop(dev.analysis_vjp(&gpu, &alm)));
    let derivatives = best(&mut || drop(dev.derivatives(&gpu, &alm)));
    let derivatives_vjp = best(&mut || drop(dev.derivatives_vjp(&gpu, &d)));
    let (w, _) = lp_windows(3 * nside - 1);
    let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let cic = Cic::new(lo, hi, 100);
    let mk = Minkowski::default();
    let mut terms = GpuTerms::new(gpu.clone(), &sht, &mk).unwrap();
    let (cps, cpdf, cmf) = (vec![1.0; w.len()], vec![1.0; 100], vec![1.0; mk.count()]);
    let mut fwd = [0.0; 2];
    let forward = best(&mut || fwd = terms.forward(&x, &w, &cic).3);
    let mut rev = [0.0; 2];
    let reverse = best(&mut || rev = terms.reverse(&w, &cic, &cps, &cpdf, &cmf).1);
    let report = serde_json::json!({
        "nside": nside, "lmax": 3 * nside - 1, "device": gpu.name, "reps": reps, "setup": setup,
        "seconds": {
            "sht_analysis": analysis, "sht_synthesis": synthesis, "sht_derivatives": derivatives, "sht_derivatives_vjp": derivatives_vjp,
            "forward": forward, "forward_power_pdf": fwd[0], "forward_mf": fwd[1],
            "reverse": reverse, "reverse_power_pdf": rev[0], "reverse_mf": rev[1],
        },
        "per_member_evaluation": {
            "sht": 2.0 * analysis + derivatives + derivatives_vjp + 2.0 * synthesis,
            "total": forward + reverse,
        },
    });
    println!("{report}");
}

/// Wall time of the host terms of one member and one evaluation, part by
/// part (`--ignored`; `STL_PROFILE_NSIDE`, default 128; `STL_THREADS`):
/// the SHT passes, the Minkowski thresholds (the functionals minus their
/// SHTs), the CIC PDF and the C_l.  Best of `STL_PROFILE_REPS` runs
/// (default 3, 1 above nside 256).
#[test]
#[ignore]
fn host_profile() {
    use stl_cli::ensemble::fields::{band_powers, band_powers_vjp, lp_windows, Cic, Minkowski};
    use std::time::Instant;
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let nside = env("STL_PROFILE_NSIDE").unwrap_or(128);
    let reps = env("STL_PROFILE_REPS").unwrap_or(if nside > 256 { 1 } else { 3 });
    let best = |f: &mut dyn FnMut()| {
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let t = Instant::now();
    let sht = Sht::new(nside, 3 * nside - 1);
    let setup = t.elapsed().as_secs_f64();
    // A lognormal-like field: the cost does not depend on its content.
    let x: Vec<f64> = (0..sht.npix).map(|i| (0.5 * stl_cli::pixel_normal(3, i as i64)).exp() - 1.0).collect();
    let alm = sht.analysis(&x);
    let d = sht.derivatives(&alm);
    let dv = sht.derivatives_vjp([&d[0], &d[1], &d[2], &d[3], &d[4]]);
    let analysis = best(&mut || drop(sht.analysis(&x)));
    let synthesis = best(&mut || drop(sht.analysis_vjp(&dv)));
    let derivatives = best(&mut || drop(sht.derivatives(&alm)));
    let derivatives_vjp = best(&mut || drop(sht.derivatives_vjp([&d[0], &d[1], &d[2], &d[3], &d[4]])));
    let (w, _) = lp_windows(3 * nside - 1);
    let cot_w = vec![1.0; w.len()];
    let power = best(&mut || drop(band_powers(&sht, &w, &x)));
    let power_vjp = best(&mut || drop(band_powers_vjp(&sht, &w, &alm, &cot_w)));
    let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let cic = Cic::new(lo, hi, 100);
    let cot_h = vec![1.0; 100];
    let cic_forward = best(&mut || drop(cic.forward(&x)));
    let cic_vjp = best(&mut || drop(cic.vjp(&x, &cot_h)));
    let mk = Minkowski::default();
    let (v, s) = mk.forward(&sht, &x);
    let mf_forward = best(&mut || drop(mk.forward(&sht, &x)));
    let mf_vjp = best(&mut || drop(mk.vjp(&sht, &s, &v)));
    // Per member and evaluation: C_l (analysis, synthesis), Minkowski
    // (analysis, derivatives; derivatives_vjp, synthesis), CIC both ways.
    let sht_total = 2.0 * analysis + derivatives + derivatives_vjp + 2.0 * synthesis;
    let thresholds = (mf_forward - analysis - derivatives) + (mf_vjp - derivatives_vjp - synthesis);
    let pdf = cic_forward + cic_vjp;
    let report = serde_json::json!({
        "nside": nside, "lmax": 3 * nside - 1, "threads": stl_cli::ensemble::fields::host_threads(), "reps": reps,
        "setup": setup,
        "seconds": {
            "sht_analysis": analysis, "sht_synthesis": synthesis, "sht_derivatives": derivatives, "sht_derivatives_vjp": derivatives_vjp,
            "power_forward": power, "power_vjp": power_vjp, "cic_forward": cic_forward, "cic_vjp": cic_vjp,
            "mf_forward": mf_forward, "mf_vjp": mf_vjp,
        },
        "per_member_evaluation": {
            "sht": sht_total, "mf_thresholds": thresholds, "pdf": pdf,
            "power_only": power + power_vjp - analysis - synthesis,
            "total": sht_total + thresholds + pdf + (power + power_vjp - analysis - synthesis),
        },
    });
    println!("{report}");
}
