//! `stl-synth2d` objectives against STL-Dev's planar synthesis
//! (`Synthesis.optimize_from_maps`, fixture `planar/loss_*` of
//! `tools/dump_planar_reference.py`), and the gradients of the ensemble
//! objective against finite differences.

use std::path::PathBuf;
use stl_cli::planar::{nyquist_filter, standardize, white_noise, Config, Mode, PlanarObjective};
use stl_core::lbfgs::{minimize, LbfgsOptions, Local};
use stl_mpi::io::read_npy_shape;

fn fixture(case: &str) -> (PathBuf, serde_json::Value) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/planar").join(case);
    let meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    (dir, meta)
}

fn array(dir: &std::path::Path, name: &str) -> Vec<f64> {
    read_npy_shape(&dir.join(format!("{name}.npy"))).unwrap().1
}

#[test]
fn stl_dev_loss_gradient_and_trajectory() {
    let (dir, meta) = fixture("loss_64_t4_b3");
    let size = meta["size"].as_u64().unwrap() as usize;
    let target = nyquist_filter(&array(&dir, "target"), size, size);
    let (t, _, _) = standardize(&target, 1, size * size);
    let cfg = Config::new(size, size, 1, Mode::StlDev);
    assert_eq!(cfg.j, meta["J"].as_u64().unwrap() as usize);
    assert_eq!(cfg.n_bins, meta["n_bins"].as_u64().unwrap() as usize);
    let obj = PlanarObjective::new(cfg, &t).unwrap();
    let u0 = array(&dir, "u0");
    let (loss, grad) = obj.loss_and_grad(&u0);
    let loss0 = meta["loss0"].as_f64().unwrap();
    assert!((loss - loss0).abs() <= 1e-10 * loss0, "loss {loss} vs {loss0}");
    let g0 = array(&dir, "grad0");
    let scale = g0.iter().fold(0f64, |m, v| m.max(v.abs()));
    let err = grad.iter().zip(&g0).fold(0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(err <= 1e-10 * scale, "gradient error {err:e} for a scale {scale:e}");

    let niter = meta["niter"].as_u64().unwrap() as usize;
    let opts = LbfgsOptions { lr: 1.0, max_iter: niter, max_eval: None, tolerance_grad: 1e-12, tolerance_change: 1e-15, history_size: 50 };
    let mut u = u0.clone();
    let mut history = Vec::new();
    let mut f = |x: &[f64]| {
        let r = obj.loss_and_grad(x);
        history.push(r.0);
        r
    };
    minimize(&mut u, &opts, &Local, &mut f);
    let reference: Vec<f64> = meta["history"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(history.len(), reference.len(), "evaluations: {history:?} vs {reference:?}");
    for (k, (a, b)) in history.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() <= 1e-7 * b.abs(), "evaluation {k}: {a} vs {b}\n{history:?}\n{reference:?}");
    }
}

/// Directional derivative of the ensemble loss against central differences.
#[test]
fn ensemble_gradient() {
    let (size, bt, b) = (32, 4, 3);
    let target: Vec<f64> = white_noise(bt * size * size, 7).iter().map(|v| (0.5 * v).exp()).collect();
    let (t, _, _) = standardize(&nyquist_filter(&target, size, size), 1, size * size);
    let mut cfg = Config::new(size, size, 1, Mode::Ensemble);
    cfg.tile = 8;
    let obj = PlanarObjective::new(cfg, &t).unwrap();
    let u = nyquist_filter(&white_noise(b * size * size, 3), size, size);
    let (_, g) = obj.loss_and_grad(&u);
    let dir = white_noise(u.len(), 11);
    let slope: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
    let h = 1e-5;
    let at = |s: f64| obj.loss_and_grad(&u.iter().zip(&dir).map(|(a, d)| a + s * d).collect::<Vec<_>>()).0;
    let fd = (at(h) - at(-h)) / (2.0 * h);
    assert!((fd - slope).abs() <= 1e-5 * slope.abs().max(1e-3), "slope {slope} vs finite differences {fd}");
}
