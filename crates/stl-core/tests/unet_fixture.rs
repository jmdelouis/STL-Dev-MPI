//! The native U-Net (`stl_core::nn`) against `stlia.unet.UNet` (torch,
//! `tools/dump_unet_reference.py`): output, parameter gradients and one
//! Adam step, gauges phi and cosmo, nside 16, three levels, one
//! conditioning map, batch of two.

use std::path::PathBuf;
use stl_core::nn::{Adam, UNet, UNetConfig};
use stl_core::stencil::Gauge;

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/unet").join(name)
}

fn npy<T: npyz::Deserialize>(d: &PathBuf, name: &str) -> (Vec<T>, Vec<u64>) {
    let file = std::fs::File::open(d.join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}: {e}"));
    let f = npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap();
    let shape = f.shape().to_vec();
    (f.into_vec::<T>().unwrap(), shape)
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// The network of a fixture directory, its parameters loaded by name.
pub fn load(d: &PathBuf) -> (UNet, Vec<f32>, serde_json::Value) {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    let net = UNet::new(UNetConfig {
        nside: cfg["nside"].as_u64().unwrap() as usize,
        channels: cfg["channels"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap() as usize).collect(),
        ksz: cfg["ksz"].as_u64().unwrap() as usize,
        gauge: Gauge::parse(cfg["gauge"].as_str().unwrap()).unwrap(),
        slope: 0.2,
        cond: cfg["cond"].as_u64().unwrap() as usize,
        maps: cfg["maps"].as_u64().unwrap_or(1) as usize,
        scale: cfg["scale"].as_f64(),
    });
    let names: Vec<&str> = cfg["params"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, net.params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), "state_dict order");
    let mut w = vec![0f32; net.size];
    for p in &net.params {
        let (v, shape) = npy::<f32>(d, &format!("p_{}", p.name));
        assert_eq!(shape.iter().map(|&s| s as usize).collect::<Vec<_>>(), p.shape, "{}", p.name);
        w[p.offset..p.offset + p.len].copy_from_slice(&v);
    }
    (net, w, cfg)
}

fn check(name: &str) {
    check_at(name, dir(name), 1e-4);
}

/// Output, gradients (tolerance `tol`) and Adam step of the fixture `d`.
fn check_at(name: &str, d: PathBuf, tol: f64) {
    let (net, w, _) = load(&d);
    let (dm, _) = npy::<f64>(&d, "d");
    let (cond, cshape) = npy::<f32>(&d, "cond");
    let n = net.pixels(0);
    let cond: Vec<Vec<f32>> = (0..cshape[0] as usize).map(|k| cond[k * n..(k + 1) * n].to_vec()).collect();
    let (u, tapes, scale) = net.apply(&w, &dm, &cond);
    let (ut, _) = npy::<f64>(&d, "u");
    let (st, _) = npy::<f64>(&d, "scale");
    assert!((scale - st[0]).abs() <= 1e-12 * st[0], "scale {scale} vs {}", st[0]);
    // The correction d - u, in units of the input.
    let r: Vec<f64> = dm.iter().zip(&u).map(|(a, b)| a - b).collect();
    let rt: Vec<f64> = dm.iter().zip(&ut).map(|(a, b)| a - b).collect();
    let er = max_rel(&r, &rt);
    let (cot, _) = npy::<f64>(&d, "cot");
    let mut g = vec![0f32; net.size];
    net.apply_backward(&w, &tapes, scale, &cot, &mut g);
    let mut eg = 0f64;
    // Relative to the tensor's largest value, or to 1e-2 of the largest
    // gradient: the biases of a conv followed by a GroupNorm of one channel
    // per group have a zero gradient (rounding against rounding).
    let mut all = Vec::new();
    for p in &net.params {
        all.extend(npy::<f32>(&d, &format!("g_{}", p.name)).0);
    }
    let floor = 1e-2 * all.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    for p in &net.params {
        let (gt, _) = npy::<f32>(&d, &format!("g_{}", p.name));
        let scale = gt.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
        let diff = g[p.offset..p.offset + p.len].iter().zip(&gt).fold(0f64, |m, (x, y)| m.max((x - y).abs() as f64));
        let e = diff / scale.max(floor);
        if e > 1e-4 {
            eprintln!("{name} gradient {}: {e:e} (correction {er:e})", p.name);
        }
        eg = eg.max(e);
    }
    // One Adam step from torch's gradient.
    let mut gt = vec![0f32; net.size];
    for p in &net.params {
        gt[p.offset..p.offset + p.len].copy_from_slice(&npy::<f32>(&d, &format!("g_{}", p.name)).0);
    }
    let mut w1 = w.clone();
    Adam::new(net.size).step(&mut w1, &gt, 1e-3);
    let mut ea = 0f64;
    for p in &net.params {
        let (at, _) = npy::<f32>(&d, &format!("adam_{}", p.name));
        for (x, y) in w1[p.offset..p.offset + p.len].iter().zip(&at) {
            ea = ea.max((x - y).abs() as f64);
        }
    }
    eprintln!("{name}: correction {er:e}, gradients {eg:e}, Adam step {ea:e}");
    assert!(er <= 1e-5, "{name} correction {er:e}");
    assert!(eg <= tol, "{name} gradients {eg:e}");
    assert!(ea <= 1e-7, "{name} Adam {ea:e}");
}

#[test]
fn unet_matches_torch_phi() {
    check("n16_phi");
}

#[test]
fn unet_matches_torch_cosmo() {
    check("n16_cosmo");
}

/// The fixture in `STL_UNET_FIXTURE` (e.g. nside 64, four levels, from
/// `tools/dump_unet_reference.py`), if set: gradients to 1e-2 (f32 on both
/// sides, through four GroupNorm levels).
#[test]
fn unet_matches_torch_extra() {
    if let Ok(p) = std::env::var("STL_UNET_FIXTURE") {
        check_at("extra", PathBuf::from(p), 1e-2);
    }
}
