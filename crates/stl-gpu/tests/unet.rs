//! The U-Net on the device (`stl_gpu::nn`) against torch
//! (`fixtures/unet`, `tools/dump_unet_reference.py`) and against the host
//! network (`stl_core::nn`) at nside 64.

use std::path::PathBuf;
use std::sync::Arc;
use stl_core::nn::{std_unbiased, Adam, UNet, UNetConfig};
use stl_core::stencil::Gauge;
use stl_gpu::nn::GpuUNet;
use stl_gpu::Gpu;

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::new(0) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            eprintln!("no device, test skipped: {e}");
            None
        }
    }
}

fn npy<T: npyz::Deserialize>(d: &PathBuf, name: &str) -> (Vec<T>, Vec<u64>) {
    let file = std::fs::File::open(d.join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}: {e}"));
    let f = npyz::NpyFile::new(std::io::BufReader::new(file)).unwrap();
    let shape = f.shape().to_vec();
    (f.into_vec::<T>().unwrap(), shape)
}

/// Largest difference over the largest value of `b`, or over `floor`.
fn rel(a: &[f64], b: &[f64], floor: f64) -> f64 {
    let s = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(floor);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / s
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Output and gradients of `net` (parameters `w`) for the batch `d` on the
/// device: (u, parameter cotangents of sum(cot u)).
fn device_run(gpu: &Arc<Gpu>, net: UNet, w: &[f32], d: &[f64], cond: &[f32], cot: &[f64]) -> (Vec<f64>, Vec<f32>, GpuUNet) {
    let n = net.pixels(0);
    let nc = net.cfg.cond;
    let scale = std_unbiased(d);
    let mut dev = GpuUNet::new(gpu.clone(), net, w).unwrap();
    // Conditioning maps pixel-major [n, nc].
    let mut pm = vec![0f32; n * nc];
    for k in 0..nc {
        for i in 0..n {
            pm[i * nc + k] = cond[k * n + i];
        }
    }
    let levels = if nc > 0 { dev.cond_levels(&gpu.upload(&pm)) } else { Vec::new() };
    dev.zero_grad();
    let mut u = Vec::new();
    for (di, ci) in d.chunks(n).zip(cot.chunks(n)) {
        let (ui, tape) = dev.forward(&gpu.upload(di), &[scale], &levels);
        dev.backward(&tape, &[scale], &gpu.upload(ci));
        u.extend(gpu.download(&ui));
    }
    let g = dev.grads();
    (u, g, dev)
}

fn fixture(name: &str) {
    fixture_at(name, PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/unet").join(name), 1e-4);
}

fn fixture_at(name: &str, d: PathBuf, tol: f64) {
    let Some(gpu) = gpu() else { return };
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    let net = UNet::new(UNetConfig {
        nside: cfg["nside"].as_u64().unwrap() as usize,
        channels: cfg["channels"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap() as usize).collect(),
        ksz: cfg["ksz"].as_u64().unwrap() as usize,
        gauge: Gauge::parse(cfg["gauge"].as_str().unwrap()).unwrap(),
        slope: 0.2,
        cond: cfg["cond"].as_u64().unwrap() as usize,
        maps: cfg["maps"].as_u64().unwrap_or(1) as usize,
        scale: None,
    });
    let mut w = vec![0f32; net.size];
    let mut gt = vec![0f32; net.size];
    let mut at = vec![0f32; net.size];
    let specs = net.params.clone();
    for p in &specs {
        w[p.offset..p.offset + p.len].copy_from_slice(&npy::<f32>(&d, &format!("p_{}", p.name)).0);
        gt[p.offset..p.offset + p.len].copy_from_slice(&npy::<f32>(&d, &format!("g_{}", p.name)).0);
        at[p.offset..p.offset + p.len].copy_from_slice(&npy::<f32>(&d, &format!("adam_{}", p.name)).0);
    }
    let (dm, _) = npy::<f64>(&d, "d");
    let (cond, _) = npy::<f32>(&d, "cond");
    let (cot, _) = npy::<f64>(&d, "cot");
    let (ut, _) = npy::<f64>(&d, "u");
    let (u, g, mut dev) = device_run(&gpu, net, &w, &dm, &cond, &cot);
    let r: Vec<f64> = dm.iter().zip(&u).map(|(a, b)| a - b).collect();
    let rt: Vec<f64> = dm.iter().zip(&ut).map(|(a, b)| a - b).collect();
    let er = rel(&r, &rt, 0.0);
    let floor = 1e-2 * gt.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let mut eg = 0f64;
    for p in &specs {
        let e = rel(&f64s(&g[p.offset..p.offset + p.len]), &f64s(&gt[p.offset..p.offset + p.len]), floor);
        if e > 1e-4 {
            eprintln!("  {}: {e:e}", p.name);
        }
        eg = eg.max(e);
    }
    // One device Adam step from torch's gradient.
    gpu.copy_in(&gt, &mut dev.g);
    dev.adam_step(1e-3);
    let w1 = dev.params();
    let ea = w1.iter().zip(&at).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("{name} device: correction {er:e}, gradients {eg:e}, Adam {ea:e}");
    assert!(er <= 1e-5 && eg <= tol && ea <= 1e-7);
}

/// The fixture in `STL_UNET_FIXTURE`, if set (gradients to 1e-2).
#[test]
fn device_unet_matches_torch_extra() {
    if let Ok(p) = std::env::var("STL_UNET_FIXTURE") {
        fixture_at("extra", PathBuf::from(p), 1e-2);
    }
}

#[test]
fn device_unet_matches_torch_phi() {
    fixture("n16_phi");
}

#[test]
fn device_unet_matches_torch_cosmo() {
    fixture("n16_cosmo");
}

/// Device against host at nside 64 (four levels, the paper's first
/// channels, one conditioning map), and host Adam against device Adam.
/// Against torch in float64 (`dump_unet_reference.py --double`), the
/// gradients of the host, of the device and of torch in float32 are all
/// within about 1.5e-3: the conditioning of the GroupNorm reverse in single
/// precision.
#[test]
fn device_unet_matches_host_n64() {
    let Some(gpu) = gpu() else { return };
    let cfg = UNetConfig { nside: 64, channels: vec![32, 64, 96, 128], ksz: 3, gauge: Gauge::Phi, slope: 0.2, cond: 1, maps: 1, scale: None };
    let net = UNet::new(cfg.clone());
    let n = net.pixels(0);
    let mut w = net.init(7);
    // A nonzero head, so that every parameter has a gradient.
    let mut s = 1u64;
    for v in w[net.head[0]..net.head[1] + 1].iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
    }
    let noise = |seed: u64| -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    };
    let d = noise(1);
    let cond: Vec<f32> = noise(2).iter().map(|&v| v as f32).collect();
    let cot = noise(3);
    let (u, tapes, scale) = net.apply(&w, &d, &[cond.clone()]);
    let mut gh = vec![0f32; net.size];
    net.apply_backward(&w, &tapes, scale, &cot, &mut gh);
    let host = UNet::new(cfg);
    let (ud, gd, mut dev) = device_run(&gpu, host, &w, &d, &cond, &cot);
    let r: Vec<f64> = d.iter().zip(&u).map(|(a, b)| a - b).collect();
    let rd: Vec<f64> = d.iter().zip(&ud).map(|(a, b)| a - b).collect();
    let er = rel(&rd, &r, 0.0);
    let floor = 1e-2 * gh.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let mut eg = 0f64;
    for p in &net.params {
        let e = rel(&f64s(&gd[p.offset..p.offset + p.len]), &f64s(&gh[p.offset..p.offset + p.len]), floor);
        if e > 1e-4 {
            eprintln!("  {}: {e:e}", p.name);
        }
        eg = eg.max(e);
    }
    let mut adam = Adam::new(net.size);
    let mut wh = w.clone();
    adam.step(&mut wh, &gh, 1e-3);
    gpu.copy_in(&gh, &mut dev.g);
    dev.adam_step(1e-3);
    let ea = dev.params().iter().zip(&wh).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("n64 device vs host: correction {er:e}, gradients {eg:e}, Adam {ea:e}");
    // f32 gradients through four GroupNorm levels are conditioned to about
    // 1e-3 (all of host, device and torch float32 against torch float64).
    assert!(er <= 1e-5 && eg <= 1e-2 && ea <= 1e-7);
}
