//! Planar engine on the device against the CPU engine
//! (`stl_core::planar::PlanarEngine`, itself tested against STL-Dev):
//! statistics and gradients to 1e-12, whole-level and small tiles, one and
//! two channels, both S3/S4 definitions, batches of images.
//!
//! Without a device these tests fail (set `STL_GPU_SKIP=1` to skip them).

use num_complex::Complex64;
use std::sync::Arc;
use stl_core::planar::PlanarEngine;
use stl_gpu::planar::GpuPlanar;
use stl_gpu::Gpu;

type C64 = Complex64;

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::new(0) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) if std::env::var_os("STL_GPU_SKIP").is_some() => {
            eprintln!("no device, test skipped (STL_GPU_SKIP): {e}");
            None
        }
        Err(e) => panic!("no CUDA device ({e}); set STL_GPU_SKIP=1 to skip the device tests"),
    }
}

/// Smooth positive random field, `[B, C, H, W]`.
fn field(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut u = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n).map(|_| (0.6 * (u() - 0.5) + 0.3 * (u() - 0.5)).exp()).collect()
}

fn worst(a: &[C64], b: &[C64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.norm()));
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).norm())) / scale
}

#[test]
fn planar_device_matches_cpu() {
    let Some(gpu) = gpu() else { return };
    for (h, w, j, c, b, fewer) in [(32, 32, 3, 1, 2, false), (32, 32, 3, 1, 3, true), (32, 16, 2, 2, 2, false), (32, 32, 3, 2, 1, true), (64, 64, 4, 1, 2, false)] {
        let x = field(b * c * h * w, (h * 7 + j * 3 + c) as u64);
        for tile in [h.max(w), 8] {
            let cpu = PlanarEngine::new(h, w, c, j, fewer, tile);
            let np = cpu.plan().len();
            let cot: Vec<C64> = field(2 * b * np, 99).chunks(2).map(|v| C64::new(v[0] - 1.0, v[1] - 1.0)).collect();
            let dev = GpuPlanar::new(gpu.clone(), cpu.clone(), b);
            let (s, g) = dev.gradient(&x, &cot);
            let npix = c * h * w;
            let what = format!("{h}x{w} J{j} C{c} B{b} fewer {fewer} tile {tile}");
            for i in 0..b {
                let (sc, pyr) = cpu.forward(&x[i * npix..][..npix]);
                let gc = cpu.reverse(&pyr, &cot[i * np..][..np]);
                let es = worst(&s[i * np..][..np], &sc);
                assert!(es < 1e-12, "{what} image {i}: statistics {es:e}");
                let gd: Vec<C64> = g[i * npix..][..npix].iter().map(|&v| C64::new(v, 0.0)).collect();
                let gcc: Vec<C64> = gc.iter().map(|&v| C64::new(v, 0.0)).collect();
                let eg = worst(&gd, &gcc);
                assert!(eg < 1e-12, "{what} image {i}: gradient {eg:e}");
            }
        }
    }
}
