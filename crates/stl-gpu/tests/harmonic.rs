//! The device spherical harmonic transforms against `stl_core::sht`: every
//! transform and transpose, the derivative maps and the C_l, at several
//! nside and lmax (including polar rings where the recursion is scaled).

use num_complex::Complex64;
use stl_core::sht::Sht;
use stl_gpu::harmonic::GpuSht;
use stl_gpu::Gpu;

fn gpu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("no device, test skipped: {e}");
            None
        }
    }
}

fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn random_alm(sht: &Sht, seed: u64) -> Vec<Complex64> {
    let re = noise(sht.nalm(), seed);
    let im = noise(sht.nalm(), seed + 1);
    sht.lm().iter().enumerate().map(|(i, &(_, m))| Complex64::new(re[i], if m == 0 { 0.0 } else { im[i] })).collect()
}

fn flat(a: &[Complex64]) -> Vec<f64> {
    a.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// Largest difference relative to the largest reference value.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / s
}

#[test]
fn device_transforms_match_the_host() {
    let Some(gpu) = gpu() else { return };
    for (nside, lmax) in [(1usize, 2usize), (4, 11), (16, 47), (16, 20), (64, 191), (128, 383)] {
        let sht = Sht::new(nside, lmax);
        let dev = GpuSht::new(&gpu, &sht).expect("device transforms");
        let a = random_alm(&sht, 3);
        let x = noise(sht.npix, 5);
        let cot: Vec<f64> = (0..5).flat_map(|k| noise(sht.npix, 7 + k as u64)).collect();
        let (da, dx, dc) = (gpu.upload(&flat(&a)), gpu.upload(&x), gpu.upload(&cot));
        let mut worst = 0f64;
        let mut check = |name: &str, got: Vec<f64>, want: Vec<f64>| {
            let e = rel(&got, &want);
            eprintln!("nside {nside} lmax {lmax} {name}: {e:.2e}");
            worst = worst.max(e);
            assert!(e < 1e-12, "nside {nside} lmax {lmax} {name}: {e:e}");
        };
        check("synthesis", gpu.download(&dev.synthesis(&gpu, &da)), sht.synthesis(&a));
        check("adjoint synthesis", gpu.download(&dev.adjoint_synthesis(&gpu, &dx)), flat(&sht.adjoint_synthesis(&x)));
        check("analysis", gpu.download(&dev.analysis(&gpu, &dx)), flat(&sht.analysis(&x)));
        check("analysis vjp", gpu.download(&dev.analysis_vjp(&gpu, &da)), sht.analysis_vjp(&a));
        let d = sht.derivatives(&a);
        check("derivatives", gpu.download(&dev.derivatives(&gpu, &da)), d.concat());
        let c: Vec<&[f64]> = cot.chunks(sht.npix).collect();
        check("derivatives vjp", gpu.download(&dev.derivatives_vjp(&gpu, &dc)), flat(&sht.derivatives_vjp([c[0], c[1], c[2], c[3], c[4]])));
        check("power", dev.power(&gpu, &da), sht.power(&a));
        let dcl = noise(lmax + 1, 11);
        check("power vjp", gpu.download(&dev.power_vjp_map(&gpu, &da, &dcl)), sht.analysis_vjp(&sht.power_vjp(&a, &dcl)));
        eprintln!("nside {nside} lmax {lmax}: worst {worst:.2e}");
    }
}

/// The batched transforms (`*_many`, several maps per Legendre GEMM) against
/// `stl_core::sht` map by map.
#[test]
fn batched_transforms_match_the_host() {
    let Some(gpu) = gpu() else { return };
    for (nside, lmax) in [(4usize, 11usize), (16, 47), (64, 191)] {
        let sht = Sht::new(nside, lmax);
        let dev = GpuSht::new(&gpu, &sht).expect("device transforms");
        let k = 3;
        let alms: Vec<Vec<Complex64>> = (0..k).map(|i| random_alm(&sht, 3 + 2 * i as u64)).collect();
        let xs: Vec<Vec<f64>> = (0..k).map(|i| noise(sht.npix, 50 + i as u64)).collect();
        let cots: Vec<Vec<f64>> = (0..k).map(|i| (0..5).flat_map(|j| noise(sht.npix, 70 + 5 * i as u64 + j)).collect()).collect();
        let da: Vec<_> = alms.iter().map(|a| gpu.upload(&flat(a))).collect();
        let dx: Vec<_> = xs.iter().map(|x| gpu.upload(x)).collect();
        let dc: Vec<_> = cots.iter().map(|c| gpu.upload(c)).collect();
        let (ra, rx, rc): (Vec<_>, Vec<_>, Vec<_>) = (da.iter().collect(), dx.iter().collect(), dc.iter().collect());
        let check = |name: &str, got: Vec<stl_gpu::CudaSlice<f64>>, want: Vec<Vec<f64>>| {
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let e = rel(&gpu.download(g), w);
                eprintln!("nside {nside} lmax {lmax} {name} {i}: {e:.2e}");
                assert!(e < 1e-12, "nside {nside} lmax {lmax} {name} {i}: {e:e}");
            }
        };
        check("synthesis", dev.synthesis_many(&gpu, &ra), alms.iter().map(|a| sht.synthesis(a)).collect());
        check("analysis", dev.analysis_many(&gpu, &rx), xs.iter().map(|x| flat(&sht.analysis(x))).collect());
        check("derivatives", dev.derivatives_many(&gpu, &ra), alms.iter().map(|a| sht.derivatives(a).concat()).collect());
        let want: Vec<Vec<f64>> = cots
            .iter()
            .map(|c| {
                let c: Vec<&[f64]> = c.chunks(sht.npix).collect();
                flat(&sht.derivatives_vjp([c[0], c[1], c[2], c[3], c[4]]))
            })
            .collect();
        check("derivatives vjp", dev.derivatives_vjp_many(&gpu, &rc), want);
    }
}
