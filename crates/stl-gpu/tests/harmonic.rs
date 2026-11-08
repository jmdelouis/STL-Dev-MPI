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

/// The device E/B maps (`GpuEb`, spin 2 from the Hessians of scalar
/// fields) against the host `EbTransform` (spin-weighted recursions, healpy
/// to 1e-13) on a batch of pairs, iter 0 and 3, and their transposes; the
/// dot-product test on the device.
#[test]
fn eb_transform_matches_host() {
    let Some(g) = gpu() else { return };
    for (nside, iter) in [(8usize, 0usize), (16, 3), (32, 3)] {
        let lmax = 3 * nside - 1;
        let host = stl_core::sht::EbTransform::new(nside, lmax, iter);
        let dev = stl_gpu::GpuEb::new(&g, nside, lmax, iter).unwrap();
        let n = 12 * nside * nside;
        let x = noise(3 * 2 * n, 71 + nside as u64);
        let y = noise(3 * 2 * n, 73 + nside as u64);
        let (fh, th) = (host.apply(&x), host.adjoint(&y));
        let fd = g.download(&dev.apply(&g, &g.upload(&x)));
        let td = g.download(&dev.adjoint(&g, &g.upload(&y)));
        let rel = |a: &[f64], b: &[f64]| {
            let s = b.iter().fold(0f64, |m, v| m.max(v.abs()));
            a.iter().zip(b).fold(0f64, |m, (p, q)| m.max((p - q).abs())) / s
        };
        let (ef, et) = (rel(&fd, &fh), rel(&td, &th));
        let lhs: f64 = fd.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(&td).map(|(a, b)| a * b).sum();
        let dot = (lhs - rhs).abs() / lhs.abs();
        println!("nside {nside} iter {iter}: forward {ef:.2e}, transpose {et:.2e}, dot {dot:.2e}");
        assert!(ef < 1e-10 && et < 1e-10 && dot < 1e-12, "nside {nside}: {ef} {et} {dot}");
    }
}

/// The spin-2 analysis without iteration (`GpuEb::analysis_pairs`, the
/// anafast of the cl term) against `Sht::analysis_spin(2)`, with an lmax
/// below `3 nside - 1`; the synthesis against `EbTransform::synthesis_spin2`.
#[test]
fn spin2_analysis_matches_host() {
    let Some(g) = gpu() else { return };
    for (nside, lmax) in [(8usize, 15usize), (16, 31), (32, 95)] {
        let n = 12 * nside * nside;
        let host = stl_core::sht::EbTransform::new(nside, lmax, 0);
        let dev = stl_gpu::GpuEb::new(&g, nside, lmax, 0).unwrap();
        let x = noise(2 * 2 * n, 91 + nside as u64);
        let pairs: Vec<(&[f64], &[f64])> = x.chunks(2 * n).map(|p| (&p[..n], &p[n..])).collect();
        let ah = host.sht.analysis_spin(2, &pairs);
        let dp: Vec<stl_gpu::harmonic::Pair> = x.chunks(2 * n).map(|p| (g.upload(&p[..n]), g.upload(&p[n..]))).collect();
        let ad = dev.analysis_pairs(&g, &dp);
        let flat = |v: &[Complex64]| -> Vec<f64> { v.iter().flat_map(|z| [z.re, z.im]).collect() };
        let rel = |a: &[f64], b: &[f64]| {
            let s = b.iter().fold(0f64, |m, v| m.max(v.abs()));
            a.iter().zip(b).fold(0f64, |m, (p, q)| m.max((p - q).abs())) / s
        };
        let mut ea = 0f64;
        for ((e, b), (de, db)) in ah.iter().zip(&ad) {
            ea = ea.max(rel(&g.download(de), &flat(e))).max(rel(&g.download(db), &flat(b)));
        }
        let sh = host.synthesis_spin2(&ah);
        let sd = dev.synthesis_pairs(&g, &ad);
        let mut es = 0f64;
        for ((q, u), (dq, du)) in sh.iter().zip(&sd) {
            es = es.max(rel(&g.download(dq), q)).max(rel(&g.download(du), u));
        }
        println!("nside {nside} lmax {lmax}: analysis {ea:.2e}, synthesis {es:.2e}");
        assert!(ea < 1e-10 && es < 1e-10, "nside {nside}: {ea} {es}");
    }
}
