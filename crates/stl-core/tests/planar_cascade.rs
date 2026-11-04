//! Coarse-to-fine conditional detail (`stl_core::planar::cascade`):
//! Fourier resampling, conditional histograms of the draws, determinism.

use stl_core::planar::cascade::{gradient_modulus, resample, ConditionalDetail, SplitMix};

fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut r = SplitMix(seed);
    (0..n).map(|_| r.normal()).collect()
}

#[test]
fn resampling_keeps_means_and_round_trips() {
    let x = noise(64 * 64, 1);
    let d = resample(&x, 64, 32);
    let u = resample(&d, 32, 64);
    let back = resample(&u, 64, 32);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    assert!((mean(&x) - mean(&d)).abs() < 1e-13 && (mean(&d) - mean(&u)).abs() < 1e-13);
    assert!(d.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-12));
}

/// Detail that depends on the bin of the coarse value: every draw must come
/// from the pool of its own bin, so its conditional mean is the target's.
#[test]
fn draws_follow_the_conditional_histogram_and_are_reproducible() {
    let n = 32;
    let q = 4;
    let coarse: Vec<f64> = (0..3).flat_map(|k| resample(&noise(16 * 16, 2 + k), 16, n)).collect();
    let probe = ConditionalDetail::new(&coarse, &vec![0.0; coarse.len()], n, q);
    let grad: Vec<f64> = coarse.chunks(n * n).flat_map(|p| gradient_modulus(p, n)).collect();
    let bins: Vec<usize> = coarse.iter().zip(&grad).map(|(&c, &g)| probe.bin(c, g)).collect();
    // Detail = 10 * bin + U(0, 1): bins have disjoint value ranges.
    let mut r = SplitMix(3);
    let detail: Vec<f64> = bins.iter().map(|&b| 10.0 * b as f64 + r.next_f64()).collect();
    let cd = ConditionalDetail::new(&coarse, &detail, n, q);
    let a = cd.draw(&coarse, &mut SplitMix(7));
    let b = cd.draw(&coarse, &mut SplitMix(7));
    assert_eq!(a, b, "same seed, same draw");
    assert_ne!(a, cd.draw(&coarse, &mut SplitMix(8)));
    for (v, &bin) in a.iter().zip(&bins) {
        assert_eq!((v / 10.0).floor() as usize, bin, "draw outside its bin's pool");
    }
    for bin in 0..q * q {
        let t: Vec<f64> = detail.iter().zip(&bins).filter(|(_, &b)| b == bin).map(|(v, _)| *v).collect();
        let s: Vec<f64> = a.iter().zip(&bins).filter(|(_, &b)| b == bin).map(|(v, _)| *v).collect();
        if t.len() > 200 {
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            assert!((m(&t) - m(&s)).abs() < 0.08, "bin {bin}: {} vs {}", m(&t), m(&s));
        }
    }
}
