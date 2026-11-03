//! Planar engine against STL-Dev's `STL_2D_Kernel_Torch` (`pbc=True`):
//! fixtures `fixtures/planar` of `tools/dump_planar_reference.py`.

mod common;

use common::{cases, Case};
use num_complex::Complex64;
use stl_core::planar::grid::{self, conv_valid, conv_valid_adjoint, Window};
use stl_core::planar::kernels::{wavelet, K, L};
use stl_core::planar::spectrum::GaussianRings;
use stl_core::planar::PlanarEngine;

type C64 = Complex64;

fn complex(case: &Case, name: &str) -> Vec<C64> {
    case.f64(name).chunks(2).map(|v| C64::new(v[0], v[1])).collect()
}

/// `|a - b| <= atol + rtol |b|` everywhere, with the worst offender in the message.
fn close(what: &str, a: &[C64], b: &[C64], atol: f64, rtol: f64) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let (mut worst, mut at) = (0.0, 0);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let e = (x - y).norm() / (atol + rtol * y.norm());
        if e > worst {
            worst = e;
            at = i;
        }
    }
    assert!(worst <= 1.0, "{what}: element {at}: {} vs {} ({worst:.2} x tolerance)", a[at], b[at]);
}

fn close_re(what: &str, a: &[f64], b: &[f64], rel: f64) {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
    let a: Vec<C64> = a.iter().map(|&v| C64::new(v, 0.0)).collect();
    let b: Vec<C64> = b.iter().map(|&v| C64::new(v, 0.0)).collect();
    close(what, &a, &b, rel * scale, 0.0);
}

#[test]
fn kernels_match_stl_dev() {
    let case = &cases("planar", "kernels")[0];
    let w = complex(case, "wavelet");
    let ours: Vec<C64> = wavelet().iter().flat_map(|k| k.iter().copied()).collect();
    assert_eq!(ours, w, "wavelet taps");
    let s = case.f64("smooth");
    let ours: Vec<f64> = stl_core::planar::kernels::SMOOTH.iter().flatten().copied().collect();
    assert_eq!(ours, s, "smoothing taps");
}

#[test]
fn convolution_and_decimation() {
    let wav = wavelet();
    for case in cases("planar", "ops") {
        let (h, w) = (case.int("H") as usize, case.int("W") as usize);
        let x = case.f64("x");
        let full = Window::full(h, w);
        let pad = grid::read(&x, &full.grow(K / 2), 1);
        let z: Vec<C64> = (0..L).flat_map(|l| conv_valid(&pad, h, w, &wav[l])).collect();
        close(&format!("{} z", case.name()), &z, &complex(&case, "z"), 1e-14, 1e-12);
        let gz = complex(&case, "gz");
        let mut gpad = vec![0.0; pad.len()];
        for l in 0..L {
            conv_valid_adjoint(&gz[l * h * w..][..h * w], h, w, &wav[l], &mut gpad);
        }
        let mut gx = vec![0.0; h * w];
        grid::accumulate(&mut gx, &full.grow(K / 2), &gpad, 1);
        close_re(&format!("{} conv adjoint", case.name()), &gx, &case.f64("gx_conv"), 1e-13);
        let y = grid::decimate_level(&x, h, w, 1);
        close_re(&format!("{} decimation", case.name()), &y, &case.f64("y"), 1e-14);
        let gd = grid::decimate_level_adjoint(&case.f64("gy"), h, w, 1);
        close_re(&format!("{} decimation adjoint", case.name()), &gd, &case.f64("gx_down"), 1e-14);
    }
}

/// Statistics and gradients of every case for whole-level tiles and small
/// tiles (the block decomposition must not change the result).
#[test]
fn statistics_and_gradients() {
    for case in cases("planar", "stats") {
        let (h, w) = (case.int("H") as usize, case.int("W") as usize);
        let (j, c, b) = (case.int("J") as usize, case.int("C") as usize, case.int("B") as usize);
        let fewer = case.meta["fewer"].as_bool().unwrap();
        let x = case.f64("x");
        let stats = complex(&case, "stats");
        let cot = complex(&case, "cot");
        let grad = case.f64("grad");
        let n = case.int("n") as usize;
        let npix = c * h * w;
        for tile in [h.max(w), 8, 4] {
            for threads in [1, 3] {
                let e = PlanarEngine::new(h, w, c, j, fewer, tile).with_threads(threads);
                assert_eq!(e.plan().len(), n, "{}: plan length", case.name());
                for i in 0..b {
                    let what = format!("{} tile {tile} threads {threads} image {i}", case.name());
                    let (s, pyr) = e.forward(&x[i * npix..][..npix]);
                    close(&format!("{what} statistics"), &s, &stats[i * n..][..n], 1e-11, 1e-10);
                    let g = e.reverse(&pyr, &cot[i * n..][..n]);
                    close_re(&format!("{what} gradient"), &g, &grad[i * npix..][..npix], 1e-10);
                }
            }
        }
    }
}

#[test]
fn power_spectrum() {
    for case in cases("planar", "stats") {
        let (h, w) = (case.int("H") as usize, case.int("W") as usize);
        let (j, c, b) = (case.int("J") as usize, case.int("C") as usize, case.int("B") as usize);
        let rings = GaussianRings::new(h, w, case.int("n_bins") as usize, j, 0);
        let (x, ps, cot, grad) = (case.f64("x"), complex(&case, "ps"), complex(&case, "cot_ps"), case.f64("grad_ps"));
        let (n, npix) = (rings.count(c), c * h * w);
        for i in 0..b {
            let (v, xs) = rings.forward(&x[i * npix..][..npix], c);
            close(&format!("{} PS image {i}", case.name()), &v, &ps[i * n..][..n], 1e-13, 1e-11);
            let g = rings.vjp(&xs, &cot[i * n..][..n]);
            close_re(&format!("{} PS gradient image {i}", case.name()), &g, &grad[i * npix..][..npix], 1e-11);
        }
    }
}
