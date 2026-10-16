//! Milestone 2: wavelet on a block and its adjoint, block-wise decimation
//! and its transpose, decimation pieces and their transposes, against
//! STL-Dev's `LocalGeometry` (fixtures `fixtures/block`).

mod common;

use common::{cases, conv_noise, max_abs_diff, Case};
use num_complex::Complex64;
use stl_core::local::{modulus_vjp, LocalGeometry};
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;

/// Decimation weights are exact to 1e-12 (CLAUDE.md, validation 1).
const TOL: f64 = 1e-12;

fn geometry(case: &Case) -> LocalGeometry {
    let geo = LocalGeometry::new(
        case.int("nside"),
        case.int("kernel_sz") as usize,
        case.int("L") as usize,
        Gauge::parse(case.str("gauge")).unwrap(),
        case.int("block_pixels") as usize,
    );
    assert_eq!(geo.conv_rings as i64, case.int("conv_rings"));
    geo
}

fn complex(case: &Case, name: &str) -> Vec<Complex64> {
    let (re, im) = (case.f64(&format!("{name}_re")), case.f64(&format!("{name}_im")));
    re.iter().zip(&im).map(|(&a, &b)| Complex64::new(a, b)).collect()
}

/// The `level` case of the same configuration and level.
fn level(case: &Case) -> Vec<f64> {
    let name = case.name();
    let tag = &name[name.find("_n").unwrap()..name.find("_d").unwrap()];
    let want = format!("level{tag}_d{}", case.int("dg"));
    cases("block", "level").into_iter().find(|c| c.name() == want).unwrap().f64("x")
}

#[test]
fn wavelet_kernel() {
    for case in cases("block", "wavelet") {
        let w = wavelet_weights(case.int("kernel_sz") as usize);
        let err = max_abs_diff(&w, &case.f64("weights"));
        assert!(err <= 1e-15, "{}: {err:e}", case.name());
    }
}

#[test]
fn wave_and_adjoint() {
    for case in cases("block", "wave") {
        let geo = geometry(&case);
        let dg = case.int("dg") as usize;
        let nside = geo.nside(dg);
        let x = level(&case);
        let block = case.i64("block");
        let wave = geo.wave(&x, &block, dg);
        assert_eq!(wave.domain, case.i64("domain"), "{}: domain", case.name());
        let z = complex(&case, "z");
        let m = block.len();
        let mut ratio = 0f64;
        for (i, (a, b)) in wave.z.iter().zip(&z).enumerate() {
            ratio = ratio.max((a - b).norm() / conv_noise(block[i % m], nside));
        }
        let zerr = wave.z.iter().zip(&z).fold(0f64, |m, (a, b)| m.max((a - b).norm()));
        eprintln!("{}: z max error {zerr:.2e} ({ratio:.2} x noise)", case.name());
        assert!(ratio <= 1.0, "{}: z error {ratio} x noise", case.name());

        let gz = complex(&case, "gz");
        let gx = geo.wave_adjoint(&gz, &block, &wave);
        let reference = case.f64("gx");
        let kd = wave.domain.len();
        let mut ratio = 0f64;
        for (i, (a, b)) in gx.iter().zip(&reference).enumerate() {
            ratio = ratio.max((a - b).abs() / conv_noise(wave.domain[i % kd], nside));
        }
        assert!(ratio <= 1.0, "{}: adjoint error {ratio} x noise", case.name());

        // The adjoint is the transpose of the forward: <W x, gz> = <x, W^T gz>
        // (real part of the complex pairing), independently of the reference.
        let n = geo.pixels(dg);
        let lhs: f64 = wave.z.iter().zip(&gz).map(|(a, b)| a.re * b.re + a.im * b.im).sum();
        let rhs: f64 = (0..gx.len())
            .map(|i| gx[i] * x[(i / kd) * n + wave.domain[i % kd] as usize])
            .sum();
        assert!((lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0), "{}: {lhs} vs {rhs}", case.name());
    }
}

#[test]
fn decimate_and_adjoint() {
    for case in cases("block", "decimate") {
        let geo = geometry(&case);
        let dg = case.int("dg") as usize;
        assert_eq!(geo.down_rings(dg) as i64, case.int("down_rings"));
        let y = geo.decimate(&level(&case), dg);
        let err = max_abs_diff(&y, &case.f64("y"));
        eprintln!("{}: decimate max error {err:.2e}", case.name());
        assert!(err <= TOL, "{}: decimate {err:e}", case.name());
        let gx = geo.decimate_adjoint(&case.f64("g"), dg);
        let err = max_abs_diff(&gx, &case.f64("gx"));
        assert!(err <= TOL, "{}: decimate_adjoint {err:e}", case.name());
    }
}

#[test]
fn pieces_and_adjoints() {
    for case in cases("block", "pieces") {
        let geo = geometry(&case);
        let dg = case.int("dg") as usize;
        let steps = case.int("steps") as usize;
        // Every step from the reference input, so that errors do not compound.
        let mut supports = vec![case.i64("block")];
        for i in 1..=steps {
            let (a, next) = geo.piece_forward(&case.f64(&format!("a{}", i - 1)), &supports[i - 1], dg + i - 1);
            assert_eq!(next, case.i64(&format!("support{i}")), "{}: support {i}", case.name());
            let err = max_abs_diff(&a, &case.f64(&format!("a{i}")));
            assert!(err <= TOL, "{}: piece {i} {err:e}", case.name());
            supports.push(next);
        }
        for i in (0..steps).rev() {
            let g = geo.piece_adjoint(&case.f64(&format!("g{}", i + 1)), &supports[i], &supports[i + 1], dg + i);
            let err = max_abs_diff(&g, &case.f64(&format!("g{i}")));
            assert!(err <= TOL, "{}: piece adjoint {i} {err:e}", case.name());
        }
        let gz = modulus_vjp(&complex_from_wave(&case, &geo), &case.f64("g0"));
        let reference = complex(&case, "gz");
        let err = gz.iter().zip(&reference).fold(0f64, |m, (a, b)| m.max((a - b).norm()));
        assert!(err <= TOL, "{}: modulus vjp {err:e}", case.name());
    }
}

/// The reference `z` of a pieces case: the matching `wave` case.
fn complex_from_wave(case: &Case, _geo: &LocalGeometry) -> Vec<Complex64> {
    let want = case.name().replacen("pieces_", "wave_", 1);
    let wave = cases("block", "wave").into_iter().find(|c| c.name() == want).unwrap();
    let z = complex(&wave, "z");
    let a0 = case.f64("a0");
    // torch's complex abs and hypot agree to the ulp.
    let err = z.iter().zip(&a0).fold(0f64, |m, (v, &a)| m.max((v.norm() - a).abs() / a.max(1e-300)));
    assert!(err <= 4.0 * f64::EPSILON, "{}: |z| vs a0 {err:e}", case.name());
    z
}
