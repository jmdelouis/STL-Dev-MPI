//! Disc-patch statistics (`stl_core::disc`): a disc of radius 180 degrees
//! is the full sky (values and gradients of `LocalEngine`), the VJP on a
//! disc with a core against central finite differences (one and two
//! channels, both S3/S4 definitions), and the cross terms of a map with
//! itself against its auto terms.

use num_complex::Complex64;
use std::f64::consts::PI;
use stl_core::disc::{disc_pixels, pixel_vector, DiscGeometry, DiscModel};
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Coefficient, Family};
use stl_core::stencil::Gauge;

/// Deterministic uniform numbers in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn close(a: Complex64, b: Complex64, abs: f64, rel: f64) -> bool {
    (a - b).norm() <= abs + rel * b.norm()
}

fn cross_plan(plan: Vec<Coefficient>) -> Vec<Coefficient> {
    plan.into_iter().filter(|c| c.c1 != c.c2).collect()
}

#[test]
fn disc_pixels_are_the_centres_within_the_radius() {
    let nside = 32;
    let centre = pixel_vector(1234, nside);
    let r = 0.3;
    let got = disc_pixels(nside, centre, r);
    let brute: Vec<i64> = (0..12 * nside * nside)
        .filter(|&p| {
            let v = pixel_vector(p, nside);
            let d = centre[0] * v[0] + centre[1] * v[1] + centre[2] * v[2];
            d.clamp(-1.0, 1.0).acos() <= r
        })
        .collect();
    assert_eq!(got, brute);
    assert_eq!(disc_pixels(nside, centre, PI).len(), 12 * 32 * 32);
}

/// R = 180 degrees: statistics, variances and map gradient equal to the
/// full-sky engine.
#[test]
fn full_disc_equals_full_sky() {
    let nside = 16i64;
    for (channels, fewer) in [(1usize, false), (2, false), (2, true)] {
        let j = 3;
        let geo = LocalGeometry::new(nside, 5, 4, Gauge::Cosmo, 256);
        let cfg = EngineConfig::new(j, fewer);
        let plan = build_plan(j, geo.l, &cfg.j_to_dg, channels, None, &Family::ALL);
        let npix = 12 * (nside * nside) as usize;
        let x = Lcg(7 + channels as u64).vec(channels * npix);
        let mut eng = LocalEngine::new(&geo, &cfg, x.clone(), 1, channels);
        let (v_ref, var_ref) = eng.forward_moments(&plan);
        let mut rng = Lcg(99);
        let cot: Vec<Complex64> = plan.iter().map(|_| Complex64::new(rng.next(), rng.next())).collect();
        let mg = eng.reverse(&plan, &cot);
        let g_ref = eng.map_cotangent(mg);

        let disc = DiscGeometry::new(&geo, cfg.levels(), [0.3, -0.2, 0.9], PI, None);
        let model = DiscModel { geo: &geo, cfg: &cfg, channels };
        let (v, var, state) = model.forward(&disc, &x, &plan, true);
        for (i, (a, b)) in v.iter().zip(&v_ref).enumerate() {
            assert!(close(*a, *b, 1e-11, 1e-10), "C{channels} fewer {fewer} coefficient {i} {:?}: {a} vs {b}", plan[i]);
        }
        for (i, (a, b)) in var.iter().zip(&var_ref).enumerate() {
            assert!((a - b).abs() <= 1e-11 + 1e-10 * b.abs(), "variance {i}: {a} vs {b}");
        }
        let g = model.reverse(&disc, &state, &plan, &cot);
        let scale = g_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in g.iter().zip(&g_ref).enumerate() {
            assert!((a - b).abs() <= 1e-10 * scale, "C{channels} fewer {fewer} gradient {i}: {a} vs {b}");
        }
    }
}

/// `Re sum(conj(cot) * S(x))` on the patch.
fn objective(model: &DiscModel, disc: &DiscGeometry, x: &[f64], plan: &[Coefficient], cot: &[Complex64]) -> f64 {
    let (v, _, _) = model.forward(disc, x, plan, false);
    v.iter().zip(cot).map(|(s, c)| (c.conj() * s).re).sum()
}

#[test]
fn disc_gradient_matches_finite_differences() {
    let nside = 16i64;
    for (channels, fewer, cross) in [(1usize, false, false), (1, true, false), (2, false, true), (2, true, true)] {
        let j = 3;
        let geo = LocalGeometry::new(nside, 5, 4, Gauge::Cosmo, 256);
        let cfg = EngineConfig::new(j, fewer);
        let mut plan = build_plan(j, geo.l, &cfg.j_to_dg, channels, None, &Family::ALL);
        if cross {
            plan = cross_plan(plan);
        }
        let disc = DiscGeometry::new(&geo, cfg.levels(), pixel_vector(700, nside), 50f64.to_radians(), Some(35f64.to_radians()));
        assert!(disc.core_pixels(0) < disc.ids[0].len());
        let model = DiscModel { geo: &geo, cfg: &cfg, channels };
        let n = channels * disc.ids[0].len();
        let mut rng = Lcg(3 + channels as u64 + fewer as u64);
        let x = rng.vec(n);
        let cot: Vec<Complex64> = plan.iter().map(|_| Complex64::new(rng.next(), rng.next())).collect();
        let (_, _, state) = model.forward(&disc, &x, &plan, false);
        let g = model.reverse(&disc, &state, &plan, &cot);
        for trial in 0..3 {
            let dir = Lcg(1000 + trial).vec(n);
            let h = 1e-5;
            let xp: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + h * d).collect();
            let xm: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a - h * d).collect();
            let fd = (objective(&model, &disc, &xp, &plan, &cot) - objective(&model, &disc, &xm, &plan, &cot)) / (2.0 * h);
            let an: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
            assert!((fd - an).abs() <= 1e-6 * an.abs().max(1.0), "C{channels} fewer {fewer} trial {trial}: fd {fd} vs vjp {an}");
        }
    }
}

/// Cross terms of `(a, a)` equal the auto terms of `a` (S1 to the 1e-8 of
/// its root normalisation).
#[test]
fn cross_of_a_map_with_itself_is_its_auto() {
    let nside = 16i64;
    let j = 3;
    let geo = LocalGeometry::new(nside, 5, 4, Gauge::Cosmo, 256);
    let cfg = EngineConfig::new(j, false);
    let auto = build_plan(j, geo.l, &cfg.j_to_dg, 1, None, &Family::ALL);
    let cross = cross_plan(build_plan(j, geo.l, &cfg.j_to_dg, 2, None, &Family::ALL));
    let disc = DiscGeometry::new(&geo, cfg.levels(), pixel_vector(100, nside), 60f64.to_radians(), None);
    let a = Lcg(5).vec(disc.ids[0].len());
    let both: Vec<f64> = a.iter().chain(&a).copied().collect();
    let (va, _, _) = DiscModel { geo: &geo, cfg: &cfg, channels: 1 }.forward(&disc, &a, &auto, false);
    let (vc, _, _) = DiscModel { geo: &geo, cfg: &cfg, channels: 2 }.forward(&disc, &both, &cross, false);
    let key = |c: &Coefficient| (c.family.name(), c.native_dg, c.j1, c.j2, c.j3, c.l1, c.l2, c.l3);
    let mut matched = 0;
    for (c, v) in cross.iter().zip(&vc) {
        let i = auto.iter().position(|d| key(d) == key(c)).expect("auto coefficient");
        let tol = if c.family == Family::S1 { 1e-7 } else { 1e-12 };
        assert!(close(*v, va[i], tol, 1e-10), "{c:?}: {v} vs {}", va[i]);
        matched += 1;
    }
    assert_eq!(matched, cross.len());
}
