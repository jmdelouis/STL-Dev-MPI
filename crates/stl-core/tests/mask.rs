//! Masked averages of the full-sky statistics (`stl_core::mask`): values,
//! variances and map gradient of every mask equal those of a full-sky disc
//! patch whose core is the mask (one and two channels, both S3/S4
//! definitions, a batch of two maps), and the full-sky mask equals the
//! unmasked engine.

use num_complex::Complex64;
use stl_core::disc::{pixel_vector, DiscGeometry, DiscModel};
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::mask::Masks;
use stl_core::plan::{build_plan, Family};
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

/// Pixels of galactic-like latitude `|b| > b0` (degrees) about the z axis.
fn latitude_cut(nside: i64, b0: f64) -> Vec<i64> {
    (0..12 * nside * nside).filter(|&p| pixel_vector(p, nside)[2].abs() > b0.to_radians().sin()).collect()
}

#[test]
fn masked_averages_equal_full_sky_discs_with_the_mask_as_core() {
    let nside = 16i64;
    let npix = 12 * (nside * nside) as usize;
    let all: Vec<i64> = (0..npix as i64).collect();
    let sets = vec![all.clone(), latitude_cut(nside, 20.0), latitude_cut(nside, 45.0)];
    let batch = 2;
    for (channels, fewer) in [(1usize, false), (2, false), (2, true)] {
        let j = 3;
        let geo = LocalGeometry::new(nside, 5, 4, Gauge::Cosmo, 256);
        let cfg = EngineConfig::new(j, fewer);
        let plan = build_plan(j, geo.l, &cfg.j_to_dg, channels, None, &Family::ALL);
        let (k_count, np) = (sets.len(), plan.len());
        let x = Lcg(11 + channels as u64).vec(batch * channels * npix);
        let mut eng = LocalEngine::new(&geo, &cfg, x.clone(), batch, channels);
        let (v_full, var_full) = eng.forward_moments(&plan);
        eng.set_masks(Some(Masks::new(nside, cfg.levels(), &sets)));
        eng.reload(&x);
        let (v, var) = eng.forward_moments(&plan);
        assert_eq!(v.len(), batch * k_count * np);
        let mut rng = Lcg(5);
        let cot: Vec<Complex64> = (0..batch * k_count * np).map(|_| Complex64::new(rng.next(), rng.next())).collect();
        let mg = eng.reverse(&plan, &cot);
        let g = eng.map_cotangent(mg);

        let model = DiscModel { geo: &geo, cfg: &cfg, channels };
        let mut g_ref = vec![0.0; batch * channels * npix];
        for (k, set) in sets.iter().enumerate() {
            let disc = DiscGeometry::from_support(&geo, cfg.levels(), all.clone(), Some(set.clone()));
            for b in 0..batch {
                let xb = &x[b * channels * npix..][..channels * npix];
                let (vr, varr, state) = model.forward(&disc, xb, &plan, true);
                let at = (b * k_count + k) * np;
                for i in 0..np {
                    let (a, r) = (v[at + i], vr[i]);
                    assert!(close(a, r, 1e-11, 1e-10), "C{channels} fewer {fewer} mask {k} batch {b} coefficient {i} {:?}: {a} vs {r}", plan[i]);
                    let (a, r) = (var[at + i], varr[i]);
                    assert!((a - r).abs() <= 1e-11 + 1e-10 * r.abs(), "variance mask {k} batch {b} {i}: {a} vs {r}");
                }
                if k == 0 {
                    // The full-sky mask is the unmasked engine.
                    for i in 0..np {
                        let (a, r) = (v[at + i], v_full[b * np + i]);
                        assert!(close(a, r, 1e-13, 1e-12), "full-sky mask {i}: {a} vs {r}");
                        let (a, r) = (var[at + i], var_full[b * np + i]);
                        assert!((a - r).abs() <= 1e-13 + 1e-12 * r.abs(), "full-sky mask variance {i}: {a} vs {r}");
                    }
                }
                let gr = model.reverse(&disc, &state, &plan, &cot[at..at + np]);
                for (o, v) in g_ref[b * channels * npix..][..channels * npix].iter_mut().zip(&gr) {
                    *o += v;
                }
            }
        }
        let scale = g_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in g.iter().zip(&g_ref).enumerate() {
            assert!((a - b).abs() <= 1e-10 * scale, "C{channels} fewer {fewer} gradient {i}: {a} vs {b}");
        }
    }
}

#[test]
fn masks_on_coarser_levels_are_the_parents() {
    let nside = 8i64;
    let set = latitude_cut(nside, 30.0);
    let m = Masks::new(nside, 3, std::slice::from_ref(&set));
    let mut level = set.clone();
    for n in 0..3 {
        assert_eq!(m.counts[0][n], level.len());
        for &p in &level {
            assert_eq!(m.weights[0][n][p as usize], 1.0);
        }
        level = stl_core::neighbours::parents(&level);
    }
}
