//! Redundancy of the DIA coefficients between rows: how many rows of the
//! full-sky convolution have the same coefficients (kernel folded in, on
//! the same diagonals of their face frame), by ring.
//!
//! `cargo run --release -p stl-core --example dia_redundancy -- 64 128`
//! (`STL_DIA_L`: L = 4 by default; `STL_DIA_GAUGE`: cosmo by default;
//! kernel 5).

use std::collections::HashMap;
use stl_core::conv::{ConvGeometry, Kernel};
use stl_core::dia::{DiaGeometry, Diagonals};
use stl_core::healpix::{nest2xyf, npix, order_of};
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;

const JRLL: [i64; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];

fn main() {
    let l: usize = std::env::var("STL_DIA_L").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let ksz = 5;
    let nsides: Vec<i64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let gauge = Gauge::parse(&std::env::var("STL_DIA_GAUGE").unwrap_or_else(|_| "cosmo".into())).expect("gauge cosmo or phi");
    let diagonals = Diagonals::observed(&[8, 16, 32], ksz, l, gauge);
    let ker = Kernel::shared(&wavelet_weights(ksz), l, 1, 2, ksz * ksz);
    for nside in if nsides.is_empty() { vec![64] } else { nsides } {
        let order = order_of(nside);
        let ids: Vec<i64> = (0..npix(nside)).collect();
        let geo = ConvGeometry::rows(&ids, &ids, nside, ksz, l, gauge);
        let dia = DiaGeometry::from_conv(&geo, &ker, &ids, &ids, &diagonals).expect("read off the diagonals");
        let (k, row) = (ids.len(), l * 2);
        let coef = |r: usize| -> Vec<f64> { (0..diagonals.len()).flat_map(|d| dia.coef[(d * k + r) * row..][..row].to_vec()).collect() };
        let scale = dia.coef.iter().fold(0f64, |m, v| m.max(v.abs()));
        // Rows by ring and face.
        let mut rings: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (r, &id) in ids.iter().enumerate() {
            let (x, y, f) = nest2xyf(id, order);
            rings.entry((JRLL[f as usize] * nside - x - y - 1, f)).or_default().push(r);
        }
        let mut keys: Vec<(i64, i64)> = rings.keys().copied().collect();
        keys.sort_unstable();
        // Within a ring: largest difference from the ring's first row.
        let (mut eq_max, mut cap_max, mut eq_rings, mut eq_same) = (0f64, 0f64, 0usize, 0usize);
        let mut worst: Vec<((i64, i64), f64)> = Vec::new();
        for &key in &keys {
            let jr = key.0;
            let rs = &rings[&key];
            let c0 = coef(rs[0]);
            let mut m = 0f64;
            for &r in &rs[1..] {
                let c = coef(r);
                m = m.max(c.iter().zip(&c0).fold(0f64, |a, (p, q)| a.max((p - q).abs())));
            }
            m /= scale;
            let equatorial = jr > nside + 5 && jr < 3 * nside - 5;
            if equatorial {
                eq_rings += 1;
                eq_same += (m <= 1e-12) as usize;
                eq_max = eq_max.max(m);
            } else {
                cap_max = cap_max.max(m);
            }
            worst.push((key, m));
        }
        // Distinct rows over the sphere (coefficients rounded to 1e-12 of the scale).
        let mut distinct: HashMap<Vec<i64>, usize> = HashMap::new();
        for r in 0..k {
            let key: Vec<i64> = coef(r).iter().map(|v| (v / scale * 1e12).round() as i64).collect();
            *distinct.entry(key).or_default() += 1;
        }
        println!("nside {nside}, L {l}, gauge {gauge:?}: {k} rows, {} (ring, face) groups, {} diagonals", keys.len(), diagonals.len());
        println!(
            "  equatorial rings at least 6 rings from the caps: {eq_rings}, {eq_same} with all their rows equal to 1e-12; largest difference within such a ring {eq_max:.1e} (relative to the largest coefficient)"
        );
        println!("  other rings: largest difference within a ring {cap_max:.1e}");
        println!("  distinct coefficient rows (to 1e-12): {} of {k}", distinct.len());
        for ((jr, f), m) in worst.iter().filter(|((jr, f), _)| *f == 5 && (*jr % (nside / 4).max(1) == 0 || *jr == 2 * nside)) {
            println!("    ring {jr:5}, face {f}: {:.1e}", m);
        }
    }
}
