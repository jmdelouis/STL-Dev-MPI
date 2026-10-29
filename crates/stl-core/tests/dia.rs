//! The DIA form of the convolution (`stl_core::dia`) against STL-Dev's
//! `TorchConv` outputs and the CPU convolution it rewrites.

mod common;

use common::{cases, conv_noise};
use std::collections::BTreeMap;
use stl_core::conv::{convolution_adjoint, convolution_forward, ConvGeometry, Kernel};
use stl_core::dia::{DiaGeometry, Diagonals};
use stl_core::stencil::Gauge;

fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

#[test]
fn dia_matches_the_convolution_fixtures() {
    let mut sets: BTreeMap<(usize, usize, bool), Diagonals> = BTreeMap::new();
    for case in cases("geometry", "conv") {
        let nside = case.int("nside");
        let ksz = case.int("kernel_sz") as usize;
        let g = case.int("G") as usize;
        let gauge = Gauge::parse(case.str("gauge")).unwrap();
        let ids = case.i64("ids");
        let p = ksz * ksz;
        let geo = ConvGeometry::new(&ids, nside, ksz, g, gauge);
        let ker = Kernel::shared(&case.f64("weights"), g, 1, 2, p);
        // The diagonals read at nside 8, 16 and 32 cover every level.
        let diagonals =
            sets.entry((ksz, g, gauge == Gauge::Cosmo)).or_insert_with(|| Diagonals::observed(&[8, 16, 32], ksz, g, gauge));
        let dia = DiaGeometry::from_conv(&geo, &ker, &ids, &ids, diagonals)
            .unwrap_or_else(|| panic!("{}: a read off the {} diagonals", case.name(), diagonals.len()));

        // Forward: STL-Dev on complete rows, the CPU convolution on all rows.
        let (x, y) = (case.f64("x"), case.f64("y"));
        let mine = dia.forward(&ids, &ids, &x, 1);
        let conv = convolution_forward(&geo, &ker, &x, 1);
        let kk = ids.len();
        let complete = case.bool("complete");
        let (mut vs_ref, mut vs_conv) = (0f64, 0f64);
        for row in 0..g * 2 {
            for k in 0..kk {
                let tol = conv_noise(ids[k], nside);
                if complete[k] {
                    vs_ref = vs_ref.max((mine[row * kk + k] - y[row * kk + k]).abs() / tol);
                }
                vs_conv = vs_conv.max((mine[row * kk + k] - conv[row * kk + k]).abs() / tol);
            }
        }
        // Adjoint on random cotangents of every row.
        let gy = noise(nside as u64 * 31 + ksz as u64 + g as u64, 2 * g * 2 * kk);
        let gx = dia.adjoint(&ids, &ids, &gy, 2);
        let gx_conv = convolution_adjoint(&geo, &ker, &gy, 2);
        let scale = gx_conv.iter().fold(0f64, |m, v| m.max(v.abs()));
        let adj = gx.iter().zip(&gx_conv).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
        eprintln!(
            "{}: {} diagonals; forward {vs_ref:.3} x noise from STL-Dev, {vs_conv:.3} x noise from the convolution; adjoint {adj:.1e} relative",
            case.name(),
            diagonals.len()
        );
        assert!(vs_ref <= 1.0 && vs_conv <= 1.0, "{}: forward beyond the rounding noise", case.name());
        assert!(adj <= 1e-13, "{}: adjoint {adj:e}", case.name());
    }
}
