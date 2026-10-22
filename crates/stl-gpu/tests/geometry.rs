//! Device geometry and convolution against the CPU port
//! (`stl_core::conv`), itself tested against STL-Dev's `TorchConv`.

mod common;

use common::conv_noise;
use stl_core::conv::{convolution_adjoint, convolution_forward, ConvGeometry, Kernel};
use stl_core::neighbours::dilate_fine;
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;
use stl_gpu::device::NULL;
use stl_gpu::geometry::{DeviceGeometry, Grid, Spec};
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
fn geometry_and_convolution_match_cpu() {
    let Some(gpu) = gpu() else { return };
    eprintln!("device: {}", gpu.name);
    let cases: [(i64, usize, usize, i64, usize); 7] = [
        // nside, kernel, L, block start, block size
        (8, 5, 2, 0, 64),
        (8, 3, 4, 300, 40),
        (32, 5, 2, 5000, 256),
        (32, 5, 4, 2048, 512),
        (128, 5, 4, 100_000, 1024),
        (1024, 5, 2, 6_000_000, 2048),
        (1024, 3, 4, 12_500_000, 1024),
    ];
    for (nside, ksz, l, start, size) in cases {
        let rings = ((ksz - 1) / 2 + 2) as u32;
        let block: Vec<i64> = (start..start + size as i64).collect();
        let a1 = dilate_fine(&block, nside, rings);
        let domain = dilate_fine(&a1, nside, rings);
        let p = ksz * ksz;
        for (rows, own) in [(&a1, None), (&domain, Some((start, start + size as i64)))] {
            let cpu = ConvGeometry::rows(&domain, rows, nside, ksz, l, Gauge::Cosmo);
            let grid = Grid::new(&gpu, &domain, own);
            let rows_d = gpu.upload(rows);
            let spec = Spec { nside, kernel_sz: ksz, g: l, gauge: Gauge::Cosmo };
            let mut dev = DeviceGeometry::build(&gpu, &grid, &rows_d, rows.len(), &spec, true);
            let (pos, w, comp) = (
                gpu.download(dev.pos.as_ref().unwrap()),
                gpu.download(dev.w.as_ref().unwrap()),
                gpu.download(dev.complete.as_ref().unwrap()),
            );
            // Dropping and recomputing the weights gives them back to the bit.
            let wt = gpu.download(dev.wt.as_ref().unwrap());
            dev.strip();
            dev.refresh(&gpu, &grid, &rows_d, &spec);
            assert_eq!(gpu.download(dev.w.as_ref().unwrap()), w);
            assert_eq!(gpu.download(dev.wt.as_ref().unwrap()), wt);
            let k = rows.len();
            let (mut werr, mut bad_pos, mut bad_complete) = (0f64, 0usize, 0usize);
            for kk in 0..k {
                let tol = conv_noise(rows[kk], nside);
                let mut row_complete = true;
                for g in 0..l {
                    row_complete &= comp[kk * l + g] != 0;
                    for pp in 0..p {
                        // The interpolation of one tap as a map cell -> weight: a
                        // direction on a ring boundary may come out with the two
                        // rings in other slots (a weight of 1e-16 instead of 0).
                        let mut diff: Vec<(u32, f64)> = Vec::new();
                        for n in 0..4 {
                            let c = cpu.at(n, g, kk, pp);
                            let d = ((g * 4 + n) * p + pp) * k + kk;
                            diff.push((cpu.pos[c], cpu.w[c]));
                            diff.push((pos[d], -w[d]));
                        }
                        diff.sort_by_key(|e| e.0);
                        let mut i = 0;
                        while i < diff.len() {
                            let (cell, mut s) = (diff[i].0, 0.0);
                            while i < diff.len() && diff[i].0 == cell {
                                s += diff[i].1;
                                i += 1;
                            }
                            // Incomplete rows renormalise over the cells found, where
                            // a 1e-16 weight may decide; the engine reads complete rows only.
                            if cpu.complete[kk] {
                                werr = werr.max(s.abs() / tol);
                                if s.abs() > 1e-9 {
                                    bad_pos += 1;
                                }
                            }
                        }
                    }
                }
                bad_complete += (row_complete != cpu.complete[kk]) as usize;
            }
            // Convolution forward and adjoint on random data, 2 rows.
            let ker = Kernel::shared(&wavelet_weights(ksz), l, 1, 2, p);
            let ker_d = gpu.upload(&ker.w);
            let x = noise(nside as u64 * 7 + ksz as u64, 2 * domain.len());
            let y_cpu = convolution_forward(&cpu, &ker, &x, 2);
            let x_d = gpu.upload(&x);
            let y_d = gpu.zeros::<f64>(2 * l * 2 * k);
            dev.forward(&gpu, &ker_d, gpu.a(&x_d), domain.len(), 2, 0, k, NULL, gpu.a(&y_d), k);
            let y = gpu.download(&y_d);
            let mut yerr = 0f64;
            for (i, (a, b)) in y_cpu.iter().zip(&y).enumerate() {
                let kk = i % k;
                if cpu.complete[kk] {
                    yerr = yerr.max((a - b).abs() / conv_noise(rows[kk], nside));
                }
            }
            let mut g = noise(nside as u64 * 13 + l as u64, 2 * l * 2 * k);
            for (i, v) in g.iter_mut().enumerate() {
                if !cpu.complete[i % k] {
                    *v = 0.0;
                }
            }
            let gx_cpu = convolution_adjoint(&cpu, &ker, &g, 2);
            let g_d = gpu.upload(&g);
            let gx_d = gpu.zeros::<f64>(2 * domain.len());
            dev.adjoint(&gpu, &ker_d, gpu.a(&g_d), k, 2, 0, k, NULL, gpu.a(&gx_d), domain.len(), false);
            let gx = gpu.download(&gx_d);
            let scale = gx_cpu.iter().fold(0f64, |m, v| m.max(v.abs()));
            let gerr = gx_cpu.iter().zip(&gx).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
            eprintln!(
                "nside {nside} kernel {ksz} L {l} rows {k}: weights {werr:.2} x noise, {bad_pos} positions, \
                 {bad_complete} complete flags differ; forward {yerr:.2} x noise; adjoint {gerr:.1e} relative"
            );
            assert_eq!(bad_pos, 0);
            assert_eq!(bad_complete, 0);
            assert!(werr <= 1.0 && yerr <= 1.0, "weights / outputs beyond the rounding noise");
            assert!(gerr <= 1e-12);
        }
    }
}
