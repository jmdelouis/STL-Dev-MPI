//! The device DIA convolution (`stl_gpu::dia`) against the current device
//! convolution (`conv_fwd`, `conv_adj`), the CPU convolution
//! (`stl_core::conv`) and the CPU DIA reference (`stl_core::dia`).

mod common;

use common::conv_noise;
use stl_core::conv::{convolution_adjoint, convolution_forward, ConvGeometry, Kernel};
use stl_core::dia::{DiaGeometry as CpuDia, Diagonals};
use stl_core::neighbours::dilate_fine;
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;
use stl_gpu::device::NULL;
use stl_gpu::dia::{DeviceDiagonals, DiaGeometry, OwnRows};
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

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn dia_matches_the_convolution() {
    let Some(gpu) = gpu() else { return };
    eprintln!("device: {}", gpu.name);
    let cases: [(i64, usize, usize, i64, usize); 9] = [
        // nside, kernel, L, block start, block size
        (32, 5, 2, 5000, 256),
        (32, 5, 4, 2048, 512),
        (64, 5, 4, 4096 - 512, 512),      // north pole (corner of 4 faces)
        (64, 5, 4, 5 * 4096 - 512, 512),  // north corner of face 4 (3 faces)
        (64, 5, 4, 4 * 4096, 256),        // south corner of face 4 (3 faces)
        (128, 5, 4, 100_000, 1024),
        (1024, 5, 2, 6_000_000, 2048),
        (1024, 3, 4, 12_500_000, 1024),
        (1024, 5, 4, 3 * 1024 * 1024 - 1000, 3000), // faces 2 and 3, equator
    ];
    for (nside, ksz, l, start, size) in cases {
        let rings = ((ksz - 1) / 2 + 2) as u32;
        let block: Vec<i64> = (start..start + size as i64).collect();
        let a1 = dilate_fine(&block, nside, rings);
        let domain = dilate_fine(&a1, nside, rings);
        let p = ksz * ksz;
        let diagonals = Diagonals::observed(&[8, 16, 32], ksz, l, Gauge::Cosmo);
        let diag = DeviceDiagonals::new(&gpu, &diagonals);
        let ker = Kernel::shared(&wavelet_weights(ksz), l, 1, 2, p);
        let ker_d = gpu.upload(&ker.w);
        for (rows, own) in [(&a1, None), (&domain, Some((start, start + size as i64)))] {
            let k = rows.len();
            let cpu = ConvGeometry::rows(&domain, rows, nside, ksz, l, Gauge::Cosmo);
            let cpu_dia = CpuDia::from_conv(&cpu, &ker, rows, &domain, &diagonals).expect("read off the diagonals");
            let grid = Grid::new(&gpu, &domain, own);
            let rows_d = gpu.upload(rows);
            let spec = Spec { nside, kernel_sz: ksz, g: l, gauge: Gauge::Cosmo };
            let old = DeviceGeometry::build(&gpu, &grid, &rows_d, k, &spec, false);
            let own_rows = OwnRows::longest_run(rows);
            let mut dia = DiaGeometry::build(&gpu, &grid, &rows_d, k, &spec, &diag, &ker_d, own_rows);
            assert_eq!(dia.misses, 0, "reads off the diagonals");
            // Dropping and recomputing the coefficients gives them back to the bit.
            let coef = dia.dense(&gpu);
            dia.strip();
            dia.refresh(&gpu, &grid, &rows_d, &spec, &diag, &ker_d);
            assert_eq!(dia.dense(&gpu), coef);
            // Complete rows only: incomplete ones renormalise over the cells
            // found, where a 1e-16 weight may decide (geometry.rs).
            let on_complete = |v: &[f64]| -> Vec<f64> { v.iter().enumerate().map(|(i, &a)| if cpu.complete[i % k] { a } else { 0.0 }).collect() };
            let coef_err = max_rel(&on_complete(&coef), &on_complete(&cpu_dia.coef));

            // Forward on random data, 2 input rows.
            let x = noise(nside as u64 * 7 + ksz as u64, 2 * domain.len());
            let x_d = gpu.upload(&x);
            let run_fwd = |which: u8| {
                let y_d = gpu.zeros::<f64>(2 * l * 2 * k);
                if which == 0 {
                    dia.forward(&gpu, &grid, &rows_d, &diag, gpu.a(&x_d), domain.len(), 2, 0, k, NULL, gpu.a(&y_d), k);
                } else {
                    old.forward(&gpu, &ker_d, gpu.a(&x_d), domain.len(), 2, 0, k, NULL, gpu.a(&y_d), k);
                }
                gpu.download(&y_d)
            };
            let (y, y_old) = (run_fwd(0), run_fwd(1));
            let y_cpu = convolution_forward(&cpu, &ker, &x, 2);
            let y_cpu_dia = cpu_dia.forward(rows, &domain, &x, 2);
            let (mut vs_cpu, mut vs_old) = (0f64, 0f64);
            for i in 0..y.len() {
                let kk = i % k;
                let tol = conv_noise(rows[kk], nside);
                if cpu.complete[kk] {
                    vs_cpu = vs_cpu.max((y[i] - y_cpu[i]).abs() / tol);
                    vs_old = vs_old.max((y[i] - y_old[i]).abs() / tol);
                }
            }
            let vs_cpu_dia = max_rel(&on_complete(&y), &on_complete(&y_cpu_dia));

            // Adjoint on cotangents of the complete rows.
            let mut g = noise(nside as u64 * 13 + l as u64, 2 * l * 2 * k);
            for (i, v) in g.iter_mut().enumerate() {
                if !cpu.complete[i % k] {
                    *v = 0.0;
                }
            }
            let g_d = gpu.upload(&g);
            let run_adj = |which: u8, row0: usize, n: usize, gd: &cudarc::driver::CudaSlice<f64>, stride: usize| {
                let gx_d = gpu.zeros::<f64>(2 * domain.len());
                if which == 0 {
                    dia.adjoint(&gpu, &grid, &rows_d, &diag, gpu.a(gd), stride, 2, row0, n, NULL, gpu.a(&gx_d), domain.len(), false);
                } else {
                    old.adjoint(&gpu, &ker_d, gpu.a(gd), stride, 2, row0, n, NULL, gpu.a(&gx_d), domain.len(), false);
                }
                gpu.download(&gx_d)
            };
            let gx = run_adj(0, 0, k, &g_d, k);
            let gx_old = run_adj(1, 0, k, &g_d, k);
            let gx_cpu = convolution_adjoint(&cpu, &ker, &g, 2);
            let gx_cpu_dia = cpu_dia.adjoint(rows, &domain, &g, 2);
            let (adj_cpu, adj_old, adj_cpu_dia) = (max_rel(&gx, &gx_cpu), max_rel(&gx, &gx_old), max_rel(&gx, &gx_cpu_dia));
            let old_cpu = max_rel(&gx_old, &gx_cpu);
            // The rounding noise of the weights (relative to O(1) outputs)
            // grows near the poles (conv_noise).
            let noise = (0..k).filter(|&kk| cpu.complete[kk]).map(|kk| conv_noise(rows[kk], nside)).fold(0f64, f64::max);

            // The block's own rows only (as for the second wavelets): rows
            // f0 .. f0 + len of the cotangent, the others zero on the CPU.
            let (f0, len) = (own_rows.f0 as usize, own_rows.len as usize);
            let mut g_sub = vec![0f64; 2 * l * 2 * len];
            let mut g_cpu = vec![0f64; g.len()];
            for row in 0..2 * l * 2 {
                for i in 0..len {
                    g_sub[row * len + i] = g[row * k + f0 + i];
                    g_cpu[row * k + f0 + i] = g[row * k + f0 + i];
                }
            }
            let g_sub_d = gpu.upload(&g_sub);
            let gx_sub = run_adj(0, f0, len, &g_sub_d, len);
            let adj_sub = max_rel(&gx_sub, &convolution_adjoint(&cpu, &ker, &g_cpu, 2));

            eprintln!(
                "nside {nside} kernel {ksz} L {l} rows {k} ({} own), {} diagonals ({} slots), {} cells: coefficients {coef_err:.1e} from the CPU DIA; \
                 forward {vs_cpu:.3} x noise from the CPU, {vs_old:.3} from conv_fwd, {vs_cpu_dia:.1e} relative from the CPU DIA; \
                 adjoint {adj_cpu:.1e} / {adj_old:.1e} / {adj_cpu_dia:.1e} relative (CPU, conv_adj, CPU DIA; conv_adj from the CPU {old_cpu:.1e}, noise {noise:.1e}), own rows {adj_sub:.1e}",
                own_rows.len,
                diag.d,
                dia.slots,
                dia.n_dest
            );
            assert!(vs_cpu <= 1.0 && vs_old <= 1.0, "forward beyond the rounding noise");
            // Same device weights as conv_adj: only the summation order differs.
            assert!(adj_old <= 1e-14, "adjoint {adj_old:e} from conv_adj");
            assert!(adj_cpu <= noise && adj_sub <= noise, "adjoint beyond the rounding noise");
        }
    }
}
