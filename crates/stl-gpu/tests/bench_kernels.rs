//! Timing of the device kernels on one block (not a check):
//! `cargo test --release -p stl-gpu --test bench_kernels -- --ignored --nocapture`
//! (`STL_BENCH_NSIDE`, `STL_BENCH_L`, `STL_BENCH_ROWS` to change the case).

use std::time::Instant;
use stl_core::conv::Kernel;
use stl_core::neighbours::dilate_fine;
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;
use stl_gpu::device::{Arg, NULL};
use stl_gpu::geometry::{DeviceGeometry, Grid, Spec};
use stl_gpu::Gpu;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Host-side setup of one rank: partition (two CPU block sizes) and device topology.
#[test]
#[ignore]
fn bench_setup() {
    use stl_core::engine::EngineConfig;
    use stl_core::local::LocalGeometry;
    use stl_gpu::topology::Topology;
    use stl_mpi::decomposition::{build_partition, Decomposition};
    let nside = env("STL_BENCH_NSIDE", 1024) as i64;
    let ranks = env("STL_BENCH_RANKS", 1);
    let j = (63 - nside.leading_zeros() as usize).saturating_sub(1);
    let cfg = EngineConfig::new(j, false);
    for block in [16384usize, 1 << 18] {
        let geo = LocalGeometry::new(nside, 5, 2, Gauge::Cosmo, block);
        let dec = Decomposition::new(&geo, cfg.levels(), ranks);
        let t = Instant::now();
        let part = build_partition(&geo, &dec, 0);
        eprintln!("nside {nside}, {ranks} ranks, CPU blocks of {block}: build_partition {:.1} s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let topo = Topology::new(&geo, &part, 1 << 20, threads);
        eprintln!("  device topology ({threads} threads): {:.1} s, blocks {:?}", t.elapsed().as_secs_f64(), topo.levels.iter().map(|l| l.blocks.len()).collect::<Vec<_>>());
    }
}

/// The bound stencils with their transposition (`geometry`) against the
/// DIA geometry (`dia`) on one block of `STL_BENCH_ROWS` rows from pixel 0:
/// build, weights / coefficients recomputed, forward and adjoint for 1, 4
/// and 16 input rows, bytes kept; and, as an upper bound only, a constant
/// kernel on face images (FP64 and FP32, not the operator):
/// `cargo test --release -p stl-gpu --test bench_kernels bench_dia -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_dia() {
    use stl_core::dia::Diagonals;
    use stl_gpu::dia::{DeviceDiagonals, DiaGeometry, OwnRows};
    let gpu = Gpu::new(0).unwrap();
    let nside = env("STL_BENCH_NSIDE", 256) as i64;
    let l = env("STL_BENCH_L", 4);
    let n = env("STL_BENCH_ROWS", (12 * nside * nside) as usize).min((12 * nside * nside) as usize);
    let reps = env("STL_BENCH_REPS", 3);
    let ksz = 5;
    let rings = ((ksz - 1) / 2 + 2) as u32;
    let block: Vec<i64> = (0..n as i64).collect();
    let a1 = dilate_fine(&block, nside, rings);
    let grid_ids = dilate_fine(&a1, nside, rings);
    let grid = Grid::new(&gpu, &grid_ids, Some((0, n as i64)));
    let rows = gpu.upload(&a1);
    let k = a1.len();
    let spec = Spec { nside, kernel_sz: ksz, g: l, gauge: Gauge::Cosmo };
    let ker_h = Kernel::shared(&wavelet_weights(ksz), l, 1, 2, ksz * ksz);
    let ker = gpu.upload(&ker_h.w);
    let diagonals = Diagonals::observed(&[8, 16, 32], ksz, l, Gauge::Cosmo);
    let diag = DeviceDiagonals::new(&gpu, &diagonals);
    let own = OwnRows::longest_run(&a1);
    let time = |f: &mut dyn FnMut()| -> f64 {
        f();
        gpu.sync();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        gpu.sync();
        t.elapsed().as_secs_f64() / reps as f64 * 1e3
    };
    eprintln!("{} ({}), nside {nside}, L {l}: {k} rows, {} grid cells, {} diagonals", gpu.name, gpu.ordinal(), grid.n, diag.d);
    let mut table: Vec<(String, f64, f64)> = Vec::new();
    let t_old = time(&mut || drop(DeviceGeometry::build(&gpu, &grid, &rows, k, &spec, false)));
    let t_dia = time(&mut || drop(DiaGeometry::build(&gpu, &grid, &rows, k, &spec, &diag, &ker, own)));
    table.push(("build".into(), t_old, t_dia));
    let mut old = DeviceGeometry::build(&gpu, &grid, &rows, k, &spec, false);
    let mut dia = DiaGeometry::build(&gpu, &grid, &rows, k, &spec, &diag, &ker, own);
    assert_eq!(dia.misses, 0);
    let (old_full, old_kept, dia_full) = (old.bytes(), DeviceGeometry::stripped_bytes(&old), dia.bytes());
    let t_old = time(&mut || {
        old.strip();
        old.refresh(&gpu, &grid, &rows, &spec);
    });
    let t_dia = time(&mut || {
        dia.strip();
        dia.refresh(&gpu, &grid, &rows, &spec, &diag, &ker);
    });
    table.push(("weights / coefficients recomputed".into(), t_old, t_dia));
    let dia_kept = dia.stripped_bytes();
    for r in [1usize, 4, 16] {
        let x = gpu.upload(&(0..r * grid.n).map(|i| ((i % 97) as f64) / 97.0 - 0.5).collect::<Vec<_>>());
        let y_old = gpu.zeros::<f64>(r * l * 2 * k);
        let y_dia = gpu.zeros::<f64>(r * l * 2 * k);
        let t_old = time(&mut || old.forward(&gpu, &ker, gpu.a(&x), grid.n, r, 0, k, NULL, gpu.a(&y_old), k));
        let t_dia = time(&mut || dia.forward(&gpu, &grid, &rows, &diag, gpu.a(&x), grid.n, r, 0, k, NULL, gpu.a(&y_dia), k));
        let (a, b) = (gpu.download(&y_old), gpu.download(&y_dia));
        let fwd_diff = a.iter().zip(&b).fold(0f64, |m, (p, q)| m.max((p - q).abs()));
        table.push((format!("forward, {r} input rows"), t_old, t_dia));
        let out_old = gpu.zeros::<f64>(r * grid.n);
        let out_dia = gpu.zeros::<f64>(r * grid.n);
        let t_old = time(&mut || old.adjoint(&gpu, &ker, gpu.a(&y_old), k, r, 0, k, NULL, gpu.a(&out_old), grid.n, false));
        let t_dia = time(&mut || dia.adjoint(&gpu, &grid, &rows, &diag, gpu.a(&y_old), k, r, 0, k, NULL, gpu.a(&out_dia), grid.n, false));
        let (a, b) = (gpu.download(&out_old), gpu.download(&out_dia));
        let scale = a.iter().fold(0f64, |m, v| m.max(v.abs()));
        let adj_diff = a.iter().zip(&b).fold(0f64, |m, (p, q)| m.max((p - q).abs())) / scale;
        table.push((format!("adjoint, {r} input rows"), t_old, t_dia));
        eprintln!("  {r} input rows: forward max |old - DIA| {fwd_diff:.1e}, adjoint {adj_diff:.1e} relative");
    }
    // Upper bound: constant kernel on face images, same number of output rows.
    let stride = n + 3 * nside as usize;
    let mut bound: Vec<(usize, f64, f64)> = Vec::new();
    for r in [1usize, 4, 16] {
        let x64 = gpu.upload(&vec![0.5f64; r * stride]);
        let x32 = gpu.upload(&vec![0.5f32; r * stride]);
        let k64 = gpu.upload(&ker_h.w);
        let k32 = gpu.upload(&ker_h.w.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let o64 = gpu.zeros::<f64>(r * l * 2 * n);
        let o32 = gpu.zeros::<f32>(r * l * 2 * n);
        let args = |x: Arg, kk: Arg, o: Arg| [x, stride.into(), (r as i32).into(), nside.into(), (l as i32).into(), (ksz as i32).into(), kk, n.into(), o];
        let t64 = time(&mut || gpu.launch1("face_conv_f64", n * r, &args(gpu.a(&x64), gpu.a(&k64), gpu.a(&o64))));
        let t32 = time(&mut || gpu.launch1("face_conv_f32", n * r, &args(gpu.a(&x32), gpu.a(&k32), gpu.a(&o32))));
        bound.push((r, t64, t32));
    }
    let per_row = |b: usize| b as f64 / k as f64;
    eprintln!("\n| kernel | bound stencils + transposition (ms) | DIA (ms) | ratio |");
    eprintln!("|---|---|---|---|");
    for (what, a, b) in &table {
        eprintln!("| {what} | {a:.2} | {b:.2} | {:.2} |", a / b);
    }
    eprintln!("\n| kept on the device | bound stencils + transposition | DIA |");
    eprintln!("|---|---|---|");
    eprintln!("| everything | {:.1} MB ({:.0} B / row) | {:.1} MB ({:.0} B / row) |", old_full as f64 / 1e6, per_row(old_full), dia_full as f64 / 1e6, per_row(dia_full));
    eprintln!("| between sweeps when stripped | {:.1} MB ({:.0} B / row) | {:.1} MB ({:.1} B / row) |", old_kept as f64 / 1e6, per_row(old_kept), dia_kept as f64 / 1e6, per_row(dia_kept));
    eprintln!("\nconstant kernel on face images ({n} rows; not the operator), ms: ");
    for (r, t64, t32) in bound {
        eprintln!("  {r} input rows: FP64 {t64:.2}, FP32 {t32:.2}");
    }
}

#[test]
#[ignore]
fn bench_kernels() {
    let gpu = Gpu::new(0).unwrap();
    let nside = env("STL_BENCH_NSIDE", 256) as i64;
    let l = env("STL_BENCH_L", 2);
    let n = env("STL_BENCH_ROWS", (12 * nside * nside) as usize).min((12 * nside * nside) as usize);
    let ksz = 5;
    let rings = ((ksz - 1) / 2 + 2) as u32;
    let block: Vec<i64> = (0..n as i64).collect();
    let a1 = dilate_fine(&block, nside, rings);
    let grid_ids = dilate_fine(&a1, nside, rings);
    let grid = Grid::new(&gpu, &grid_ids, Some((0, n as i64)));
    let rows = gpu.upload(&a1);
    let k = a1.len();
    let time = |what: &str, f: &dyn Fn()| {
        f();
        gpu.sync();
        let t = Instant::now();
        let reps = 3;
        for _ in 0..reps {
            f();
        }
        gpu.sync();
        let dt = t.elapsed().as_secs_f64() / reps as f64;
        eprintln!("{what:<40} {:8.2} ms  ({:.1} ns per row)", dt * 1e3, dt * 1e9 / k as f64);
    };
    let spec = Spec { nside, kernel_sz: ksz, g: l, gauge: Gauge::Cosmo };
    let t = Instant::now();
    let geom = DeviceGeometry::build(&gpu, &grid, &rows, k, &spec, false);
    gpu.sync();
    eprintln!("nside {nside} L {l}: {k} rows, geometry {:.1} MB", geom.bytes() as f64 / 1e6);
    eprintln!("{:<40} {:8.2} ms", "geometry build + transpose (first)", t.elapsed().as_secs_f64() * 1e3);
    time("geometry build + transpose", &|| {
        let _ = DeviceGeometry::build(&gpu, &grid, &rows, k, &spec, false);
    });
    let mut stripped = DeviceGeometry::build(&gpu, &grid, &rows, k, &spec, false);
    stripped.strip();
    eprintln!("transposition only: {:.1} MB", stripped.bytes() as f64 / 1e6);
    let cell = std::cell::RefCell::new(stripped);
    time("weights refresh (transposition kept)", &|| {
        let mut g = cell.borrow_mut();
        g.strip();
        g.refresh(&gpu, &grid, &rows, &spec);
    });
    let ker = gpu.upload(&Kernel::shared(&wavelet_weights(ksz), l, 1, 2, ksz * ksz).w);
    for r in [1usize, l] {
        let x = gpu.upload(&vec![1.0f64; r * grid.n]);
        let y = gpu.zeros::<f64>(r * l * 2 * k);
        time(&format!("forward, {r} input rows"), &|| geom.forward(&gpu, &ker, gpu.a(&x), grid.n, r, 0, k, NULL, gpu.a(&y), k));
        let out = gpu.zeros::<f64>(r * grid.n);
        time(&format!("adjoint, {r} input rows"), &|| {
            geom.adjoint(&gpu, &ker, gpu.a(&y), k, r, 0, k, NULL, gpu.a(&out), grid.n, true)
        });
    }
}
