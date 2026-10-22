//! Timing of the device kernels on one block (not a check):
//! `cargo test --release -p stl-gpu --test bench_kernels -- --ignored --nocapture`
//! (`STL_BENCH_NSIDE`, `STL_BENCH_L`, `STL_BENCH_ROWS` to change the case).

use std::time::Instant;
use stl_core::conv::Kernel;
use stl_core::neighbours::dilate_fine;
use stl_core::stencil::Gauge;
use stl_core::wavelet::wavelet_weights;
use stl_gpu::device::NULL;
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
