//! Map gradient of the ensemble loss of `stl-synth --weights none
//! --ensemble B` (sum_i |(1/B) sum_m S_i(x_m) - S_i(target)|^2), serial CPU
//! engine, written as raw little-endian f64 [B, N]; for the gradient
//! comparison of docs/bench_n128.md.
//!
//!     cargo run --release -p stl-core --example ensemble_gradient -- \
//!         TARGET.npy INIT_PREFIX B J L OUT.f64
//!
//! TARGET.npy and INIT_PREFIX_mMMM.npy are float64 NESTED maps; the target
//! is standardised here (population moments), the inits are used as they are.

use num_complex::Complex64;
use stl_core::engine::{EngineConfig, LocalEngine};
use stl_core::local::LocalGeometry;
use stl_core::plan::{build_plan, Family};
use stl_core::stencil::Gauge;

fn read(path: &str) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    npyz::NpyFile::new(&bytes[..]).unwrap().into_vec::<f64>().unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (b, j, l): (usize, usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
    let raw = read(&a[1]);
    let n = raw.len();
    let nside = ((n / 12) as f64).sqrt().round() as i64;
    let mean = raw.iter().sum::<f64>() / n as f64;
    let std = (raw.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64).sqrt();
    let target: Vec<f64> = raw.iter().map(|x| (x - mean) / std).collect();
    let mut x = Vec::with_capacity(b * n);
    for m in 0..b {
        x.extend(read(&format!("{}_m{m:03}.npy", a[2])));
    }
    let geo = LocalGeometry::new(nside, 5, l, Gauge::parse("cosmo").unwrap(), 16384);
    let cfg = EngineConfig::new(j, false);
    let plan = build_plan(j, l, &cfg.j_to_dg, 1, None, &Family::ALL);
    let t = LocalEngine::new(&geo, &cfg, target, 1, 1).forward(&plan);
    let mut engine = LocalEngine::new(&geo, &cfg, x, b, 1);
    let v = engine.forward(&plan);
    let p = plan.len();
    let residual: Vec<Complex64> = (0..p).map(|i| (0..b).map(|m| v[m * p + i]).sum::<Complex64>() / b as f64 - t[i]).collect();
    let loss: f64 = residual.iter().map(|r| r.norm_sqr()).sum();
    // d loss = Re sum_m sum_i conj(2 r_i / B) dS_i(x_m).
    let g: Vec<Complex64> = (0..b).flat_map(|_| residual.iter().map(|r| r * (2.0 / b as f64))).collect();
    let maps_gradient = engine.reverse(&plan, &g);
    let (grad, _) = engine.analysis_vjp(maps_gradient);
    println!("nside {nside} J {j} L {l} B {b}: {p} coefficients, loss {loss:.15e}");
    let bytes: Vec<u8> = grad.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[6], bytes).unwrap();
}
