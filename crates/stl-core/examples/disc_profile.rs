//! Cost of the parts of a disc geometry (support, decimations,
//! convolution binding) on one thread: `disc_profile NSIDE RADIUS_DEG`.

use std::time::Instant;
use stl_core::disc::{disc_pixels, pixel_vector};
use stl_core::local::LocalGeometry;
use stl_core::stencil::Gauge;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nside: i64 = args.get(1).map_or(256, |s| s.parse().unwrap());
    let radius = args.get(2).map_or(10.0, |s| s.parse::<f64>().unwrap()).to_radians();
    let levels = nside.ilog2() as usize - 1;
    let geo = LocalGeometry::new(nside, 5, 4, Gauge::Cosmo, 4096);
    let centre = pixel_vector(12345 % (12 * nside * nside), nside);
    let t = Instant::now();
    let ids0 = disc_pixels(nside, centre, radius);
    let t_ids = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut ids = vec![ids0];
    for d in 0..levels - 1 {
        let dn = geo.down(d, &ids[d]);
        ids.push(dn.out_ids);
    }
    let t_down = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for (n, s) in ids.iter().enumerate() {
        std::hint::black_box(geo.conv_plan(n, s, s));
    }
    let t_conv = t.elapsed().as_secs_f64();
    let rows: usize = ids.iter().map(Vec::len).sum();
    println!(
        "nside {nside}: {} pixels at level 0, {rows} over {levels} levels; support {:.4} s, decimations {:.4} s, convolutions {:.4} s ({:.1} us per row)",
        ids[0].len(),
        t_ids,
        t_down,
        t_conv,
        1e6 * t_conv / rows as f64
    );
}
