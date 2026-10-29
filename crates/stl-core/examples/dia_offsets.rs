//! Feasibility of the DIA convolution: the cells read by every row of the
//! full-sky convolution geometry, as offsets `(dx, dy)` in the extended
//! frame of the row's face (`stl_core::dia::rel_offset`).
//!
//! `cargo run --release -p stl-core --example dia_offsets -- 64 256 1024`
//! (`STL_DIA_L`, `STL_DIA_KERNEL`: L = 4 and kernel 5 by default; gauge
//! cosmo; `STL_DIA_STRIDE` takes every n-th row).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use stl_core::conv::ConvGeometry;
use stl_core::dia::{ext_pix, rel_offset};
use stl_core::healpix::{nest2xyf, npix, order_of};
use stl_core::stencil::Gauge;

/// Half-width of the offset histograms.
const W: i64 = 12;
const SIDE: usize = (2 * W + 1) as usize;
const CHUNK: usize = 4096;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[inline]
fn slot(dx: i64, dy: i64) -> usize {
    ((dy + W) as usize) * SIDE + (dx + W) as usize
}

const CLASSES: [&str; 4] = ["interior", "edge", "corner (4 faces)", "corner (3 faces)"];

#[derive(Clone)]
struct Acc {
    rows: u64,
    entries: u64,
    zero_entries: u64,
    /// Entries by offset (nonzero weight).
    entries_by_off: Vec<u64>,
    /// (row, gauge) pairs reading an offset with a nonzero weight, by gauge.
    pairs_by_off: Vec<Vec<u64>>,
    /// Rows reading an offset (nonzero weight, any gauge).
    rows_by_off: Vec<u64>,
    /// Histograms: distinct cells per (row, gauge) with nonzero weight, the
    /// same with the zero weights, distinct cells per row (all gauges).
    distinct_rg: Vec<u64>,
    distinct_rg_all: Vec<u64>,
    distinct_row: Vec<u64>,
    /// Rows by class, and by class and max-norm of their offsets.
    class: [u64; 4],
    class_norm: [[u64; SIDE]; 4],
    /// Rows by (max |dx + dy|, max |dx - dy|) over their nonzero reads.
    band: Vec<u64>,
    /// Reads with no offset (face not adjacent) or beyond W; offsets that do
    /// not lead back to the cell; offsets whose reverse has another max-norm.
    failed: u64,
    roundtrip: u64,
    asym: u64,
}

impl Acc {
    fn new(g: usize) -> Self {
        Acc {
            rows: 0,
            entries: 0,
            zero_entries: 0,
            entries_by_off: vec![0; SIDE * SIDE],
            pairs_by_off: vec![vec![0; SIDE * SIDE]; g],
            rows_by_off: vec![0; SIDE * SIDE],
            distinct_rg: vec![0; 101],
            distinct_rg_all: vec![0; 101],
            distinct_row: vec![0; 801],
            class: [0; 4],
            class_norm: [[0; SIDE]; 4],
            band: vec![0; SIDE * SIDE],
            failed: 0,
            roundtrip: 0,
            asym: 0,
        }
    }

    fn merge(&mut self, o: &Acc) {
        fn add(a: &mut [u64], b: &[u64]) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        self.rows += o.rows;
        self.entries += o.entries;
        self.zero_entries += o.zero_entries;
        add(&mut self.entries_by_off, &o.entries_by_off);
        for (a, b) in self.pairs_by_off.iter_mut().zip(&o.pairs_by_off) {
            add(a, b);
        }
        add(&mut self.rows_by_off, &o.rows_by_off);
        add(&mut self.distinct_rg, &o.distinct_rg);
        add(&mut self.distinct_rg_all, &o.distinct_rg_all);
        add(&mut self.distinct_row, &o.distinct_row);
        add(&mut self.class, &o.class);
        for c in 0..4 {
            add(&mut self.class_norm[c], &o.class_norm[c]);
        }
        add(&mut self.band, &o.band);
        self.failed += o.failed;
        self.roundtrip += o.roundtrip;
        self.asym += o.asym;
    }
}

fn analyse(nside: i64, ksz: usize, l: usize, stride: usize) -> Acc {
    let order = order_of(nside);
    let grid: Vec<i64> = (0..npix(nside)).collect();
    let rows: Vec<i64> = (0..npix(nside)).step_by(stride).collect();
    let next = AtomicUsize::new(0);
    let total = Mutex::new(Acc::new(l));
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut acc = Acc::new(l);
                let mut cells_rg: Vec<i64> = Vec::new();
                let mut cells_rg_all: Vec<i64> = Vec::new();
                let mut cells_row: Vec<i64> = Vec::new();
                let mut row_offs: Vec<bool> = vec![false; SIDE * SIDE];
                loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c * CHUNK >= rows.len() {
                        break;
                    }
                    let part = &rows[c * CHUNK..((c + 1) * CHUNK).min(rows.len())];
                    let geo = ConvGeometry::rows(&grid, part, nside, ksz, l, Gauge::Cosmo);
                    for (k, &row) in part.iter().enumerate() {
                        let (x, y, f) = nest2xyf(row, order);
                        acc.rows += 1;
                        cells_row.clear();
                        row_offs.iter_mut().for_each(|v| *v = false);
                        let mut faces: Vec<i64> = Vec::new();
                        let (mut norm, mut ring, mut along) = (0i64, 0i64, 0i64);
                        for g in 0..l {
                            cells_rg.clear();
                            cells_rg_all.clear();
                            let mut pair_offs: Vec<usize> = Vec::new();
                            for p in 0..geo.p {
                                for n in 0..4 {
                                    let a = geo.at(n, g, k, p);
                                    let cell = grid[geo.pos[a] as usize];
                                    let w = geo.w[a];
                                    acc.entries += 1;
                                    cells_rg_all.push(cell);
                                    let fq = cell >> (2 * order);
                                    if fq != f && !faces.contains(&fq) {
                                        faces.push(fq);
                                    }
                                    let Some((dx, dy)) = rel_offset(row, cell, nside, order).filter(|o| o.0.abs() <= W && o.1.abs() <= W)
                                    else {
                                        acc.failed += 1;
                                        continue;
                                    };
                                    if ext_pix(f, x + dx, y + dy, nside, order) != Some(cell) {
                                        acc.roundtrip += 1;
                                    }
                                    if let Some((bx, by)) = rel_offset(cell, row, nside, order) {
                                        if bx.abs().max(by.abs()) != dx.abs().max(dy.abs()) {
                                            acc.asym += 1;
                                        }
                                    } else {
                                        acc.asym += 1;
                                    }
                                    norm = norm.max(dx.abs().max(dy.abs()));
                                    if w == 0.0 {
                                        acc.zero_entries += 1;
                                        continue;
                                    }
                                    ring = ring.max((dx + dy).abs());
                                    along = along.max((dx - dy).abs());
                                    acc.entries_by_off[slot(dx, dy)] += 1;
                                    pair_offs.push(slot(dx, dy));
                                    row_offs[slot(dx, dy)] = true;
                                    cells_rg.push(cell);
                                    cells_row.push(cell);
                                }
                            }
                            pair_offs.sort_unstable();
                            pair_offs.dedup();
                            for &s in &pair_offs {
                                acc.pairs_by_off[g][s] += 1;
                            }
                            for v in [&mut cells_rg, &mut cells_rg_all] {
                                v.sort_unstable();
                                v.dedup();
                            }
                            acc.distinct_rg[cells_rg.len()] += 1;
                            acc.distinct_rg_all[cells_rg_all.len()] += 1;
                        }
                        cells_row.sort_unstable();
                        cells_row.dedup();
                        acc.distinct_row[cells_row.len()] += 1;
                        for (s, &v) in row_offs.iter().enumerate() {
                            acc.rows_by_off[s] += v as u64;
                        }
                        let class = match faces.len() {
                            0 => 0,
                            1 => 1,
                            _ => {
                                let cx = if 2 * x < nside { -1 } else { nside };
                                let cy = if 2 * y < nside { -1 } else { nside };
                                if ext_pix(f, cx, cy, nside, order).is_some() {
                                    2
                                } else {
                                    3
                                }
                            }
                        };
                        acc.class[class] += 1;
                        acc.class_norm[class][norm.min(2 * W) as usize] += 1;
                        acc.band[(ring.min(2 * W) as usize) * SIDE + along.min(2 * W) as usize] += 1;
                    }
                }
                total.lock().unwrap().merge(&acc);
            });
        }
    });
    total.into_inner().unwrap()
}

fn mean(h: &[u64]) -> f64 {
    let n: u64 = h.iter().sum();
    h.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum::<f64>() / n as f64
}

fn range(h: &[u64]) -> (usize, usize) {
    let lo = h.iter().position(|&c| c > 0).unwrap_or(0);
    let hi = h.iter().rposition(|&c| c > 0).unwrap_or(0);
    (lo, hi)
}

fn report(nside: i64, ksz: usize, l: usize, a: &Acc) {
    println!("\n## nside {nside}, kernel {ksz}, L = {l}: {} rows, {} entries ({} with a zero weight)", a.rows, a.entries, a.zero_entries);
    println!("\nreads without an offset within {W}: {}; offsets not leading back: {}; reverse max-norm differs: {}", a.failed, a.roundtrip, a.asym);
    let union: Vec<usize> = (0..SIDE * SIDE).filter(|&s| a.rows_by_off[s] > 0).collect();
    let per_gauge: Vec<usize> = a.pairs_by_off.iter().map(|h| h.iter().filter(|&&c| c > 0).count()).collect();
    let (rg_lo, rg_hi) = range(&a.distinct_rg);
    let (all_lo, all_hi) = range(&a.distinct_rg_all);
    let (row_lo, row_hi) = range(&a.distinct_row);
    println!("\ndistinct offsets used (nonzero weight): {} over all rows and gauges; per gauge {:?}", union.len(), per_gauge);
    println!(
        "distinct cells per (row, gauge): {:.2} mean, {rg_lo}..{rg_hi} (with zero weights {:.2}, {all_lo}..{all_hi}); per row over the {l} gauges: {:.2}, {row_lo}..{row_hi}",
        mean(&a.distinct_rg),
        mean(&a.distinct_rg_all),
        mean(&a.distinct_row)
    );
    println!("\nrows by class (cells read on other faces), max-norm of the offsets:\n");
    println!("| class | rows | fraction | max-norm histogram |");
    println!("|---|---|---|---|");
    for c in 0..4 {
        let hist: Vec<String> =
            a.class_norm[c].iter().enumerate().filter(|(_, &n)| n > 0).map(|(i, n)| format!("{i}: {n}")).collect();
        println!("| {} | {} | {:.3e} | {} |", CLASSES[c], a.class[c], a.class[c] as f64 / a.rows as f64, hist.join(", "));
    }
    println!("\nrows by (max |dx + dy|, max |dx - dy|) of their nonzero reads:\n");
    for r in 0..SIDE {
        for t in 0..SIDE {
            let n = a.band[r * SIDE + t];
            if n > 0 {
                println!("  ring {r:2}, along {t:2}: {n}");
            }
        }
    }
    println!("\nper mille of the rows reading each offset (dx right, dy up; '.' never):\n");
    let (lo, hi) = union.iter().fold((W, -W), |(lo, hi), &s| {
        let (dx, dy) = ((s % SIDE) as i64 - W, (s / SIDE) as i64 - W);
        (lo.min(dx).min(dy), hi.max(dx).max(dy))
    });
    print!("     ");
    for dx in lo..=hi {
        print!("{dx:>5}");
    }
    println!();
    for dy in (lo..=hi).rev() {
        print!("{dy:>5}");
        for dx in lo..=hi {
            let n = a.rows_by_off[slot(dx, dy)];
            if n == 0 {
                print!("{:>5}", ".");
            } else {
                print!("{:>5}", ((n as f64 / a.rows as f64) * 1000.0).round() as u64);
            }
        }
        println!();
    }
    // Rows outside the candidate diagonal sets |dx + dy| <= ring, |dx - dy| <= along.
    println!("\nrows with a nonzero read outside |dx + dy| <= ring and |dx - dy| <= along:\n");
    for (ring, along) in [(5, 4), (5, 5), (6, 4), (6, 5), (6, 6), (7, 5), (7, 7), (8, 8)] {
        let mut out = 0u64;
        for r in 0..SIDE {
            for t in 0..SIDE {
                if r > ring || t > along {
                    out += a.band[r * SIDE + t];
                }
            }
        }
        let d = (-(W)..=W).flat_map(|dy| (-(W)..=W).map(move |dx| (dx, dy))).filter(|(dx, dy)| (dx + dy).abs() <= ring as i64 && (dx - dy).abs() <= along as i64).count();
        println!("  ring {ring}, along {along}: {d} diagonals, {out} rows ({:.2e})", out as f64 / a.rows as f64);
    }
}

fn main() {
    let ksz = env("STL_DIA_KERNEL", 5);
    let l = env("STL_DIA_L", 4);
    let stride = env("STL_DIA_STRIDE", 1);
    let nsides: Vec<i64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    for nside in if nsides.is_empty() { vec![64, 256, 1024] } else { nsides } {
        let t = std::time::Instant::now();
        let a = analyse(nside, ksz, l, stride);
        report(nside, ksz, l, &a);
        eprintln!("nside {nside}: {:.1} s", t.elapsed().as_secs_f64());
    }
}
