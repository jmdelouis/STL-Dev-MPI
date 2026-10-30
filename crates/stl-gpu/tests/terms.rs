//! The objective terms on the device (`stl_gpu::terms`) against host
//! references written as the host code of `stl-cli` computes them: the
//! radix sort (the same permutation as a sort by value then index), the
//! 4-children pyramid, its moments and the level-power and kurtosis
//! gradients, the marginal term.

use stl_gpu::terms::{add_product, marginal_row, Pyramid, RadixSort};
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

/// Heavy-tailed values with ties, signed zeros and a few extremes.
fn field(seed: u64, n: usize) -> Vec<f64> {
    noise(seed, n)
        .iter()
        .enumerate()
        .map(|(i, v)| match i % 97 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.25,
            3 => -1e300,
            4 => 1e-310,
            _ => v * v * v * 40.0 + if i % 1013 == 0 { 9.0 } else { 0.0 },
        })
        .collect()
}

fn host_pyramid(row: &[f64], levels: usize) -> Vec<(Vec<f64>, f64)> {
    let mut cur = row.to_vec();
    let mut out = Vec::new();
    for _ in 0..levels {
        let parents: Vec<f64> = cur.chunks(4).map(|c| c.iter().sum::<f64>() / 4.0).collect();
        let d: Vec<f64> = cur.chunks(4).zip(&parents).flat_map(|(c, &m)| c.iter().map(move |v| v - m)).collect();
        let s = d.iter().map(|v| v * v).sum();
        out.push((d, s));
        cur = parents;
    }
    out
}

/// The device sort gives the order of `sort_by(total_cmp, then index)`,
/// on sizes around the tile and scan boundaries.
#[test]
fn radix_sort_matches_host_order() {
    let Some(gpu) = gpu() else { return };
    for &n in &[1usize, 7, 2047, 2048, 2049, 300_001, 1 << 20] {
        let x = field(n as u64 + 3, n);
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.sort_by(|&a, &b| x[a as usize].total_cmp(&x[b as usize]).then(a.cmp(&b)));
        let dx = gpu.upload(&x);
        let mut sorter = RadixSort::new(&gpu, n);
        let perm = gpu.download(sorter.sort(&gpu, gpu.a(&dx)));
        assert_eq!(perm, idx, "n = {n}");
    }
}

/// Pyramid means bit for bit, moments to rounding, the level-power and
/// kurtosis gradients (with and without the chain factor) against the
/// host expressions.
#[test]
fn pyramid_terms_match_host() {
    let Some(gpu) = gpu() else { return };
    let (rows, len, depth) = (3usize, 4usize.pow(7), 6usize);
    let x: Vec<f64> = (0..rows).flat_map(|r| field(11 + r as u64, len)).map(|v| v.clamp(-50.0, 50.0)).collect();
    let dk: Vec<f64> = noise(5, rows * len).iter().map(|v| 1.0 + v).collect();
    let dx = gpu.upload(&x);
    let pyr = Pyramid::new(&gpu, &dx, rows, len, depth);
    let mom = pyr.moments(&gpu, &dx);
    let host: Vec<Vec<(Vec<f64>, f64)>> = x.chunks(len).map(|row| host_pyramid(row, depth)).collect();
    for r in 0..rows {
        for n in 0..depth {
            let (d, s2) = &host[r][n];
            let s4: f64 = d.iter().map(|v| v * v * v * v).sum();
            let [a2, a4] = mom[r * depth + n];
            assert!((a2 - s2).abs() <= 1e-13 * s2, "row {r} level {n}: {a2} vs {s2}");
            assert!((a4 - s4).abs() <= 1e-13 * s4, "row {r} level {n}: {a4} vs {s4}");
        }
    }
    // Level power: coefficients per row and level.
    let coef: Vec<f64> = (0..rows * depth).map(|k| 0.3 + 0.01 * k as f64).collect();
    let g0 = noise(9, rows * len);
    let mut dg = gpu.upload(&g0);
    pyr.level_power_grad(&gpu, &dx, &coef, &mut dg);
    let got = gpu.download(&dg);
    for (r, (p, g)) in host.iter().zip(g0.chunks(len)).enumerate() {
        let mut g2 = vec![0.0; len];
        for (n, (d, _)) in p.iter().enumerate() {
            let k = 1usize << (2 * n);
            for (i, gi) in g2.iter_mut().enumerate() {
                *gi += coef[r * depth + n] * d[i / k];
            }
        }
        for i in 0..len {
            assert_eq!(got[r * len + i], g[i] + g2[i], "level power, row {r} pixel {i}");
        }
    }
    // Kurtosis: levels 0, 2, 5, (coef, K, V) per row and level.
    let lev = [0usize, 2, 5];
    let cf: Vec<f64> = (0..rows * lev.len()).flat_map(|k| [0.7 - 0.05 * k as f64, 3.0 + 0.1 * k as f64, 0.2 + 0.01 * k as f64]).collect();
    for chain in [false, true] {
        let mut dg = gpu.upload(&g0);
        let ddk = gpu.upload(&dk);
        pyr.kurtosis_grad(&gpu, &dx, &lev, &cf, chain.then_some(&ddk), &mut dg);
        let got = gpu.download(&dg);
        for (r, p) in host.iter().enumerate() {
            let mut g4 = vec![0.0; len];
            for (t, &n) in lev.iter().enumerate() {
                let q = &cf[(r * lev.len() + t) * 3..][..3];
                let (coef, kn, v) = (q[0], q[1], q[2]);
                let gd: Vec<f64> = p[n]
                    .0
                    .chunks(4)
                    .flat_map(|sib| {
                        let c = sib.iter().map(|di| di * di * di).sum::<f64>() / 4.0;
                        sib.iter().map(move |di| coef * (di * di * di - c - kn * v * di))
                    })
                    .collect();
                let k = 1usize << (2 * n);
                for (i, gi) in g4.iter_mut().enumerate() {
                    *gi += gd[i / k];
                }
            }
            for i in 0..len {
                let want = if chain { g0[r * len + i] + g4[i] * dk[r * len + i] } else { g0[r * len + i] + g4[i] };
                assert_eq!(got[r * len + i], want, "kurtosis (chain {chain}), row {r} pixel {i}");
            }
        }
    }
    // g += a b.
    let mut dg = gpu.upload(&g0);
    add_product(&gpu, &mut dg, &dx, Some(&gpu.upload(&dk)));
    let got = gpu.download(&dg);
    assert!(got.iter().enumerate().all(|(i, v)| *v == g0[i] + x[i] * dk[i]));
}

/// The marginal term of one row: gradient bit for bit, loss to rounding.
#[test]
fn marginal_matches_host() {
    let Some(gpu) = gpu() else { return };
    let n = 200_003;
    let x = field(21, n);
    let mut t = field(22, n);
    t.sort_by(f64::total_cmp);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    let coef = 2.0 * 0.37 / n as f64;
    let g0 = noise(4, n);
    let (mut want, mut loss) = (g0.clone(), 0.0);
    for (q, &p) in idx.iter().enumerate() {
        let d = x[p] - t[q];
        loss += d * d;
        want[p] += coef * d;
    }
    let (dx, dt, dg) = (gpu.upload(&x), gpu.upload(&t), gpu.upload(&g0));
    let mut sorter = RadixSort::new(&gpu, n);
    let perm = sorter.sort(&gpu, gpu.a(&dx));
    let got_loss = marginal_row(&gpu, gpu.a(&dx), perm, &dt, coef, gpu.a(&dg));
    assert!((got_loss - loss).abs() <= 1e-13 * loss, "{got_loss} vs {loss}");
    assert_eq!(gpu.download(&dg), want);
}
