//! Cost of the overlapping-patch mode (prototype study): every rank
//! computes its own pyramid locally on extended regions, the levels from
//! `n_c` on are replicated on every rank (their stored moduli summed by one
//! allreduce), so an evaluation needs a handful of collectives instead of
//! per-level halo exchanges.  Prints, for a middle rank, the work regions
//! `W_n` (blocks evaluated) and map regions `M_n` against the owned cells.
//!
//!     cargo run --release -p stl-mpi --example overlap_cost -- NSIDE RANKS [KERNEL]

use std::collections::BTreeSet;
use stl_core::local::LocalGeometry;
use stl_core::neighbours::{children, dilate_fine};
use stl_core::stencil::Gauge;
use stl_mpi::Decomposition;

fn union(a: &[i64], b: &[i64]) -> Vec<i64> {
    let s: BTreeSet<i64> = a.iter().chain(b).copied().collect();
    s.into_iter().collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nside: i64 = args[1].parse().unwrap();
    let ranks: usize = args[2].parse().unwrap();
    let kernel: usize = args.get(3).map_or(5, |v| v.parse().unwrap());
    let geo = LocalGeometry::new(nside, kernel, 4, Gauge::Cosmo, 16384);
    let levels = (nside.trailing_zeros() as usize) - 1; // J = log2(nside) - 1, dg(j) = j
    let dec = Decomposition::new(&geo, levels, ranks);
    let rank = ranks / 2;
    let owned: Vec<Vec<i64>> = (0..levels).map(|d| { let (s, e) = dec.owned(rank, d); (s..e).collect() }).collect();
    let c = geo.conv_rings;
    let pre = |k: usize, t: &[i64]| dilate_fine(&children(t), geo.nside(k), geo.down_rings(k));
    let base: usize = owned.iter().map(Vec::len).sum();
    println!("nside {nside}, {ranks} ranks, rank {rank}, levels {levels}, conv rings {c}, owned level 0: {}", owned[0].len());
    println!("n_c  overhead(W)  |W_0|/|owned_0|  |M_0|/|owned_0|  replicated pixels");
    for n_c in (1..=levels).rev() {
        // W_k = owned_k U S_k, S_k = pre(S_{k+1} U dilate(W_{k+1}, c)), S = {} at n_c - 1.
        let mut w: Vec<Vec<i64>> = vec![Vec::new(); levels];
        let mut s_next: Vec<i64> = Vec::new();
        for k in (0..n_c).rev() {
            w[k] = union(&owned[k], &s_next);
            if k > 0 {
                let need = union(&s_next, &dilate_fine(&w[k], geo.nside(k), c));
                s_next = pre(k - 1, &need);
            }
        }
        // M_k = dilate(W_k, 2c) U decimation domain of M_{k+1}; M_{n_c} = owned (allgathered).
        let mut m_next: Vec<i64> = if n_c < levels { owned[n_c].clone() } else { Vec::new() };
        let mut m0 = Vec::new();
        for k in (0..n_c).rev() {
            let mut mk = dilate_fine(&dilate_fine(&w[k], geo.nside(k), c), geo.nside(k), c);
            if !m_next.is_empty() {
                mk = union(&mk, &dilate_fine(&children(&m_next), geo.nside(k), 2 * geo.down_rings(k)));
            }
            if k == 0 { m0 = mk.clone(); }
            m_next = mk;
        }
        let replicated: usize = (n_c..levels).map(|d| geo.pixels(d)).sum();
        let work: usize = (0..n_c).map(|k| w[k].len()).sum::<usize>() + replicated;
        println!(
            "{n_c:3}  {:10.2}  {:15.2}  {:15.2}  {replicated}",
            work as f64 / base as f64,
            w[0].len() as f64 / owned[0].len() as f64,
            m0.len() as f64 / owned[0].len() as f64
        );
    }
}
