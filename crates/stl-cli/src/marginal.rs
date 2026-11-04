//! Pixel histogram of the map (`stl-synth --marginal-weight`): the squared
//! 1-D Wasserstein distance between the values of every member and those of
//! the target, `w / B sum_m (1/N) sum_i (x_m,(i) - t_(i))^2` over the sorted
//! values, with gradient `2 w / (B N) (x_p - t_(rank of p))`.  The
//! statistics leave the tails of the gaussianised map loose (too many pixels
//! beyond 4 sigma): the histogram matching back to kappa then moves the
//! statistics of the final map, and a term on `kappa = f(g)` acts on values
//! that the matching replaces.  The ranks are found by a distributed sample
//! sort: memory and messages per rank stay of the size of its slice.

use stl_mpi::Comm;

/// Sample sort of the values `v` (global pixel ids `ids`) over the ranks of
/// `comm`, ordered by (value, id): this rank's part of the global order and
/// the global position of its first element.
fn sort_global<C: Comm>(v: &[f64], ids: &[i64], comm: &C) -> (Vec<(f64, i64)>, usize) {
    let mut local: Vec<(f64, i64)> = v.iter().copied().zip(ids.iter().copied()).collect();
    local.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    distribute(local, comm)
}

/// The global order from this rank's pairs `local`, sorted by (value, id).
fn distribute<C: Comm>(local: Vec<(f64, i64)>, comm: &C) -> (Vec<(f64, i64)>, usize) {
    let p = comm.size();
    // Splitters from 4 P regular samples per rank.
    let k = 4 * p;
    let samples: Vec<f64> = if local.is_empty() { Vec::new() } else { (0..k).map(|i| local[i * local.len() / k].0).collect() };
    let mut all = comm.alltoallv_f64(vec![samples; p]).concat();
    all.sort_by(f64::total_cmp);
    let splitters: Vec<f64> = (1..p).map(|i| all[i * all.len() / p]).collect();
    let mut send = vec![Vec::new(); p];
    for &(x, id) in &local {
        send[splitters.partition_point(|s| *s < x)].extend([x, id as f64]);
    }
    let mut mine: Vec<(f64, i64)> = comm.alltoallv_f64(send).iter().flat_map(|m| m.chunks(2).map(|c| (c[0], c[1] as i64))).collect();
    mine.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let counts = comm.allgather_i64(mine.len() as i64);
    let start = counts[..comm.rank()].iter().sum::<i64>() as usize;
    (mine, start)
}

pub struct Marginal {
    pub weight: f64,
    /// This rank's part of the sorted target values (standardised) and the
    /// global position of its first one, for every rank.
    pub(crate) chunk: Vec<f64>,
    chunk_starts: Vec<usize>,
    pub(crate) npix: usize,
    pub(crate) members: usize,
}

impl Marginal {
    /// `target`: owned slice of the standardised target, from global pixel
    /// `start`; `comm`: the ranks sharing the sky.
    pub fn new<C: Comm>(weight: f64, target: &[f64], start: usize, npix: usize, members: usize, comm: &C) -> Self {
        let ids: Vec<i64> = (0..target.len()).map(|i| (start + i) as i64).collect();
        let (sorted, first) = sort_global(target, &ids, comm);
        let chunk_starts = comm.allgather_i64(first as i64).into_iter().map(|v| v as usize).collect();
        Marginal { weight, chunk: sorted.into_iter().map(|(x, _)| x).collect(), chunk_starts, npix, members }
    }

    /// Target value of the global rank of every owned value of `row`
    /// (global pixels `start..start + row.len()`).
    fn matched<C: Comm>(&self, row: &[f64], start: usize, comm: &C) -> Vec<f64> {
        let ids: Vec<i64> = (0..row.len()).map(|i| (start + i) as i64).collect();
        let (mine, first) = sort_global(row, &ids, comm);
        self.matched_order(mine, first, start, row.len(), comm)
    }

    /// [`matched`](Self::matched) from the row's local order: `perm` lists its
    /// positions by (value, position), as the device sort gives them.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn matched_sorted<C: Comm>(&self, row: &[f64], perm: &[u32], start: usize, comm: &C) -> Vec<f64> {
        let local: Vec<(f64, i64)> = perm.iter().map(|&q| (row[q as usize], (start + q as usize) as i64)).collect();
        let (mine, first) = distribute(local, comm);
        self.matched_order(mine, first, start, row.len(), comm)
    }

    /// Target values of the owned pixels of a row (`len` from global pixel
    /// `start`) from this rank's part `mine` of the global order, starting at
    /// global position `first`.
    fn matched_order<C: Comm>(&self, mine: Vec<(f64, i64)>, first: usize, start: usize, len: usize, comm: &C) -> Vec<f64> {
        let p = comm.size();
        // Target values of the global positions [first, first + len) of every rank.
        let firsts = comm.allgather_i64(first as i64);
        let lens = comm.allgather_i64(mine.len() as i64);
        let (c0, c1) = (self.chunk_starts[comm.rank()], self.chunk_starts[comm.rank()] + self.chunk.len());
        let send: Vec<Vec<f64>> = (0..p)
            .map(|q| {
                let (lo, hi) = ((firsts[q] as usize).max(c0), ((firsts[q] + lens[q]) as usize).min(c1));
                if lo < hi { self.chunk[lo - c0..hi - c0].to_vec() } else { Vec::new() }
            })
            .collect();
        let targets = comm.alltoallv_f64(send).concat();
        // Back to the owners of the pixels (ranges of the owned slices).
        let starts = comm.allgather_i64(start as i64);
        let mut back = vec![Vec::new(); p];
        for (&(_, id), &t) in mine.iter().zip(&targets) {
            back[starts.partition_point(|&s| s <= id) - 1].extend([id as f64, t]);
        }
        let mut out = vec![0.0; len];
        for m in comm.alltoallv_f64(back) {
            for c in m.chunks(2) {
                out[c[0] as usize - start] = c[1];
            }
        }
        out
    }

    /// The term and its gradient on the owned slice `x` of this group's
    /// members (rows of `len` values from global pixel `start`); `cross`
    /// links the groups.
    pub fn loss_and_grad<C: Comm>(&self, x: &[f64], len: usize, start: usize, comm: &C, cross: Option<&C>) -> (f64, Vec<f64>) {
        let n = self.npix as f64;
        let w = self.weight / self.members as f64;
        let mut sums = [0.0];
        let mut grad = vec![0.0; x.len()];
        for (row, g) in x.chunks(len).zip(grad.chunks_mut(len)) {
            let t = self.matched(row, start, comm);
            for ((gi, xi), ti) in g.iter_mut().zip(row).zip(&t) {
                let d = xi - ti;
                sums[0] += d * d;
                *gi = 2.0 * w / n * d;
            }
        }
        comm.allreduce_sum(&mut sums);
        let mut total = [w / n * sums[0]];
        if let Some(c) = cross {
            c.allreduce_sum(&mut total);
        }
        (total[0], grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stl_mpi::threads::run_ranks;

    fn field(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|p| ((p as f64 * 0.731 + seed).sin() + 0.4 * (p as f64 * 0.0137 + seed).cos()) * (1.0 + 0.3 * ((p % 7) as f64))).collect()
    }

    /// The term by a sort of the whole map in one place.
    fn reference(x: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut t = target.to_vec();
        t.sort_by(f64::total_cmp);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
        let mut d = vec![0.0; n];
        for (r, &p) in idx.iter().enumerate() {
            d[p] = x[p] - t[r];
        }
        (d.iter().map(|v| v * v).sum::<f64>() / n as f64, d.iter().map(|v| 2.0 * v / n as f64).collect())
    }

    /// Same loss and gradient as one sort of the whole map, on 1 to 5 ranks
    /// of unequal slices; zero for any permutation of the target.
    #[test]
    fn distributed_sort_matches_a_global_sort() {
        let npix = 12 * 8 * 8;
        let target = field(npix, 0.3);
        let x = field(npix, 1.1);
        let (l_ref, g_ref) = reference(&x, &target);
        let permuted: Vec<f64> = (0..npix).map(|p| target[(p * 7 + 3) % npix]).collect();
        for ranks in 1..=5 {
            run_ranks(ranks, |comm| {
                let cut = |r: usize| if r == ranks { npix } else { r * npix / ranks + 17 * (r % 2) };
                let (s, e) = (cut(comm.rank()), cut(comm.rank() + 1));
                let m = Marginal::new(1.0, &target[s..e], s, npix, 1, comm);
                let (l, g) = m.loss_and_grad(&x[s..e], e - s, s, comm, None);
                assert!((l - l_ref).abs() <= 1e-13 * l_ref, "P={ranks}: loss {l} vs {l_ref}");
                for (k, gk) in g.iter().enumerate() {
                    assert!((gk - g_ref[s + k]).abs() <= 1e-15, "P={ranks}: pixel {}", s + k);
                }
                let (l0, _) = m.loss_and_grad(&permuted[s..e], e - s, s, comm, None);
                assert!(l0.abs() < 1e-14, "P={ranks}: permuted target, loss {l0}");
            });
        }
    }

    /// Gradient against central differences, 2 members on 3 ranks.
    #[test]
    fn gradient_matches_finite_differences() {
        let npix = 12 * 8 * 8;
        let target = field(npix, 0.3);
        let maps = [field(npix, 1.1), field(npix, 2.7)];
        run_ranks(3, |comm| {
            let (s, e) = (comm.rank() * npix / 3, (comm.rank() + 1) * npix / 3);
            let len = e - s;
            let m = Marginal::new(0.7, &target[s..e], s, npix, 2, comm);
            let x: Vec<f64> = maps.iter().flat_map(|v| v[s..e].iter().copied()).collect();
            let (_, g) = m.loss_and_grad(&x, len, s, comm, None);
            for &k in &[0, 5, len - 1, len + 9, 2 * len - 3] {
                let h = 1e-7;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                if comm.rank() == 1 {
                    xp[k] += h;
                    xm[k] -= h;
                }
                let (lp, _) = m.loss_and_grad(&xp, len, s, comm, None);
                let (lm, _) = m.loss_and_grad(&xm, len, s, comm, None);
                if comm.rank() == 1 {
                    let fd = (lp - lm) / (2.0 * h);
                    let rel = (fd - g[k]).abs() / g[k].abs().max(1e-12);
                    assert!(rel <= 1e-5, "pixel {k}: gradient {} vs finite difference {fd} (rel {rel:.1e})", g[k]);
                }
            }
        });
    }
}
