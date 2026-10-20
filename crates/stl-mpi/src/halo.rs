//! Halo exchange and ghost accumulation over a [`Comm`].

use crate::comm::Comm;
use crate::decomposition::Decomposition;
use stl_core::engine::{Exchange, Partition};

/// Per level and peer: the halo positions of this rank owned by the peer
/// (`ghosts`) and the owned positions the peer holds as halo (`serve`),
/// both in increasing cell id order.
pub struct HaloExchange<'c, C: Comm> {
    comm: &'c C,
    ghosts: Vec<Vec<Vec<usize>>>,
    serve: Vec<Vec<Vec<usize>>>,
    lens: Vec<usize>,
}

impl<'c, C: Comm> HaloExchange<'c, C> {
    /// Collective: every rank announces the halo cells it holds to their owners.
    pub fn new(comm: &'c C, dec: &Decomposition, part: &Partition) -> Self {
        let size = comm.size();
        assert_eq!(size, dec.size());
        let mut ghosts = Vec::new();
        let mut serve = Vec::new();
        let mut lens = Vec::new();
        for (level, layout) in part.layouts.iter().enumerate() {
            let mut requests: Vec<Vec<i64>> = vec![Vec::new(); size];
            let mut gpos: Vec<Vec<usize>> = vec![Vec::new(); size];
            for p in 0..layout.len() {
                let id = layout.id(p);
                if !part.owns(level, id) {
                    let owner = dec.owner(level, id);
                    requests[owner].push(id);
                    gpos[owner].push(p);
                }
            }
            let asked = comm.alltoallv_i64(requests);
            serve.push(asked.iter().map(|ids| ids.iter().map(|&id| layout.pos(id)).collect()).collect());
            ghosts.push(gpos);
            lens.push(layout.len());
        }
        // The point-to-point exchanges trust both sides' counts: check once
        // that what each peer will send is what this rank expects.
        let sizes: Vec<Vec<i64>> = (0..size)
            .map(|p| serve.iter().chain(ghosts.iter()).map(|lv: &Vec<Vec<usize>>| lv[p].len() as i64).collect())
            .collect();
        let peer_sizes = comm.alltoallv_i64(sizes);
        let levels = ghosts.len();
        for (p, theirs) in peer_sizes.iter().enumerate() {
            for l in 0..levels {
                assert_eq!(theirs[l] as usize, ghosts[l][p].len(), "level {l}: rank {p} serves {} cells, rank {} expects {}", theirs[l], comm.rank(), ghosts[l][p].len());
                assert_eq!(theirs[levels + l] as usize, serve[l][p].len(), "level {l}: rank {p} sends {} ghosts, rank {} expects {}", theirs[levels + l], comm.rank(), serve[l][p].len());
            }
        }
        HaloExchange { comm, ghosts, serve, lens }
    }

    /// Number of halo cells per level (cells held but not owned).
    pub fn halo_cells(&self) -> Vec<usize> {
        self.ghosts.iter().map(|g| g.iter().map(Vec::len).sum()).collect()
    }

    fn pack(field: &[f64], len: usize, rows: usize, pos: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * pos.len());
        for r in 0..rows {
            out.extend(pos.iter().map(|&p| field[r * len + p]));
        }
        out
    }
}

impl<C: Comm> Exchange for HaloExchange<'_, C> {
    fn halo(&self, level: usize, field: &mut [f64], rows: usize) {
        let len = self.lens[level];
        assert_eq!(field.len(), rows * len);
        let send = self.serve[level].iter().map(|pos| Self::pack(field, len, rows, pos)).collect();
        let counts: Vec<usize> = self.ghosts[level].iter().map(|pos| rows * pos.len()).collect();
        let recv = self.comm.exchange_f64(send, &counts);
        for (peer, data) in recv.iter().enumerate() {
            let pos = &self.ghosts[level][peer];
            assert_eq!(data.len(), rows * pos.len());
            for r in 0..rows {
                for (i, &p) in pos.iter().enumerate() {
                    field[r * len + p] = data[r * pos.len() + i];
                }
            }
        }
    }

    fn reduce_ghosts(&self, level: usize, field: &mut [f64], rows: usize) {
        let len = self.lens[level];
        assert_eq!(field.len(), rows * len);
        let send = self.ghosts[level].iter().map(|pos| Self::pack(field, len, rows, pos)).collect();
        for pos in &self.ghosts[level] {
            for r in 0..rows {
                pos.iter().for_each(|&p| field[r * len + p] = 0.0);
            }
        }
        let counts: Vec<usize> = self.serve[level].iter().map(|pos| rows * pos.len()).collect();
        let recv = self.comm.exchange_f64(send, &counts);
        // Owner's own value first, then the peers' contributions in rank order.
        for (peer, data) in recv.iter().enumerate() {
            let pos = &self.serve[level][peer];
            assert_eq!(data.len(), rows * pos.len());
            for r in 0..rows {
                for (i, &p) in pos.iter().enumerate() {
                    field[r * len + p] += data[r * pos.len() + i];
                }
            }
        }
    }

    fn allreduce(&self, values: &mut [f64]) {
        self.comm.allreduce_sum(values);
    }

    fn trivial(&self) -> bool {
        self.comm.size() == 1
    }
}
