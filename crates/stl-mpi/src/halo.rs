//! Halo exchange and ghost accumulation over a [`Comm`].

use crate::comm::Comm;
use crate::decomposition::Decomposition;
use std::cell::Cell;
use stl_core::engine::{Exchange, Partition};

/// Per level and peer: the halo positions of this rank owned by the peer
/// (`ghosts`) and the owned positions the peer holds as halo (`serve`),
/// both in increasing cell id order.
pub struct HaloExchange<'c, C: Comm> {
    comm: &'c C,
    ghosts: Vec<Vec<Vec<usize>>>,
    serve: Vec<Vec<Vec<usize>>>,
    lens: Vec<usize>,
    /// Neighbour exchanges and global sums issued so far.
    exchanges: Cell<u64>,
    allreduces: Cell<u64>,
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
        HaloExchange { comm, ghosts, serve, lens, exchanges: Cell::new(0), allreduces: Cell::new(0) }
    }

    /// Number of halo cells per level (cells held but not owned).
    pub fn halo_cells(&self) -> Vec<usize> {
        self.ghosts.iter().map(|g| g.iter().map(Vec::len).sum()).collect()
    }

    /// `(neighbour exchanges, global sums)` issued since the start.
    pub fn calls(&self) -> (u64, u64) {
        (self.exchanges.get(), self.allreduces.get())
    }

    fn pack_into(out: &mut Vec<f64>, field: &[f64], len: usize, rows: usize, pos: &[usize]) {
        for r in 0..rows {
            out.extend(pos.iter().map(|&p| field[r * len + p]));
        }
    }

    /// Values of every field at `from[level][peer]`, concatenated per peer
    /// in the order of `fields` (then zeroed there when `clear_sent`); one
    /// exchange; then `unpack(field, len, rows, to[level][peer], data)` per
    /// peer in rank order and per field in the order of `fields`.
    fn exchange_many(
        &self,
        fields: &mut [(usize, &mut [f64], usize)],
        from: &[Vec<Vec<usize>>],
        to: &[Vec<Vec<usize>>],
        clear_sent: bool,
        unpack: impl Fn(&mut [f64], usize, usize, &[usize], &[f64]),
    ) {
        let size = self.comm.size();
        for (level, field, rows) in fields.iter() {
            assert_eq!(field.len(), rows * self.lens[*level]);
        }
        let send: Vec<Vec<f64>> = (0..size)
            .map(|peer| {
                let mut out = Vec::new();
                for (level, field, rows) in fields.iter() {
                    Self::pack_into(&mut out, field, self.lens[*level], *rows, &from[*level][peer]);
                }
                out
            })
            .collect();
        if clear_sent {
            for (level, field, rows) in fields.iter_mut() {
                let len = self.lens[*level];
                for pos in &from[*level] {
                    for r in 0..*rows {
                        pos.iter().for_each(|&p| field[r * len + p] = 0.0);
                    }
                }
            }
        }
        let counts: Vec<usize> =
            (0..size).map(|peer| fields.iter().map(|(level, _, rows)| rows * to[*level][peer].len()).sum()).collect();
        self.exchanges.set(self.exchanges.get() + 1);
        let recv = self.comm.exchange_f64(send, &counts);
        for (peer, data) in recv.iter().enumerate() {
            assert_eq!(data.len(), counts[peer]);
            let mut offset = 0;
            for (level, field, rows) in fields.iter_mut() {
                let pos = &to[*level][peer];
                let n = *rows * pos.len();
                unpack(field, self.lens[*level], *rows, pos, &data[offset..offset + n]);
                offset += n;
            }
        }
    }
}

impl<C: Comm> Exchange for HaloExchange<'_, C> {
    fn halo(&self, level: usize, field: &mut [f64], rows: usize) {
        self.halo_many(&mut [(level, field, rows)]);
    }

    fn reduce_ghosts(&self, level: usize, field: &mut [f64], rows: usize) {
        self.reduce_ghosts_many(&mut [(level, field, rows)]);
    }

    fn allreduce(&self, values: &mut [f64]) {
        self.allreduces.set(self.allreduces.get() + 1);
        self.comm.allreduce_sum(values);
    }

    /// Owners send their cells, one message per peer for all the fields.
    fn halo_many(&self, fields: &mut [(usize, &mut [f64], usize)]) {
        self.exchange_many(fields, &self.serve, &self.ghosts, false, |field, len, rows, pos, data| {
            for r in 0..rows {
                for (i, &p) in pos.iter().enumerate() {
                    field[r * len + p] = data[r * pos.len() + i];
                }
            }
        });
    }

    /// Halo contributions to their owners, one message per peer for all
    /// the fields.  Per cell: the owner's own value first, then the peers'
    /// contributions in rank order (as with one field at a time).
    fn reduce_ghosts_many(&self, fields: &mut [(usize, &mut [f64], usize)]) {
        self.exchange_many(fields, &self.ghosts, &self.serve, true, |field, len, rows, pos, data| {
            for r in 0..rows {
                for (i, &p) in pos.iter().enumerate() {
                    field[r * len + p] += data[r * pos.len() + i];
                }
            }
        });
    }
}
