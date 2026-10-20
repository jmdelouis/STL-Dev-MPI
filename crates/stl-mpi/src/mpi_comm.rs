//! MPI backend (rsmpi).  General exchanges: `MPI_Alltoall` of the counts
//! then `MPI_Alltoallv` (setup only).  In the sweeps: halo and ghost
//! exchanges by nonblocking point-to-point messages to the neighbours only,
//! sums by `MPI_Gather` to rank 0 (added there in rank order, as the
//! threads backend does) and `MPI_Bcast`, maxima by `MPI_Allreduce`.  No
//! per-evaluation operation costs O(P) messages per rank.
//!
//! Every point-to-point exchange carries its own tag (a counter, the same
//! on every rank since all ranks make the same sequence of calls), and
//! receives name that tag: a message can only match the exchange it
//! belongs to.  `STL_MPI_HALO=alltoall` switches the halo exchanges back
//! to `MPI_Alltoallv` (fallback, O(P) per rank).

use crate::comm::Comm;
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use mpi::{Count, Tag};
use std::cell::Cell;

pub struct MpiComm {
    world: SimpleCommunicator,
    seq: Cell<u64>,
    alltoall_halo: bool,
}

impl MpiComm {
    pub fn new(world: SimpleCommunicator) -> Self {
        let alltoall_halo = std::env::var("STL_MPI_HALO").map_or(false, |v| v == "alltoall");
        MpiComm { world, seq: Cell::new(0), alltoall_halo }
    }

    /// Tag of the next point-to-point exchange, in 1..=30000 (MPI
    /// guarantees tags up to 32767).
    fn next_tag(&self) -> Tag {
        let s = self.seq.get();
        self.seq.set(s + 1);
        (s % 30000) as Tag + 1
    }

    fn alltoallv<T: Equivalence + Default + Clone>(&self, send: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let size = self.size();
        assert_eq!(send.len(), size);
        let counts: Vec<Count> = send.iter().map(|v| v.len() as Count).collect();
        let mut recv_counts = vec![0 as Count; size];
        self.world.all_to_all_into(&counts[..], &mut recv_counts[..]);
        let displs = |c: &[Count]| -> Vec<Count> {
            c.iter().scan(0, |acc, &n| {
                let d = *acc;
                *acc += n;
                Some(d)
            })
            .collect()
        };
        let (sd, rd) = (displs(&counts), displs(&recv_counts));
        let flat: Vec<T> = send.concat();
        let total: Count = recv_counts.iter().sum();
        let mut buf = vec![T::default(); total as usize];
        {
            let sp = Partition::new(&flat[..], &counts[..], &sd[..]);
            let mut rp = PartitionMut::new(&mut buf[..], &recv_counts[..], &rd[..]);
            self.world.all_to_all_varcount_into(&sp, &mut rp);
        }
        (0..size).map(|p| buf[rd[p] as usize..(rd[p] + recv_counts[p]) as usize].to_vec()).collect()
    }
}

impl Comm for MpiComm {
    fn rank(&self) -> usize {
        self.world.rank() as usize
    }

    fn size(&self) -> usize {
        self.world.size() as usize
    }

    fn alltoallv_f64(&self, send: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.alltoallv(send)
    }

    fn alltoallv_i64(&self, send: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
        self.alltoallv(send)
    }

    fn exchange_f64(&self, mut send: Vec<Vec<f64>>, recv_counts: &[usize]) -> Vec<Vec<f64>> {
        let me = self.rank();
        assert_eq!(send.len(), self.size());
        assert_eq!(recv_counts.len(), self.size());
        if self.alltoall_halo {
            let recv = self.alltoallv(send);
            assert!(recv.iter().zip(recv_counts).all(|(r, &n)| r.len() == n), "halo counts differ from the setup");
            return recv;
        }
        let tag = self.next_tag();
        let mut recv: Vec<Vec<f64>> = recv_counts.iter().map(|&n| vec![0.0; n]).collect();
        recv[me] = std::mem::take(&mut send[me]);
        // One message per (source, destination) pair and exchange, with this
        // exchange's tag, all completed before returning.
        mpi::request::scope(|scope| {
            let mut receives = Vec::new();
            for (p, buf) in recv.iter_mut().enumerate() {
                if p != me && !buf.is_empty() {
                    receives.push(self.world.process_at_rank(p as i32).immediate_receive_into_with_tag(scope, &mut buf[..], tag));
                }
            }
            let mut sends = Vec::new();
            for (p, buf) in send.iter().enumerate() {
                if p != me && !buf.is_empty() {
                    sends.push(self.world.process_at_rank(p as i32).immediate_send_with_tag(scope, &buf[..], tag));
                }
            }
            let expected: Vec<(usize, usize)> =
                recv_counts.iter().enumerate().filter(|&(p, &n)| p != me && n > 0).map(|(p, &n)| (p, n)).collect();
            receives.into_iter().zip(expected).for_each(|(r, (p, n))| {
                let status = r.wait();
                let got = status.count(f64::equivalent_datatype()) as usize;
                assert!(status.tag() == tag && got == n, "rank {me}: from {p} tag {} ({tag} expected), {got} values ({n} expected)", status.tag());
            });
            sends.into_iter().for_each(|r| {
                r.wait();
            });
        });
        recv
    }

    fn allreduce_sum(&self, values: &mut [f64]) {
        let n = values.len();
        if n == 0 {
            return;
        }
        let root = self.world.process_at_rank(0);
        if self.rank() == 0 {
            let mut all = vec![0.0; n * self.size()];
            root.gather_into_root(&values[..], &mut all[..]);
            // Same additions, in the same order, as the default implementation.
            values.iter_mut().for_each(|v| *v = 0.0);
            for part in all.chunks(n) {
                values.iter_mut().zip(part).for_each(|(v, x)| *v += x);
            }
        } else {
            root.gather_into(&values[..]);
        }
        root.broadcast_into(values);
    }

    fn allreduce_max(&self, local: f64) -> f64 {
        let mut out = f64::NEG_INFINITY;
        self.world.all_reduce_into(&local, &mut out, SystemOperation::max());
        out
    }
}
