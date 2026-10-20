//! MPI backend (rsmpi).  General exchanges: `MPI_Alltoall` of the counts
//! then `MPI_Alltoallv` (setup only).  In the sweeps: halo and ghost
//! exchanges by nonblocking point-to-point messages to the neighbours only,
//! sums by `MPI_Gather` to rank 0 (added there in rank order, as the
//! threads backend does) and `MPI_Bcast`, maxima by `MPI_Allreduce`.  No
//! per-evaluation operation costs O(P) messages per rank.

use crate::comm::Comm;
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use mpi::Count;

pub struct MpiComm {
    world: SimpleCommunicator,
}

impl MpiComm {
    pub fn new(world: SimpleCommunicator) -> Self {
        MpiComm { world }
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
        let mut recv: Vec<Vec<f64>> = recv_counts.iter().map(|&n| vec![0.0; n]).collect();
        recv[me] = std::mem::take(&mut send[me]);
        // One message per (source, destination) pair and exchange, all
        // completed before returning: MPI's non-overtaking order matches
        // every receive with this exchange's message.
        mpi::request::scope(|scope| {
            let mut receives = Vec::new();
            for (p, buf) in recv.iter_mut().enumerate() {
                if p != me && !buf.is_empty() {
                    receives.push(self.world.process_at_rank(p as i32).immediate_receive_into(scope, &mut buf[..]));
                }
            }
            let mut sends = Vec::new();
            for (p, buf) in send.iter().enumerate() {
                if p != me && !buf.is_empty() {
                    sends.push(self.world.process_at_rank(p as i32).immediate_send(scope, &buf[..]));
                }
            }
            receives.into_iter().for_each(|r| {
                r.wait();
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
