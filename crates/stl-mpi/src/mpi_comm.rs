//! MPI backend (rsmpi): `MPI_Alltoall` of the counts then `MPI_Alltoallv`.

use crate::comm::Comm;
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
}
