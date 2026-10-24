//! The communication primitive every backend provides.

/// A group of ranks exchanging variable-size messages.  Every rank calls
/// the same sequence of collective operations.
pub trait Comm {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// All-to-all with variable counts: `send[p]` goes to rank `p`; returns
    /// `recv[p]`, the message from rank `p` (the own slot is passed through).
    fn alltoallv_f64(&self, send: Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn alltoallv_i64(&self, send: Vec<Vec<i64>>) -> Vec<Vec<i64>>;

    /// Sparse exchange with counts known on both sides: `send[p]` goes to
    /// rank `p`, `recv_counts[p]` values come from rank `p` (the own slot
    /// is passed through).  Empty messages are not sent: the halo pattern
    /// touches a few neighbours per rank, so the cost stays independent of
    /// the number of ranks (an all-to-all costs O(P) per rank).
    fn exchange_f64(&self, send: Vec<Vec<f64>>, recv_counts: &[usize]) -> Vec<Vec<f64>> {
        let recv = self.alltoallv_f64(send);
        debug_assert!(recv.iter().zip(recv_counts).all(|(r, &n)| r.len() == n));
        recv
    }

    /// Element-wise sum over the ranks, added in rank order on every rank so
    /// that the result is identical everywhere (and independent of the
    /// backend's reduction tree).
    fn allreduce_sum(&self, values: &mut [f64]) {
        let all = self.alltoallv_f64(vec![values.to_vec(); self.size()]);
        values.iter_mut().for_each(|v| *v = 0.0);
        for part in &all {
            values.iter_mut().zip(part).for_each(|(v, x)| *v += x);
        }
    }

    /// Maximum over the ranks (order-independent, so any reduction tree).
    fn allreduce_max(&self, local: f64) -> f64 {
        let all = self.alltoallv_f64(vec![vec![local]; self.size()]);
        all.iter().fold(f64::NEG_INFINITY, |m, v| m.max(v[0]))
    }

    /// `value` of every rank, in rank order.
    fn allgather_i64(&self, value: i64) -> Vec<i64> {
        self.alltoallv_i64(vec![vec![value]; self.size()]).into_iter().map(|v| v[0]).collect()
    }

    /// Sub-communicator of the ranks passing the same `color`, ranked by
    /// `key` (collective: every rank calls it).  None when the backend cannot
    /// split.
    fn split(&self, _color: usize, _key: usize) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
}
