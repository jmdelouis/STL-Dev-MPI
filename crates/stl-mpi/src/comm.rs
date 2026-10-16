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

    /// `value` of every rank, in rank order.
    fn allgather_i64(&self, value: i64) -> Vec<i64> {
        self.alltoallv_i64(vec![vec![value]; self.size()]).into_iter().map(|v| v[0]).collect()
    }
}
