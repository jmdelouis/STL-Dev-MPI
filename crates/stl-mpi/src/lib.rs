//! Domain decomposition and communication for the `stl-core` sweeps.
//!
//! The sky is cut into contiguous NESTED ranges at level 0, one per rank,
//! aligned so that every rank owns the parents of its pixels at every
//! coarser level ([`Decomposition`]).  A rank holds, per level, its owned
//! range plus the halo its blocks read or write ([`build_partition`]);
//! [`HaloExchange`] moves owned values out to halo holders and adds halo
//! contributions back into owners (ghost-cell pattern, additive), on top of
//! any [`Comm`] backend: threads in one process ([`threads`]) or MPI
//! (feature `mpi`).

pub mod check;
pub mod comm;
pub mod decomposition;
pub mod halo;
#[cfg(feature = "mpi")]
pub mod mpi_comm;
pub mod threads;

pub use comm::Comm;
pub use decomposition::{build_partition, Decomposition};
pub use halo::HaloExchange;
