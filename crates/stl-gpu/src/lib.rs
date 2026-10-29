//! CUDA backend of the pyramidal scattering-covariance engine: the sweeps
//! of `stl_core::engine::LocalEngine` with the fields of every level on one
//! GPU per rank, the convolution geometry computed on the device, and the
//! halo / ghost exchanges of `stl_core::engine::Exchange` between ranks.
//! See `docs/gpu.md`.

pub mod device;
pub mod dia;
pub mod engine;
pub mod geometry;
pub mod topology;

pub use device::Gpu;
pub use engine::{GeometryCache, GpuEngine, GpuEvaluator, GpuOptions};
