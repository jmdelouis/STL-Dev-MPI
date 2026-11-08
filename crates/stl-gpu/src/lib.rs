//! CUDA backend of the pyramidal scattering-covariance engine: the sweeps
//! of `stl_core::engine::LocalEngine` with the fields of every level on one
//! GPU per rank, the convolution geometry computed on the device, and the
//! halo / ghost exchanges of `stl_core::engine::Exchange` between ranks.
//! See `docs/gpu.md`.

#![warn(missing_docs)]

pub mod cublas;
pub mod cufft;
pub mod device;
pub mod dia;
pub mod disc;
pub mod engine;
pub mod geometry;
pub mod harmonic;
pub mod nn;
pub mod planar;
pub mod terms;
pub mod topology;

pub use cudarc::driver::{CudaSlice, PinnedHostSlice};
pub use device::Gpu;
pub use engine::{host_cache_budget, ConvForm, FieldStore, GeometryCache, GpuEngine, GpuEvaluator, GpuOptions, HostForwardState};
pub use harmonic::{GpuEb, GpuSht};
