//! Planar (2D) images: the block-wise pyramidal engine on the kernels of
//! STL-Dev's `STL_2D_Kernel_Torch` with periodic boundaries, the power
//! spectrum of its `CS_operator_2D_Kernel_Torch` (Gaussian rings) and the
//! Fourier tools of the planar objective.  See `docs/planar_2d.md`.

pub mod cascade;
pub mod engine;
pub mod grid;
pub mod kernels;
pub mod spectrum;
pub mod warp;

pub use engine::{PlanarEngine, Pyramid};
