//! HEALPix geometry and block-wise kernels of the pyramidal
//! scattering-covariance engine (no MPI, `f64`).
//!
//! Geometry is computed on the fly for each domain (block plus halo),
//! never stored for the full sky.  Every module is a port of an STL-Dev
//! reference (`STL_main/TorchGeometry.py`, `LocalGeometry.py`) and is
//! tested against fixtures dumped from it (`tools/dump_reference.py`).

pub mod conv;
pub mod decimation;
pub mod healpix;
pub mod local;
pub mod neighbours;
pub mod stencil;
pub mod wavelet;
