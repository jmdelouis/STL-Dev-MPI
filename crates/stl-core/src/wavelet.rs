//! The complex directional wavelet on the tangent stencil
//! (`WaveletOperatorHealpixKernel_torch._wavelet_kernel`).  Orientations come
//! from the convolution gauges, so a single kernel serves all of them and
//! every level.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Zero-mean, l1-normalised Gabor kernel `[P]`, tap `p = i * kernel_sz + j`
/// at `(x, y) = (j - (kernel_sz - 1) / 2, i - (kernel_sz - 1) / 2)`.
pub fn wavelet_kernel(kernel_sz: usize, sigma: f64) -> Vec<Complex64> {
    let c = |i: usize| i as f64 - (kernel_sz as f64 - 1.0) / 2.0;
    let mut k = Vec::with_capacity(kernel_sz * kernel_sz);
    for i in 0..kernel_sz {
        for j in 0..kernel_sz {
            let (x, y) = (c(j), c(i));
            let mother = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
            let angle = 0.5 * PI * x;
            k.push(Complex64::new(angle.cos() * mother, angle.sin() * mother));
        }
    }
    let n = k.len() as f64;
    let mean = k.iter().fold(Complex64::new(0.0, 0.0), |s, &v| s + v) / n;
    k.iter_mut().for_each(|v| *v -= mean);
    let l1 = k.iter().fold(0.0, |s, v| s + v.norm());
    k.iter_mut().for_each(|v| *v /= l1);
    k
}

/// Convolution weights `[C_in = 1, C_out = 2, P]` carrying (real, imag),
/// the layout of `_wav_weights`.
pub fn wavelet_weights(kernel_sz: usize) -> Vec<f64> {
    let k = wavelet_kernel(kernel_sz, 1.0);
    k.iter().map(|v| v.re).chain(k.iter().map(|v| v.im)).collect()
}
