# GPU guide

`--gpu` runs the sweeps on CUDA devices, **one device per rank**, with the same decomposition,
exchanges, objective and L-BFGS as the CPU engine. Everything stays `f64`; the kernels are
compiled at start-up by NVRTC with `--fmad=false`, without floating-point atomics, so the results
are deterministic and match the CPU engine to 1e-12 (statistics to 6e-16). The design and every
measurement are in the [GPU engine report](../gpu.md).

## Build and run

```bash
export CUDARC_CUDA_VERSION=12040                    # the driver's CUDA version (nvidia-smi)
cargo build --release -p stl-cli --features gpu     # add mpi for several devices
stl-synth --target map.npy --gpu --niter 100 --out run
mpirun -n 3 stl-synth --target map.npy --gpu --niter 100 --out run     # 3 devices
stl-synth2d --target images.npy --gpu --out run2d
```

No CUDA toolkit is needed to build. At run time the process needs `libcuda.so` (driver) and
`libnvrtc.so`; when the system NVRTC is older than the driver, the PyPI wheel
`nvidia-cuda-nvrtc-cu11` / `-cu12` works (put its `lib` directory in `LD_LIBRARY_PATH`;
`tools/garoupe/setup.sh` does it). `stl-ensemble --harmonic gpu` also needs cuFFT
(`libcufft.so` or `STL_CUFFT`). In WSL, add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH`, otherwise
the GPU tests skip silently with `CUDA_ERROR_NO_DEVICE`.

The device of a rank is its local rank modulo the visible devices (`CUDA_VISIBLE_DEVICES`), or
`STL_GPU_DEVICE`.

## Choosing the options

| Situation | Options |
|---|---|
| default | `--gpu`: fields `auto` (kept when they take at most half the free memory), extra terms on the device, geometry cache `auto`, DIA convolution |
| fastest, when it fits | `--gpu-fields keep --geometry-cache device` |
| least device memory | `--gpu-fields recompute --geometry-cache off --gpu-block-rows 1024 --block-pixels 1024` |
| geometry does not fit on the device, much host memory | `--geometry-cache auto` (or `host`): weights in page-locked host memory, uploaded instead of recomputed when that is faster |
| several members per device | `--ensemble B --groups G`: each group of ranks holds `B / G` members as one batch |
| debugging a kernel failure | `STL_GPU_SYNC=1` |

**Fields.** `keep` holds `z_n` and `|z_n|` of every level from the forward to the reverse sweep
(188 bytes per pixel and member at `L = 4`: 2.4 GB at nside 1024, 9.5 GB at 2048); `recompute`
rebuilds them per block in the reverse sweep (60 bytes per pixel and member). On a T4 at nside
1024 both take the same time (22.6 and 23.1 s per evaluation, one member); on an A100 at nside
128, `keep` with a full cache is 8 times faster than the minimum-memory configuration.

**Geometry cache.** What is not taken by the fields and the decimation rows is used to keep the
convolution geometry of the blocks between evaluations (about 3.3 kB per row with DIA at
`L = 4`). Recomputing what does not fit is the main cost at nside 2048. The policies give
bit-identical results ([report](../geometry_cache_preconditioner.md)).

**Extra terms.** With `--gpu-terms device` (default), the ensemble mean and scatter, level power,
marginal, kappa transform and kurtosis and the gradient sums run on the device, and the maps go
to the device once per evaluation. Minkowski terms and `--freeze-nside` fall back to `host`.

## Measured

| Case | Device | s / evaluation | Device memory |
|---|---|---|---|
| nside 128 × 4 members, `L = 4`, full plan (vs STL-Dev legacy 1.23 s, 19.8 GB) | A100 | 0.106 (`keep`) / 0.885 (`recompute`, small blocks) | 2.2 GB / 0.58 GB |
| nside 256, `L = 4` | 1 / 3 T4 | 1.26 / 0.34 | 9.1 / 4.0 GB |
| nside 1024, `L = 4` | 1 / 3 T4 | 66.8 / 13.3 | 12.3 / 10.9 GB |
| nside 2048, `L = 4` | 3 T4 | 114 | 12.9 GB |
| 10 images 256², `stl-synth2d` (vs STL-Dev 0.64 s, 2.8 GB on the same T4) | T4 | 0.18 | |
| one image 4096², `stl-synth2d` | T4 / A100 | 4.9 / 0.6 | 1.9 GB / |

Sources: [bench_n128](../bench_n128.md), [gpu](../gpu.md) section 5,
[planar_2d](../planar_2d.md) sections 5b and 5c. The T4 has 1/32 of its FP32 throughput in FP64;
data-centre GPUs (A100, H100) are 40 to 130 times faster in FP64.

## Large productions

- nside 1024 × 10 members with P. Campeti's method on one A100:
  [step by step](../campeti_production_1024.md);
- cost model and projections to nside 2048 × 10:
  [harmonic speed](../harmonic_speed.md), [geometry cache](../geometry_cache_preconditioner.md)
  section 4;
- the DIA convolution: [design and measurements](../dia_convolution.md).
