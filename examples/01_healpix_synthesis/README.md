# HEALPix synthesis of one map, on CPU and GPU

Synthesis of a map whose scattering-covariance statistics match those of a target: the WebSky
convergence map κ at nside 64 (`fixtures/synthesis/synthesis_kap_nside64_nested_niter20/target_map.npy`,
NESTED, shipped with the repository), starting from white noise, with `stl-synth`.

```bash
cargo build --release -p stl-cli --features gpu      # without a GPU: no feature, and GPU=0
bash examples/01_healpix_synthesis/run.sh            # NITER=100 RANKS=8 GPU=1 by default
```

The commands it runs:

```bash
stl-synth --target target_map.npy --L 4 --weights ergodic --niter 100 --seed 1 --ranks 8 --out out/cpu
stl-synth --target target_map.npy --L 4 --weights ergodic --niter 100 --seed 1 --gpu --out out/gpu
```

- `--L 4`: four orientations (the default; two leave anisotropic textures);
- `J` defaults to `log2(nside) − 1 = 5` scales, the kernel to 5 × 5: 2522 coefficients;
- `--weights ergodic`: each squared residual is divided by the target's spatial variance of that
  coefficient, so the loss is a reduced χ² and every family is fitted;
- `--ranks 8`: eight ranks as threads of one process (with the `mpi` feature, use `mpirun -n 8`
  instead: [example 05](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/05_mpi));
- `--gpu`: one CUDA device.

## Results

![maps](maps.png)

Top: Mollweide views; bottom: one base face (64 × 64 pixels). At nside 64 (pixels of 0.9°) the
WebSky κ map is dominated by small-scale fluctuations and looks almost Gaussian; the strongly
non-Gaussian case is [example 02](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/02_healpix_ensemble).

![loss](loss.png)

| Run | Evaluations | s / evaluation | Optimisation | Wall clock (setup included) | Loss | Peak memory |
|---|---|---|---|---|---|---|
| CPU, 8 thread ranks | 107 | 3.95 | 422 s | 427 s | 132.1 → 0.2403 | 0.42 GiB host |
| GPU (Tesla T4) | 107 | 0.061 | 6.5 s | 24 s | 132.1 → 0.2403 | 0.31 GiB host, 0.99 GiB device |

garoupe: 2 × Xeon Bronze 3204 (12 cores at 1.9 GHz) shared with other jobs during the run (load
about 16), Tesla T4 (whose FP64 is 1/32 of its FP32). The two runs follow the same trajectory: the
final losses agree to 6e-7 relative (the GPU and CPU gradients differ by about 1e-12, which
L-BFGS amplifies slowly; see the [GPU report](../../docs/gpu.md), section 5).

## Outputs (`out/`)

| File | Content |
|---|---|
| `cpu.json`, `gpu.json` | run report: `loss_history`, `seconds_per_evaluation`, `timers_seconds`, `memory_peak_bytes_per_rank`, `final_residual_in_ergodic_sigma` per family, ... |
| `*_synthesis_nested.npy` | the synthesis in the target's units |
| `*_synthesis_standardized.npy` | the same, standardised |
| `*_synthesis.fits`, `*_target.fits` | HEALPix FITS binary tables |

## Going further

- Non-Gaussian targets (weak lensing at low redshift, density): Gaussian anamorphosis, synthesis
  at `2N`, histogram matching ([USAGE](../../docs/USAGE.md), section 4).
- Large nside: coarse-to-fine starts (`--init` from a lower-nside synthesis), MPI ranks or several
  GPUs ([MPI guide](../../docs/user-guide/mpi.md), [GPU guide](../../docs/user-guide/gpu.md)).
- Statistics of the result: [example 03](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/03_statistics).
