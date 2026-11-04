# HEALPix synthesis of one map, on CPU and GPU

Synthesis of a map whose scattering-covariance statistics match those of a target: the
weak-lensing convergence κ of Takahashi et al. (2017), realisation r000 at source redshift zs1,
block-averaged to nside 64 (`fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy`, NESTED,
shipped with the repository), with `stl-synth`, in two ways:

1. **directly**, from white noise, on the CPU and on a GPU (the two follow the same trajectory);
2. with **the recipe for skewed fields** ([USAGE](../../docs/USAGE.md), section 4): Gaussian
   anamorphosis of the target, synthesis at `2 nside`, back to `nside`, histogram matching.

```bash
cargo build --release -p stl-cli --features gpu      # without a GPU: no feature, and GPU=0
bash examples/01_healpix_synthesis/run.sh            # NITER=100 RANKS=8 GPU=1 by default
```

The commands it runs:

```bash
stl-synth --target takahashi_r000_zs1_nside64_nested.npy --L 4 --weights ergodic --niter 100 --seed 1 --ranks 8 --out out/cpu
stl-synth --target takahashi_r000_zs1_nside64_nested.npy --L 4 --weights ergodic --niter 100 --seed 1 --gpu --out out/gpu
# the recipe
python tools/gaussianize.py forward takahashi_r000_zs1_nside64_nested.npy out/target_gauss.npy out/table.npz
python tools/resample_nested.py up out/target_gauss.npy out/target_gauss_up.npy --keep-mean     # nside 128
stl-synth --target out/target_gauss_up.npy --L 4 --weights ergodic --niter 100 --seed 1 --gpu --out out/recipe
python tools/resample_nested.py down out/recipe_synthesis_nested.npy out/recipe_gauss.npy       # nside 64
python tools/gaussianize.py match out/table.npz out/recipe_gauss.npy out/recipe_nested.npy
```

- `--L 4`: four orientations (the default; two leave anisotropic textures);
- `J` defaults to `log2(nside) − 1 = 5` scales, the kernel to 5 × 5: 2522 coefficients;
- `--weights ergodic`: each squared residual is divided by the target's spatial variance of that
  coefficient, so the loss is a reduced χ² and every family is fitted;
- `--ranks 8`: eight ranks as threads of one process (with the `mpi` feature, use `mpirun -n 8`
  instead: [example 05](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/05_mpi));
- `--gpu`: one CUDA device (with `GPU=0` the recipe runs on the CPU thread ranks);
- `--keep-mean`: the interpolation to nside 128 shifts every group of 4 children so that its mean
  is the parent's value, so the target is recovered exactly when cut back.

## Results

![maps](maps.png)

Top: Mollweide views; bottom: one base face (64 × 64 pixels). All panels share one
histogram-equalised colour scale, set by the target (healpy's `norm='hist'`). At nside 64 (pixels of
0.9°) the Takahashi κ map is strongly non-Gaussian (pixel skewness 8.4): filaments and halos
around voids.

- **Direct synthesis.** It reproduces the filamentary network, and its statistics match the
  target's (reduced χ² 0.14, [example 03](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/03_statistics)), but it keeps pixel-scale noise and only half
  the skewness (4.1). A direct synthesis of so skewed a field does not settle the smallest scales.
- **Recipe.** The map is clean, with the target's exact pixel histogram (skewness 8.38), and its
  textures resemble the target's down to the pixel.

Power spectrum of the synthesis over that of the target (healpy `anafast`, mean over each band):

| Run | ℓ 2–29 | ℓ 30–99 | ℓ 100–191 | Skewness (target 8.38) |
|---|---|---|---|---|
| direct | 1.17 | 1.00 | 1.05 | 4.06 |
| recipe | 1.31 | 1.12 | 0.89 | 8.38 |

The direct run's spectrum includes its pixel noise. The recipe, without `--keep-mean`, gave a
smoother map (0.61 at ℓ 100–191, 1.70 at ℓ 2–29), because the plain interpolation removes most of
the target's finest scale.

![loss](loss.png)

| Run | Evaluations | s / evaluation | Optimisation | Wall clock (setup included) | Loss | Peak memory |
|---|---|---|---|---|---|---|
| CPU, 8 thread ranks | 112 | 3.58 | 401 s | 404 s | 259.2 → 0.1429 | 0.42 GiB host |
| GPU (Tesla T4) | 112 | 0.043 | 4.8 s | 22 s | 259.2 → 0.1429 | 0.31 GiB host, 0.51 GiB device |
| recipe, nside 128, GPU (T4) | 106 | 0.142 | 15 s | 41 s | 803.8 → 0.691 | 0.31 GiB host, 1.32 GiB device |

garoupe: 2 × Xeon Bronze 3204 (12 cores at 1.9 GHz), Tesla T4 (whose FP64 is 1/32 of its FP32).
The two direct runs follow the same trajectory: the final losses agree to 2.3e-8 relative (the GPU and CPU gradients differ by about 1e-12, which
L-BFGS amplifies slowly; see the [GPU report](../../docs/gpu.md), section 5).

## Outputs (`out/`)

| File | Content |
|---|---|
| `cpu.json`, `gpu.json`, `recipe.json` | run report: `loss_history`, `seconds_per_evaluation`, `timers_seconds`, `memory_peak_bytes_per_rank`, `final_residual_in_ergodic_sigma` per family, ... |
| `*_synthesis_nested.npy` | the synthesis in the target's units |
| `*_synthesis_standardized.npy` | the same, standardised |
| `*_synthesis.fits`, `*_target.fits` | HEALPix FITS binary tables |
| `target_gauss*.npy`, `table.npz` | the Gaussianised target (nside 64 and 128) and its anamorphosis table |
| `recipe_nested.npy` | the recipe's synthesis at nside 64, with the target's histogram |

## Going further

- Large nside: coarse-to-fine starts (`--init` from a lower-nside synthesis), MPI ranks or several
  GPUs ([MPI guide](../../docs/user-guide/mpi.md), [GPU guide](../../docs/user-guide/gpu.md)).
- Statistics of the result: [example 03](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/03_statistics).
