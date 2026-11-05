# `stl-synth`: HEALPix synthesis

`stl-synth` synthesises a full-sky HEALPix map, or an ensemble of maps, whose scattering-covariance
statistics match those of a target map. It reads the target, standardises it (mean 0, std 1),
computes its statistics, and minimises

$$
\mathcal{L}(x) = \sum_i w_i \, |S_i(x) - S_i(\text{target})|^2 \;+\; \text{optional terms}
$$

over the pixels of `x` with L-BFGS (strong Wolfe, the line search of `torch.optim.LBFGS`), from
white noise or from an initial map. The statistics and their gradient are computed block by block
over the pyramid ({doc}`concepts`), on CPU ranks or one GPU per rank.

```bash
stl-synth --target kappa_nside128_nested.npy --weights ergodic --niter 200 --gpu --out run
```

```
run.json                        report: loss_history, seconds_per_evaluation, timers, memory, residuals
run_synthesis_nested.npy        the synthesis in the target's units (NESTED)
run_synthesis_standardized.npy  the same, standardised
run_synthesis.fits, run_target.fits   HEALPix FITS binary tables
```

---

## Key parameters

| Option | Default | Description |
|---|---|---|
| `--target MAP` | required | full-sky NESTED map, `.npy` (1-D) or HEALPix FITS |
| `--L N` | 4 | orientations; always 4 (`L = 2` gives anisotropic, blobby textures) |
| `--J N` | `log2(nside) − 1` | scales, one per level of the pyramid |
| `--weights W` | `none` | `ergodic`: each squared residual over the target's spatial variance of that coefficient (reduced χ²); fits every family |
| `--niter N` | 50 | L-BFGS iterations (at most `5N/4` evaluations) |
| `--init MAP`, `--init-noise X` | white noise, `auto` | coarse-to-fine start from a lower-nside synthesis, plus white noise of the target's missing small-scale RMS (`auto`) or details drawn from the target's, conditionally on the parent's value and local variation (`conditional`) |
| `--ensemble B`, `--groups G` | 1, 1 | `B` members whose mean statistics match the target, split over `G` groups of ranks |
| `--scatter-weight X` | 0 | holds the scatter of the members at the ergodic variance (`X = 1`: an ergodic ensemble) |
| `--block-pixels N` | 16384 | cells per block and level: the memory knob of the CPU engine |
| `--ranks N` / `mpirun -n N` | 1 | thread ranks / MPI ranks ({doc}`mpi`) |
| `--gpu` and `--gpu-*` | off | one CUDA device per rank ({doc}`gpu`) |
| `--out PREFIX` | `synthesis_rust_nside<N>` | output prefix |

Extra terms of the loss, all off by default:

| Option | Holds |
|---|---|
| `--level-power-weight X` | the variance of every level of the 4-children pyramid |
| `--marginal-weight X` | the pixel histogram (squared 1-D Wasserstein distance) |
| `--marginal-tail-weight X`, `--marginal-tail K` | the `K` smallest and largest pixel values (the tails of the histogram) |
| `--kappa-table KNOTS`, `--kappa-weight`, `--kappa-families` | the statistics of `κ = f(g)` when the variable is a Gaussianised map `g` |
| `--kurtosis-weight X` | the kurtosis of the details of every level |
| `--minkowski-weight X` | the Minkowski functionals V0, V1, V2 of every level |
| `--family-weights F` | factors per family, e.g. `S1=10,S2=10` |

Every option, with its exact text: {doc}`../api/cli/stl-synth`.

---

## Minimal example

The Takahashi weak-lensing map at nside 64 is shipped in the repository:

```bash
T=fixtures/takahashi/takahashi_r000_zs1_nside64_nested.npy
stl-synth --target $T --weights ergodic --niter 100 --ranks 8 --out cpu      # 3.6 s / evaluation on 12 cores
stl-synth --target $T --weights ergodic --niter 100 --gpu --out gpu          # 0.043 s / evaluation on a T4
```

Both runs reach the same loss (259 → 0.143 after 112 evaluations). The complete example, with
figures: {doc}`../tutorials/healpix_synthesis`; on real lensing data:
{doc}`../tutorials/takahashi_synthesis`.

---

## Non-Gaussian targets

For skewed fields such as weak-lensing convergence at low redshift, synthesise a Gaussianised
version of the target and transform back:

1. `tools/gaussianize.py forward T.npy T_gauss.npy table.npz` (rank to normal, as FOSCAT's
   `to_gaussian`);
2. synthesise `T_gauss` (optionally at `2N`, then cut back to `N`: the finest level is the least
   well synthesised);
3. `tools/gaussianize.py match table.npz S_gauss.npy S.npy` restores the target's histogram.

Alternatively, keep the Gaussian map as the variable and add the statistics of the physical map
with `--kappa-table` (and `--level-power-weight`, `--marginal-weight`), as in the ensemble recipes
of {doc}`../gpu` section 5.1. Details: {doc}`../USAGE`, section 4.

---

## Pitfalls and costs

- **Masks and NaNs** are not supported: every pixel must be finite. Masked statistics are in the
  Python engines ({doc}`native-engine`).
- **Sorted NESTED input.** The map must be NESTED; `tools/prepare_healpix_target.py` converts RING
  maps and degrades them.
- **Large nside**: do not start from white noise; chain `N/8 → N/4 → N/2 → N` with `--init`
  (a few hundred iterations at the coarsest, tens at the finest). `--init-noise conditional`
  starts each level closer to the solution than white details and avoids their pixel-scale
  checkerboard ({doc}`../lagrangian_synthesis`, section 6.6).
- **Rank counts with `--init`**: every rank's range must fall on whole pixels of the init map
  ("rank ranges are not aligned"): use fewer ranks or a closer init nside.
- **Cost per evaluation** (forward + gradient, `L = 4`): nside 64, 0.043 s on a T4, 3.6 s on
  12 cores; nside 128 × 4 members, 0.106 s on an A100; nside 1024, 13 s on three T4s. Memory per
  rank scales with its share of the sky: maps, stored moduli and L-BFGS history shrink as `1/P`,
  plus the transients of one block (80 to 110 MB at blocks of 16384).
- **Trajectories** of runs with different rank counts or devices agree to about 1e-13 at first;
  from a white-noise start L-BFGS can amplify differences of 1e-12 to visible ones after tens of
  iterations, to the same final loss.
