# Command-line tools

All tools are in `crates/stl-cli` (and `stl-mpi-check` in `crates/stl-mpi`). Each prints its
complete option list with `--help`; the tables below group the options and are taken from the
same text in the code (`stl_cli::USAGE`, `STATS_USAGE`, `ensemble::cli::USAGE`, and the
binaries' own usage strings).

| Tool | Does | Input | Main outputs |
|---|---|---|---|
| [`stl-synth`](#stl-synth) | synthesis of one HEALPix map or an ensemble from a target map | NESTED map | `PREFIX.json`, `PREFIX_synthesis_nested.npy`, FITS |
| [`stl-stats`](#stl-stats) | statistics of HEALPix maps | NESTED maps | `STATS.json` (plan + values) |
| [`stl-ensemble`](#stl-ensemble) | ensemble synthesis with P. Campeti's method | reference maps | `PREFIX.json`, per-member maps |
| [`stl-synth2d`](#stl-synth2d) | synthesis of periodic 2D images | `.npy` image stack | `PREFIX_maps.npy`, features, `PREFIX.json` |
| [`stl-minkowski`](#stl-minkowski) | Minkowski functionals per pyramid level | NESTED maps | text, JSON |
| [`stl-mpi-check`](#stl-mpi-check) | distributed statistics and gradients vs an STL-Dev fixture | fixture directory | text, exit status |

Common rules:

- maps are **full-sky NESTED** HEALPix maps, `.npy` (1-D, `float32` or `float64`) or HEALPix
  FITS binary tables; `nside` a power of 2, every pixel finite (no masks yet);
- without the `mpi` feature, `--ranks N` runs `N` ranks as threads of one process; with it, the
  ranks are those of `mpirun`;
- with the `gpu` feature, `--gpu` gives every rank one CUDA device (local rank modulo the visible
  devices, or `STL_GPU_DEVICE`);
- the defaults are those of the validated runs: `L = 4`, `kernel_size = 5`,
  `J = log2(nside) − 1`, blocks of 16384 cells, L-BFGS history 10.

(stl-synth)=
## `stl-synth`

```
stl-synth --target MAP [options]
mpirun -n P stl-synth --target MAP [options]          # feature mpi
```

Reads the target, standardises it (mean 0, std 1), computes its statistics, and minimises the
loss from white noise or from `--init`. One line per evaluation on stderr unless `--quiet`; a
short JSON summary on stdout at the end.

**Statistics**

| Option | Default | Meaning |
|---|---|---|
| `--J N` | `log2(nside) − 1` | number of scales |
| `--L N` | 4 | orientations (`L = 2` leaves anisotropic, blobby syntheses) |
| `--kernel-size N` | 5 | wavelet stencil |
| `--fewer` | off | `has_fewer_convolutions` definitions of S3/S4 |

**Start**

| Option | Default | Meaning |
|---|---|---|
| `--init MAP` | white noise | initial map, `nside ≤` target's (coarse-to-fine); `{m}` in the path gives one per member |
| `--init-units U` | `target` | `target` (standardised like the target) or `standardized` |
| `--init-noise X` | `auto` | noise added to `--init`: `auto` is the RMS of the target's scales missing at the init nside |
| `--seed N` | 11 | white-noise seed (member `m` uses `seed + m`) |
| `--freeze-nside M` | off | hold the means over the pixels of nside `M` at those of the init: only finer scales move |

**Loss**

| Option | Default | Meaning |
|---|---|---|
| `--weights W` | `none` | `none`: `Σ |S − S*|²`; `ergodic`: reduced χ², each residual over the target's spatial variance |
| `--family-weights F` | 1 | factors per family, e.g. `S1=10,S2=10` |
| `--ensemble B` | 1 | synthesise `B` members whose **mean** statistics match the target |
| `--groups G` | 1 | split the ranks into `G` groups, each holding `B / G` members |
| `--scatter-weight X` | 0 | hold the scatter of the members at the ergodic variance (`X = 1` gives an ergodic ensemble) |
| `--level-power-weight X` | 0 | hold the variance of every level of the 4-children pyramid |
| `--marginal-weight X` | 0 | squared 1-D Wasserstein distance to the target's pixel histogram |
| `--kappa-table KNOTS` | off | add the statistics of `kappa = f(g)`, `f` the monotone interpolation of the knots (`tools/gaussianize.py`) |
| `--kappa-weight X` | 0.1 | factor of that term |
| `--kappa-families F` | `mean,var,S1,S2` | its families |
| `--kappa-s3-weight X`, `--kappa-s3-nside A-B` | 0, all | S3 of kappa as a separate reduced χ², restricted to levels `A..B` |
| `--kurtosis-weight X`, `--kurtosis-nside A-B` | 0 | kurtosis of the details of every level of kappa |
| `--minkowski-weight X`, `--minkowski-nside A-B` | 0, 64 to nside/2 | soft Minkowski functionals V0, V1, V2 per level |
| `--minkowski-standardise`, `--minkowski-sigma F` | off | standardise every level first; their scatter (`stl-minkowski --sigma-out`) |

**Optimiser**

| Option | Default | Meaning |
|---|---|---|
| `--niter N` | 50 | L-BFGS iterations (at most `5N/4` evaluations) |
| `--history-size N` | 10 | L-BFGS history |
| `--precondition P` | `none` | initial inverse Hessian over the Haar bands: `bands:G`, `power` or `auto` ([report](../geometry_cache_preconditioner.md)) |

**Engine and memory**

| Option | Default | Meaning |
|---|---|---|
| `--block-pixels N` | 16384 | cells per block and level (smaller: less memory, more time) |
| `--conv-chunk N` | whole block | rebuild the convolution geometry by sub-blocks of `N` rows |
| `--overlap N` | off | levels `< N` with redundant halos (no exchange during the sweeps), for a few ranks |
| `--ranks N` | 1 | thread ranks (builds without MPI) |

**GPU** (feature `gpu`; see the [GPU guide](gpu.md))

| Option | Default | Meaning |
|---|---|---|
| `--gpu` | off | one CUDA device per rank |
| `--gpu-fields F` | `auto` | wavelet fields `keep` (fastest), `recompute` (a third of the memory) or `auto` |
| `--gpu-terms T` | `device` | the objective's extra terms on the `device` or the `host` |
| `--geometry-cache P` | from `--gpu-cache` | convolution geometry between evaluations: `device`, `host` (page-locked host memory), `auto`, `off`; bit-identical results |
| `--gpu-cache C` | `auto` | `full`, `transposition`, `none`, `auto` |
| `--gpu-conv F` | `dia` | geometry form: masked diagonals (`dia`) or bound stencils (`stencils`) |
| `--gpu-block-rows N` | from the device memory | owned cells per device block |

**Output**

`--out PREFIX` (default `synthesis_rust_nside<N>`) writes:

| File | Content |
|---|---|
| `PREFIX.json` | run report: `loss_history`, `evaluations`, `seconds_per_evaluation`, `timers_seconds`, `memory_peak_bytes_per_rank`, final residuals per family in ergodic σ, GPU details |
| `PREFIX_synthesis_nested.npy` | the synthesis in the target's units (NESTED) |
| `PREFIX_synthesis.fits`, `PREFIX_target.fits` | HEALPix FITS binary tables |
| `PREFIX_mMMM_synthesis_nested.npy`, `_standardized.npy` | per member with `--ensemble B > 1` |

The field names of the report are those of STL-Dev's `benchmark_synthesis.py`, so that both
can be compared by the same scripts.

Example: [HEALPix synthesis on CPU and GPU](../tutorials/healpix_synthesis.md). The recipe for
non-Gaussian targets (Gaussian anamorphosis, synthesis at `2N`, histogram matching) and batch
scripts are in [USAGE](../USAGE.md), sections 4 to 6.

(stl-stats)=
## `stl-stats`

```
stl-stats --out STATS.json [--norm MAP] [engine options] -- MAP [MAP ...]
```

Computes the statistics of every map with the plan and the engine of `stl-synth`.

| Option | Meaning |
|---|---|
| `--out FILE` | JSON: `plan` (family, `j1`, `j2`, `j3`, `l1`, `l2`, `l3`, `native_dg` per coefficient), and per map `file`, `mean`, `std`, `seconds`, `re`, `im` |
| `--norm MAP` | standardise every map with the mean and std of `MAP` (e.g. the target), so that all are in the same units; default: each map with its own moments |
| engine options | `--J`, `--L`, `--kernel-size`, `--fewer`, `--block-pixels`, `--conv-chunk`, `--ranks`, `--gpu`, `--gpu-block-rows`, `--gpu-cache`, `--geometry-cache`, `--gpu-conv`, `--gpu-fields`, `--quiet` |

The maps come after `--`, all at the same nside. Example:
[statistics of maps](../tutorials/statistics.md).

(stl-ensemble)=
## `stl-ensemble`

```
stl-ensemble calibrate --references MAP... --nside N --out CAL.json [options]
stl-ensemble run --calibration CAL.json --niter N --out PREFIX [options]
```

P. Campeti's ensemble method (STL_ensemble v2, variance ratio): `B` members whose Gaussian maps
`g` are the variable, `kappa = f(g) − mean` through the LUT pooled over the references, and

$$
\mathcal{L} = (L_\text{GSC} + L_\text{PSC} + L_\text{PS} + L_\text{PDF} + L_\text{MF}) / 5
$$

on the ScatCov statistics of `g` and of `kappa` (full plan, `L = 4`, kernel 5), the 16 log C_ℓ
bands, the pixel PDF and the harmonic Minkowski functionals of `kappa`; each block on the mean over
the members (against a bootstrap draw of the references) and on their variance (against that of
the references). Details in [the merge report](../campeti_merge.md); the nside 1024 production in
[step by step](../campeti_production_1024.md).

**calibrate**

| Option | Default | Meaning |
|---|---|---|
| `--references MAP...` | | reference maps (NESTED, nside ≥ N), up to the next option |
| `--nside N` | | nside of the synthesis (maps block-averaged) |
| `--ensemble B` | 10 | members |
| `--bootstrap-ids I,J,..` | all / a draw | references whose mean is the target of the means |
| `--seed N` | 20260914 | seed of the bootstrap draw |
| `--knots N` | 8192 | LUT knots |

**run**

| Option | Default | Meaning |
|---|---|---|
| `--calibration F` | | calibration JSON |
| `--niter N`, `--max-eval N` | , `5N/4` | L-BFGS iterations and evaluations |
| `--ensemble B`, `--groups G` | 10, 1 | members; MPI groups sharing `B / G` members |
| `--init MAP` | Gaussian random fields | Gaussian maps of the members (`{m}`: one per member) |
| `--seed N` | 61022026 | seed of the initial fields |
| `--ps-variance-upper X` | 383 at nside 128 | upper band centre of the C_ℓ variance term (required at other nsides; production: `3 nside − 1`) |
| `--history-size N` | 10 | L-BFGS history |
| `--checkpoint N` | off | every `N` evaluations, the maps of the lowest loss so far (restart with `--init`) |
| `--harmonic cpu\|gpu` | `gpu` with `--gpu` | where the C_ℓ, PDF and Minkowski terms run (needs cuFFT) |
| `--out PREFIX` | | `PREFIX.json`, `PREFIX_mMMM_gaussian_nested.npy`, `PREFIX_mMMM_physical_nested.npy` |

Both subcommands accept `--block-pixels`, `--ranks`, `--gpu`, `--gpu-conv`, `--gpu-fields`,
`--gpu-cache`, `--geometry-cache`, `--gpu-block-rows`, `--quiet`. Example:
[ensemble synthesis](../tutorials/healpix_ensemble.md).

(stl-synth2d)=
## `stl-synth2d`

```
stl-synth2d --target IMAGES.npy [options]
```

Synthesis of periodic planar images (L-BFGS on the images), with the 2D engine
(`stl_core::planar`); threads from `STL_THREADS` (default: all cores).

| Option | Default | Meaning |
|---|---|---|
| `--target PATH` | | target images, `.npy` `[Bt, H, W]` or `[Bt, C, H, W]` |
| `--ntarget N`, `--crop S` | all, none | first `N` target images; top-left `S × S` |
| `--nbatch B` | 10 | images synthesised |
| `--mode M` | `ensemble` | `ensemble`: ScatCov + power spectrum + Minkowski, batch mean and variance against the targets'; `stldev`: the loss of STL-Dev's `synthesize_from_maps` |
| `--J N` | `log2(min(H, W)) − 2` | scales |
| `--tile T` | 64 | tile side of the block-wise engine |
| `--fewer` | off | `has_fewer_convolutions` |
| `--bins N` | `2^(log2(min) − 4)` | power-spectrum bins (0: none) |
| `--w-sc X --w-ps X --w-mf X` | 1 | ensemble block weights |
| `--niter N`, `--history N` | 100, 50 | L-BFGS |
| `--warm N` | 0 | first `N` iterations on the `stldev` loss (a well-conditioned start for `ensemble`) |
| `--init PATH`, `--seed N` | white noise, 26 | initial images (standardised units) |
| `--bench K` | | only time `K` evaluations |
| `--gpu` | off | the batch on a CUDA device (feature `gpu`; `STL_GPU_DEVICE`) |
| `--out PREFIX` | | `PREFIX_maps.npy` `[B, C, H, W]` (target units), `PREFIX_target.npy`, `PREFIX_features_{target,init,final}.npy`, `PREFIX_mf_*.npy`, `PREFIX.json` |

`tools/plot_planar.py --run PREFIX --fig DIR` draws the images and the statistics. Example:
[2D synthesis](../tutorials/planar_synthesis.md); design and benchmarks in
[the planar report](../planar_2d.md).

(stl-minkowski)=
## `stl-minkowski`

```
stl-minkowski REF MAP... --sigma-out sigma.json      # std of the functionals over REF and the MAPs
stl-minkowski REF MAP... --sigma sigma.json          # z of the MAPs against REF
```

Minkowski functionals per pyramid level with the estimator of `stl-synth --minkowski-weight`,
thresholds from the reference map. Options: `--nside A-B` (levels, default 64 to nside/2),
`--h X` and `--eps X` (0.2, 0.01), `--standardise`, `--json OUT`.

(stl-mpi-check)=
## `stl-mpi-check`

```
mpirun -n P stl-mpi-check fixtures/stats/<case> [--save out.bin] [--reference ref.bin]
```

Built with `stl-mpi`'s `mpi` feature. Computes the distributed statistics and gradients of an
STL-Dev `stats` fixture, compares them with the fixture (and with a saved run), prints the error
in units of the tolerance and the cells held per rank, and exits with status 1 when a comparison
fails. `tools/mpi_check.sh 2 4` runs it on every fixture with 1, 2 and 4 ranks
(`STL_CHECK_GPU=1`: one device per rank). This is the acceptance test of the multi-rank code.
Example: [multi-rank runs](../tutorials/mpi.md).

## Environment variables

| Variable | Effect |
|---|---|
| `STL_GPU_DEVICE` | device of a rank (default: local rank modulo the visible devices) |
| `STL_GPU_FIELDS`, `STL_GPU_CONV` | defaults of `--gpu-fields`, `--gpu-conv` |
| `STL_GPU_CACHE_MB`, `STL_HOST_CACHE_MB` | budgets of the device and host geometry caches |
| `STL_GPU_THREADS` | host threads of the GPU setup per rank |
| `STL_GPU_SYNC=1`, `STL_GPU_PROFILE=1` | synchronise after every kernel; profile the geometry phases |
| `STL_THREADS` | threads of `stl-synth2d` and of the harmonic terms of `stl-ensemble` |
| `STL_CUFFT` | path of `libcufft.so` |
| `CUDARC_CUDA_VERSION` | build time: the driver API version (e.g. 12040) |
| `MALLOC_MMAP_THRESHOLD_` | overrides the 128 kB mmap threshold `stl-synth` sets so that block buffers are returned to the system |
