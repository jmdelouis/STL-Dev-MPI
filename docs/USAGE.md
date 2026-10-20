# Using STL-Dev-MPI on a computing centre

This guide is for a user who wants to synthesise HEALPix maps from
scattering-covariance statistics with STL-Dev-MPI on their own cluster. It
covers installation without root rights, building with MPI, preparing a
target map, running `stl-synth`, and the recipe that gives good syntheses on
non-Gaussian fields. The model itself is described in `CLAUDE.md` and the
validation in `README.md`.

## 1. What the package does

`stl-synth` reads a full-sky HEALPix map (the *target*), computes its
pyramidal scattering-covariance statistics (`mean`, `var`, S1, S2, S3, S4),
and minimises with L-BFGS the distance between the statistics of a map and
those of the target, starting from white noise or from a lower-resolution
synthesis. The statistics and their gradient are computed block by block,
and the sky is split between MPI ranks, so the memory per rank is set by
its share of the sky, not by the full sky.

The package contains:

| Path | Content |
|---|---|
| `crates/stl-core` | geometry, convolutions, block-wise statistics and gradients (pure Rust, no MPI) |
| `crates/stl-mpi` | domain decomposition, halo exchange, distributed objective, npy/FITS I/O |
| `crates/stl-cli` | the `stl-synth` binary |
| `tools/*.py` | target preparation, Gaussian anamorphosis, resampling, figures |
| `tools/hpc/`, `tools/datarmor/` | environments and batch scripts for two clusters (Slurm and PBS), to adapt |

## 2. Requirements

| Need | Why | Without root rights |
|---|---|---|
| A recent stable Rust | build | `rustup` in your work directory (§3.1) |
| An MPI library with `mpicc` (OpenMPI 4/5, Intel MPI, MPICH) | multi-rank runs | usually a `module load` |
| `libclang` | build time only: the `mpi` crate generates its bindings with bindgen | a module, or the PyPI `libclang` wheel (§3.2) |
| A C compiler (`gcc`) | linking | system compiler |
| Python 3.10+ with `numpy`, `scipy`, `healpix-geo`, `matplotlib` | target preparation and figures | a venv; see §3.4 when the cluster's Python is older |

Without MPI, everything still runs on one node with thread ranks
(`--ranks N`), which is enough up to nside 128 to 256.

## 3. Installation

Put everything in a work directory with enough quota (toolchain about
1.5 GB, build about 1 GB). Home directories are often too small. Below,
`W` is that directory.

```bash
export W=/path/to/work/stl-mpi
mkdir -p $W && cd $W
```

### 3.1 Rust toolchain

```bash
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
source $CARGO_HOME/env
rustc --version
```

### 3.2 libclang

First look for a module (`module avail clang`, `module avail llvm`). If
there is none, install the PyPI wheel, which ships `libclang.so`:

```bash
python3 -m pip install --target $W/libclang libclang
export LIBCLANG_PATH=$W/libclang/clang/native
```

bindgen also needs the compiler's own headers:

```bash
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
```

### 3.3 Sources and dependencies

```bash
git clone https://github.com/jmdelouis/STL-Dev-MPI.git $W/src   # private repository: ask for access
cd $W/src
cargo fetch                                                      # on a node with internet access
```

Compute nodes often have no internet access. `cargo fetch` on the login
node downloads every crate into `$CARGO_HOME`; the build then runs anywhere
with `--offline`. Without `git` access, copy the source tree with `rsync`
(skip `target/`).

### 3.4 Python environment for the tools

```bash
python3 -m venv $W/pyenv
$W/pyenv/bin/pip install numpy scipy healpix-geo matplotlib
```

`healpix-geo` needs Python 3.10 or later. If the cluster only offers an
older Python, two choices:

- install a recent Python through conda, mamba or `uv` in `$W`;
- run the two steps that use `healpix-geo` (`resample_nested.py up` and
  `plot_mollweide.py`) on another machine, and copy the files. The steps
  that run on compute nodes after the synthesis (`resample_nested.py down`,
  `gaussianize.py match`) only need numpy and scipy.

Make sure the venv's Python is the one the compute nodes run: some
clusters' compute nodes do not see the login node's Python packages.

### 3.5 Environment file

Write an `env.sh` sourced by every build and batch script.
`tools/hpc/env.sh` (Slurm, OpenMPI) and `tools/datarmor/env.sh` (PBS,
Intel MPI) are examples. A generic version:

```bash
# $W/src/tools/<site>/env.sh
W=${W:-/path/to/work/stl-mpi}
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo CARGO_TARGET_DIR=$W/target
source $CARGO_HOME/env
module load openmpi            # your site's MPI module
export LIBCLANG_PATH=$W/libclang/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
```

Pitfalls met on the two clusters:

- **Mismatched compiler.** A conda environment loaded by the login profile
  can put a `gcc` first in `PATH` that cannot link position-independent
  executables. Remove it from `PATH` and force the system compiler
  (`CC=/usr/bin/gcc`, `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc`;
  for Intel MPI, also `I_MPI_CC=/usr/bin/gcc`).
- **csh login shell.** Send bash scripts with `ssh host 'bash -l -s' < script.sh`,
  or put `#!/bin/bash` at the top of batch scripts.

### 3.6 Build

```bash
source $W/src/tools/<site>/env.sh
cd $W/src
cargo build --release --offline -p stl-cli --features mpi
ls $CARGO_TARGET_DIR/release/stl-synth
```

Without `--features mpi`, the binary runs thread ranks only (`--ranks N`).
Building takes a few minutes. Where login nodes limit CPU time, build on a
compute node (`tools/hpc/build.slurm`).

### 3.7 Check the installation

```bash
cargo test --release --offline                 # fixture tests against STL-Dev, threads backend
tools/mpi_check.sh 2 4                         # same statistics under mpirun with 1, 2 and 4 ranks
```

Every test must pass. The MPI check compares each rank count with the
1-rank run (differences below 1e-15).

## 4. Preparing a target

`stl-synth` expects a **full-sky map in NESTED ordering**, as a `.npy`
(1-D, `float64` or `float32`, either endianness) or a HEALPix FITS binary
table. `nside` must be a power of 2. Masks and NaNs are not supported:
every pixel must be finite.

### 4.1 Ordering and resolution

```bash
$W/pyenv/bin/python tools/prepare_healpix_target.py IN.npy OUT_PREFIX 256 512 1024 [--ordering ring|nested|auto]
```

The script writes `OUT_PREFIX_nside<N>_nested.npy` for every requested
`N`, averaging sub-pixels. It is memory-mapped and works by chunks, so it
handles nside 8192 inputs. With `--ordering auto` (the default), it detects
whether the input is RING or NESTED.

### 4.2 Gaussian anamorphosis (strongly recommended)

Fields such as weak-lensing convergence or density have a very skewed
pixel distribution. Synthesise a Gaussianised version of the target, then
transform back. This is the transform of FOSCAT
(`scat_cov.to_gaussian` / `from_gaussian`):

```bash
python tools/gaussianize.py forward  T.npy T_gauss.npy table.npz   # rank -> norm.ppf((r + 0.5) / N)
python tools/gaussianize.py inverse  table.npz S_gauss.npy S.npy    # cubic interpolation back
python tools/gaussianize.py match    table.npz S_any.npy S.npy      # rank-based histogram matching
```

`inverse` is the exact inverse of `forward` (round-trip error 1e-8 of the
std). `match` gives a map exactly the target's pixel histogram, whatever
transforms were applied in between.

### 4.3 Removing the smallest scales: synthesise at 2 N

The smallest scales (those of the finest level, seen by a 5 x 5 stencil
with bilinear interpolation) are the least well synthesised. The validated
recipe is:

1. gaussianise the target `T_N`;
2. interpolate it bilinearly to `2N`, so the new finest level carries no
   power of its own:
   ```bash
   python tools/resample_nested.py up T_gauss.npy T_gauss_up.npy      # needs healpix-geo; chunked
   ```
3. run the synthesis at `2N` (§5);
4. cut back to `N` (mean of the 4 children):
   ```bash
   python tools/resample_nested.py down up_synthesis_nested.npy S_gauss.npy
   ```
5. match the target's histogram:
   ```bash
   python tools/gaussianize.py match table.npz S_gauss.npy S.npy
   ```

`tools/hpc/chain_up.slurm` runs the whole chain in one Slurm job. When
step 2 has to run elsewhere (old Python on the cluster), pass the directory
holding `target_gauss_up_nested.npy` and `gauss_table.npz` as `PREP`.

## 5. Running `stl-synth`

```bash
# one node, thread ranks (binary built with or without --features mpi)
stl-synth --target T_gauss_up.npy --L 2 --weights ergodic --niter 400 --ranks 8 --out run

# MPI
mpirun -np 192 --map-by core --bind-to core stl-synth --target T_gauss_up.npy --L 2 --weights ergodic --niter 400 --out run
```

### 5.1 Options

| Option | Default | Meaning |
|---|---|---|
| `--target MAP` | (required) | NESTED full-sky map, `.npy` or `.fits`; standardised internally (mean 0, std 1) |
| `--init MAP` | white noise | initial map (NESTED, nside <= target): coarse-to-fine start, replicated on the children |
| `--init-units U` | `target` | `target`: the init is in the target's units and gets the same standardisation; `standardized`: already standardised |
| `--init-noise X` | `auto` | noise added to `--init`; `auto` = RMS of the target's scales missing at the init nside |
| `--seed N` | 11 | white-noise seed (the same map whatever the number of ranks) |
| `--J N` | log2(nside) - 1 | number of scales |
| `--L N` | 2 | wavelet orientations |
| `--kernel-size N` | 5 | stencil size |
| `--fewer` | off | `has_fewer_convolutions` definitions of S3/S4 |
| `--weights W` | `none` | `none`: sum of squared differences; `ergodic`: reduced chi^2 (below) |
| `--niter N` | 50 | L-BFGS iterations (strong Wolfe line search, as `torch.optim.LBFGS`) |
| `--history-size N` | 10 | L-BFGS history |
| `--block-pixels N` | 16384 | block size per level: smaller blocks need less memory and run a little slower |
| `--conv-chunk N` | whole block | convolution geometry rebuilt by sub-blocks of N rows (least memory, slowest) |
| `--out PREFIX` | `synthesis_rust_nside<N>` | output prefix |
| `--ranks N` | 1 | thread ranks in this process (non-MPI builds) |
| `--quiet` | off | no line per evaluation |

**Ergodic weights.** With `--weights ergodic`, every coefficient is a pixel
mean `S = <f_p>` over the `n` pixels of its native level. Its variance under
ergodicity is `sigma^2 = (<|f_p|^2> - |S|^2) / n`, computed on the target.
The loss becomes `mean_i |S_i - S_i(target)|^2 / sigma_i^2`. Without these
weights the loss is dominated by the low-order coefficients, and S3/S4
stay at 1 to 3 sigma even after 400 iterations. With them, every family
converges to well below 1 sigma. Use them for any real synthesis.

### 5.2 Outputs

| File | Content |
|---|---|
| `PREFIX.json` | report: `loss_history`, `evaluations`, `seconds`, `timers_seconds`, `memory_floor_bytes`, peak memory per rank, plan size, `weights`, `final_residual_in_ergodic_sigma` (rms and max per family) |
| `PREFIX_synthesis_nested.npy` | the synthesis, NESTED, in the target's units |
| `PREFIX_synthesis.fits`, `PREFIX_target.fits` | NESTED HEALPix binary tables |

Every rank reads and writes only its own slice: no rank holds a full-sky
map. `final_residual_in_ergodic_sigma` is the quickest convergence check:
every family at a few tenths of a sigma or less is a converged synthesis.

### 5.3 Rank count and coarse-to-fine starts

- Any number of ranks works. Statistics and loss are identical to 16 digits
  whatever the rank count.
- With `--init` at a lower nside, every rank's range must fall on whole
  pixels of the init map. If the program stops with "rank ranges are not
  aligned on its pixels", use fewer ranks or an init closer in nside.
- Large nside: do not start from white noise. Chain syntheses
  (`N/8 -> N/4 -> N/2 -> N`), each initialised with the previous one
  (`--init`, default `--init-noise auto`), with fewer iterations at each
  step (a few hundred at the coarsest, tens at the finest).

## 6. Batch jobs

### 6.1 Slurm

```bash
#!/bin/bash
#SBATCH --job-name=stl-synth
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G            # not --mem=0 where the site sets --mem-per-cpu
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
set -eu
source $W/src/tools/<site>/env.sh
OUT=$W/runs/$SLURM_JOB_ID; mkdir -p $OUT; cd $OUT
mpirun -np $SLURM_NTASKS --map-by core --bind-to core \
    $CARGO_TARGET_DIR/release/stl-synth --target $TARGET --L 2 --weights ergodic --niter 400 --out run \
    2> run.err | tee run.log
```

```bash
sbatch --export=ALL,W=$W,TARGET=$W/data/T_gauss_up.npy job.slurm
```

Check the per-user CPU limit of your QOS (`sacctmgr show qos`). A job
asking for more than the limit stays pending forever. To spread the
allowed CPUs over several nodes, use `--nodes=2 --ntasks-per-node=96`.

### 6.2 PBS

`tools/datarmor/synth_nside64.pbs` is a working example (Intel MPI,
`mpirun -np $NCPUS`). Submit it with `qsub -v W=$W script.pbs`.

## 7. Sizing

Measured with L = 2, kernel 5, blocks of 16384:

| nside | Ranks | s / evaluation | Code memory per rank | Note |
|---|---|---|---|---|
| 64 | 1 | 3.3 | 87 MB | AMD EPYC 9655, OpenMPI 5 |
| 64 | 28 | 0.8 | 21 MB | Xeon E5-2680 v4, Intel MPI |
| 128 | 8 threads | 7 | n/a | 20-core workstation |
| 1024 | 24 | 61 | 155 MB | |
| 1024 | 192 | 10.2 | 100 MB | 75 % efficiency from 24 to 192 ranks |

- Time scales as the number of pixels divided by the number of ranks, as
  long as a rank owns at least about 16k level-0 pixels.
- L = 4 costs about 1.3 to 1.5 times L = 2.
- An L-BFGS iteration costs about 1.1 evaluations on average.
- The memory per rank is the state (maps, stored moduli, their cotangents,
  L-BFGS history: about `250 bytes x N_pix / P` at L = 2), plus about
  100 MB of block transients, plus the MPI runtime. OpenMPI 5 starts each
  rank at 200 to 350 MB; Intel MPI at 20 to 30 MB.
- To lower the peak, reduce `--block-pixels` (4096: about half the
  transients) or use `--conv-chunk 2048`.

## 8. Figures

```bash
python tools/plot_mollweide.py out.png "Target=T.npy" "Synthesis=S.npy" [...]
python tools/compare_syntheses.py compare.png T.npy "label=S1.npy" "label=S2.npy" [...]
```

`plot_mollweide.py` draws Mollweide views on a common colour scale (the
first map's percentiles). `compare_syntheses.py` draws full-sky and zoomed
rasters, pixel histograms, and a small-scale anisotropy measure: the ratio
of east-west to north-south power in gnomonic patches, which is 1 for an
isotropic map.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `bindgen` / `libclang` errors at build | `LIBCLANG_PATH` not set, or compiler headers not found: see §3.2 |
| link error about `Scrt1.o` | a conda `gcc` is first in `PATH`: use the system compiler (§3.5) |
| `cargo build --offline` cannot find a crate | run `cargo fetch` on a node with internet access first |
| `ModuleNotFoundError: numpy` on compute nodes | the nodes do not see the login node's packages: use a venv under `$W` |
| `healpix-geo` not installable | Python older than 3.10 (§3.4) |
| job pending with `QOSMaxCpuPerUserLimit` | ask for fewer CPUs |
| `--mem=0` rejected | the site sets `--mem-per-cpu`: give an explicit `--mem` |
| `N non-finite values` in the target | masks and NaNs are not supported: fill or inpaint the map first |
| loss stops after one or two evaluations | the loss is already below the gradient tolerance (1e-12): check the weights and the target's standardisation |
| synthesis elongated east-west, or too smooth | use `--weights ergodic`, gaussianise the target, synthesise at 2N (§4) |
| background runs die when the SSH session ends | `setsid nohup ... < /dev/null &`, or use the batch system |
