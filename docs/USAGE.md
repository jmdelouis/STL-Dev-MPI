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

With CUDA GPUs, `stl-synth --gpu` (feature `gpu`) runs one device per
rank; it needs only the NVIDIA driver and an NVRTC library at run time
(no toolkit to build). See [`docs/gpu.md`](gpu.md) §4.

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
git clone https://github.com/jmdelouis/STL-Dev-MPI.git $W/src   # private repository: ask jmdelouis for access
cd $W/src
git checkout main
cargo fetch                                                      # on a node with internet access
```

`main` is the reference branch: everything in this guide is on `main`,
and every merge into it has passed the fixture tests. Work in progress
lives on feature branches (`m6-scaling`, ...), which are merged into `main`
when their tests pass. To update an installation, run `git pull` on `main`,
then `cargo fetch` and rebuild (§3.6).

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
   With `--keep-mean`, every group of 4 children is shifted so its mean is
   the parent's value: cutting back gives the target exactly. The plain
   bilinear interpolation smooths the target's finest scale (on the
   Takahashi map at nside 2048, only 35 % of the variance of the finest
   details survives up then down).
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
stl-synth --target T_gauss_up.npy --L 4 --weights ergodic --niter 400 --ranks 8 --out run

# MPI
mpirun -np 192 --map-by core --bind-to core stl-synth --target T_gauss_up.npy --L 4 --weights ergodic --niter 400 --out run
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
| `--L N` | 4 | wavelet orientations (`--L 2` reproduces STL-Dev's reference runs, but see below) |
| `--kernel-size N` | 5 | stencil size |
| `--fewer` | off | `has_fewer_convolutions` definitions of S3/S4 |
| `--weights W` | `none` | `none`: sum of squared differences; `ergodic`: reduced chi^2 (below) |
| `--family-weights F` | 1 | extra factors on whole families, e.g. `S1=10,S2=10` (families `mean`, `var`, `S1`..`S4`) |
| `--ensemble B` | 1 | synthesise B maps whose mean statistics match the target (§5.4) |
| `--groups G` | 1 | split the ranks into G groups, each holding B / G members (MPI; §5.4) |
| `--scatter-weight X` | 0 | hold the scatter of the members at the ergodic variance (second moment, §5.4) |
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

**Orientations.** Use L = 4 (the default). With L = 2, even with ergodic
weights and a gaussianised target, the syntheses come out elongated
east-west and the high values clump into blobs. On the Takahashi
convergence map at nside 256, the east-west / north-south small-scale power
ratio is 0.89 +- 0.03 against 1.02 +- 0.05 for the target. L = 4 costs
about 1.4 times L = 2.

### 5.2 Outputs

| File | Content |
|---|---|
| `PREFIX.json` | report: `loss_history`, `evaluations`, `seconds`, `timers_seconds`, `memory_floor_bytes`, peak memory per rank, plan size, `weights`, `final_residual_in_ergodic_sigma` (rms and max per family) |
| `PREFIX_synthesis_nested.npy` | the synthesis, NESTED, in the target's units |
| `PREFIX_synthesis.fits`, `PREFIX_target.fits` | NESTED HEALPix binary tables |
| `PREFIX_mMMM_synthesis_nested.npy` | member MMM of an ensemble (`--ensemble B > 1`, §5.4) |

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

### 5.4 Ensembles of syntheses

`--ensemble B` synthesises B maps together, with the loss on the mean of
their statistics:

    loss = sum_i w_i | (1/B) sum_m S_i(map_m) - S_i(target) |^2

The target's statistics are then matched by the ensemble, not by every map:
the ergodic scatter goes into the differences between the members. With
`--weights ergodic` the weights are multiplied by B (variance sigma^2 / B of
a mean of B maps), so the loss stays the reduced chi^2 of the mean. Member m
starts from white noise with seed `--seed + m`, or from `--init` plus noise
drawn with that seed; `--init-noise 0` is refused (identical members stay
identical). Member 0 is the single-map synthesis, and `--ensemble 1` gives
exactly the previous runs.

The mean alone leaves the scatter of the members free: at nside 64 (4
members, 300 iterations) every member ends about 3 sigma from the target,
where an ergodic realisation sits about 1 sigma away. `--scatter-weight X`
adds the second moment of the ensemble,

    X / n sum_i (D_i / sigma_i^2 - 1)^2,   D_i = sum_m |S_i(map_m) - mean_i|^2 / (B - 1)

which holds the scatter of the members around their mean at the ergodic
variance of one map (`sigma_i^2`, the same as in the ergodic weights). The
report gives, per family, `scatter_over_ergodic_variance` (the mean of
`D_i / sigma_i^2`, 1 for an ergodic ensemble) and the two terms of the
final loss. With few members `D_i` is a noisy estimate: hold it per
coefficient only with B of about 8 or more, or keep X small.

Measured at nside 64 (Takahashi kappa, L = 4, ergodic weights, 300
iterations); residuals per family S1..S4, var:

| Run | mean vs target (sigma / sqrt B) | each member vs target (sigma) | `scatter_over_ergodic_variance` |
|---|---|---|---|
| B = 4, X = 0 | 0.24 - 0.45 | 3.0 - 4.6 | free |
| B = 4, X = 0.1 | 0.21 - 0.59 | 0.47 - 1.06 | 1.07 - 1.37 (var 0.29) |
| B = 4, X = 1 | 0.02 - 0.72 | 0.89 - 0.96 | 0.94 - 1.12 |
| B = 4, X = 10 | 0.81 - 3.0 | 0.97 - 1.73 | 1.00 - 1.03 |
| B = 8, X = 1 | 0.45 - 0.80 | 0.95 - 1.03 | 0.95 - 1.15 |

X = 1 gives an ergodic ensemble (members about 1 sigma from the target,
scatter at the ergodic variance) with the mean still within one sigma of
the mean; X = 10 holds the scatter exactly but loosens the mean.

Placement: `--groups G` splits the ranks into G groups of consecutive ranks
(G divides B and the number of ranks). The ranks of a group share the sky
of B / G members, held as a batch by the engine (CPU or GPU); the groups
exchange one vector of `len(plan)` complex numbers per evaluation, the
partial sums of the statistics. So:

- `--groups 1`: every rank holds every member: memory and time per
  evaluation grow with B, no extra communication. Small maps.
- `--groups B`: one member per group: memory per rank and time per
  evaluation of a single map, with B times the ranks (or GPUs). Large maps.

Any G gives the same synthesis up to the order of the sums (relative
differences around 1e-12 after 30 iterations on 4 ranks, nside 64).

Outputs: `PREFIX_mMMM_synthesis_nested.npy` and
`PREFIX_mMMM_synthesis_standardized.npy` per member, `PREFIX_target.fits`,
and in `PREFIX.json`: `final_residual_in_ergodic_sigma` for the mean (in
units of sigma / sqrt(B)), and `ensemble.per_member_in_ergodic_sigma`, per
family, the RMS of every member's residual to the target and of its scatter
around the mean (in units of sigma).

```bash
# 8 members on 8 GPUs (one per group), from the coarse-to-fine init
mpirun -np 8 stl-synth --gpu --target T_gauss_up.npy --weights ergodic --niter 400 \
    --ensemble 8 --groups 8 --init INIT.npy --out ens
```

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
    $CARGO_TARGET_DIR/release/stl-synth --target $TARGET --L 4 --weights ergodic --niter 400 --out run \
    2> run.err | tee run.log
```

```bash
sbatch --export=ALL,W=$W,TARGET=$W/data/T_gauss_up.npy job.slurm
```

Check the per-user CPU limit of your QOS (`sacctmgr show qos`). A job
asking for more than the limit stays pending forever. To spread the
allowed CPUs over several nodes, use `--nodes=2 --ntasks-per-node=96`.

### 6.2 PBS

`tools/datarmor/chain_up.pbs` (synthesis chain of §4.3) and
`tools/datarmor/synth_nside64.pbs` are working examples (Intel MPI,
`mpirun -np $NCPUS`). Submit with `qsub -v W=$W,... script.pbs`.

### 6.3 TGCC Irene (`ccc_msub`)

`tools/irene/` holds the environment (`rust`, `mpi/openmpi/4.1.6` and
`python3/3.12` modules, the system's libclang 17), the chain job
`chain_up.msub` and `submit_c2f.sh`. The latter submits a coarse-to-fine
series, one job per target nside, chained with `afterok`. Irene has no
internet access: ship the sources with their crates vendored (`cargo
vendor`, plus the `.cargo/config.toml` it prints), then build `--offline`.

## 7. Sizing

Measured with kernel 5 and blocks of 16384:

| nside synthesised | L | Ranks | s / evaluation | Code memory per rank | Machine |
|---|---|---|---|---|---|
| 64 | 2 | 1 | 3.3 | 87 MB | AMD EPYC 9655, OpenMPI 5 |
| 64 | 2 | 28 | 0.8 | 21 MB | Xeon E5-2680 v4, Intel MPI |
| 128 | 2 | 8 threads | 7 | n/a | 20-core workstation |
| 1024 | 2 | 24 | 61 | 155 MB | EPYC 9655 |
| 1024 | 2 | 192 | 10.2 | 100 MB | EPYC 9655, 75 % efficiency from 24 to 192 ranks |
| 1024 | 2 | 128 | 30.2 | | Irene Rome (EPYC 7H12), OpenMPI 4.1 |
| 1024 | 2 | 512 | 8.9 | | Irene Rome, 85 % efficiency from 128 ranks |
| 1024 | 4 | 512 | 12.8 | | Irene Rome |

- Time scales as the number of pixels divided by the number of ranks, as
  long as a rank owns at least about 16k level-0 pixels. Every exchange of
  an evaluation goes to neighbours only (point-to-point halos, sums by
  gather and broadcast), so this holds beyond a few hundred ranks. The
  first versions, which used all-to-all exchanges, ran slower at 512 ranks
  than at 128.
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
