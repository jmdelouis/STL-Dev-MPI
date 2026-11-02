# Production at nside 1024 with P. Campeti's ensemble method: step by step

This note records how the nside-1024 ensemble of P. Campeti's method
(`stl-ensemble`, docs/campeti_merge.md sections 8 to 10) was produced on
one A100 of the hpc cluster (Ifremer, Slurm), and how it was judged.  It
gives every command with its options and the durations measured.

The run used ten members, ten Takahashi realisations (zs1) as references,
2500 L-BFGS iterations, the harmonic terms on the device, and one MPI rank
on one A100 80 GB.  Revision: branch `claude/ensemble-gpu-terms`, commit
223ace8, which reached main through PR #5 (merge commit 521fc8a).  The
code of main is faster for the same results: about 28 s per evaluation
instead of 69 s, so about 22 h for the same run (docs/harmonic_speed.md).

## 1. Environment and modules

The hpc cluster runs RHEL 9 with Slurm.  Its `gpu` partition has nodes of
8 x A100 80 GB (and one of 8 x H200).  The toolchain is not a module:

| What | Where |
|---|---|
| Rust (rustup, stable) | `$HOME/stl-mpi/rustup`, `$HOME/stl-mpi/cargo` (`W=$HOME/stl-mpi`) |
| libclang (build time only, MPI bindings) | PyPI wheel `libclang` under `$W/libclang` |
| OpenMPI | `module load openmpi/5.0.11` |
| CUDA (NVRTC for the kernels, cuFFT for the harmonic terms) | `module load cuda/13.0.0` (`/app/cuda/13.0.0/lib64/libcufft.so.12`) |
| sources and build of this production | `G=/scratch/jmdeloui/stl-campeti` (`$G/src`, `$G/target`) |

`tools/hpc/env.sh` and `tools/hpc/env_gpu.sh` set this up (sourced by
the Slurm scripts):

    export G=/scratch/jmdeloui/stl-campeti W=$HOME/stl-mpi
    source $G/src/tools/hpc/env_gpu.sh
    # RUSTUP_HOME, CARGO_HOME = $W/...; CARGO_TARGET_DIR = $G/target;
    # modules openmpi/5.0.11 and cuda/13.0.0; LIBCLANG_PATH;
    # CUDARC_CUDA_VERSION=13000 (driver API of cudarc)

cuFFT is loaded at run time (`crates/stl-gpu/src/cufft.rs`): the CUDA
module puts `libcufft.so` on `LD_LIBRARY_PATH`.  `STL_CUFFT=/path/libcufft.so.N`
overrides it.

## 2. Build

The sources come from the git branch.  The compute nodes have no network,
so the crates are fetched on the login node, then the build runs on 16
cores of the `cpu` partition:

    # on the workstation (the repository is private):
    git archive --format=tar claude/ensemble-gpu-terms | ssh hpc 'mkdir -p /scratch/jmdeloui/stl-campeti/src && tar xf - -C /scratch/jmdeloui/stl-campeti/src'
    # on the hpc login node:
    cd $G/src && cargo fetch
    sbatch --export=ALL,G=$G,W=$W $G/src/tools/hpc/build_gpu.slurm
    # = cargo build --release --offline -j 16 -p stl-cli --features mpi,gpu

The binary is `$G/target/release/stl-ensemble`.  Build time: 18 s (108 crates, job 2570447).

## 3. Inputs and calibration

**References.** These are the ten Takahashi realisations r000-r009
(source redshift zs1) at nside 1024: `/scratch/jmdeloui/takahashi/r00?_nside1024_nested.npy`,
NESTED, `float64`.  Each is exactly the mean of the 4 children of the
nside-2048 map (checked: maximum difference 0).  The same files are on
garoupe in `/tmp/jmdeloui-irene1024/real` (identical md5).

**Calibration.** The calibration builds the LUT pooled over the ten maps,
the targets of the five blocks, and the C_l of the initial Gaussian
fields:

    stl-ensemble calibrate --references r000_nside1024_nested.npy ... r009_nside1024_nested.npy \
        --nside 1024 --gpu --out cal1024.json

On garoupe (one T4 for the SC statistics, 12 threads for the transforms)
it takes 13 min, 8.8 GB of host memory, and gives a 3.7 MB JSON.  The
production used this file, copied to `$G/cal1024.json`
(md5 60a91029b58c222f7309ad42db8a1fe8).  The Slurm script computes it
when `$CAL` is missing.

**C_l variance cutoff.** `--ps-variance-upper 3071` keeps every band from
l = 20 to `3 nside - 1`, the source's rule at nside 128.  The realisations'
band powers scatter by 8-15 % at every l, up to 134 times the Gaussian
value.  docs/campeti_merge.md section 10.2 has the table.

**Initial fields.** These are the default Gaussian random fields with the
mean C_l of the Gaussianised references (`--seed 61022026 + m`).  A warm
start from the nside-128 run started higher (docs/campeti_merge.md
section 10.3).

## 4. The Slurm job

`tools/hpc/ensemble_a100.slurm` (listing below).  It runs one MPI rank on
one A100, with the ten members in one batch:

    sbatch --dependency=afterany:2535705,afterok:2570447 \
        --export=ALL,G=/scratch/jmdeloui/stl-campeti,W=$HOME/stl-mpi \
        /scratch/jmdeloui/stl-campeti/src/tools/hpc/ensemble_a100.slurm

The dependency waited for the production job then on the A100 and for the
build.  Options, with the values used:

| Option | Value | Role |
|---|---|---|
| `--partition gpu --gres gpu:a100:1` | | one A100 80 GB |
| `--cpus-per-task 16`, `STL_THREADS`, `STL_GPU_THREADS` | 16 | host threads (LUT chain, geometry) |
| `--mem` | 128G | host memory (L-BFGS history: 2 x 10 vectors of 10 x 12.6 M doubles) |
| `--time` | 4-00:00:00 | |
| `STL_GPU_CACHE_MB` | 40960 | geometry cache of the SC engine; the rest of the device for the harmonic terms |
| `--niter`, `--max-eval` | 2500, 50000 | as the source's runs |
| `--ps-variance-upper` | 3071 | section 3 |
| `--gpu --gpu-fields recompute` | | SC engine on the device; level fields recomputed in the reverse sweep |
| `--harmonic` | gpu (the default with `--gpu`) | C_l, PDF and Minkowski terms on the device |
| `--checkpoint` | 100 | every 100 evaluations, the maps of the lowest loss so far |
| `--groups` | 1 (default) | one group: the ten members on the one device |

```bash
#!/bin/bash
#SBATCH --job-name=stl-ens1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out

set -eu
G=${G:-/scratch/jmdeloui/stl-campeti}
W=${W:-$HOME/stl-mpi}
source "$G/src/tools/hpc/env_gpu.sh"
NSIDE=${NSIDE:-1024}
NITER=${NITER:-2500}
CUTOFF=${CUTOFF:-3071}
CKPT=${CKPT:-100}
REF=${REF:-/scratch/jmdeloui/takahashi}
CAL=${CAL:-$G/cal${NSIDE}.json}
OUT=$G/runs/${RUN:-$SLURM_JOB_ID}
B=$CARGO_TARGET_DIR/release/stl-ensemble
mkdir -p "$OUT"
cd "$OUT"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, GPU ${CUDA_VISIBLE_DEVICES:-?}, nside $NSIDE, niter $NITER, cutoff $CUTOFF, init ${INIT:-random fields}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export STL_THREADS=$SLURM_CPUS_PER_TASK STL_GPU_THREADS=$SLURM_CPUS_PER_TASK
# Geometry cache of the SC engine: 40 GB, the rest of the 80 GB for the
# harmonic terms (transforms and Minkowski states) next to the engine.
export STL_GPU_CACHE_MB=${STL_GPU_CACHE_MB:-40960}
if [ ! -f "$CAL" ]; then
    time mpirun -np 1 --bind-to none "$B" calibrate --references $(ls "$REF"/r00?_nside${NSIDE}_nested.npy) \
        --nside "$NSIDE" --gpu --out "$CAL" 2>&1 | grep -v Authorization
fi
INIT_OPT=""
[ -n "${INIT:-}" ] && INIT_OPT="--init $INIT"
(while sleep 600; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) > gpu_usage.log 2>&1 &
MON=$!
time mpirun -np 1 --bind-to none "$B" run --calibration "$CAL" --ps-variance-upper "$CUTOFF" \
    --niter "$NITER" --max-eval 50000 --gpu --gpu-fields recompute --checkpoint "$CKPT" $INIT_OPT ${EXTRA:-} \
    --out ens 2> ens.err | tee ens.log
kill $MON 2>/dev/null || true
tail -3 ens.err
md5sum ens_m0*_nested.npy > md5.txt
touch READY
echo "done $(date)"
```

The production was job 2570602, on node compute-114-15.  It started on
31 Oct 2026 at 09:50 and ended on 2 Nov at 10:50 (COMPLETED, exit 0).

## 5. Monitoring and restart

Below, `JOB` is the Slurm job id (2570602 for this production).

    squeue -u $USER                      # the job
    tail -f $G/runs/JOB/ens.log          # one line per evaluation: loss, five blocks, time
    cat $G/runs/JOB/gpu_usage.log        # device memory and utilisation every 10 min
    cat $G/runs/JOB/ens_checkpoint.json  # evaluation and loss of the last checkpoint

Each `ens.log` line reads
`evaluation N  loss = L  blocks = [GSC, PSC, C_l, PDF, MF]  t = T s`.  The
loss is the mean of the five blocks.  Each block is a mean term plus a
variance term.

**Restart.** Every 100 evaluations the run writes the Gaussian and the
physical maps of the lowest loss reached so far: `ens_checkpoint_mMMM_*`,
through a temporary name, so that an interrupted write leaves the previous
checkpoint whole.  It also writes `ens_checkpoint.json`.  A job that stops
early (time limit, node failure) restarts from them:

    sbatch --export=ALL,G=$G,W=$W,RUN=JOB_restart,NITER=REMAINING,\
INIT=$G/runs/JOB/ens_checkpoint_m{m}_gaussian_nested.npy \
        $G/src/tools/hpc/ensemble_a100.slurm

The first evaluation of the restart gives the checkpoint's loss to the bit
(tested on garoupe).  The L-BFGS history starts again empty.

## 6. Results and post-processing

At the end the run writes the following in `$G/runs/JOB`:
- `ens_mMMM_gaussian_nested.npy` and `ens_mMMM_physical_nested.npy` (ten
  members, NESTED, `float64`, kappa with the mean removed);
- `ens.json` (report: `loss_history`, `blocks_history`, `blocks_final`,
  timers, memory);
- `md5.txt` and `READY`.

The judgement runs on garoupe (analyses there, not on hpc).  The ten
physical maps are copied there and linked under the names the tools read:

    # on garoupe, J=/export/home1/jmdeloui/stl-campeti/n1024/judge
    scp -o ProxyJump=draco "jmdeloui@access.hpc.ifremer.fr:$G/runs/2570602/ens_m00?_physical_nested.npy" \
        "jmdeloui@access.hpc.ifremer.fr:$G/runs/2570602/md5.txt" $J/a100/
    cd $J/a100 && grep physical md5.txt | md5sum -c -
    for m in 0 1 2 3 4 5 6 7 8 9; do ln -s ens_m00${m}_physical_nested.npy synthesis_m00${m}_nested.npy; done

The two judgement tools are `tools/cv10.py` and `text/scripts/mf_levels.py`
(with `text/scripts/common.py` and `tools/minkowski.py`), run with the
Python environment `/tmp/jmdeloui-diffusion/venv` (healpy 1.20, numpy 2.2):

    # I=/tmp/jmdeloui-irene1024: tools/ (copies of the two scripts), real/ (r000-r009 at 1024), cache/
    export OMP_NUM_THREADS=4
    cd $I/tools
    python cv10.py 1024 $J/cv10_n1024_campeti_a100 $(ls $I/real/r00?_nside1024_nested.npy) -- \
        "Campeti A100, 2500 it=$J/a100"
    python mf_levels.py $I/cache 1024 $J/mfl_a100 $I/real "campeti_a100_2500it=$J/a100" \
        --levels 64,128,256,512 --procs 4 --threads 1

What they measure:
- `cv10.py`, per level of the 4-children pyramid: the variance and the
  excess kurtosis of the details of kappa, and the variance of the details
  of the Gaussianised map.  It also counts the peaks above 3, 5 and 10
  sigma and computes the C_l in log bands.
- `mf_levels.py`: V0, V1 and V2 of kappa degraded to nside 64-512, at
  thresholds nu from -2 to 5.

Both express the members' deviation from r000 in units of the scatter of
the ten realisations.  Durations: the copy takes 3 min, `cv10.py` 10 min,
and `mf_levels.py` 2.4 min (the realisations' functionals come from the
cache).

## 7. Durations

| Step | Where | Duration |
|---|---|---|
| build (16 cores) | hpc `cpu` | 18 s |
| calibration | garoupe, 1 T4 + 12 threads | 13 min |
| setup of the run (geometry, transforms) | A100 | 90 s |
| one evaluation (ten members), wall time | A100 | 69.2 s: objective 65.5 s, L-BFGS step 3.7 s |
| 2500 iterations (2547 evaluations) | A100 | 49.0 h |
| copy of the ten physical maps (1 GB, through draco) | hpc to garoupe | 3 min |
| judgement (cv10 + mf_levels) | garoupe, 4 processes | 12.4 min |

## 8. Results

### 8.1 Optimisation

The run stopped on the iteration limit, at 2500 iterations (2547
evaluations), and the loss was still going down: it fell by 4 % over the
last 47 evaluations.

| Evaluation | 1 | 100 | 500 | 1000 | 1500 | 2000 | 2500 | 2547 |
|---|---|---|---|---|---|---|---|---|
| loss | 831.9 | 225.7 | 31.43 | 11.90 | 6.352 | 3.987 | 2.656 | 2.548 |

| Block | GSC | PSC | C_l | PDF | MF |
|---|---|---|---|---|---|
| first evaluation | 2169 | 1498 | 4.839 | 2.448 | 485.0 |
| last evaluation | 11.29 | 1.082 | 0.0696 | 0.0728 | 0.228 |
| of which the mean term | 10.38 | 0.934 | 0.0670 | 0.0488 | 0.201 |

Peak memory was 62.3 GB on the device (the 40 GB geometry cache included)
and 46.6 GB on the host.  An evaluation takes 65.5 s:
- the SC timers 45 s, of which 22 s are the single-threaded layout
  (docs/harmonic_speed.md section 1.1);
- the harmonic terms 14 s;
- the host passes 6 s.

The code of main removes most of the layout, harmonic and host time,
which brings an evaluation to about 28 s.

### 8.2 Judgement against the ten realisations

The outputs are in `benchmarks/campeti_n1024_a100/`: `cv10.py` (txt and
png), `mf_levels.py` (txt and pdf), and `tools/tails.py` (txt).

The tables give the deviation of the members from r000, in units of the
scatter of r000-r009: the median over the ten members, with [min, max]
for this run.  The other two columns are our own recipe at nside 1024, from
`runs/cv10_n1024_datarmor.txt` and `runs/mf_levels_n1024_datarmor.txt`:
600 iterations on Datarmor, and 300 on Irene.  The C_l, the PDF and the
Minkowski functionals of the full-resolution map are terms of this
method's loss.  The statistics per level of the pyramid, the peaks and the
functionals of the degraded maps are not.

**`cv10.py`**, details of the 4-children pyramid:

| level | variance kappa: this run | recipe, Datarmor | excess kurtosis kappa: this run | recipe, Datarmor | variance gaussianised: this run | recipe, Datarmor |
|---|---|---|---|---|---|---|
| 8 | -0.6 [-1.6, +0.8] | -0.0 | -1.3 [-2.7, -0.5] | -1.7 | -1.0 [-2.4, +0.3] | +0.4 |
| 16 | +0.1 [-1.4, +1.5] | +0.5 | -1.3 [-2.9, +2.2] | +0.2 | +1.8 [-0.8, +3.9] | +1.1 |
| 32 | +0.1 [-1.7, +1.3] | +0.4 | -0.6 [-1.6, +11.2] | -0.0 | +3.9 [+0.6, +5.2] | +2.4 |
| 64 | +0.6 [-1.4, +1.4] | +0.1 | +5.7 [-1.5, +51.5] | -0.0 | +3.9 [+1.5, +5.6] | +3.3 |
| 128 | -0.5 [-1.8, +1.2] | -0.1 | +4.7 [-2.9, +110] | -0.1 | +1.6 [-2.5, +5.7] | +3.9 |
| 256 | -0.4 [-1.8, +1.8] | -1.8 | +12.7 [-3.2, +243] | -0.1 | +1.1 [-3.0, +2.7] | -1.2 |
| 512 | -0.3 [-1.5, +1.3] | -0.9 | +10.7 [-3.7, +232] | +0.0 | +0.1 [-1.4, +1.1] | -2.1 |
| 1024 | +2.6 [+1.0, +4.1] | +2.9 | +10.7 [-3.5, +112] | +0.3 | -4.9 [-5.6, -4.3] | -2.5 |

Peaks above 3, 5 and 10 sigma: +1.2 [-1.3, +3.2], +0.1 [-2.3, +2.2] and
+0.3 [-1.2, +1.4], against +1.2, +0.6 and -0.7 for the recipe.  For the
C_l bands with l >= 31, the median |z| is 0.4 and the maximum 3.7, against
0.3 and 1.3.

**`mf_levels.py`**, median |z| over members and thresholds nu = -2 ... 5.
The first column is r001-r009 against r000, which is what an ensemble
within cosmic variance gives:

| level | functional | r001-r009 | this run | recipe, Datarmor 600 it | recipe, Irene 300 it |
|---|---|---|---|---|---|
| 64 | V0 / V1 / V2 | 0.91 / 1.05 / 1.06 | 1.09 / 1.13 / 1.09 | 0.88 / 1.31 / 1.05 | 0.90 / 1.20 / 1.04 |
| 128 | V0 / V1 / V2 | 1.49 / 1.14 / 0.94 | 3.12 / 1.42 / 1.27 | 2.10 / 1.78 / 1.67 | 2.21 / 1.95 / 1.74 |
| 256 | V0 / V1 / V2 | 1.93 / 1.20 / 1.10 | 3.53 / 1.68 / 1.55 | 4.08 / 1.26 / 1.78 | 4.09 / 2.17 / 2.99 |
| 512 | V0 / V1 / V2 | 2.07 / 1.15 / 0.94 | 4.19 / 1.63 / 1.80 | 3.07 / 1.90 / 1.64 | 2.42 / 4.16 / 3.50 |

**Reading.**
- Variance per level: the members scatter around r000 as the realisations
  do, from -1.8 to +1.8 sigma, from level 8 to 512.  Their medians are
  within 0.6 sigma, closer than the recipe's at 256 and 512.  Level 1024
  is high by 2.6 sigma, as for the recipe (2.9).
- Variance of the gaussianised details: closer to the realisations than
  the recipe at 128 and 512, similar at 256.  It is further from 16 to 64,
  and at 1024 (-4.9 against -2.5).
- Peaks and Minkowski functionals of the degraded maps: comparable to the
  recipe.  V1 and V2 are closer to the realisations at 128; V0 is
  further at 128 and 512.
- Excess kurtosis of the details: this is the clear defect.  The medians
  are +4.7 to +12.7 sigma from level 64 up, with one member up to +243
  sigma, where the recipe stays within 0.3 sigma.

**Tails.**  The kurtosis comes from a few pixels.  Kappa with the mean
removed, measured by `tools/tails.py` on garoupe:
- The ten realisations stay within [-0.0003, 0.148].
- Seven members out of ten have 1 to 7 pixels above 0.148, up to 0.45:
  three times the highest realisation.  Their excess kurtosis is 2250 to
  46150, against 1065 to 1773 for the realisations.  The other three
  members (m004, m005, m007) have 795, 1121 and 2215.
- All the members have 430 to 1210 pixels below the realisations'
  minimum, which is the empty-beam floor of kappa.  They go down to
  -0.0014.

Two features of the method explain this:
- The LUT extrapolates linearly beyond its knots and nothing clips the
  result (as at nside 128, docs/campeti_merge.md section 8.5).
- The PDF block bins kappa in 100 linear bins over the references' range
  plus 2 %, so each bin is 1.6e-3 wide, against a standard deviation of
  5.6e-4.  The bulk and the floor fall into three or four bins.  Pixels above the
  references' maximum fall outside the bins and do not enter the loss.

A clip at the references' extremes, or a PDF in finer or logarithmic bins
in the tails, would change P. Campeti's loss; that is his call.
