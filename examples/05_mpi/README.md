# Multi-rank runs under MPI

The sky is split into contiguous NESTED ranges, one per MPI rank; each rank holds its share of
the pyramid, exchanges halos with its neighbours and sums the statistics in rank order. So any
number of ranks gives the same statistics, to the order of the sums. This example runs the
synthesis of [example 01](https://github.com/jmdelouis/STL-Dev-MPI/tree/main/examples/01_healpix_synthesis) (WebSky κ, nside 64, `L = 4`, 10 L-BFGS
iterations) on 1, 2 and 4 MPI ranks, then on 3 ranks with one GPU each. It then checks the
distributed statistics and gradients against an STL-Dev fixture with `stl-mpi-check`.

```bash
# libclang for the mpi crate (build time), e.g. the PyPI wheel
export LIBCLANG_PATH=... BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
cargo build --release -p stl-cli -p stl-mpi --features stl-cli/mpi,stl-cli/gpu,stl-mpi/mpi
MPIRUN="mpirun --bind-to none" GPUS=3 bash examples/05_mpi/run.sh       # GPUS=0: no GPU run
```

The commands it runs:

```bash
mpirun -n 4 stl-synth --target target_map.npy --L 4 --weights ergodic --niter 10 --seed 1 --out out/mpi4
mpirun -n 3 stl-synth --target target_map.npy --L 4 --weights ergodic --niter 10 --seed 1 --gpu --out out/mpi_gpu3
mpirun -n 4 stl-mpi-check fixtures/stats/stats_n32_J4_L4_k5_C1_B1_full_b256
```

`--bind-to none` lets OpenMPI's ranks use every core (by default they are bound to one core each,
and the per-rank threads of the setup then share it). With one GPU per rank, rank `r` takes
device `r` modulo the visible devices. OpenMPI does not need to be CUDA-aware.

## Results (garoupe, 12 cores, 3 × Tesla T4, OpenMPI 4.1.2)

| Ranks | s / evaluation | Speed-up | Wall clock (12 evaluations + setup) | Peak RSS per rank | Loss history vs 1 rank |
|---|---|---|---|---|---|
| 1 | 17.3 | 1.0 | 213 s | 0.16 GiB | – |
| 2 | 9.50 | 1.8 | 119 s | 0.15 GiB | 7.3e-14 |
| 4 | 5.57 | 3.1 | 72 s | 0.13 GiB | 4.2e-14 |
| 3, one T4 each | 0.143 | 121 | 21 s | 0.33 GiB | 7.9e-14 |

Every run ends at the same loss (132.1 → 6.801 after 12 evaluations). The loss histories agree
with the 1-rank run to below 1e-13. An MPI rank runs one thread, so one rank is slower than the
8 thread ranks of example 01. The machine was shared with other jobs (CPU load 5 to 16, the T4s
busy with a training run), so the absolute times are pessimistic. Cluster scaling (Datarmor,
hpc: up to 192 ranks over 4 nodes) is in the [MPI guide](../../docs/user-guide/mpi.md).

`stl-mpi-check` on the nside 32, `J = 4`, `L = 4` fixture:

| Ranks | Time | Statistics error | Gradient error | Level-0 cells held per rank (of 12288) |
|---|---|---|---|---|
| 1 | 10.2 s | 2.3e-6 × tolerance | 2.6e-7 × tolerance | 12288 |
| 2 | 5.2 s | 2.8e-6 × tolerance | 2.6e-7 × tolerance | 8928, 8928 |
| 4 | 2.8 s | 2.8e-6 × tolerance | 2.6e-7 × tolerance | 6144, 6768, 7968, 6144 |

The tolerance is that of the acceptance tests (statistics 1e-11 + 1e-10 relative). At nside 32
the halos are a large share of each rank's cells (73 % more than the owned 3072 at 4 ranks), which
limits the speed-up at that size. `tools/mpi_check.sh 2 4` runs the check on every fixture.

OpenMPI may print `Authorization required, but no authorization protocol specified` once per
rank when `DISPLAY` points to an X server it cannot open (hwloc probing the GPUs through X). This
message is harmless; `unset DISPLAY` silences it.
