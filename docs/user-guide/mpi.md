# MPI

Every tool runs on several ranks. Without the `mpi` feature the ranks are threads of one process
(`--ranks N`); with it they are the processes of `mpirun`, on one or many nodes. Both backends
implement the same `stl_mpi::Comm` trait, and the results do not depend on the backend nor on
the number of ranks (to the order of the sums, below 1e-15 for the statistics).

---

## Minimal example

```bash
export LIBCLANG_PATH=...                                    # bindgen (build time only)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
cargo build --release -p stl-cli -p stl-mpi --features stl-cli/mpi,stl-mpi/mpi
mpirun -n 16 stl-synth --target map.npy --niter 100 --out run
mpirun -n 4 stl-synth --target map.npy --gpu --out run        # with --features gpu too: one device per rank
mpirun -n 4 stl-mpi-check fixtures/stats/stats_n32_J4_L4_k5_C1_B1_full_b256
```

Any MPI library works (OpenMPI 4 and 5, Intel MPI 2019 were used); CUDA-aware MPI is not needed
(the exchanges go through the host).

---

## Key parameters

| Option | Default | Description |
|---|---|---|
| `mpirun -n P` | | ranks (with the `mpi` feature); `--ranks P` runs them as threads without it |
| `--block-pixels N` | 16384 | cells per block: the per-rank transient memory |
| `--groups G` | 1 | groups of ranks for ensembles, `B / G` members each |
| `--overlap N` | off | levels `< N` with redundant halos (no exchange during the sweeps), for a few ranks |
| `--gpu` | off | one device per rank: local rank modulo the visible devices, or `STL_GPU_DEVICE` |
| `mpirun --bind-to none` | | let each rank use every core (OpenMPI binds ranks to one core by default) |

### Decomposition

Rank `r` owns a contiguous range of NESTED cells of level 0, aligned so that it also owns the
parents of its cells at every coarser level ([concepts](concepts.md#distributed-memory)).
Ranks may own no cell of the coarsest levels: the decomposition goes beyond the 192 cells of
nside 4. Before each sweep a rank receives the halos of its blocks from its neighbours; the
pieces it pushes onto a neighbour's cells are summed by the owner after the level. Coefficient
partial sums are reduced in rank order on every rank.

No rank reads or writes a full-sky map: inputs and outputs are read and written by owned slices,
so the memory per rank shrinks as `1/P` down to the block transients (80 to 110 MB at blocks of
16384) and the MPI runtime itself (OpenMPI 5 starts every rank at 190 to 350 MB, Intel MPI at
20 to 30 MB).

---

## Pitfalls and costs

### How many ranks

| nside | Typical | Measured |
|---|---|---|
| 64 | 1 to 28 | Datarmor, 28 ranks: 0.81 s / evaluation (`L = 2`), 7.7× on one node |
| 1024 | 48 to 192 | hpc, 2 × EPYC 9655: 24 ranks 60.9 s, 192 ranks 10.2 s (`L = 2`), 75 % efficiency; 192 ranks over 4 nodes cost 5 % |
| 2048 | GPUs | 3 T4 (3 ranks): 114 s at `L = 4`; one A100 for 10 members (`--gpu-fields recompute`) |

At small nside the work saturates when a rank holds only a few hundred cells of level 0 (beyond
64 ranks at nside 64): the halos of the small levels dominate.

### Groups of ranks for ensembles

`--ensemble B --groups G` splits the ranks into `G` groups of consecutive ranks; each group holds
`B / G` members (as one batch on its CPU or GPU engine) and the groups exchange one vector of
`len(plan)` complex numbers per evaluation. `G = 1` keeps every member on every rank (small
maps); `G = B` gives one member per group (large maps, `B` times the ranks or GPUs). Every `G`
gives the same synthesis to about 1e-12.

### Coarse-to-fine starts

With `--init` at a lower nside, every rank's range must fall on whole pixels of the init map; if
the program stops with "rank ranges are not aligned on its pixels", use fewer ranks or an init
closer in nside. At large nside chain `N/8 → N/4 → N/2 → N`, each step initialised by the
previous one (`--init`, `--init-noise auto`).

### Batch systems

Slurm, PBS and TGCC scripts are in `tools/hpc/`, `tools/datarmor/`, `tools/irene/`, and
explained in [USAGE](../USAGE.md) section 6, with the sizing rules in section 7. Typical pitfalls:
a conda `gcc` first in `PATH` (force `CC=/usr/bin/gcc`), csh login shells (send scripts with
`bash -s`), `--mem=0` refused where the site sets `--mem-per-cpu`.
