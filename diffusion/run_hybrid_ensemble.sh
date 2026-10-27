#!/bin/bash
# Hybrid ensemble: an STL ensemble at a coarse nside, the diffusion models
# for the finer levels of every member, then the STL ensemble minimisation
# at the target's nside from these maps, and a control minimisation from the
# coarse members plus white noise.
#   diffusion/run_hybrid_ensemble.sh STL_SYNTH TARGET.npy MODEL_DIR OUT_DIR COARSE_NSIDE B NITER_COARSE NITER_FINE
# RANKS thread ranks (default 3, stl-synth built with --features gpu, without
# mpi; --groups needs MPI), every rank holding its share of the sky of all
# the members, one GPU per rank round-robin.  PYTHON and FINE_OPTS (extra options of
# the two full-nside minimisations, e.g. --gpu-cache transposition) from the environment.
set -e
here=$(cd "$(dirname "$0")" && pwd)
stl=$1 target=$2 models=$3 out=$4 coarse=$5 B=$6 nc=$7 nf=$8
py=${PYTHON:-python}
mkdir -p "$out"
export STL_GPU_THREADS=${STL_GPU_THREADS:-1}
common=(--gpu --ranks "${RANKS:-3}" --ensemble "$B" --L 4 --weights ergodic --scatter-weight 1 --quiet)

# 1. STL ensemble at the coarse nside, with the level-power term so that its
#    finest level is right.
$py -c "
import sys; sys.path.insert(0, '$here')
import numpy as np, healpix_faces as hf
x = np.load('$target').astype(np.float64)
while hf.nside_of(x.size) > $coarse: x = hf.down(x)
np.save('$out/target_coarse.npy', x)"
[ -f "$out/coarse.json" ] || "$stl" "${common[@]}" --target "$out/target_coarse.npy" --niter "$nc" \
    --level-power-weight 10 --out "$out/coarse" > "$out/coarse.log" 2>&1

# 2. Diffusion levels above the coarse nside, one sample per member.
$py -c "
import numpy as np
np.save('$out/coarse_stack.npy', np.array([np.load('$out/coarse_m%03d_synthesis_nested.npy' % m) for m in range($B)]))"
nside=$($py -c "import numpy as np; print(int(round((np.load('$target', mmap_mode='r').size / 12) ** 0.5)))")
$py -u "$here/sample.py" "$target" "$models" "$nside" "$out/hybrid.npy" --coarse-nside "$coarse" \
    --coarse "$out/coarse_stack.npy" 2>&1 | grep --line-buffered -v Warning | tee "$out/sample.log"
$py -c "
import numpy as np
for m, x in enumerate(np.load('$out/hybrid.npy')): np.save('$out/hybrid_m%03d.npy' % m, x)"

# 3. STL ensemble minimisation at full nside from the hybrid maps, and the
#    control from the coarse members plus white noise.
[ -f "$out/polished.json" ] || "$stl" "${common[@]}" --target "$target" --niter "$nf" \
    --init "$out/hybrid_m{m}.npy" --init-noise 0 ${FINE_OPTS:-} --out "$out/polished" > "$out/polished.log" 2>&1
[ -f "$out/control.json" ] || "$stl" "${common[@]}" --target "$target" --niter "$nf" \
    --init "$out/coarse_m{m}_synthesis_nested.npy" --init-noise auto ${FINE_OPTS:-} --out "$out/control" > "$out/control.log" 2>&1
echo HYBRID-ENSEMBLE-DONE
