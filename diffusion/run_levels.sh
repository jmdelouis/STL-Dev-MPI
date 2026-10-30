#!/bin/bash
# Train per-level models one after the other on one GPU.
#   diffusion/run_levels.sh TARGET.npy RUN_DIR STEPS LEVEL:CROP:CHANNELS[:BATCH] [...]
# e.g. CUDA_VISIBLE_DEVICES=0 diffusion/run_levels.sh g256.npy runs/a 10000 16:8:32,64,64 32:16:64,128,128
# PYTHON (default python) and TRAIN_OPTS (extra train.py options) from the environment.
set -e
here=$(cd "$(dirname "$0")" && pwd)
target=$1 dir=$2 steps=$3
shift 3
mkdir -p "$dir"
[ -d /usr/lib/wsl/lib ] && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
for spec in "$@"; do
    IFS=: read -r level crop channels batch <<< "$spec"
    ${PYTHON:-python} -u "$here/train.py" "$target" "$level" "$dir/level$level.pt" --steps "$steps" --crop "$crop" \
        --channels "$channels" --batch "${batch:-64}" --eval-every 500 ${TRAIN_OPTS:-} 2>&1 \
        | grep --line-buffered -v Warning | tee "$dir/level$level.log"
done
