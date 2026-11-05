#!/bin/bash
# Builds the documentation site into docs/_build/html, with the rustdoc of
# the workspace under docs/_build/html/rust.  Warnings are errors, in both
# rustdoc and Sphinx (-W -n, as GRID4EARTH/healpix-resample).
#   pip install -r docs/requirements.txt
#   docs/build.sh            # add --no-rustdoc to skip cargo doc
# The notebooks of docs/tutorials are rendered with their committed outputs,
# never executed here.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${CARGO_TARGET_DIR:-target}
if [ "${1:-}" != "--no-rustdoc" ]; then
    # The default features (no MPI, no CUDA driver needed to document stl-gpu).
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --no-deps --workspace
fi
rm -rf docs/_build/html docs/api/generated
python -m sphinx -W -n --keep-going -b html docs docs/_build/html
if [ -d "$TARGET/doc" ]; then
    rm -rf docs/_build/html/rust
    cp -r "$TARGET/doc" docs/_build/html/rust
fi
echo "open docs/_build/html/index.html"
