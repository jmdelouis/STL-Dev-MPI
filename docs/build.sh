#!/bin/bash
# Builds the documentation site into docs/_build/html, with the rustdoc of
# the workspace under docs/_build/html/rust.
#   pip install -r docs/requirements.txt
#   docs/build.sh            # add --no-rustdoc to skip cargo doc
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${CARGO_TARGET_DIR:-target}
if [ "${1:-}" != "--no-rustdoc" ]; then
    # The default features (no MPI, no CUDA driver needed to document stl-gpu).
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --no-deps --workspace
fi
sphinx-build -b html docs docs/_build/html
if [ -d "$TARGET/doc" ]; then
    rm -rf docs/_build/html/rust
    cp -r "$TARGET/doc" docs/_build/html/rust
fi
echo "open docs/_build/html/index.html"
