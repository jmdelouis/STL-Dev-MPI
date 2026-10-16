#!/usr/bin/env bash
# Regenerate the STL-Dev fixtures (see tools/dump_reference.py).
# STL_DEV defaults to the synced STL-Dev tree, STL_PYTHON to its CPU venv.
set -euo pipefail
cd "$(dirname "$0")/.."
STL_DEV="${STL_DEV:-$HOME/stl-work}"
STL_PYTHON="${STL_PYTHON:-$HOME/stl-deps/venv-cpu/bin/python}"
PYTHONPATH="$STL_DEV" "$STL_PYTHON" tools/dump_reference.py "${@:-geometry}"
