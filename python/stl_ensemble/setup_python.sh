#!/usr/bin/env bash
set -euo pipefail
echo 'SCRIPT VERSION: setup_python.sh 2026-10-28-v1'
port_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -e "$port_dir/.venv" ]]; then
    echo 'The isolated .venv already exists; leaving it intact. Use its Python or create a fresh package directory.' >&2
    exit 1
fi
python3 -m venv "$port_dir/.venv"
"$port_dir/.venv/bin/python" -m pip install -r "$port_dir/requirements.txt"
echo "Python ready: $port_dir/.venv/bin/python"
