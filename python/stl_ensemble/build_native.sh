#!/usr/bin/env bash
set -euo pipefail
echo 'SCRIPT VERSION: build_native.sh 2026-10-28-v1'
port_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend="${1:-cpu}"
case "$backend" in cpu|gpu) ;; *) echo 'Usage: bash build_native.sh cpu|gpu' >&2; exit 2;; esac
command -v cargo >/dev/null || { echo 'Rust cargo is required; use an isolated toolchain or your cluster Rust module.' >&2; exit 1; }
cd "$port_dir/STL-Dev-MPI"
if [[ "$backend" == gpu ]]; then
    cargo build --release -p stl-bridge --features gpu
else
    cargo build --release -p stl-bridge
fi
mkdir -p "$port_dir/native"
cp target/release/libstl_bridge.so "$port_dir/native/libstl_bridge.so"
echo "Built $backend native bridge: $port_dir/native/libstl_bridge.so"
