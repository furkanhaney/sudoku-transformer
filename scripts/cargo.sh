#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-$root/.cuda}"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo_command="${1:?usage: cargo.sh <command> [arguments...]}"
shift
exec cargo "$cargo_command" --manifest-path "$root/Cargo.toml" "$@"
