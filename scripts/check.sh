#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v "$root/scripts/test_setup_cuda.py"
bash "$root/scripts/cargo.sh" fmt --all -- --check
bash "$root/scripts/cargo.sh" clippy --release --locked --all-targets -- -D warnings
bash "$root/scripts/cargo.sh" test --release --locked
bash "$root/scripts/train.sh" --smoke
