#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
axis_log="$(mktemp)"
trap 'rm -f "$axis_log"' EXIT

batch="${BATCH:-256}"
embedding="${EMBEDDING:-64}"
heads="${HEADS:-2}"
layers="${LAYERS:-4}"
steps="${STEPS:-12}"

bash "$root/scripts/train.sh" \
  --steps "$steps" --batch "$batch" --eval-size 1 --eval-batch 1 \
  --log-every "$steps" --embedding "$embedding" --heads "$heads" \
  --layers "$layers" --profile-steps | tee "$axis_log"

python3 - "$axis_log" "$batch" "$embedding" "$heads" "$layers" <<'PY'
import json, pathlib, re, statistics, sys
values = [float(x) for x in re.findall(r"trainer_ms=([0-9.]+)", pathlib.Path(sys.argv[1]).read_text())]
if len(values) < 3:
    raise SystemExit("Axis benchmark needs at least three timed steps")
steady = values[2:]
print(json.dumps({
    "backend": "axis",
    "dtype": "fp32",
    "batch": int(sys.argv[2]),
    "embedding": int(sys.argv[3]),
    "heads": int(sys.argv[4]),
    "layers": int(sys.argv[5]),
    "discarded_warmup_steps": 2,
    "median_step_ms": statistics.median(steady),
    "min_step_ms": min(steady),
    "steps": len(steady),
}, sort_keys=True))
PY

python3 "$root/scripts/perf/torch_sudoku.py" \
  --batch "$batch" --embedding "$embedding" --heads "$heads" \
  --layers "$layers" --warmup 2 --steps "$((steps - 2))"
