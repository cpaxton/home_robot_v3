#!/usr/bin/env bash
# Re-run the existing WBC clean-suite benchmark against the recorded baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CHECKPOINTS="${CHECKPOINTS:-logs/eval/wbc_multirobot_20261126}"
OUT="${OUT:-/tmp/wbc_benchmark_regression}"
EPISODES="${EPISODES:-50}"
SEEDS="${SEEDS:-30000,40000,50000}"
ROBOTS="${ROBOTS:-mars stretch sourccey rby1 pandaomron}"

mkdir -p "$OUT"
for robot in $ROBOTS; do
  echo "=== WBC benchmark: $robot ==="
  "$PYTHON" scripts/eval_wbc_policy.py \
    --robot "$robot" \
    --checkpoint "$CHECKPOINTS/$robot/best.pt" \
    --output "$OUT/$robot" \
    --episodes "$EPISODES" \
    --seeds "$SEEDS" \
    --device cpu \
    --allow-missing-contract
 done

"$PYTHON" scripts/compare_wbc_benchmark.py \
  --baseline "$CHECKPOINTS/manifest.json" \
  --results "$OUT" \
  --episodes "$EPISODES" \
  --robots "$ROBOTS" \
  --seeds "$SEEDS"
