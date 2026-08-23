#!/usr/bin/env bash
# Time-limited HM-EQA iteration on a fixed random subset (P0/P2 debugging).
# Usage:
#   ./scripts/run_habitat_iter_subset.sh                 # default subset + tag
#   TAG=iter2 IDS=3,14,17 TIMEOUT=900 ./scripts/run_habitat_iter_subset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
mkdir -p "$OUT"

IDS="${IDS:-3,14,17,28,31,35,81,94}"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"
METHOD="${METHOD:-graph_eqa}"
MAX_PLANNING="${MAX_PLANNING:-20}"
MAX_MOVEMENT="${MAX_MOVEMENT:-10}"
TAG="${TAG:-iter}"
TIMEOUT="${TIMEOUT:-1500}"
FRONTIER_FLAG="${FRONTIER_FLAG:---frontier-nodes}"
KW="${KW:-2}"

JSONL="${OUT}/subset_${TAG}_${FAMILY}.jsonl"
LOG="${OUT}/subset_${TAG}_${FAMILY}.log"

echo "subset ids: ${IDS}"
echo "  output: ${JSONL}"
echo "  log:    ${LOG}"
echo "  timeout: ${TIMEOUT}s  frontier: ${FRONTIER_FLAG} kw=${KW}"

timeout "${TIMEOUT}" "$HAB" run-batch \
    --method "$METHOD" \
    --question-ids "$IDS" \
    --max-planning-steps "$MAX_PLANNING" \
    --max-movement-step "$MAX_MOVEMENT" \
    --eqa-vl-family "$FAMILY" \
    --eqa-hf-model-id "$HF_ID" \
    --device cuda \
    ${FRONTIER_FLAG} \
    --frontier-keyword-weight "$KW" \
    --resume \
    --output "$JSONL" \
    2>&1 | tee "$LOG"
