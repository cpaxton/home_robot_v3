#!/usr/bin/env bash
# HM-EQA frontier exploration sweeps (graph_eqa). Requires .venv-habitat and a free GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
mkdir -p "$OUT"

QSTART="${QSTART:-0}"
QEND="${QEND:-19}"
METHOD="${METHOD:-graph_eqa}"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_PLANNING="${MAX_PLANNING:-20}"
MAX_MOVEMENT="${MAX_MOVEMENT:-10}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
EXTRA=()
if [[ -n "$HF_ID" ]]; then
    EXTRA+=(--eqa-hf-model-id "$HF_ID")
fi

TAG="frontier_v2_${FAMILY}_q${QSTART}-${QEND}${RUN_SUFFIX}"
LOG="${OUT}/${TAG}.log"
JSONL="${OUT}/${TAG}.jsonl"

echo "Running ${METHOD} ${FAMILY} questions ${QSTART}-${QEND} (planning=${MAX_PLANNING} movement=${MAX_MOVEMENT})"
echo "  output:   ${JSONL}"
echo "  manifest: ${OUT}/${TAG}_manifest.json"
echo "  log:      ${LOG}"
echo "  episodes: ${HOME}/.cache/habitat_eqa/episodes/${TAG}/"

exec "$HAB" run-batch \
    --method "$METHOD" \
    --question-start "$QSTART" \
    --question-end "$QEND" \
    --paper-subset \
    --max-planning-steps "$MAX_PLANNING" \
    --max-movement-step "$MAX_MOVEMENT" \
    --eqa-vl-family "$FAMILY" \
    --device cuda \
    --resume \
    --output "$JSONL" \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG"
