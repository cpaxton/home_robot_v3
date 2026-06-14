#!/usr/bin/env bash
# HM-EQA frontier ablation (P4): fluid vs graph frontier nodes on Q0-19.
# Requires .venv-habitat, GPU, and HM-EQA assets.
#
# Arms (see docs/plans/2026-06-03_habitat_eqa_exploration_improvements.md):
#   A fluid      — no frontier nodes, keyword_weight=0 (time-heuristic fluid frontier only)
#   B fluid_kw   — no frontier nodes, keyword_weight=2 (question-biased fluid frontier)
#   C nodes      — frontier graph nodes + keyword_weight=2 (frontier v2 default)
#
# Usage:
#   ./scripts/run_habitat_frontier_ablation.sh
#   FAMILY=qwen2_5_vl HF_ID=Qwen/Qwen2.5-VL-3B-Instruct QSTART=0 QEND=19 ./scripts/run_habitat_frontier_ablation.sh
#   ARM=fluid_kw ./scripts/run_habitat_frontier_ablation.sh   # single arm
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
ARM="${ARM:-all}"

run_arm() {
  local tag="$1"
  shift
  local jsonl="${OUT}/ablation_${tag}_${FAMILY}_q${QSTART}-${QEND}.jsonl"
  local log="${OUT}/ablation_${tag}_${FAMILY}_q${QSTART}-${QEND}.log"
  echo "=== ablation arm: ${tag} ==="
  echo "  output: ${jsonl}"
  "$HAB" run-batch \
    --method "$METHOD" \
    --question-start "$QSTART" \
    --question-end "$QEND" \
    --paper-subset \
    --max-planning-steps "$MAX_PLANNING" \
    --max-movement-step "$MAX_MOVEMENT" \
    --eqa-vl-family "$FAMILY" \
    --eqa-hf-model-id "$HF_ID" \
    --device cuda \
    --resume \
    --output "$jsonl" \
    "$@" \
    2>&1 | tee "$log"
}

maybe_run() {
  local name="$1"
  shift
  if [[ "$ARM" == "all" || "$ARM" == "$name" ]]; then
    run_arm "$name" "$@"
  fi
}

maybe_run "fluid" --no-frontier-nodes --frontier-keyword-weight 0
maybe_run "fluid_kw" --no-frontier-nodes --frontier-keyword-weight 2
maybe_run "nodes" --frontier-nodes --frontier-keyword-weight 2

echo "Done. Summarize with:"
echo "  uv run python scripts/summarize_frontier_ablation.py --q-start ${QSTART} --q-end ${QEND}"
