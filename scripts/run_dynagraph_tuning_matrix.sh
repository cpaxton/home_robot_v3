#!/usr/bin/env bash
# Dynagraph HM-EQA ablation matrix on fixed slices (holdout-8 + canonical-8).
# Uses tuned habitat_eqa harness defaults; per-arm CLI overrides for ablations.
#
# Usage:
#   nohup ./scripts/run_dynagraph_tuning_matrix.sh >> ~/runs/emet/dynagraph_tuning/nohup.log 2>&1 &
#
# Env:
#   RUN_ID          default dynagraph_tune_YYYYMMDD_HHMMSS
#   ARMS            comma list: baseline,no_debias,no_memory,no_explore,graph_eqa_like,all_off
#   TIMEOUT         per-arm seconds (default 14400)
#   NEED_MIB        GPU preflight (default 12000)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-dynagraph_tune_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${HOME}/runs/emet/dynagraph_tuning/${RUN_ID}"
mkdir -p "$LOG_ROOT"
SUMMARY="${LOG_ROOT}/summary.txt"
ARMS="${ARMS:-baseline,no_debias,no_memory,no_explore,graph_eqa_like}"
TIMEOUT="${TIMEOUT:-14400}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HOLDOUT_IDS="15,56,65,68,79,88,104,105"
CANONICAL_IDS="3,14,17,28,31,35,81,94"
METHOD="dynagraph"

echo "[$(date -Is)] ========== DYNAGRAPH TUNING MATRIX run_id=${RUN_ID} ==========" | tee -a "$SUMMARY"
echo "arms=${ARMS}" | tee -a "$SUMMARY"

arm_flags() {
    local arm="$1"
    case "$arm" in
        baseline)
            echo ""
            ;;
        no_debias)
            echo "--no-mcq-debias"
            ;;
        no_memory)
            echo "--no-memory-summary"
            ;;
        no_explore)
            echo "--explore-when-uncovered off"
            ;;
        graph_eqa_like)
            echo "--no-mcq-debias --no-memory-summary --explore-when-uncovered off"
            ;;
        all_off)
            echo "--no-mcq-debias --no-memory-summary --explore-when-uncovered off"
            ;;
        *)
            echo "unknown arm ${arm}" >&2
            return 1
            ;;
    esac
}

score_jsonl() {
    local jsonl="$1"
    if [[ ! -f "$jsonl" ]]; then
        echo "0/0"
        return
    fi
    uv run python - <<PY
import json
from pathlib import Path
p = Path("${jsonl}")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
ok = sum(1 for r in rows if r.get("correct"))
print(f"{ok}/{len(rows)}")
PY
}

run_slice() {
    local arm="$1"
    local slice="$2"
    local ids="$3"
    local extra
    extra="$(arm_flags "$arm")"
    local tag="${RUN_ID}_${arm}_${slice}"
    local jsonl="${OUT}/subset_${tag}_${FAMILY}.jsonl"
    local log="${OUT}/subset_${tag}_${FAMILY}.log"
    echo "[$(date -Is)] arm=${arm} slice=${slice} ids=${ids}" | tee -a "$SUMMARY"
    emet_kill_stale_eval_processes || true
    NEED_MIB="${NEED_MIB:-12000}" emet_gpu_wait_mib || true
    # shellcheck disable=SC2086
    timeout "${TIMEOUT}" "$HAB" run-batch \
        --method "$METHOD" \
        --question-ids "$ids" \
        --max-planning-steps 20 \
        --max-movement-step 10 \
        --eqa-vl-family "$FAMILY" \
        --eqa-hf-model-id "$HF_ID" \
        --device cuda \
        --frontier-nodes \
        --frontier-keyword-weight 2 \
        --resume \
        $extra \
        --output "$jsonl" \
        2>&1 | tee "$log" || true
    emet_kill_stale_eval_processes || true
    local score
    score="$(score_jsonl "$jsonl")"
    echo "  ${slice}: ${score} -> ${jsonl}" | tee -a "$SUMMARY"
    cp -f "$jsonl" "${LOG_ROOT}/${arm}_${slice}.jsonl" 2>/dev/null || true
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    arm="$(echo "$arm" | xargs)"
    [[ -z "$arm" ]] && continue
    run_slice "$arm" holdout8 "$HOLDOUT_IDS"
    run_slice "$arm" canonical8 "$CANONICAL_IDS"
done

echo "[$(date -Is)] ========== DYNAGRAPH TUNING MATRIX END ==========" | tee -a "$SUMMARY"
echo "summary: ${SUMMARY}"
