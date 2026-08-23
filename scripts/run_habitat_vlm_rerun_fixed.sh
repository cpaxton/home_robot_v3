#!/usr/bin/env bash
# Re-run VLMs that were broken before IMAGE_DESCRIPTIONS + Qwen2.5 device_map fixes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
mkdir -p "$OUT"

QSTART="${QSTART:-0}"
QEND="${QEND:-9}"
MAX_PLANNING="${MAX_PLANNING:-20}"
MAX_MOVEMENT="${MAX_MOVEMENT:-10}"
SUFFIX="${SUFFIX:-_fixed}"

run_one() {
    local slug="$1" family="$2" hf_id="${3:-}"
    local tag="vlm_sweep_${slug}_q${QSTART}-${QEND}${SUFFIX}"
    local jsonl="${OUT}/${tag}.jsonl"
    local log="${OUT}/${tag}.log"
    echo "=== ${slug} -> ${jsonl}"
    local extra=()
    if [[ -n "$hf_id" ]]; then
        extra+=(--eqa-hf-model-id "$hf_id")
    fi
    rm -f "$jsonl"
    "$HAB" run-batch \
        --method static_graph \
        --question-start "$QSTART" \
        --question-end "$QEND" \
        --paper-subset \
        --max-planning-steps "$MAX_PLANNING" \
        --max-movement-step "$MAX_MOVEMENT" \
        --eqa-vl-family "$family" \
        --device cuda \
        --output "$jsonl" \
        "${extra[@]}" \
        2>&1 | tee "$log"
}

run_one gemma4_e4b gemma4 google/gemma-4-E4B-it
run_one qwen25_vl_3b qwen2_5_vl
run_one gemma3_4b gemma4 google/gemma-3-4b-it
run_one gemma4_e2b gemma4 google/gemma-4-e2b-it

echo "Done. Summarize:"
echo "  python3 ${ROOT}/scripts/summarize_vlm_sweep_fixed.py --q-start ${QSTART} --q-end ${QEND} --suffix ${SUFFIX}"
