#!/usr/bin/env bash
# Run the currently supported learned HM-EQA paper rows with fixed, auditable
# settings.  DynaMem and the oracle are intentionally not silently substituted:
# their HM-EQA adapters must be implemented before they join this launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_paper/${RUN_ID}}"
ROWS="${ROWS:-dynamem_voxel lazy_arrival dynagraph_bounded_instance dynagraph_no_instance static_no_instance}"
QIDS="${QIDS:-}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
mkdir -p "$OUT_DIR"

# This local environment has no Flash-Attn wheel; the paper manifest records
# model/runtime details, and remote fp16 runs may override this setting.
export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"

common=(run-batch --paper-subset --no-hm3d-semantics --max-planning-steps 20 --max-movement-step 10 \
    --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct --device cuda --frontier-nodes)
if [[ -n "$QIDS" ]]; then
    common+=(--question-ids "$QIDS")
fi

run_row() {
    local row="$1" method="" fusion=""
    case "$row" in
        dynamem_voxel) method="dynamem" ;;
        lazy_arrival) method="lazy_graph" ;;
        dynagraph_bounded_instance) method="dynagraph"; fusion="$ROOT/configs/benchmarks/fusion_strategy_b.yaml" ;;
        dynagraph_no_instance) method="dynagraph"; fusion="$ROOT/configs/benchmarks/fusion_strategy_c.yaml" ;;
        static_no_instance) method="static_graph" ;;
        *) echo "unknown supported HM-EQA paper row: $row" >&2; return 2 ;;
    esac
    local output="$OUT_DIR/${row}.jsonl"
    echo "[$(date -Is)] row=$row method=$method output=$output" >&2
    if [[ -n "$fusion" ]]; then
        EMET_GRAPH_FUSION_CONFIG="$fusion" "$HAB" "${common[@]}" --method "$method" --output "$output"
    else
        "$HAB" "${common[@]}" --method "$method" --output "$output"
    fi
}

for row in $ROWS; do
    run_row "$row"
done

echo "paper HM-EQA rows complete: $OUT_DIR" >&2
