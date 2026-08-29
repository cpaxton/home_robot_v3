#!/usr/bin/env bash
# Paper matrix: HM-EQA (no semantics) + OVMM find across environments.
#
# Runs the full paper numbers in one ``emet jobs`` run (serialized on the GPU
# lock), so we can compare against prior art:
#   - HM-EQA paper-113 on the shared agentic loop, NO hm3d semantics
#     (prior art GraphEQA used habitat semantics + external tools).
#   - OVMM find on multiple environments (S0 table, Robocasa S1, Molmo S2).
#
# Usage:
#   uv run emet jobs run --name paper-matrix --need-mib 12000 --gpu-exclusive -- \
    #     ./scripts/run_paper_matrix.sh
#
# Env:
#   RUN_HMEQA    "1"/"0" (default 1) — paper-113 no-semantics (METHODS from HAB_METHODS)
#   RUN_OVMM     "1"/"0" (default 1) — OVMM find multi-env sweep (S1+S2)
#   RUN_OVMM_S0  "1"/"0" (default 1) — OVMM find S0 table (rby1 gate)
#   HAB_METHODS  HM-EQA methods (default "dynagraph")
#   OVMM_PRESET  OVMM sweep preset (default molmo-robocasa)
#   RUN_ID       base run id; phases get suffixes

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_HMEQA="${RUN_HMEQA:-1}"
RUN_OVMM="${RUN_OVMM:-1}"
RUN_OVMM_S0="${RUN_OVMM_S0:-1}"
HAB_METHODS="${HAB_METHODS:-dynagraph}"
OVMM_PRESET="${OVMM_PRESET:-molmo-robocasa}"
OUT_BASE="${OUT_DIR:-$HOME/runs/emet/paper_matrix/${RUN_ID}}"
mkdir -p "$OUT_BASE"

echo "=== paper matrix ${RUN_ID} ==="
printf '{"run_id":"%s","run_hmeqa":%s,"run_ovmm":%s,"run_ovmm_s0":%s,"hab_methods":"%s","ovmm_preset":"%s"}\n' \
    "$RUN_ID" "$RUN_HMEQA" "$RUN_OVMM" "$RUN_OVMM_S0" "$HAB_METHODS" "$OVMM_PRESET" > "$OUT_BASE/meta.json"

HMEQA_OK="skipped"
OVMM_OK="skipped"
OVMM_S0_OK="skipped"

if [[ "$RUN_HMEQA" == "1" ]]; then
    echo "--- phase 1/3: HM-EQA paper-113 no-semantics (METHODS=${HAB_METHODS}) ---"
    set +e
    # flash-attn is not installed here; the VLM must load via PyTorch SDPA.
    # Without EMET_ALLOW_SDPA_ATTN=1 every episode fails at LLM client init
    # (batch summary accuracy 0.0, all qids 'failed: Flash-Attn 2 required').
    env EMET_ALLOW_SDPA_ATTN=1 RUN_ID="${RUN_ID}_hmeqa" METHODS="$HAB_METHODS" OUT_DIR="$OUT_BASE/hmeqa" \
        ./scripts/run_hmeqa_paper113_h2h.sh
    rc=$?
    set -e
    HMEQA_OK="pass"
    [[ "$rc" -eq 0 ]] || HMEQA_OK="fail"
    echo "HMEQA phase rc=${rc} -> ${HMEQA_OK}"
fi

if [[ "$RUN_OVMM" == "1" ]]; then
    echo "--- phase 2/3: OVMM find multi-env sweep (preset=${OVMM_PRESET}) ---"
    set +e
    uv run emet ovmm sweep --preset "$OVMM_PRESET" --out "$OUT_BASE/ovmm_sweep" --backend dynagraph
    rc=$?
    set -e
    OVMM_OK="pass"
    [[ "$rc" -eq 0 ]] || OVMM_OK="fail"
    echo "OVMM sweep rc=${rc} -> ${OVMM_OK}"
fi

if [[ "$RUN_OVMM_S0" == "1" ]]; then
    echo "--- phase 3/3: OVMM find S0 table (rby1 gate) ---"
    set +e
    env PROFILE=slice OUT_DIR="$OUT_BASE/ovmm_s0" RUN_ID="${RUN_ID}_ovmm_s0" \
        ./scripts/run_ovmm_find_recep_slice.sh
    rc=$?
    set -e
    OVMM_S0_OK="pass"
    [[ "$rc" -eq 0 ]] || OVMM_S0_OK="fail"
    echo "OVMM S0 rc=${rc} -> ${OVMM_S0_OK}"
fi

printf '{"hmeqa":"%s","ovmm":"%s","ovmm_s0":"%s"}\n' "$HMEQA_OK" "$OVMM_OK" "$OVMM_S0_OK" | tee "$OUT_BASE/summary.json"
echo "=== paper matrix ${RUN_ID} done: hmeqa=${HMEQA_OK} ovmm=${OVMM_OK} ovmm_s0=${OVMM_S0_OK} (OUT=${OUT_BASE}) ==="
exit 0
