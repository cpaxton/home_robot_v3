#!/usr/bin/env bash
# Paper-facing eval battery with tuned Dynagraph harness (sequential GPU jobs).
#
# Tracks: seven-track sim smoke gate, Habitat holdout+balanced-32,
# Habitat OVMM dynagraph agentic find (Phase 0 already ran Habitat OVMM GT),
# Robocasa/Molmo OVMM find-phase (dynagraph), SQA3D val q0-30.
#
# Usage:
#   nohup ./scripts/run_dynagraph_tuned_paper_battery.sh >> ~/runs/emet/dynagraph_tuning/nohup_battery.log 2>&1 &
#
# Env:
#   RUN_ID           default tuned_paper_YYYYMMDD_HHMMSS
#   SKIP_SMOKE       1 to skip seven-track battery
#   SKIP_HABITAT_DEEP 1 to skip holdout+bal32
#   SKIP_OVMM        1 to skip OVMM legs
#   SKIP_SQA3D       1 to skip SQA3D val sweep
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-tuned_paper_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${HOME}/runs/emet/dynagraph_tuning/${RUN_ID}"
mkdir -p "$LOG_ROOT"
SUMMARY="${LOG_ROOT}/summary.txt"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
FAMILY="qwen3_vl"
HF_ID="Qwen/Qwen3-VL-8B-Instruct"

echo "[$(date -Is)] ========== TUNED PAPER BATTERY run_id=${RUN_ID} ==========" | tee "$SUMMARY"

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    echo "[$(date -Is)] Phase 0: seven-track simulation smoke" | tee -a "$SUMMARY"
    RUN_ID="sim_smoke_${RUN_ID}" "${ROOT}/scripts/run_simulation_smoke_battery.sh" \
        2>&1 | tee "${LOG_ROOT}/sim_smoke_battery.log" || true
    uv run python "${ROOT}/scripts/inspect_simulation_smoke_battery.py" \
        --run-id "sim_smoke_${RUN_ID}" --write-report \
        2>&1 | tee -a "${LOG_ROOT}/sim_smoke_inspection.log" || true
fi

if [[ "${SKIP_HABITAT_DEEP:-0}" != "1" ]]; then
    echo "[$(date -Is)] Phase 1: Habitat holdout-8 (tuned dynagraph)" | tee -a "$SUMMARY"
    emet_kill_stale_eval_processes || true
    NEED_MIB="${NEED_MIB:-12000}" emet_gpu_wait_mib || true
    TAG="${RUN_ID}_holdout8" METHOD=dynagraph IDS="15,56,65,68,79,88,104,105" TIMEOUT=14400 \
        "${ROOT}/scripts/run_habitat_iter_subset.sh" \
        2>&1 | tee "${LOG_ROOT}/holdout8.log" || true

    echo "[$(date -Is)] Phase 2: Habitat balanced-32 (tuned dynagraph)" | tee -a "$SUMMARY"
    emet_kill_stale_eval_processes || true
    NEED_MIB="${NEED_MIB:-12000}" emet_gpu_wait_mib || true
    TAG="${RUN_ID}_bal32" METHOD=dynagraph \
        IDS="2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84" \
        TIMEOUT=21600 \
        "${ROOT}/scripts/run_habitat_iter_subset.sh" \
        2>&1 | tee "${LOG_ROOT}/balanced32.log" || true
fi

if [[ "${SKIP_OVMM:-0}" != "1" ]]; then
    echo "[$(date -Is)] Phase 3: Habitat OVMM dynagraph agentic find" | tee -a "$SUMMARY"
    # Product path on HM3D (not a second GT check — that is Phase 0 track 2).
    # Cap 4/4 like overnight / habitat smoke so timeout 3600 cannot eat a full 8/8 VLM episode.
    timeout 3600 "$HAB" run-ovmm-find-episode \
        --episode-id hm3d_lamp_bed_00006 \
        --backend dynagraph --device cuda \
        --agentic-find --agentic-max-rounds 4 --agentic-max-nav-steps 4 \
        --output "${LOG_ROOT}/habitat_ovmm_dynagraph.json" \
        2>&1 | tee "${LOG_ROOT}/habitat_ovmm.log" || true

    echo "[$(date -Is)] Phase 4: Robocasa OVMM find S1 (dynagraph)" | tee -a "$SUMMARY"
    uv run python "${ROOT}/scripts/eval_ovmm_find_phases.py" \
        --episode-id robocasa_pp_s1 \
        --backend dynagraph --device cuda \
        --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_robocasa" \
        2>&1 | tee "${LOG_ROOT}/robocasa_ovmm.log" || true

    echo "[$(date -Is)] Phase 5: Molmo iTHOR OVMM find S2 (dynagraph)" | tee -a "$SUMMARY"
    uv run python "${ROOT}/scripts/eval_ovmm_find_phases.py" \
        --episode-id molmo_ithor_s2_idx0 \
        --backend dynagraph --device cuda \
        --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_molmo" \
        2>&1 | tee "${LOG_ROOT}/molmo_ovmm.log" || true
fi

if [[ "${SKIP_SQA3D:-0}" != "1" ]]; then
    echo "[$(date -Is)] Phase 6: SQA3D val q0-30 (dynagraph, tuned harness)" | tee -a "$SUMMARY"
    emet_kill_stale_eval_processes || true
    NEED_MIB="${NEED_MIB:-12000}" emet_gpu_wait_mib || true
    uv run emet sqa3d run-real-sweep \
        --split val --question-start 0 --question-end 30 \
        --method dynagraph --profile tuned \
        --eqa-vl-family qwen3_vl --eqa-hf-model-id "$HF_ID" \
        --device cuda \
        --output-dir "${HOME}/runs/emet/sqa3d/${RUN_ID}_val_q0_30" \
        2>&1 | tee "${LOG_ROOT}/sqa3d_val.log" || true
fi

echo "[$(date -Is)] Phase 7: figure pack" | tee -a "$SUMMARY"
uv run python "${ROOT}/scripts/build_eval_figure_pack.py" \
    --run-id "$RUN_ID" \
    --output-dir "${LOG_ROOT}/figures" \
    2>&1 | tee "${LOG_ROOT}/figures.log" || true

echo "[$(date -Is)] ========== TUNED PAPER BATTERY END ==========" | tee -a "$SUMMARY"
echo "artifacts: ${LOG_ROOT}"
