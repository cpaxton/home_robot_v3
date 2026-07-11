#!/usr/bin/env bash
# Representative cross-benchmark sample (all paper tracks, subset sizes).
#
# Covers: Habitat HM-EQA (graph_eqa comparison), Habitat OVMM find, OVMM sim S0/S1/S2,
# SQA3D val q0-10, dynamic exploration smoke, backend localization figure.
# Ingests completed dynagraph tuning matrix via build_representative_results_tables.py.
#
# Usage:
#   nohup ./scripts/run_representative_benchmark_sample.sh >> ~/runs/emet/representative_sample/nohup.log 2>&1 &
#
# Env:
#   RUN_ID              default rep_sample_YYYYMMDD_HHMMSS
#   TUNING_RUN_ID       dynagraph ablation run to ingest (default dynagraph_tune_20260706_110513)
#   SKIP_HABITAT        1 to skip Habitat HM-EQA graph_eqa comparison runs
#   SKIP_OVMM           1 to skip sim + Habitat OVMM legs
#   SKIP_SQA3D          1 to skip SQA3D val slice
#   SKIP_DYNAMIC        1 to skip dynamic exploration smoke
#   SKIP_BACKEND_FIG    1 to skip backend localization figure
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-rep_sample_$(date +%Y%m%d_%H%M%S)}"
TUNING_RUN_ID="${TUNING_RUN_ID:-dynagraph_tune_20260706_110513}"
LOG_ROOT="${HOME}/runs/emet/representative_sample/${RUN_ID}"
mkdir -p "$LOG_ROOT"
SUMMARY="${LOG_ROOT}/summary.txt"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
HAB_OUT="${HOME}/.cache/habitat_eqa/results"
FAMILY="qwen3_vl"
HF_ID="Qwen/Qwen3-VL-8B-Instruct"
HOLDOUT_IDS="15,56,65,68,79,88,104,105"
CANONICAL_IDS="3,14,17,28,31,35,81,94"

log() { echo "[$(date -Is)] $*" | tee -a "$SUMMARY"; }

gpu_step() {
  NEED_MIB="${NEED_MIB:-12000}" emet_gpu_between_steps "$NEED_MIB"
}

log "========== REPRESENTATIVE BENCHMARK SAMPLE run_id=${RUN_ID} tuning=${TUNING_RUN_ID} =========="

# --- Habitat HM-EQA: graph_eqa comparison on fixed slices (dynagraph ablations from TUNING_RUN_ID) ---
if [[ "${SKIP_HABITAT:-0}" != "1" ]] && [[ -x "$HAB" ]]; then
  gpu_step
  log "Track A: Habitat HM-EQA graph_eqa holdout-8"
  timeout 14400 "$HAB" run-batch \
    --method graph_eqa \
    --question-ids "$HOLDOUT_IDS" \
    --max-planning-steps 20 --max-movement-step 10 \
    --eqa-vl-family "$FAMILY" --eqa-hf-model-id "$HF_ID" \
    --device cuda --frontier-nodes --frontier-keyword-weight 2 --resume \
    --output "${HAB_OUT}/subset_${RUN_ID}_holdout8_graph_eqa_${FAMILY}.jsonl" \
    2>&1 | tee "${LOG_ROOT}/habitat_holdout8_graph_eqa.log" || true

  gpu_step
  log "Track A: Habitat HM-EQA graph_eqa canonical-8"
  timeout 14400 "$HAB" run-batch \
    --method graph_eqa \
    --question-ids "$CANONICAL_IDS" \
    --max-planning-steps 20 --max-movement-step 10 \
    --eqa-vl-family "$FAMILY" --eqa-hf-model-id "$HF_ID" \
    --device cuda --frontier-nodes --frontier-keyword-weight 2 --resume \
    --output "${HAB_OUT}/subset_${RUN_ID}_canonical8_graph_eqa_${FAMILY}.jsonl" \
    2>&1 | tee "${LOG_ROOT}/habitat_canonical8_graph_eqa.log" || true

  gpu_step
  log "Track A: Habitat HM-EQA tuned dynagraph holdout-8 (harness default)"
  timeout 14400 "$HAB" run-batch \
    --method dynagraph \
    --question-ids "$HOLDOUT_IDS" \
    --max-planning-steps 20 --max-movement-step 10 \
    --eqa-vl-family "$FAMILY" --eqa-hf-model-id "$HF_ID" \
    --device cuda --frontier-nodes --frontier-keyword-weight 2 --resume \
    --output "${HAB_OUT}/subset_${RUN_ID}_holdout8_dynagraph_${FAMILY}.jsonl" \
    2>&1 | tee "${LOG_ROOT}/habitat_holdout8_dynagraph.log" || true
else
  log "SKIP Habitat HM-EQA (SKIP_HABITAT=${SKIP_HABITAT:-0})"
fi

# --- Habitat OVMM find ---
if [[ "${SKIP_OVMM:-0}" != "1" ]] && [[ -x "$HAB" ]]; then
  gpu_step
  log "Track B: Habitat OVMM find (dynagraph + graph_eqa)"
  mkdir -p "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}"
  for backend in dynagraph graph_eqa; do
    timeout 3600 "$HAB" run-ovmm-find-episode \
      --episode-id hm3d_lamp_bed_00006 \
      --backend "$backend" --device cuda \
      --output "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}/hm3d_lamp_bed_00006_${backend}.json" \
      2>&1 | tee "${LOG_ROOT}/habitat_ovmm_${backend}.log" || true
    gpu_step
  done
fi

# --- OVMM sim find-phase: S0 all backends, S1 Robocasa, S2 Molmo (dynagraph) ---
if [[ "${SKIP_OVMM:-0}" != "1" ]]; then
  gpu_step
  log "Track C: OVMM sim S0 (all backends)"
  uv run python "${ROOT}/scripts/eval_ovmm_find_phases.py" \
    --tier S0 \
    --backend dynamem --backend graph_eqa --backend dynagraph --backend ground_truth \
    --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_s0" \
    2>&1 | tee "${LOG_ROOT}/ovmm_s0.log" || true

  gpu_step
  log "Track C: OVMM sim S1 Robocasa (dynagraph)"
  uv run python "${ROOT}/scripts/eval_ovmm_find_phases.py" \
    --episode-id robocasa_pp_s1 \
    --backend dynagraph \
    --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_robocasa" \
    2>&1 | tee "${LOG_ROOT}/ovmm_robocasa.log" || true

  gpu_step
  log "Track C: OVMM sim S2 Molmo iTHOR (dynagraph)"
  uv run python "${ROOT}/scripts/eval_ovmm_find_phases.py" \
    --episode-id molmo_ithor_s2_idx0 \
    --backend dynagraph \
    --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_molmo" \
    2>&1 | tee "${LOG_ROOT}/ovmm_molmo.log" || true
fi

# --- SQA3D val q0-10 ---
if [[ "${SKIP_SQA3D:-0}" != "1" ]]; then
  gpu_step
  log "Track D: SQA3D val q0-10 (dynagraph + dynamem)"
  for method in dynagraph dynamem; do
    uv run emet sqa3d run-real-sweep \
      --split val --question-start 0 --question-end 10 \
      --method "$method" --profile tuned \
      --eqa-vl-family "$FAMILY" --eqa-hf-model-id "$HF_ID" \
      --device cuda \
      --output-dir "${HOME}/runs/emet/sqa3d/${RUN_ID}_val_q0_10_${method}" \
      2>&1 | tee "${LOG_ROOT}/sqa3d_${method}.log" || true
    gpu_step
  done
fi

# --- Dynamic exploration smoke (dynagraph + graph_eqa) ---
if [[ "${SKIP_DYNAMIC:-0}" != "1" ]]; then
  gpu_step
  log "Track E: dynamic exploration smoke"
  DYN_OUT="${HOME}/runs/emet/dynamic_exploration/${RUN_ID}_explore"
  mkdir -p "$DYN_OUT"
  for backend in dynagraph graph_eqa; do
    timeout 7200 uv run python "${ROOT}/scripts/eval_dynamic_exploration.py" \
      --smoke --backend "$backend" \
      --output-dir "${DYN_OUT}/${backend}" \
      2>&1 | tee "${LOG_ROOT}/dynamic_explore_${backend}.log" || true
    gpu_step
  done
fi

# --- Backend localization figure (quick) ---
if [[ "${SKIP_BACKEND_FIG:-0}" != "1" ]]; then
  gpu_step
  log "Track F: backend localization figure (--quick)"
  uv run python "${ROOT}/scripts/smoke_backend_localization_figure.py" \
    --quick \
    --output-dir "${LOG_ROOT}/backend_localization" \
    2>&1 | tee "${LOG_ROOT}/backend_localization.log" || true
fi

# --- Results tables + figures ---
log "Track G: build representative results tables"
uv run python "${ROOT}/scripts/build_representative_results_tables.py" \
  --run-id "$RUN_ID" \
  --tuning-run-id "$TUNING_RUN_ID" \
  --output-dir "${LOG_ROOT}/tables" \
  2>&1 | tee "${LOG_ROOT}/tables.log" || true

log "Track H: figure pack"
uv run python "${ROOT}/scripts/build_eval_figure_pack.py" \
  --run-id "$RUN_ID" \
  --output-dir "${LOG_ROOT}/figures" \
  --render-retrieval-panels \
  2>&1 | tee "${LOG_ROOT}/figures.log" || true

# Map figures from tuning matrix episode bundles
  if [[ -d "${HOME}/runs/emet/dynagraph_tuning/${TUNING_RUN_ID}" ]]; then
  "${ROOT}/.venv-habitat/bin/python" "${ROOT}/scripts/render_paper_map_figures.py" \
    --run-id "$TUNING_RUN_ID" \
    --output-dir "${LOG_ROOT}/figures/paper_maps_tuning" \
    --with-overlay --max-bundles 16 \
    2>&1 | tee -a "${LOG_ROOT}/figures.log" || true
fi

log "========== REPRESENTATIVE SAMPLE END artifacts=${LOG_ROOT} =========="
