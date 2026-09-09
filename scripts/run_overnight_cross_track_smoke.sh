#!/usr/bin/env bash
# Overnight five-track smoke validation + safe no-sim pytest.
#
# Usage:
#   ./scripts/run_overnight_cross_track_smoke.sh
#   RUN_ID=my_run TIMEOUT_DYNAMIC=28800 ./scripts/run_overnight_cross_track_smoke.sh
#
# Do not chain Habitat VLM after this (driver hangs). Separate night:
#   bash scripts/smoke_habitat_ovmm_agentic_find.sh
#   uv run emet hmeqa overnight   # or emet-habitat run-batch --question-ids 3,14,17
#
# Logs: ~/runs/emet/overnight_cross_track/<RUN_ID>/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-cross_track_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${HOME}/runs/emet/overnight_cross_track/${RUN_ID}"
mkdir -p "$LOG_DIR"

NEED_MIB="${NEED_MIB:-12000}"
TIMEOUT_UNIT="${TIMEOUT_UNIT:-3600}"
TIMEOUT_SQA3D="${TIMEOUT_SQA3D:-1800}"
TIMEOUT_HMEQA="${TIMEOUT_HMEQA:-7200}"
TIMEOUT_OVMM="${TIMEOUT_OVMM:-3600}"
TIMEOUT_DYNAMIC="${TIMEOUT_DYNAMIC:-28800}"
TIMEOUT_WORLD="${TIMEOUT_WORLD:-28800}"
TIMEOUT_PYTEST="${TIMEOUT_PYTEST:-14400}"

HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
SUMMARY="${LOG_DIR}/summary.txt"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"; }

gpu_between_tracks() {
    emet_gpu_between_steps "$NEED_MIB"
}

run_step() {
    local name="$1" timeout_s="$2"
    shift 2
    local logf="${LOG_DIR}/${name}.log"
    log "=== START ${name} (timeout=${timeout_s}s) ==="
    if timeout "$timeout_s" "$@" > >(tee -a "$logf") 2>&1; then
        log "=== PASS ${name} ==="
        return 0
    else
        local ec=$?
        log "=== FAIL ${name} exit=${ec} (see ${logf}) ==="
        return 0
    fi
}

log "RUN_ID=$RUN_ID LOG_DIR=$LOG_DIR"
emet_kill_stale_eval_processes

# Tier 0 — focused unit tests (no MuJoCo/Habitat sim)
run_step tier0_unit_tests "$TIMEOUT_UNIT" \
    uv run emet test \
    src/test/config/ \
    src/test/benchmarks/sqa3d/ \
    src/test/habitat/test_metrics.py \
    src/test/eval/test_dynagraph_vram.py \
    src/test/memory/test_graph_eqa_memory.py \
    src/test/memory/test_mcq_debias.py \
    src/test/memory/test_ovmm_find_phase_metrics.py \
    src/test/memory/test_habitat_ovmm_find_loader.py \
    src/test/eval/test_dynamic_exploration_config.py \
    src/test/eval/test_dynamic_exploration_runner.py \
    src/test/memory/test_memory_backends_smoke.py \
    src/test/app/test_stream_stats.py \
    src/test/memory/test_dynamem_graph_hooks_fusion.py \
    src/test/memory/test_graph_object_fusion_default_yaml.py \
    src/test/eval/test_episode_diagnostics_export.py \
    src/test/eval/test_habitat_cli_diagnostics.py \
    src/test/robots/test_innate_mars_backend.py \
    -q

# Track 1 — SQA3D mock
run_step track1_sqa3d_mock "$TIMEOUT_SQA3D" \
    uv run emet sqa3d run-episode --split train --mock-llm --question-id 220602000000

gpu_between_tracks

# Track 2 — Habitat EQA Q17
if [ -x "$HAB" ]; then
    run_step track2_habitat_eqa_q17 "$TIMEOUT_HMEQA" \
        "$HAB" run-episode \
        --question-id 17 --method dynagraph \
        --eqa-vl-family qwen3_vl \
        --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
        --device cuda --export-map \
        --output "${HOME}/.cache/habitat_eqa/results/${RUN_ID}_smoke_q17.jsonl"
else
    log "SKIP track2 (no .venv-habitat)"
fi

gpu_between_tracks

# Track 3 — Habitat OVMM GT (CPU)
if [ -x "$HAB" ]; then
    mkdir -p "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}"
    run_step track3_habitat_ovmm_gt "$TIMEOUT_OVMM" \
        "$HAB" run-ovmm-find-episode \
        --episode-id hm3d_lamp_bed_00006 \
        --backend ground_truth --cpu-only --not-rotate \
        --output "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}/hm3d_lamp_bed_00006_gt.json"
else
    log "SKIP track3 (no .venv-habitat)"
fi

gpu_between_tracks

# Track 4 — Robocasa explore smoke
DYN_OUT="${HOME}/runs/emet/dynamic_exploration/${RUN_ID}_explore"
mkdir -p "$DYN_OUT"
run_step track4_robocasa_explore "$TIMEOUT_DYNAMIC" \
    uv run python scripts/eval_dynamic_exploration.py --smoke \
    --output-dir "$DYN_OUT"

gpu_between_tracks

# Track 5 — World-change
WC_OUT="${HOME}/runs/emet/dynamic_exploration/${RUN_ID}_world_change"
mkdir -p "$WC_OUT"
run_step track5_robocasa_world_change "$TIMEOUT_WORLD" \
    uv run python scripts/eval_dynamic_exploration.py \
    --phase world-change \
    --episode-id robocasa_seed0_world_change \
    --backend dynagraph \
    --output-dir "$WC_OUT"

gpu_between_tracks

# Full no-sim pytest (excludes MuJoCo/Habitat sim paths even if unmarked)
mapfile -t _pytest_ignore_args < <(emet_pytest_no_sim_ignore_args)
run_step full_pytest_no_sim "$TIMEOUT_PYTEST" \
    uv run pytest src/test -m "not sim" --tb=no -q "${_pytest_ignore_args[@]}"
unset _pytest_ignore_args

log "=== OVERNIGHT COMPLETE ==="
log "Review: ${SUMMARY} and per-step logs in ${LOG_DIR}/"

if [ "${RUN_DEEP_EVAL:-0}" = "1" ]; then
    log "RUN_DEEP_EVAL is removed (run_overnight_eval_smoke.sh deleted). Not chaining VLM."
    log "Separate night: bash scripts/smoke_habitat_ovmm_agentic_find.sh"
    log "HM-EQA: uv run emet hmeqa overnight   (or emet-habitat run-batch --question-ids 3,14,17)"
    log "Figures: uv run python scripts/build_eval_figure_pack.py --run-id …"
fi

emet_kill_stale_eval_processes
