#!/usr/bin/env bash
# Sequential seven-track simulation smoke battery (Habitat → sim search → dynamic env).
#
# Usage:
#   ./scripts/run_simulation_smoke_battery.sh
#   RUN_ID=my_run MOCK_LLM=1 nohup ./scripts/run_simulation_smoke_battery.sh >> ~/runs/emet/simulation_smoke/nohup.log 2>&1 &
#
# Docs: docs/simulation_testing_plan.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-sim_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${HOME}/runs/emet/simulation_smoke/${RUN_ID}"
mkdir -p "$LOG_DIR"

NEED_MIB="${NEED_MIB:-12000}"
MOCK_LLM="${MOCK_LLM:-1}"
HABITAT_GPU="${HABITAT_GPU:-1}"
DYNAMIC_GPU="${DYNAMIC_GPU:-1}"
SKIP_TRACKS="${SKIP_TRACKS:-}"

TIMEOUT_HMEQA="${TIMEOUT_HMEQA:-7200}"
TIMEOUT_OVMM_HAB="${TIMEOUT_OVMM_HAB:-1800}"
TIMEOUT_OVMM_SIM="${TIMEOUT_OVMM_SIM:-3600}"
TIMEOUT_SQA3D="${TIMEOUT_SQA3D:-1800}"
TIMEOUT_DYN="${TIMEOUT_DYN:-28800}"

HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
SUMMARY="${LOG_DIR}/summary.txt"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUMMARY"; }

should_skip() {
    local n="$1"
    [[ ",${SKIP_TRACKS}," == *",${n},"* ]]
}

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
        return "$ec"
    fi
}

log "RUN_ID=$RUN_ID LOG_DIR=$LOG_DIR MOCK_LLM=$MOCK_LLM HABITAT_GPU=$HABITAT_GPU DYNAMIC_GPU=$DYNAMIC_GPU"
emet_kill_stale_eval_processes

FAIL=0

# Track 1 — Habitat EQA
if should_skip 1; then
    log "SKIP track1 (Habitat EQA)"
elif [ ! -x "$HAB" ]; then
    log "SKIP track1 (no .venv-habitat)"
else
    gpu_between_tracks
    if [ "$HABITAT_GPU" = "1" ]; then
        run_step track1_habitat_eqa "$TIMEOUT_HMEQA" \
            "$HAB" run-episode \
            --question-id 17 --method dynagraph \
            --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
            --device cuda --export-map \
            --output "${HOME}/.cache/habitat_eqa/results/${RUN_ID}_hmeqa_q17.jsonl" \
            || FAIL=1
    else
        run_step track1_habitat_eqa "$TIMEOUT_HMEQA" \
            "$HAB" run-episode \
            --question-id 17 --method dynagraph --mock-llm \
            --output "${HOME}/.cache/habitat_eqa/results/${RUN_ID}_hmeqa_q17.jsonl" \
            || FAIL=1
    fi
fi

# Track 2 — Habitat OVMM GT
if should_skip 2; then
    log "SKIP track2 (Habitat OVMM)"
elif [ ! -x "$HAB" ]; then
    log "SKIP track2 (no .venv-habitat)"
else
    gpu_between_tracks
    mkdir -p "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}"
    run_step track2_habitat_ovmm "$TIMEOUT_OVMM_HAB" \
        "$HAB" run-ovmm-find-episode \
        --episode-id hm3d_lamp_bed_00006 \
        --backend ground_truth --cpu-only --not-rotate \
        --output "${HOME}/runs/emet/ovmm_habitat/${RUN_ID}/hm3d_lamp_bed_00006_gt.json" \
        || FAIL=1
fi

# Track 3 — Robocasa search (OVMM S1)
if should_skip 3; then
    log "SKIP track3 (Robocasa search)"
else
    gpu_between_tracks
    run_step track3_robocasa_search "$TIMEOUT_OVMM_SIM" \
        uv run python scripts/eval_ovmm_find_phases.py \
        --episode-id robocasa_pp_s1 \
        --backend ground_truth --cpu-only --not-rotate \
        --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_robocasa" \
        || FAIL=1
fi

# Track 4 — MolmoSpaces / iTHOR search (OVMM S2)
if should_skip 4; then
    log "SKIP track4 (Molmo search)"
elif [ ! -d "${ROOT}/.venv-molmospaces" ]; then
    log "SKIP track4 (no .venv-molmospaces)"
else
    gpu_between_tracks
    run_step track4_molmo_search "$TIMEOUT_OVMM_SIM" \
        uv run python scripts/eval_ovmm_find_phases.py \
        --episode-id molmo_ithor_s2_idx0 \
        --backend ground_truth --cpu-only --not-rotate \
        --output-dir "${HOME}/runs/emet/ovmm_find_phase/${RUN_ID}_molmo" \
        || FAIL=1
fi

# Track 5 — SQA3D
if should_skip 5; then
    log "SKIP track5 (SQA3D)"
else
    gpu_between_tracks
    if [ "$MOCK_LLM" = "1" ]; then
        run_step track5_sqa3d "$TIMEOUT_SQA3D" \
            uv run emet sqa3d run-episode --split train --mock-llm --question-id 220602000000 \
            || FAIL=1
    else
        run_step track5_sqa3d "$TIMEOUT_SQA3D" \
            uv run emet sqa3d run-episode --split val --question-id 0 --method dynagraph --replay-mode sens \
            || FAIL=1
    fi
fi

dynamic_gpu_args() {
    if [ "$DYNAMIC_GPU" = "1" ]; then
        echo ""
    else
        echo "--cpu-only"
    fi
}

# Track 6 — Robocasa dynamic env (world-change)
if should_skip 6; then
    log "SKIP track6 (Robocasa dynamic env)"
else
    gpu_between_tracks
    # shellcheck disable=SC2046
    run_step track6_robocasa_dynamic_env "$TIMEOUT_DYN" \
        uv run python scripts/eval_dynamic_exploration.py \
        --phase world-change --episode-id robocasa_seed0_world_change \
        --backend dynagraph --explore-max-iters 3 \
        $(dynamic_gpu_args) \
        --output-dir "${HOME}/runs/emet/dynamic_exploration/${RUN_ID}_world_change" \
        || FAIL=1
fi

# Track 7 — MolmoSpaces dynamic search (explore-loop)
if should_skip 7; then
    log "SKIP track7 (Molmo dynamic search)"
elif [ ! -d "${ROOT}/.venv-molmospaces" ]; then
    log "SKIP track7 (no .venv-molmospaces)"
else
    gpu_between_tracks
    # shellcheck disable=SC2046
    run_step track7_molmo_dynamic_search "$TIMEOUT_DYN" \
        uv run python scripts/eval_dynamic_exploration.py \
        --phase explore --env molmospaces --episode-id molmo_ithor0 \
        --backend dynagraph --explore-max-iters 3 --mapping-mode explore \
        --skip-eqa \
        $(dynamic_gpu_args) \
        --output-dir "${HOME}/runs/emet/dynamic_exploration/${RUN_ID}_molmo_explore" \
        || FAIL=1
fi

emet_kill_stale_eval_processes

if [ "$FAIL" -eq 0 ]; then
    log "=== SIMULATION SMOKE BATTERY COMPLETE (all tracks pass) ==="
else
    log "=== SIMULATION SMOKE BATTERY COMPLETE (failures — see logs) ==="
fi

if uv run python scripts/inspect_simulation_smoke_battery.py --run-id "$RUN_ID" --write-report >> "$SUMMARY" 2>&1; then
    log "Inspection report: ${LOG_DIR}/inspection_report.md (semantic checks pass)"
else
    log "Inspection report: ${LOG_DIR}/inspection_report.md (semantic WARN/FAIL — open for metrics + artifact paths)"
fi

exit "$FAIL"
