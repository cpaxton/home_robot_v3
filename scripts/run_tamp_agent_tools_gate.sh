#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Managed simulator gate for TAMP agent-tools acceptance (run via `emet jobs`).
#
# Runs the three managed-simulator acceptance items sequentially and reports
# per-item pass/fail via `emet jobs update` (when EMET_JOB_ID is set):
#
#   1. MolmoSpaces iTHOR + rby1 kinematic CHAT plan/execute smoke
#      (plan_pick_place -> execute_pick_place_plan on the real arm)
#   2. Stretch teleport control (plan_pick_place -> execute_pick_place_plan)
#   3. RoboCasa floor suite with per-episode mcts/sim/skip modes
#
# Usage (GPU-mutexed, crash-safe):
#
#   NEED_MIB=12000 uv run emet jobs run --name tamp-agent-tools-gate \
    #     --delay-minutes 240 --need-mib 12000 --gpu-exclusive -- \
    #     ./scripts/run_tamp_agent_tools_gate.sh
#
# Env:
#   OUT_DIR    artifact dir (default ~/runs/emet/tamp_agent_tools_gate/<ts>)
#   ITEMS      space-separated item names to run (default "kinematic stretch floor")
#   TIMEOUT    per-item wall timeout seconds (default 2700)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/tamp_agent_tools_gate/${RUN_ID}}"
mkdir -p "$OUT"
TIMEOUT="${TIMEOUT:-2700}"
ITEMS="${ITEMS:-kinematic stretch floor}"
TOTAL=3
UNITS=0

log() { echo "[$(date -Is)] $*" | tee -a "$OUT/gate.log"; }

progress() { # phase units_done item
    if [[ -n "${EMET_JOB_ID:-}" ]]; then
        uv run emet jobs update "$EMET_JOB_ID" --phase "$1" --units-done "$2" --units-total "$TOTAL" --current-id "$3" >/dev/null 2>&1 || true
    fi
}

gate_result=0

# CUDA VL / EQA loads need either flash-attn or the SDPA fallback; this box has
# no flash-attn wheel, so allow SDPA for the GPU-dependent floor item.
export EMET_ALLOW_SDPA_ATTN=1

run_item() { # name phase cmd...
    local name="$1"
    local phase="$1"
    shift
    UNITS=$((UNITS + 1))
    log "=== [$UNITS/$TOTAL] $name ==="
    progress "$phase" "$UNITS" "$name"
    log "cmd: $*"
    if timeout "$TIMEOUT" bash -c "$*" >"$OUT/${name}.log" 2>&1; then
        log "PASS $name"
    else
        log "FAIL $name (see $OUT/${name}.log)"
        tail -25 "$OUT/${name}.log" | tee -a "$OUT/gate.log" || true
        gate_result=1
    fi
}

if [[ "$ITEMS" == *"kinematic"* ]]; then
    run_item molmospaces_kinematic_chat \
        env EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py \
        --start-sim --sim configs/sim/molmospaces_ithor_train_0.yaml --manip-mode kinematic \
        --object bowl --receptacle microwave --cpu-only \
        --tool-calls-json '[{"name":"plan_pick_place","arguments":{"object_name":"bowl","receptacle_name":"microwave"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'
fi

if [[ "$ITEMS" == *"stretch"* ]]; then
    run_item stretch_teleport_control \
        env EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py \
        --start-sim --sim configs/sim/default_table_stretch.yaml --manip-mode teleport \
        --object "red cylinder" --receptacle "blue cube" --cpu-only \
        --tool-calls-json '[{"name":"plan_pick_place","arguments":{"object_name":"red cylinder","receptacle_name":"blue cube"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'
fi

if [[ "$ITEMS" == *"floor"* ]]; then
    run_item robocasa_floor_suite \
        uv run python scripts/eval_tamp_floor.py
fi

log "gate complete: pass/fail summary above (exit=$gate_result)"
progress "gate_done" "$TOTAL" "gate"
exit "$gate_result"
