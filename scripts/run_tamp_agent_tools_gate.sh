#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Managed simulator gate for TAMP agent-tools acceptance (run via `emet jobs`).
#
# Runs managed-simulator acceptance items sequentially and reports per-item
# pass/fail via `emet jobs update` (when EMET_JOB_ID is set):
#
#   chat       MolmoSpaces iTHOR + rby1 CHAT scene_tasks → plan → execute (teleport)
#   kinematic  MolmoSpaces iTHOR + rby1 CHAT scene_tasks → plan → execute (IK + base snap)
#   stretch    Stretch default-table teleport plan_pick_place → execute
#   floor-smoke  One fast rby1 MCTS floor episode (~5–8 min; no Stretch agentic sweeps)
#   floor        Full RoboCasa floor matrix (slow; Stretch + find-only explore)
#
# Profiles (env PROFILE):
#   smoke (default)  kinematic                            (~2–5 min; agent contract)
#   full             chat kinematic stretch floor          (1–2 h; overnight)
#
# Usage (GPU-mutexed, crash-safe):
#
#   ./scripts/run_tamp_agent_tools_gate.sh
#   PROFILE=full ./scripts/run_tamp_agent_tools_gate.sh
#
# Queue for later (no terminal/agent must stay alive):
#
#   ./scripts/schedule_tamp_agent_tools_gate.sh
#   # or: uv run emet jobs run --name tamp-agent-tools-gate --delay-minutes 240 \
    #   #       --need-mib 12000 --gpu-exclusive -- ./scripts/run_tamp_agent_tools_gate.sh
#
# Dry-run (log commands only; for quoting regressions):
#
#   DRY_RUN=1 ITEMS="chat kinematic stretch" ./scripts/run_tamp_agent_tools_gate.sh
#
# Env:
#   OUT_DIR    artifact dir (default ~/runs/emet/tamp_agent_tools_gate/<ts>)
#   PROFILE    smoke (default) or full — sets ITEMS when ITEMS unset
#   ITEMS      override profile tokens: chat kinematic stretch floor-smoke floor
#   TIMEOUT    per-item wall timeout seconds (default 2700; floor uses FLOOR_TIMEOUT)
#   FLOOR_TIMEOUT  wall timeout for floor / floor-smoke (default 5400 / 1200)
#   DRY_RUN    when 1, log each item command without executing (quoting smoke)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/tamp_agent_tools_gate/${RUN_ID}}"
mkdir -p "$OUT"
TIMEOUT="${TIMEOUT:-2700}"
PROFILE="${PROFILE:-smoke}"
case "$PROFILE" in
    smoke) DEFAULT_ITEMS="kinematic" ;;
    full) DEFAULT_ITEMS="chat kinematic stretch floor" ;;
    *)
        echo "FATAL: unknown PROFILE=$PROFILE (want smoke or full)" >&2
        exit 2
        ;;
esac
ITEMS="${ITEMS:-$DEFAULT_ITEMS}"
FLOOR_TIMEOUT="${FLOOR_TIMEOUT:-$([ "$PROFILE" = smoke ] && echo 1200 || echo 5400)}"
UNITS=0
gate_result=0

items_has() {
    local token="$1"
    local t
    for t in $ITEMS; do
        [[ "$t" == "$token" ]] && return 0
    done
    return 1
}

count_item() {
    items_has "$1" || return 0
    TOTAL=$((TOTAL + 1))
}

TOTAL=0
count_item chat
count_item kinematic
count_item stretch
count_item floor-smoke
count_item floor
if [[ "$TOTAL" -eq 0 ]]; then
    echo "FATAL: ITEMS='$ITEMS' selects no gate items (tokens: chat kinematic stretch floor-smoke floor)" >&2
    exit 2
fi

_STATUS_LABEL="${STATUS_LABEL:-tamp-agent-tools-gate}"
status_open "$OUT" "$_STATUS_LABEL"
STATUS_PROGRESS="0/$TOTAL starting"
STATUS_RESUME_CMD="${STATUS_RESUME_CMD:-uv run emet jobs logs ${EMET_JOB_ID:-JOB_ID} --tail 80}"

log() { echo "[$(date -Is)] $*" | tee -a "$OUT/gate.log"; }

progress() { # phase units_done item
    STATUS_PROGRESS="$2/$TOTAL $3"
    if [[ -n "${EMET_JOB_ID:-}" ]]; then
        uv run emet jobs update "$EMET_JOB_ID" --phase "$1" --units-done "$2" --units-total "$TOTAL" --current-id "$3" >/dev/null 2>&1 || true
    fi
}

write_meta() {
    {
        echo "run_id=$RUN_ID"
        echo "profile=$PROFILE"
        echo "items=$ITEMS"
        echo "git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "job=${EMET_JOB_ID:-}"
        echo "dry_run=${DRY_RUN:-0}"
    } | tee "$OUT/META.txt"
}

write_summary() {
    local summary="$OUT/gate_summary.txt"
    {
        echo "TAMP agent-tools gate summary ($RUN_ID)"
        echo "profile=$PROFILE items=$ITEMS total=$TOTAL exit=$gate_result"
        echo "git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "job=${EMET_JOB_ID:-}"
        echo ""
        for name in molmospaces_chat_teleport molmospaces_kinematic_chat stretch_teleport_control robocasa_floor_smoke robocasa_floor_suite; do
            if [[ -f "$OUT/${name}.result" ]]; then
                echo "${name}: $(cat "$OUT/${name}.result")"
            fi
        done
        if compgen -G "$OUT/floor_smoke/aggregate_*.csv" >/dev/null; then
            echo ""
            echo "floor_smoke aggregate (latest csv):"
            latest_csv="$(ls -1t "$OUT/floor_smoke"/aggregate_*.csv 2>/dev/null | head -1)"
            if [[ -n "$latest_csv" ]]; then
                cat "$latest_csv"
            fi
        fi
        if compgen -G "$OUT/floor/aggregate_*.csv" >/dev/null; then
            echo ""
            echo "floor aggregate (latest csv):"
            latest_csv="$(ls -1t "$OUT/floor"/aggregate_*.csv 2>/dev/null | head -1)"
            if [[ -n "$latest_csv" ]]; then
                cat "$latest_csv"
            fi
        fi
    } | tee "$summary"
}

run_item() { # name phase timeout_sec cmd...
    local name="$1"
    local phase="$2"
    local item_timeout="$3"
    shift 3
    UNITS=$((UNITS + 1))
    log "=== [$UNITS/$TOTAL] $name ==="
    progress "$phase" "$UNITS" "$name"
    log "cmd: $*"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        log "DRY_RUN: skip execution for $name"
        echo "DRY_RUN" >"$OUT/${name}.result"
        return 0
    fi
    if timeout "$item_timeout" "$@" >"$OUT/${name}.log" 2>&1; then
        log "PASS $name"
        echo "PASS" >"$OUT/${name}.result"
    else
        log "FAIL $name (see $OUT/${name}.log)"
        echo "FAIL" >"$OUT/${name}.result"
        tail -25 "$OUT/${name}.log" | tee -a "$OUT/gate.log" || true
        gate_result=1
    fi
}

write_meta
log "gate start profile=$PROFILE items='$ITEMS' total=$TOTAL out=$OUT"

CHAT_TOOL_CALLS='[{"name":"scene_tasks","arguments":{"object_filter":"bowl"}},{"name":"plan_pick_place","arguments":{"task_ref":"task:1"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'
# This is the routine agent test: its calls go through get_tools(), opaque
# task/plan handles, guarded TAMP execution, and the real rby1 IK path.
KINEMATIC_TOOL_CALLS='[{"name":"scene_tasks","arguments":{"object_filter":"bowl","robot":"rby1"}},{"name":"plan_pick_place","arguments":{"task_ref":"task:1"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'
STRETCH_TOOL_CALLS='[{"name":"plan_pick_place","arguments":{"object_name":"red cylinder","receptacle_name":"blue cube"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'

export EMET_ALLOW_SDPA_ATTN=1
export EMET_SIM_NAV_TELEPORT=1

if items_has chat; then
    run_item molmospaces_chat_teleport molmospaces_chat_teleport "$TIMEOUT" \
        uv run python scripts/scripted_sim_pick_place.py \
        --start-sim --sim configs/sim/molmospaces_ithor_train_0.yaml --manip-mode teleport \
        --object bowl --receptacle microwave --cpu-only \
        --tool-calls-json "$CHAT_TOOL_CALLS"
fi

if items_has kinematic; then
    run_item molmospaces_kinematic_chat molmospaces_kinematic_chat "$TIMEOUT" \
        uv run python scripts/scripted_sim_pick_place.py \
        --start-sim --sim configs/sim/molmospaces_ithor_train_0.yaml --manip-mode kinematic \
        --object bowl --receptacle microwave --cpu-only \
        --tool-calls-json "$KINEMATIC_TOOL_CALLS"
fi

if items_has stretch; then
    run_item stretch_teleport_control stretch_teleport_control "$TIMEOUT" \
        uv run python scripts/scripted_sim_pick_place.py \
        --start-sim --sim configs/sim/default_table_stretch.yaml --manip-mode teleport \
        --object "red cylinder" --receptacle "blue cube" --cpu-only \
        --tool-calls-json "$STRETCH_TOOL_CALLS"
fi

if items_has floor-smoke; then
    mkdir -p "$OUT/floor_smoke"
    run_item robocasa_floor_smoke robocasa_floor_smoke "$FLOOR_TIMEOUT" \
        uv run python scripts/eval_tamp_floor.py --smoke --output-dir "$OUT/floor_smoke"
fi

if items_has floor; then
    mkdir -p "$OUT/floor"
    run_item robocasa_floor_suite robocasa_floor_suite "$FLOOR_TIMEOUT" \
        uv run python scripts/eval_tamp_floor.py --output-dir "$OUT/floor"
fi

write_summary
log "gate complete: pass/fail summary in $OUT/gate_summary.txt (exit=$gate_result)"
progress "gate_done" "$TOTAL" "gate"
if [[ "$gate_result" -eq 0 ]]; then
    status_close DONE "TAMP agent-tools gate green ($TOTAL/$TOTAL)." \
        "uv run emet jobs logs ${EMET_JOB_ID:-} --tail 80; cat $OUT/gate_summary.txt"
else
    status_close FAILED "TAMP agent-tools gate failed (see gate_summary.txt)." \
        "uv run emet jobs logs ${EMET_JOB_ID:-} --tail 80; cat $OUT/gate_summary.txt"
fi
exit "$gate_result"
