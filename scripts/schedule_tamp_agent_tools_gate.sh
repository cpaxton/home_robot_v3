#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Queue the TAMP agent-tools managed simulator gate via `emet jobs run`.
#
# Uses native job scheduling (`--delay-minutes` / `--at`) and `--gpu-exclusive`
# so the gate waits for other MuJoCo/VLM work without keeping a terminal alive.
#
# Env:
#   DELAY_MIN   minutes before start (default 0 = as soon as GPU is free)
#   AT          optional wall time "YYYY-MM-DD HH:MM" (overrides DELAY_MIN)
#   NEED_MIB    VRAM gate (default 12000)
#   PROFILE     smoke (default) or full — forwarded to run_tamp_agent_tools_gate.sh
#   ITEMS       optional override (default: from PROFILE)
#   OUT_TAG     suffix for artifact dir (default timestamp)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"

DELAY_MIN="${DELAY_MIN:-0}"
NEED_MIB="${NEED_MIB:-12000}"
PROFILE="${PROFILE:-smoke}"
OUT_TAG="${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="$HOME/runs/emet/tamp_agent_tools_gate/scheduled_${OUT_TAG}"
NAME="tamp-agent-tools-gate-${OUT_TAG}"

log() { echo "[$(date -Iseconds)] $*"; }

mkdir -p "$OUT"
_STATUS_OUT="$OUT"
_STATUS_LABEL="tamp-agent-tools-gate"
STATUS_PROGRESS="0/? queued"

schedule_args=(--delay-minutes "$DELAY_MIN")
if [[ -n "${AT:-}" ]]; then
    schedule_args=(--at "$AT")
fi

gate_env=(PROFILE="$PROFILE" OUT_DIR="$OUT")
if [[ -n "${ITEMS:-}" ]]; then
    gate_env+=(ITEMS="$ITEMS")
fi

RESUME_CMD="uv run emet jobs status JOB_ID"
log "scheduling gate name=$NAME profile=$PROFILE items='${ITEMS:-<profile default>}' need_mib=$NEED_MIB out=$OUT"
if [[ -n "${AT:-}" ]]; then
    log "start at: $AT"
else
    log "start after delay_minutes=$DELAY_MIN (gpu-exclusive waits for live GPU work)"
fi

job_id="$(
    uv run emet jobs run \
        --name "$NAME" \
        -d "TAMP agent-tools managed simulator gate (PROFILE=$PROFILE). ITEMS=${ITEMS:-<profile default>}" \
        --need-mib "$NEED_MIB" \
        --gpu-exclusive \
        --out-dir "$OUT" \
        "${schedule_args[@]}" \
        -- \
        env "${gate_env[@]}" ./scripts/run_tamp_agent_tools_gate.sh \
        | tee "$OUT/launch.log" | tail -1
)"

RESUME_CMD="uv run emet jobs status $job_id"
STATUS_RESUME_CMD="$RESUME_CMD"
status_note PLANNED \
    "TAMP agent-tools gate queued as job $job_id (PROFILE=$PROFILE; gpu-exclusive + ${NEED_MIB} MiB)." \
    "$RESUME_CMD"

log "queued job_id=$job_id"
log "monitor: uv run emet jobs status $job_id"
log "logs:    uv run emet jobs logs $job_id --tail 80"
