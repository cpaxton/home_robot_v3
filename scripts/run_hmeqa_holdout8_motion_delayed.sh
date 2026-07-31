#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Delayed HM-EQA holdout-8 agentic run to measure motion-branch + stuck-goal
# marking vs recent paper-router baselines (fix4/fix11 ~0.75).
#
# Usage:
#   DELAY_SEC=7200 ./scripts/run_hmeqa_holdout8_motion_delayed.sh
#   nohup env DELAY_SEC=7200 ./scripts/run_hmeqa_holdout8_motion_delayed.sh \
    #     >~/runs/emet/hmeqa_motion_delayed_launch.log 2>&1 &

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DELAY_SEC="${DELAY_SEC:-7200}"
HOLDOUT_IDS="${HOLDOUT_IDS:-15,56,65,68,79,88,104,105}"
JOB_NAME="${JOB_NAME:-hmeqa-holdout8-motion}"
NEED_MIB="${NEED_MIB:-12000}"
HEAD="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HOME/runs/emet/hmeqa_holdout8_motion_${STAMP}}"
LAUNCH_LOG="${LAUNCH_LOG:-$HOME/runs/emet/hmeqa_motion_delayed_launch.log}"
LAUNCH_AT="$(date -d "+${DELAY_SEC} seconds" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || true)"

mkdir -p "$OUT" "$(dirname "$LAUNCH_LOG")"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/status_log.sh"
status_open "$OUT" "hmeqa-holdout8-motion-delayed"
STATUS_RESUME_CMD="cd $REPO_ROOT && DELAY_SEC=0 OUT=$OUT ./scripts/run_hmeqa_holdout8_motion_delayed.sh"
status_note "WAITING" \
    "delay=${DELAY_SEC}s launch_at=${LAUNCH_AT:-unknown} head=${HEAD} ids=${HOLDOUT_IDS}" \
    "after wake: uv run emet hmeqa status $OUT; uv run emet jobs"

{
    echo "[$(date -Is)] delayed holdout-8 motion experiment"
    echo "  head=${HEAD} ids=${HOLDOUT_IDS} job=${JOB_NAME}"
    echo "  OUT=${OUT}"
    echo "  sleep ${DELAY_SEC}s (until ~${LAUNCH_AT:-unknown})"
} | tee -a "$LAUNCH_LOG"

sleep "${DELAY_SEC}"

echo "[$(date -Is)] waking: GPU recover then h2h" | tee -a "$LAUNCH_LOG"
status_note "RECOVER" \
    "running emet eval recover --need-mib ${NEED_MIB}" \
    "uv run emet hmeqa h2h $OUT --preset paper-router --arms agentic"

uv run emet eval recover --need-mib "${NEED_MIB}"

status_note "LAUNCH" \
    "emet hmeqa h2h OUT=${OUT} head=${HEAD}" \
    "uv run emet hmeqa status $OUT; uv run emet jobs"

# h2h registers its own emet jobs entry (cpu-safe + gpu-exclusive).
uv run emet hmeqa h2h "${OUT}" \
    --preset paper-router \
    --arms agentic \
    --ids "${HOLDOUT_IDS}" \
    --job-name "${JOB_NAME}" \
    --need-mib "${NEED_MIB}"

status_close "LAUNCHED" \
    "hmeqa h2h returned (job may still be running under emet jobs)" \
    "uv run emet hmeqa status $OUT; uv run emet jobs; uv run emet hmeqa summarize $OUT"

echo "[$(date -Is)] h2h launch returned; monitor: uv run emet jobs; uv run emet hmeqa status ${OUT}" | tee -a "$LAUNCH_LOG"
