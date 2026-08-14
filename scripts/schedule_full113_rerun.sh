#!/usr/bin/env bash
# Scheduled launcher for the full-113 HM-EQA re-run (post-override-fix confirmation).
#
# Waits until TARGET (default: 4 hours from now), runs the Habitat EGL probe
# (safe-start), waits for it to be done, then launches the paper-113 H2H via
# `emet jobs run` so it is GPU-mutexed, resumable, and crash-safe.
#
# Usage:
#   nohup ./scripts/schedule_full113_rerun.sh > ~/runs/emet/scheduled_full113.log 2>&1 &
#
# Env:
#   TARGET       "YYYY-MM-DD HH:MM[:SS]" to launch at (default: now + DELAY_H)
#   DELAY_H      hours from now when TARGET unset (default 4)
#   OUT_TAG      suffix for the emet jobs name (default: rerun-fix)
#   METHODS      space-separated methods (default "dynagraph static_graph")
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DELAY_H="${DELAY_H:-4}"
OUT_TAG="${OUT_TAG:-rerun-fix}"
METHODS="${METHODS:-dynagraph static_graph}"

log() { echo "[$(date -Iseconds)] $*"; }

# --- target ----------------------------------------------------------------
if [[ -n "${TARGET:-}" ]]; then
  TARGET_STR="$TARGET"
else
  TARGET_STR="$(date -d "+${DELAY_H} hours" '+%Y-%m-%d %H:%M')"
fi
TARGET_EPOCH="$(date -d "$TARGET_STR" +%s)"
NOW="$(date +%s)"
if [[ "$TARGET_EPOCH" -le "$NOW" ]]; then
  log "FATAL: target '$TARGET_STR' (epoch $TARGET_EPOCH) is not in the future (now=$NOW). Refusing to launch early."
  exit 3
fi
SECS=$((TARGET_EPOCH - NOW))
log "scheduled full-113 launch at $TARGET_STR (now=$(date); sleeping ${SECS}s ≈ $((SECS / 3600))h $((SECS % 3600 / 60))m)"
sleep "$SECS"
log "target time reached; starting launch sequence"

# --- Habitat EGL probe (recover + safe-start) ------------------------------
uv run emet habitat safe-start
log "safe-start submitted; waiting for EGL probe to finish"
for _ in $(seq 1 90); do
  if ! uv run emet jobs 2>/dev/null | grep -q "habitat-egl"; then
    break
  fi
  sleep 15
done
sleep 20  # settle

# --- launch full-113 via emet jobs -----------------------------------------
NAME="hmeqa-paper113-${OUT_TAG}"
log "launching full-113 re-run as emet jobs '$NAME' (methods=$METHODS)"
uv run emet jobs run \
  --name "$NAME" \
  -d "Full-113 HM-EQA re-run with location_override_equip_gate fix default-on (merged #116); confirms offline-verified 52.2/44.2" \
  --need-mib 12000 \
  -- \
  env EMET_ALLOW_SDPA_ATTN=1 METHODS="$METHODS" HF_ID=Qwen/Qwen3-VL-8B-Instruct \
  ./scripts/run_hmeqa_paper113_h2h.sh

log "full-113 re-run launched"
