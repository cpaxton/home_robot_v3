#!/usr/bin/env bash
# Scheduled launcher: HM-EQA count/clock slice to validate the close-look crop.
#
# Validates the close-work improvement (dense_siglip_argmax_crop fed to VLM
# assess) on the 15 count/clock paper-113 questions (the weakest classes in the
# 95/113 partial: count 23%, clock 20%). Compares dynagraph with the crop path.
#
# Waits until the current GPU job clears + a buffer, then launches via `emet
# jobs run` so it is GPU-mutexed and crash-safe.
#
# Env:
#   TARGET  "YYYY-MM-DD HH:MM[:SS]" to launch at (default: now + DELAY_MIN)
#   DELAY_MIN  minutes from now when TARGET unset (default 30)
#   QUESTION_IDS  comma-separated ids (default: the 15 count/clock ids)
#   METHODS  space-separated methods (default "dynagraph")
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DELAY_MIN="${DELAY_MIN:-30}"
OUT_TAG="${OUT_TAG:-closelook-countclock}"
QUESTION_IDS="${QUESTION_IDS:-12,21,28,32,33,43,47,48,51,60,78,84,86,88,93}"
METHODS="${METHODS:-dynagraph}"

log() { echo "[$(date -Iseconds)] $*"; }

if [[ -n "${TARGET:-}" ]]; then
    TARGET_STR="$TARGET"
else
    TARGET_STR="$(date -d "+${DELAY_MIN} minutes" '+%Y-%m-%d %H:%M')"
fi
TARGET_EPOCH="$(date -d "$TARGET_STR" +%s)"
NOW="$(date +%s)"
if [[ "$TARGET_EPOCH" -le "$NOW" ]]; then
    log "FATAL: target '$TARGET_STR' is not in the future (now=$NOW). Refusing to launch early."
    exit 3
fi
SECS=$((TARGET_EPOCH - NOW))
log "scheduled close-look count/clock validation at $TARGET_STR (sleeping ${SECS}s ≈ $((SECS / 60))m)"
sleep "$SECS"
log "target reached; launching"

NAME="hmeqa-countclock-${OUT_TAG}"
log "launching count/clock slice as emet jobs '$NAME' (ids=$QUESTION_IDS methods=$METHODS)"
uv run emet jobs run \
    --name "$NAME" \
    -d "HM-EQA count/clock slice (15 ids) validating close-look crop (dense_siglip_argmax_crop -> VLM assess 2nd image). Branch feat/tamp-floor-experiments." \
    --need-mib 12000 \
    -- \
    env EMET_ALLOW_SDPA_ATTN=1 METHODS="$METHODS" HF_ID=Qwen/Qwen3-VL-8B-Instruct \
    QUESTION_IDS="$QUESTION_IDS" \
    ./scripts/run_hmeqa_countclock_slice.sh

log "count/clock slice launched"
