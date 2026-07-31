#!/usr/bin/env bash
# Preliminary head-to-head: answer-only EQA vs agentic (deterministic router).
# Uses the scene-map cache so Phase-1 skips live explore when available.
#
# Usage:
#   nohup ./scripts/run_agentic_vs_answeronly_h2h.sh [OUT_DIR] \
    #     >> ~/runs/emet/dynamic_exploration/agentic_h2h_nohup.log 2>&1 &
#
# Arms:
#   baseline  — EMET_EQA_AGENTIC_VERIFY=0 (original answer-only path)
#   agentic   — EMET_EQA_AGENTIC_VERIFY=1 EMET_EQA_AGENTIC_ROUTER=0
#               (nav→verify→answer via deterministic fallback; VLM router off
#               so preliminary numbers finish without ~20min/turn decode tax)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynamic_exploration/agentic_h2h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
./scripts/gpu_preflight.sh --kill-stale || true

# Both arms inherit eqa_vl/answer_max_new_tokens. A low pin (was 64) truncates the answer
# before ``answer:``, so the h2h would compare two salvage re-asks rather than the arms.
export EMET_EQA_QUESTION_TIMEOUT_S="${EMET_EQA_QUESTION_TIMEOUT_S:-2400}"
export EMET_DYNAMIC_EXPLORE_STALE_LOG_S="${EMET_DYNAMIC_EXPLORE_STALE_LOG_S:-1800}"
export EMET_DYNAMIC_EXPLORE_STALE_KILL_S="${EMET_DYNAMIC_EXPLORE_STALE_KILL_S:-3600}"
export EMET_EQA_TRACE="${EMET_EQA_TRACE:-1}"
export EMET_USE_SCENE_MAP_CACHE="${EMET_USE_SCENE_MAP_CACHE:-1}"

run_arm() {
    local name="$1"
    shift
    local arm_out="$OUT/$name"
    mkdir -p "$arm_out"
    log "ARM=$name start → $arm_out ($*)"
    NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
    # shellcheck disable=SC2086
    env "$@" uv run python scripts/eval_dynamic_exploration.py --smoke \
        --backend dynagraph \
        --output-dir "$arm_out" \
        2>&1 | tee -a "$arm_out/run.log"
    log "ARM=$name done"
}

log "OUT=$OUT HEAD=$(git rev-parse --short HEAD)"

# 1) Original answer-only EQA
run_arm baseline \
    EMET_EQA_AGENTIC_VERIFY=0 \
    EMET_EQA_TRACE=0

# 2) Agentic with deterministic tool policy (router off)
run_arm agentic_fallback \
    EMET_EQA_AGENTIC_VERIFY=1 \
    EMET_EQA_AGENTIC_ROUTER=0 \
    EMET_EQA_TRACE=1

log "Summarizing…"
uv run python scripts/summarize_agentic_eqa_h2h.py "$OUT" \
    --figure "$OUT/figures/agentic_vs_answeronly.png" \
    -o "$OUT/h2h_summary.json" | tee -a "$OUT/orchestrator.log"

echo DONE > "$OUT/DONE"
log "All arms finished → $OUT"
