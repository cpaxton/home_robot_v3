#!/usr/bin/env bash
# Targeted Phase-1 smoke (K=3) then Phase-2 world-change for DynaGraph+Qwen fixes.
# Prefer: uv run emet jobs run --name dyn-improve-eqa --need-mib 14000 -- ./scripts/run_dynagraph_dynamic_improve_smokes.sh OUT
# Job status is owned by ``emet jobs run`` (EMET_JOB_ID); do not self-register here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynamic_exploration/dyn_improve_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

log "OUT=$OUT job_id=${EMET_JOB_ID:-none}"
# Do not --kill-stale here: it can reap this orchestrator when launched under nohup.
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait

# Vision EQA prefill can exceed the default 30 min STALE_KILL when MuJoCo shares the GPU.
export EMET_DYNAMIC_EXPLORE_STALE_LOG_S="${EMET_DYNAMIC_EXPLORE_STALE_LOG_S:-1800}"
export EMET_DYNAMIC_EXPLORE_STALE_KILL_S="${EMET_DYNAMIC_EXPLORE_STALE_KILL_S:-3600}"
export EMET_EQA_QUESTION_TIMEOUT_S="${EMET_EQA_QUESTION_TIMEOUT_S:-2400}"
export EMET_EQA_AGENTIC_VERIFY="${EMET_EQA_AGENTIC_VERIFY:-1}"
export EMET_EQA_TRACE="${EMET_EQA_TRACE:-1}"
# Decode cap left to eqa_vl/answer_max_new_tokens (was pinned to 64 and truncated answers).

log "Phase-1 smoke K=3 start"
uv run python scripts/eval_dynamic_exploration.py --smoke --output-dir "$OUT/phase1_smoke" 2>&1 | tee -a "$OUT/phase1_smoke.log"
log "Phase-1 smoke done"

NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
log "Phase-2 world-change start"
uv run python scripts/eval_dynamic_exploration.py --phase world-change --episode-id robocasa_seed0_world_change --backend dynagraph --output-dir "$OUT/phase2_world_change" 2>&1 | tee -a "$OUT/phase2_world_change.log"
log "Phase-2 done"
echo DONE > "$OUT/DONE"
log "All smokes finished"
