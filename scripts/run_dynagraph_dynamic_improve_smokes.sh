#!/usr/bin/env bash
# Targeted Phase-1 smoke (K=3) then Phase-2 world-change for DynaGraph+Qwen fixes.
# Usage: nohup ./scripts/run_dynagraph_dynamic_improve_smokes.sh [OUT_DIR] &
# Prefer: uv run emet jobs run --name dyn-improve-eqa --need-mib 14000 -- ./scripts/run_dynagraph_dynamic_improve_smokes.sh OUT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynamic_exploration/dyn_improve_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

# Register only when not already managed by ``emet jobs run`` / queue wrapper.
EMET_BIN="${ROOT}/.venv/bin/emet"
[ -x "$EMET_BIN" ] || EMET_BIN="emet"
JOB_ID="${EMET_JOB_ID:-}"
if [ -z "$JOB_ID" ]; then
  JOB_ID="$("$EMET_BIN" jobs register \
    --name "dyn-improve-eqa" \
    --status running \
    --pid $$ \
    --out-dir "$OUT" \
    --log-path "$OUT/orchestrator.log" \
    --repo "$ROOT" \
    --cmd "./scripts/run_dynagraph_dynamic_improve_smokes.sh $OUT" || true)"
  export EMET_JOB_ID="${JOB_ID:-}"
fi
_finish_job() {
  local rc=$?
  [ -z "${JOB_ID:-}" ] && return 0
  if [ -f "$OUT/DONE" ]; then
    "$EMET_BIN" jobs update "$JOB_ID" --status done >/dev/null 2>&1 || true
  elif [ "$rc" -ne 0 ]; then
    "$EMET_BIN" jobs update "$JOB_ID" --status failed --error "exit $rc" >/dev/null 2>&1 || true
  fi
}
trap _finish_job EXIT

log "OUT=$OUT job_id=${JOB_ID:-none}"
# Do not --kill-stale here: it can reap this orchestrator when launched under nohup.
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait

# Vision EQA prefill can exceed the default 30 min STALE_KILL when MuJoCo shares the GPU.
# Heartbeats keep the log fresh; still raise thresholds so a slow-but-alive generate survives.
export EMET_DYNAMIC_EXPLORE_STALE_LOG_S="${EMET_DYNAMIC_EXPLORE_STALE_LOG_S:-1800}"
export EMET_DYNAMIC_EXPLORE_STALE_KILL_S="${EMET_DYNAMIC_EXPLORE_STALE_KILL_S:-3600}"
export EMET_EQA_QUESTION_TIMEOUT_S="${EMET_EQA_QUESTION_TIMEOUT_S:-2400}"

log "Phase-1 smoke K=3 start"
uv run python scripts/eval_dynamic_exploration.py --smoke \
  --output-dir "$OUT/phase1_smoke" \
  2>&1 | tee -a "$OUT/phase1_smoke.log"
log "Phase-1 smoke done"

NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
log "Phase-2 world-change start"
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph \
  --output-dir "$OUT/phase2_world_change" \
  2>&1 | tee -a "$OUT/phase2_world_change.log"
log "Phase-2 done"
echo DONE > "$OUT/DONE"
log "All smokes finished"
