#!/usr/bin/env bash
# Queue DynaGraph EQA smokes (Phase-1 K=3 + Phase-2 world-change) behind other GPU jobs.
#
# Does NOT call kill-stale (safe while unrelated evals are still finishing).
# Registers with ``emet jobs`` so you can ``emet jobs list`` / ``cancel``.
#
# Usage:
#   nohup ./scripts/queue_dynagraph_eqa_evals.sh [WAIT_PID ...] [OUT_DIR] &
#   # or:
#   uv run emet jobs run --name dyn-improve-eqa --need-mib 14000 --wait-pid PID -- \
#     ./scripts/run_dynagraph_dynamic_improve_smokes.sh OUT
#
# If WAIT_PID args are omitted, waits only for free VRAM (NEED_MIB).
# Trailing arg that looks like a path is OUT_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WAIT_PIDS=()
OUT=""
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    WAIT_PIDS+=("$arg")
  else
    OUT="$arg"
  fi
done
OUT="${OUT:-$HOME/runs/emet/dynamic_exploration/dyn_improve_eqa_queued_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
NEED_MIB="${NEED_MIB:-14000}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/queue.log"; }

_emet_bin() {
  if [ -x "$ROOT/.venv/bin/emet" ]; then
    echo "$ROOT/.venv/bin/emet"
  else
    echo "emet"
  fi
}
EMET="$(_emet_bin)"

REG_ARGS=(
  --name "dyn-improve-eqa-queue"
  --status waiting
  --pid $$
  --out-dir "$OUT"
  --log-path "$OUT/queue.log"
  --repo "$ROOT"
  --cmd "./scripts/queue_dynagraph_eqa_evals.sh $*"
)
if ((${#WAIT_PIDS[@]})); then
  for pid in "${WAIT_PIDS[@]}"; do
    REG_ARGS+=(--wait-pid "$pid")
  done
fi

JOB_ID="$("$EMET" jobs register "${REG_ARGS[@]}")"
log "OUT=$OUT ROOT=$ROOT NEED_MIB=$NEED_MIB job_id=$JOB_ID"
log "wait_pids=${WAIT_PIDS[*]:-(none)}"

cleanup_job() {
  local rc=$?
  if [ -f "$OUT/DONE" ]; then
    "$EMET" jobs update "$JOB_ID" --status done >/dev/null 2>&1 || true
  elif [ "$rc" -ne 0 ]; then
    "$EMET" jobs update "$JOB_ID" --status failed --error "exit $rc" >/dev/null 2>&1 || true
  fi
}
trap cleanup_job EXIT

if ((${#WAIT_PIDS[@]})); then
  for pid in "${WAIT_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      log "waiting for pid $pid ($(ps -p "$pid" -o args= 2>/dev/null | cut -c1-100))"
      while kill -0 "$pid" 2>/dev/null; do
        sleep 30
      done
      log "pid $pid exited"
    else
      log "pid $pid already gone"
    fi
  done
fi

# Also wait out leftover dynagraph/explore workers (other checkouts / children).
log "waiting until no foreign eval_dynamic / run_dynagraph workers…"
self=$$
while true; do
  conflict=0
  while read -r pid; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$self" ] && continue
    pp=$pid
    skip=0
    for _ in $(seq 1 32); do
      [ "$pp" = "$self" ] && skip=1 && break
      [ -z "$pp" ] || [ "$pp" -le 1 ] && break
      pp=$(ps -o ppid= -p "$pp" 2>/dev/null | tr -d ' ' || true)
    done
    [ "$skip" = 1 ] && continue
    conflict=1
    break
  done < <(pgrep -f 'eval_dynamic_exploration\.py|emet\.app\.run_dynagraph|/emet run dynagraph' 2>/dev/null || true)
  [ "$conflict" = 0 ] && break
  sleep 30
done
log "no conflicting dynagraph workers"

log "GPU wait (no kill-stale)"
NEED_MIB="$NEED_MIB" ./scripts/gpu_preflight.sh --wait
"$EMET" eval status 2>&1 | tee -a "$OUT/queue.log" || true

"$EMET" jobs update "$JOB_ID" --status running --pid $$ >/dev/null
log "starting improve smokes (Phase-1 EQA + Phase-2 world-change)"
./scripts/run_dynagraph_dynamic_improve_smokes.sh "$OUT" >>"$OUT/nohup.log" 2>&1
log "improve smokes finished (see $OUT/DONE)"
