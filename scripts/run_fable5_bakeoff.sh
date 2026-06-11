#!/usr/bin/env bash
# Fable5 model-size bake-off: bigger int4 VLMs vs the 3B control on canonical-8,
# winner auto-promotes to balanced-32. All phases run dynagraph with the
# free-form + rotation debias answering path (see docs/plans/fable5-dynagraph-habitat.md).
#
# Usage:
#   nohup ./scripts/run_fable5_bakeoff.sh >> /tmp/fable5_bakeoff.nohup.out 2>&1 &
#
# Env:
#   OVERNIGHT_DEADLINE_HOURS  Stop launching new attempts after this many hours (default 20).
#   SKIP_PHASES               Comma-separated phase names to skip.
#   WINNER_FAMILY/WINNER_HF   Override automatic winner selection for the bal32 phase.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/habitat_eqa/overnight/bakeoff_$RUN_ID}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/overnight.log"
LOCK_FILE="$HOME/.cache/habitat_eqa/overnight.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another overnight eval holds $LOCK_FILE — exiting." | tee -a "$MAIN_LOG"
  exit 1
fi

INTERVAL="${INTERVAL:-30}"
STABLE="${STABLE:-2}"
TIMEOUT="${TIMEOUT:-7200}"
DEADLINE_H="${OVERNIGHT_DEADLINE_HOURS:-20}"
GLOBAL_DEADLINE=$(( $(date +%s) + DEADLINE_H * 3600 ))
NEED_MIB="${NEED_MIB:-15000}"

IDS_CANONICAL="${IDS_CANONICAL:-3,14,17,28,31,35,81,94}"
IDS_BALANCED="${IDS_BALANCED:-2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84}"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

count_completed() {
  uv run python - <<'PY' "$1"
import json, sys
from pathlib import Path
from emet.habitat.metrics import episode_run_completed
p = Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(len({r["question_id"] for r in rows if episode_run_completed(r)}))
PY
}

count_correct() {
  uv run python - <<'PY' "$1"
import json, sys
from pathlib import Path
from emet.habitat.metrics import episode_run_completed
p = Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
by = {}
for line in p.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        if episode_run_completed(r) or r["question_id"] not in by:
            by[r["question_id"]] = r
print(sum(1 for r in by.values() if episode_run_completed(r) and r["correct"]))
PY
}

n_ids() { awk -F, '{print NF}' <<<"$1"; }

wait_for_gpu() {
  local need_mib="$1" ok=0 free
  while :; do
    if [ "$(date +%s)" -ge "$GLOBAL_DEADLINE" ]; then
      log "GLOBAL deadline during GPU wait (need=${need_mib}MiB)"; return 2
    fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${free:-0}" -ge "$need_mib" ]; then ok=$((ok + 1)); else ok=0; fi
    log "GPU free=${free}MiB need=${need_mib} stable=${ok}/${STABLE}"
    [ "$ok" -ge "$STABLE" ] && return 0
    sleep "$INTERVAL"
  done
}

run_phase() {
  local phase_name="$1" family="$2" hf_id="$3" ids="$4" tag="$5"
  if [[ ",${SKIP_PHASES:-}," == *",$phase_name,"* ]]; then
    log "SKIP phase $phase_name"; return 0
  fi
  local results="$HOME/.cache/habitat_eqa/results/subset_${tag}_${family}.jsonl"
  local phase_log="$LOG_DIR/${phase_name}.log"
  local n_target
  n_target=$(n_ids "$ids")
  log "=== PHASE $phase_name: family=$family hf_id=$hf_id tag=$tag n=$n_target ==="
  while [ "$(count_completed "$results")" -lt "$n_target" ]; do
    if [ "$(date +%s)" -ge "$GLOBAL_DEADLINE" ]; then
      log "GLOBAL deadline — stopping $phase_name at $(count_completed "$results")/$n_target"
      return 2
    fi
    if ! wait_for_gpu "$NEED_MIB"; then return 2; fi
    log "launch $phase_name attempt (done=$(count_completed "$results")/$n_target)"
    TAG="$tag" IDS="$ids" METHOD=dynagraph TIMEOUT="$TIMEOUT" FAMILY="$family" HF_ID="$hf_id" \
      ./scripts/run_habitat_iter_subset.sh 2>&1 | tee -a "$phase_log" "$MAIN_LOG" || true
    sleep 10
  done
  log "PHASE $phase_name COMPLETE $(count_completed "$results")/$n_target correct=$(count_correct "$results")"
}

log "############ FABLE5 BAKEOFF START run_id=$RUN_ID deadline=${DEADLINE_H}h ############"

# Candidates: (phase, family, hf_id). Preference order for ties = listed order (bigger first).
CAND_PHASES=(q3vl8b gemma4e4b q25vl3b)
CAND_FAMILY=(qwen3_vl gemma4 qwen2_5_vl)
CAND_HF=("Qwen/Qwen3-VL-8B-Instruct" "google/gemma-4-E4B-it" "Qwen/Qwen2.5-VL-3B-Instruct")

for i in "${!CAND_PHASES[@]}"; do
  run_phase "${CAND_PHASES[$i]}" "${CAND_FAMILY[$i]}" "${CAND_HF[$i]}" \
    "$IDS_CANONICAL" "fable5_bake_${CAND_PHASES[$i]}" || true
done

# Winner: max correct on canonical-8; ties resolved by candidate order.
WINNER_IDX=-1
BEST=-1
for i in "${!CAND_PHASES[@]}"; do
  results="$HOME/.cache/habitat_eqa/results/subset_fable5_bake_${CAND_PHASES[$i]}_${CAND_FAMILY[$i]}.jsonl"
  c=$(count_correct "$results")
  log "candidate ${CAND_PHASES[$i]}: correct=$c"
  if [ "$c" -gt "$BEST" ]; then BEST="$c"; WINNER_IDX="$i"; fi
done

if [ -n "${WINNER_FAMILY:-}" ] && [ -n "${WINNER_HF:-}" ]; then
  W_FAMILY="$WINNER_FAMILY"; W_HF="$WINNER_HF"; W_NAME="override"
elif [ "$WINNER_IDX" -ge 0 ]; then
  W_FAMILY="${CAND_FAMILY[$WINNER_IDX]}"; W_HF="${CAND_HF[$WINNER_IDX]}"; W_NAME="${CAND_PHASES[$WINNER_IDX]}"
else
  log "No winner determined — skipping bal32."; W_FAMILY=""; W_HF=""; W_NAME=""
fi

if [ -n "$W_FAMILY" ]; then
  log "WINNER: $W_NAME ($W_FAMILY / $W_HF) with $BEST/8 — promoting to balanced-32"
  run_phase "winner_bal32" "$W_FAMILY" "$W_HF" "$IDS_BALANCED" "fable5_bake_winner_bal32" || true
fi

log "############ SUMMARY ############"
{
  for i in "${!CAND_PHASES[@]}"; do
    results="$HOME/.cache/habitat_eqa/results/subset_fable5_bake_${CAND_PHASES[$i]}_${CAND_FAMILY[$i]}.jsonl"
    echo "canonical8 ${CAND_PHASES[$i]} (${CAND_HF[$i]}): $(count_correct "$results")/$(count_completed "$results")"
  done
  if [ -n "$W_FAMILY" ]; then
    results="$HOME/.cache/habitat_eqa/results/subset_fable5_bake_winner_bal32_${W_FAMILY}.jsonl"
    echo "balanced32 winner=$W_NAME: $(count_correct "$results")/$(count_completed "$results")"
  fi
} | tee "$LOG_DIR/summary.txt" | tee -a "$MAIN_LOG"
log "############ FABLE5 BAKEOFF END ############"
