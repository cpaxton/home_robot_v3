#!/usr/bin/env bash
# Fable5 overnight HM-EQA eval: validate MCQ choice-rotation debiasing for dynagraph.
# See docs/plans/fable5-dynagraph-habitat.md for the plan + failure analysis.
#
# Phases (serial, each with --resume; stricter episode_run_completed retries VLM-less stubs):
#   1. fable5_dg_debias_c8     dynagraph + debias, canonical 8
#   2. fable5_dg_debias_bal32  dynagraph + debias, balanced 32
#   3. fable5_ge_bal32_recheck graph_eqa baseline, balanced 32 (same code, if time)
#
# Usage:
#   nohup ./scripts/run_fable5_overnight.sh >> /tmp/fable5_overnight.nohup.out 2>&1 &
#
# Env:
#   WAIT_PID                 Wait for this pid to exit before starting (in-flight eval).
#   OVERNIGHT_DEADLINE_HOURS Stop launching new attempts after this many hours (default 10).
#   SKIP_PHASES              Comma-separated phase names to skip.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/habitat_eqa/overnight/fable5_$RUN_ID}"
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
DEADLINE_H="${OVERNIGHT_DEADLINE_HOURS:-10}"
GLOBAL_DEADLINE=$(( $(date +%s) + DEADLINE_H * 3600 ))

IDS_CANONICAL="${IDS_CANONICAL:-3,14,17,28,31,35,81,94}"
IDS_BALANCED="${IDS_BALANCED:-2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84}"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

# Wait for an in-flight eval (e.g. the canonical-8 rerun) to release the GPU.
if [ -n "${WAIT_PID:-}" ]; then
  log "Waiting for pid $WAIT_PID to exit before starting..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$GLOBAL_DEADLINE" ]; then
      log "Deadline reached while waiting for pid $WAIT_PID — exiting."
      exit 2
    fi
    sleep "$INTERVAL"
  done
  log "pid $WAIT_PID exited."
fi

count_completed() {
  local results="$1"
  uv run python - <<'PY' "$results"
import json, sys
from pathlib import Path
from emet.habitat.metrics import episode_run_completed
p = Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
done = {r["question_id"] for r in rows if episode_run_completed(r)}
print(len(done))
PY
}

n_ids() { awk -F, '{print NF}' <<<"$1"; }

wait_for_gpu() {
  local need_mib="$1"
  local ok=0 free
  while :; do
    if [ "$(date +%s)" -ge "$GLOBAL_DEADLINE" ]; then
      log "GLOBAL deadline reached during GPU wait (need=${need_mib}MiB)"
      return 2
    fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${free:-0}" -ge "$need_mib" ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    log "GPU free=${free}MiB need=${need_mib} stable=${ok}/${STABLE}"
    [ "$ok" -ge "$STABLE" ] && return 0
    sleep "$INTERVAL"
  done
}

run_phase() {
  local phase_name="$1" method="$2" ids="$3" tag="$4" need_mib="$5"
  if [[ ",${SKIP_PHASES:-}," == *",$phase_name,"* ]]; then
    log "SKIP phase $phase_name (SKIP_PHASES)"
    return 0
  fi
  local results="$HOME/.cache/habitat_eqa/results/subset_${tag}_qwen3_vl.jsonl"
  local phase_log="$LOG_DIR/${phase_name}.log"
  local n_target
  n_target=$(n_ids "$ids")
  log "=== PHASE $phase_name: method=$method tag=$tag n=$n_target need_gpu=${need_mib}MiB ==="
  while [ "$(count_completed "$results")" -lt "$n_target" ]; do
    if [ "$(date +%s)" -ge "$GLOBAL_DEADLINE" ]; then
      log "GLOBAL deadline — stopping phase $phase_name at $(count_completed "$results")/$n_target"
      return 2
    fi
    if ! wait_for_gpu "$need_mib"; then
      return 2
    fi
    log "launch $phase_name attempt (done=$(count_completed "$results")/$n_target)"
    TAG="$tag" IDS="$ids" METHOD="$method" TIMEOUT="$TIMEOUT" \
      ./scripts/run_habitat_iter_subset.sh 2>&1 | tee -a "$phase_log" "$MAIN_LOG" || true
    sleep 10
  done
  log "PHASE $phase_name COMPLETE $(count_completed "$results")/$n_target"
}

write_summary() {
  local out="$LOG_DIR/summary.txt"
  uv run python - <<'PY' >"$out"
import json
from collections import Counter
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

results_root = Path.home() / ".cache/habitat_eqa/results"
tags = {
    "canonical8 rerun dynagraph (no debias)": "canonical8_rerun_dg",
    "canonical8 rerun graph_eqa": "canonical8_rerun_ge",
    "canonical8 dynagraph+debias": "fable5_dg_debias_c8",
    "balanced32 dynagraph+debias": "fable5_dg_debias_bal32",
    "balanced32 graph_eqa recheck": "fable5_ge_bal32_recheck",
    "balanced32 dynagraph (pre-debias ref)": "bal32_dynagraph",
    "balanced32 graph_eqa (pre-debias ref)": "bal32_graph_eqa",
}

def load(tag):
    p = results_root / f"subset_{tag}_qwen3_vl.jsonl"
    if not p.exists():
        return {}
    by = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prev = by.get(r["question_id"])
        if prev is None or episode_run_completed(r) >= episode_run_completed(prev):
            by[r["question_id"]] = r
    return {q: r for q, r in by.items() if episode_run_completed(r)}

print("Fable5 overnight summary (docs/plans/fable5-dynagraph-habitat.md)\n")
for label, tag in tags.items():
    by = load(tag)
    if not by:
        print(f"{label}: (no results)")
        continue
    correct = sum(r["correct"] for r in by.values())
    gold_acc = Counter()
    gold_tot = Counter()
    flipped = 0
    for r in by.values():
        gold_tot[r["gold_answer_letter"]] += 1
        gold_acc[r["gold_answer_letter"]] += int(r["correct"])
        if r.get("predebias_letter") and r.get("predebias_letter") != r.get("parsed_answer_letter"):
            flipped += 1
    by_gold = ", ".join(f"{L}={gold_acc[L]}/{gold_tot[L]}" for L in sorted(gold_tot))
    extra = f" debias-flips={flipped}" if flipped else ""
    print(f"{label}: {correct}/{len(by)}  by gold: {by_gold}{extra}")
PY
  cat "$out" | tee -a "$MAIN_LOG"
}

log "############ FABLE5 OVERNIGHT START run_id=$RUN_ID deadline=${DEADLINE_H}h ############"

NEED_BASELINE="${NEED_BASELINE_MIB:-11000}"
NEED_DYNAGRAPH="${NEED_DYNAGRAPH_MIB:-15000}"

run_phase "dg_debias_c8" dynagraph "$IDS_CANONICAL" "fable5_dg_debias_c8" "$NEED_DYNAGRAPH" || true
run_phase "dg_debias_bal32" dynagraph "$IDS_BALANCED" "fable5_dg_debias_bal32" "$NEED_DYNAGRAPH" || true
run_phase "ge_bal32_recheck" graph_eqa "$IDS_BALANCED" "fable5_ge_bal32_recheck" "$NEED_BASELINE" || true

log "############ SUMMARY ############"
write_summary
log "############ FABLE5 OVERNIGHT END $(date -Is) ############"
log "Summary: $LOG_DIR/summary.txt"
