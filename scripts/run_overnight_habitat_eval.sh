#!/usr/bin/env bash
# Overnight HM-EQA evaluation: wait for GPU windows, run all configured phases with
# --resume (retries OOM/crash stubs), collect logs under ~/.cache/habitat_eqa/overnight/.
#
# Usage:
#   nohup ./scripts/run_overnight_habitat_eval.sh >> /tmp/overnight_habitat.nohup.out 2>&1 &
#
# Env:
#   OVERNIGHT_DEADLINE_HOURS  Stop launching new phases after this many hours (default 14).
#   SKIP_PHASES             Comma-separated phase names to skip (e.g. paper20_dynagraph).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/habitat_eqa/overnight/$RUN_ID}"
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
DEADLINE_H="${OVERNIGHT_DEADLINE_HOURS:-14}"
GLOBAL_DEADLINE=$(( $(date +%s) + DEADLINE_H * 3600 ))

# Question sets (comma-separated indices).
IDS_CANONICAL="${IDS_CANONICAL:-3,14,17,28,31,35,81,94}"
IDS_BALANCED="${IDS_BALANCED:-2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84}"
IDS_PAPER20="$(uv run python -c 'print(",".join(map(str,range(20))))')"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

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
print(sum(1 for r in rows if episode_run_completed(r)))
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
  local phase_name="$1"
  local method="$2"
  local ids="$3"
  local tag="$4"
  local need_mib="$5"

  if [[ ",${SKIP_PHASES:-}," == *",$phase_name,"* ]]; then
    log "SKIP phase $phase_name (SKIP_PHASES)"
    return 0
  fi

  local results="$HOME/.cache/habitat_eqa/results/subset_${tag}_qwen2_5_vl.jsonl"
  local phase_log="$LOG_DIR/${phase_name}.log"
  local n_target
  n_target=$(n_ids "$ids")

  log "=== PHASE $phase_name: method=$method tag=$tag n=$n_target need_gpu=${need_mib}MiB ==="
  log "  results=$results"
  log "  log=$phase_log"

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
  RUN_ID="$RUN_ID" LOG_DIR="$LOG_DIR" uv run python - <<PY >"$out"
import json
import os
from collections import Counter
from pathlib import Path

run_id = os.environ["RUN_ID"]
log_dir = Path(os.environ["LOG_DIR"])
tags = {
    "canonical8_graph_eqa": f"overnight_{run_id}_canonical8_graph_eqa",
    "canonical8_dynagraph": f"overnight_{run_id}_canonical8_dynagraph",
    "balanced32_graph_eqa": f"overnight_{run_id}_balanced32_graph_eqa",
    "balanced32_dynagraph": f"overnight_{run_id}_balanced32_dynagraph",
    "paper20_graph_eqa": f"overnight_{run_id}_paper20_graph_eqa",
    "paper20_dynagraph": f"overnight_{run_id}_paper20_dynagraph",
}
results_root = Path.home() / ".cache/habitat_eqa/results"

def load_rows(tag):
    p = results_root / f"subset_{tag}_qwen2_5_vl.jsonl"
    if not p.exists():
        return [], p
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # Best row per question_id: prefer completed runs, then latest line.
    from emet.habitat.metrics import episode_run_completed
    by_q = {}
    for r in rows:
        q = r["question_id"]
        prev = by_q.get(q)
        if prev is None:
            by_q[q] = r
            continue
        prev_ok = episode_run_completed(prev)
        cur_ok = episode_run_completed(r)
        if cur_ok and not prev_ok:
            by_q[q] = r
        elif cur_ok == prev_ok:
            by_q[q] = r
    return list(by_q.values()), p

print(f"Overnight run: {run_id}")
print(f"Log dir: {log_dir}")
print()
for label, tag in tags.items():
    rows, path = load_rows(tag)
    if not rows:
        print(f"{label}: (no results) {path}")
        continue
    from emet.habitat.metrics import episode_run_completed
    completed = [r for r in rows if episode_run_completed(r)]
    errors = len(rows) - len(completed)
    correct = sum(r["correct"] for r in completed)
    print(f"{label}: {correct}/{len(completed)} correct ({errors} error stubs in file)")
    by_gold = Counter()
    tot_gold = Counter()
    for r in completed:
        g = r.get("gold_answer_letter", "?")
        tot_gold[g] += 1
        by_gold[g] += int(r["correct"])
    if tot_gold:
        print("  by gold: " + ", ".join(f"{L}={by_gold[L]}/{tot_gold[L]}" for L in sorted(tot_gold)))
print()
# Head-to-head on balanced32 if both exist
dg_rows, _ = load_rows(tags["balanced32_dynagraph"])
bl_rows, _ = load_rows(tags["balanced32_graph_eqa"])
if dg_rows and bl_rows:
    dg = {r["question_id"]: r for r in dg_rows if episode_run_completed(r)}
    bl = {r["question_id"]: r for r in bl_rows if episode_run_completed(r)}
    ids = sorted(set(dg) & set(bl))
    if ids:
        d_ok = sum(dg[q]["correct"] for q in ids)
        b_ok = sum(bl[q]["correct"] for q in ids)
        agree = sum(1 for q in ids if dg[q]["correct"] == bl[q]["correct"])
        print(f"balanced32 head-to-head ({len(ids)} common): dynagraph {d_ok}/{len(ids)}  baseline {b_ok}/{len(ids)}  agree {agree}/{len(ids)}")
PY
  cat "$out" | tee -a "$MAIN_LOG"
}

# Write manifest
cat >"$LOG_DIR/manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "started": "$(date -Is)",
  "deadline_hours": $DEADLINE_H,
  "ids_canonical": "$IDS_CANONICAL",
  "ids_balanced": "$IDS_BALANCED",
  "ids_paper20": "$IDS_PAPER20",
  "timeout_per_batch_sec": $TIMEOUT,
  "gpu_stable_checks": $STABLE,
  "log_dir": "$LOG_DIR"
}
EOF

log "############ OVERNIGHT EVAL START run_id=$RUN_ID ############"
log "log_dir=$LOG_DIR deadline=${DEADLINE_H}h"

PREFIX="overnight_${RUN_ID}"

# Baseline (graph_eqa, no SigLIP) needs less VRAM; dynagraph needs more.
NEED_BASELINE="${NEED_BASELINE_MIB:-11000}"
NEED_DYNAGRAPH="${NEED_DYNAGRAPH_MIB:-15000}"

run_phase "canonical8_graph_eqa" graph_eqa "$IDS_CANONICAL" "${PREFIX}_canonical8_graph_eqa" "$NEED_BASELINE" || true
run_phase "canonical8_dynagraph" dynagraph "$IDS_CANONICAL" "${PREFIX}_canonical8_dynagraph" "$NEED_DYNAGRAPH" || true
run_phase "balanced32_graph_eqa" graph_eqa "$IDS_BALANCED" "${PREFIX}_balanced32_graph_eqa" "$NEED_BASELINE" || true
run_phase "balanced32_dynagraph" dynagraph "$IDS_BALANCED" "${PREFIX}_balanced32_dynagraph" "$NEED_DYNAGRAPH" || true
run_phase "paper20_graph_eqa" graph_eqa "$IDS_PAPER20" "${PREFIX}_paper20_graph_eqa" "$NEED_BASELINE" || true
run_phase "paper20_dynagraph" dynagraph "$IDS_PAPER20" "${PREFIX}_paper20_dynagraph" "$NEED_DYNAGRAPH" || true

log "############ SUMMARY ############"
write_summary
log "############ OVERNIGHT EVAL END $(date -Is) ############"
log "Full log: $MAIN_LOG"
log "Summary:  $LOG_DIR/summary.txt"
