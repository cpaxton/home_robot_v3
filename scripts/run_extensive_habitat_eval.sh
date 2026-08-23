#!/usr/bin/env bash
# Post-merge extensive HM-EQA evaluation:
#   1) Wait for stable GPU (16GB free)
#   2) Letter-balanced 32-q dynagraph run (resume-safe)
#   3) Same 32-q graph_eqa baseline run (resume-safe)
#   4) Print side-by-side summary
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Reduce CUDA fragmentation OOMs when SigLIP + VLM share the GPU across episodes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Letter-balanced 32 questions (8 each A/B/C/D, real 4-choice only).
IDS="${IDS:-2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84}"
TIMEOUT="${TIMEOUT:-7200}"
NEED_MIB="${NEED_MIB:-16000}"
STABLE="${STABLE:-3}"
INTERVAL="${INTERVAL:-30}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-480}"

DG_TAG="${DG_TAG:-postmerge_dynagraph}"
BL_TAG="${BL_TAG:-postmerge_graph_eqa}"
LOG="${LOG:-/tmp/postmerge_habitat_eval.log}"

wait_for_gpu() {
    local ok=0 free now deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
    while :; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        now=$(date +%H:%M:%S)
        if [ "${free:-0}" -ge "$NEED_MIB" ]; then ok=$((ok+1)); else ok=0; fi
        echo "[$now] free=${free}MiB need=${NEED_MIB} stable=${ok}/${STABLE}" | tee -a "$LOG"
        [ "$ok" -ge "$STABLE" ] && return 0
        [ "$(date +%s)" -ge "$deadline" ] && return 2
        sleep "$INTERVAL"
    done
}

run_phase() {
    local method="$1" tag="$2"
    local results="$HOME/.cache/habitat_eqa/results/subset_${tag}_qwen3_vl.jsonl"
    local n_target
    n_target=$(awk -F, '{print NF}' <<<"$IDS")
    count_done() {
        uv run python - <<'PY' "$results" "$n_target"
import json, sys
from pathlib import Path
from emet.habitat.metrics import episode_run_completed
p = Path(sys.argv[1])
target = int(sys.argv[2])
if not p.exists():
    print(0)
    raise SystemExit
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
done = sum(1 for r in rows if episode_run_completed(r))
print(done)
PY
    }
    while [ "$(count_done)" -lt "$n_target" ]; do
        if ! wait_for_gpu; then
            echo "GPU wait timeout before $method phase" | tee -a "$LOG"
            return 2
        fi
        echo "=== launching $method (tag=$tag) ===" | tee -a "$LOG"
        TAG="$tag" IDS="$IDS" METHOD="$method" TIMEOUT="$TIMEOUT" \
            ./scripts/run_habitat_iter_subset.sh 2>&1 | tee -a "$LOG" || true
        sleep 5
    done
    echo "=== $method done: $results ===" | tee -a "$LOG"
}

echo "############ EXTENSIVE EVAL START $(date -Is) ############" | tee "$LOG"
echo "IDS=$IDS" | tee -a "$LOG"

run_phase dynagraph "$DG_TAG"
run_phase graph_eqa "$BL_TAG"

echo "############ SUMMARY ############" | tee -a "$LOG"
uv run python - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

def load(tag):
    p = Path.home() / f".cache/habitat_eqa/results/subset_{tag}_qwen3_vl.jsonl"
    if not p.exists():
        return {}
    return {r["question_id"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}

dg = load("postmerge_dynagraph")
bl = load("postmerge_graph_eqa")
ids = sorted(set(dg) | set(bl))
print(f"dynagraph: {sum(r['correct'] for r in dg.values())}/{len(dg)}")
print(f"baseline:  {sum(r['correct'] for r in bl.values())}/{len(bl)}")
if ids:
    agree = sum(1 for q in ids if dg.get(q, {}).get("correct") == bl.get(q, {}).get("correct"))
    print(f"agreement: {agree}/{len(ids)}")
    d_only = [q for q in ids if dg.get(q, {}).get("correct") and not bl.get(q, {}).get("correct")]
    b_only = [q for q in ids if bl.get(q, {}).get("correct") and not dg.get(q, {}).get("correct")]
    print(f"dyna-only wins: {d_only}")
    print(f"base-only wins: {b_only}")
    from collections import Counter
    for name, d in [("dyna", dg), ("base", bl)]:
        by = Counter()
        tot = Counter()
        for q in ids:
            if q not in d:
                continue
            g = d[q]["gold_answer_letter"]
            tot[g] += 1
            by[g] += int(d[q]["correct"])
        print(f"{name} by gold: " + ", ".join(f"{L}={by[L]}/{tot[L]}" for L in "ABCD" if tot[L]))
PY

echo "############ EXTENSIVE EVAL END $(date -Is) ############" | tee -a "$LOG"
