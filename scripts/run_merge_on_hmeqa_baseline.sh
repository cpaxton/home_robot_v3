#!/usr/bin/env bash
# Merge-on HM-EQA Dynagraph re-baseline after Q17 gate passes.
# Holdout8 + smoke trio under default harness (no explore/merge CLI overrides).
#
# Usage (prefer nohup for overnight):
#   nohup ./scripts/run_merge_on_hmeqa_baseline.sh \
#     >> ~/runs/emet/branch_verify_20260711/merge_on_baseline_nohup.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-$HOME/runs/emet/branch_verify_20260711}"
BASE_ROOT="${BASE_ROOT:-$RUN_DIR/merge_on_baseline_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BASE_ROOT"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
LOG="$BASE_ROOT/baseline.log"
SUMMARY="$BASE_ROOT/SUMMARY.json"
: >"$LOG"

HOLDOUT_IDS="${HOLDOUT_IDS:-15,56,65,68,79,88,104,105}"
SMOKE_IDS="${SMOKE_IDS:-3,14,17}"
TIMEOUT="${TIMEOUT:-7200}"

if [[ ! -x "$EMET_HABITAT" ]]; then
  echo "Missing emet-habitat at $EMET_HABITAT" | tee -a "$LOG"
  exit 1
fi

echo "===== merge-on baseline $(date -Is) =====" | tee -a "$LOG"
echo "BASE_ROOT=$BASE_ROOT" | tee -a "$LOG"

./scripts/gpu_preflight.sh --kill-stale >>"$LOG" 2>&1 || true
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait >>"$LOG" 2>&1

run_ids() {
  local name="$1"
  local ids_csv="$2"
  local out="$BASE_ROOT/${name}.jsonl"
  local elog="$BASE_ROOT/${name}.log"
  : >"$out"
  echo "===== $name ids=$ids_csv $(date -Is) =====" | tee -a "$LOG"
  IFS=',' read -r -a ids <<<"$ids_csv"
  for qid in "${ids[@]}"; do
    qid="$(echo "$qid" | tr -d '[:space:]')"
    [[ -n "$qid" ]] || continue
    ep_out="$BASE_ROOT/${name}_q${qid}.jsonl"
    : >"$ep_out"
    echo "----- question $qid $(date -Is) -----" | tee -a "$LOG"
    timeout "$TIMEOUT" "$EMET_HABITAT" run-episode \
      --question-id "$qid" \
      --method dynagraph \
      --output "$ep_out" \
      >>"$elog" 2>&1 || echo "episode $qid failed exit=$?" | tee -a "$LOG"
    if [[ -s "$ep_out" ]]; then
      cat "$ep_out" >>"$out"
    fi
  done
  cat "$elog" >>"$LOG" || true
}

run_ids "holdout8" "$HOLDOUT_IDS"
run_ids "smoke_trio" "$SMOKE_IDS"

python3 - "$BASE_ROOT" "$SUMMARY" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

def load_jsonl(name: str):
    p = root / f"{name}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def summarize(name: str, rows: list):
    ok = sum(1 for r in rows if r.get("correct"))
    harness = (rows[0].get("harness") if rows else {}) or {}
    return {
        "name": name,
        "correct": ok,
        "total": len(rows),
        "accuracy": (ok / len(rows)) if rows else None,
        "harness_sample": harness,
        "per_question": [
            {
                "question_id": r.get("question_id"),
                "correct": bool(r.get("correct")),
                "predicted_answer": r.get("predicted_answer"),
                "gold_answer_letter": r.get("gold_answer_letter"),
                "graph_nodes": r.get("graph_nodes") or (r.get("graph") or {}).get("num_nodes"),
            }
            for r in rows
        ],
    }

holdout = summarize("holdout8", load_jsonl("holdout8"))
smoke = summarize("smoke_trio", load_jsonl("smoke_trio"))
out = {
    "baseline": "merge_on_hmeqa_dynagraph",
    "note": "New Dynagraph baseline under unified_eqa merge 0.45; not comparable to zero-merge 8/8.",
    "holdout8": holdout,
    "smoke_trio": smoke,
}
summary_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"holdout8: {holdout['correct']}/{holdout['total']}")
for q in holdout["per_question"]:
    print(f"  q{q['question_id']}: {'OK' if q['correct'] else 'FAIL'} pred={q['predicted_answer']} gold={q['gold_answer_letter']}")
print(f"smoke_trio: {smoke['correct']}/{smoke['total']}")
for q in smoke["per_question"]:
    print(f"  q{q['question_id']}: {'OK' if q['correct'] else 'FAIL'} pred={q['predicted_answer']} gold={q['gold_answer_letter']}")
print(f"SUMMARY={summary_path}")
PY

echo "===== DONE $(date -Is) =====" | tee -a "$LOG"
