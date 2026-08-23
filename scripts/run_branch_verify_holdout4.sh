#!/usr/bin/env bash
# Small holdout slice for branch verify (explore-off = tuned winner).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${RUN_DIR:-$HOME/runs/emet/branch_verify_20260711}"
mkdir -p "$RUN_DIR"
OUT="$RUN_DIR/holdout4_explore_off_qwen3_vl.jsonl"
LOG="$RUN_DIR/holdout4_explore_off_qwen3_vl.log"
: >"$OUT"
: >"$LOG"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
for qid in 15 68 105 17; do
    echo "===== question $qid $(date -Is) =====" | tee -a "$LOG"
    ep_out="$RUN_DIR/holdout4_q${qid}.jsonl"
    : >"$ep_out"
    "$EMET_HABITAT" run-episode \
        --question-id "$qid" \
        --method dynagraph \
        --explore-when-uncovered off \
        --no-mcq-debias \
        --memory-summary \
        --output "$ep_out" \
        >>"$LOG" 2>&1 || echo "episode $qid failed exit=$?" | tee -a "$LOG"
    if [[ -s "$ep_out" ]]; then
        cat "$ep_out" >>"$OUT"
    fi
done
echo "===== ALL DONE $(date -Is) =====" | tee -a "$LOG"
python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
ok = sum(1 for r in rows if r.get("correct"))
print(f"holdout4 summary: {ok}/{len(rows)}")
for r in rows:
    print(
        f"  q{r.get('question_id')}: "
        f"{'OK' if r.get('correct') else 'FAIL'} "
        f"pred={r.get('predicted_answer')} gold={r.get('gold_answer_letter')}"
    )
PY
