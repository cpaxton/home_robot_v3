#!/usr/bin/env bash
# Run frontier v2 Q0-19 at paper-matched 20/10 after the fast p10m5 sweep finishes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${HOME}/.cache/habitat_eqa/results"
FAST_JSONL="${OUT}/frontier_v2_gemma4_q0-19_p10m5.jsonl"
LOG="${OUT}/frontier_v2_gemma4_q0-19_p20m10.chain.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for fast sweep: ${FAST_JSONL} (20 lines)"

while true; do
  lines=0
  if [[ -f "$FAST_JSONL" ]]; then
    lines=$(wc -l < "$FAST_JSONL" | tr -d ' ')
  fi
  if [[ "$lines" -ge 20 ]] && ! pgrep -f "frontier_v2_gemma4_q0-19_p10m5" >/dev/null 2>&1; then
    break
  fi
  sleep 60
done

echo "[$(date -Is)] fast sweep done (${lines} lines). Summarizing..."
python3 "${ROOT}/scripts/summarize_habitat_run.py" \
  "${OUT}/graph_eqa_gemma3_paper_q0-112.jsonl" \
  "$FAST_JSONL" || true

sleep 15
cd "$ROOT"
export FAMILY=gemma4 QSTART=0 QEND=19 METHOD=graph_eqa MAX_PLANNING=20 MAX_MOVEMENT=10 RUN_SUFFIX=_p20m10
exec ./scripts/run_habitat_frontier_experiments.sh
