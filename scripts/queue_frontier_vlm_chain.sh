#!/usr/bin/env bash
# Run a second frontier HM-EQA sweep after the gemma4 frontier_v2 Q0-19 job finishes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${HOME}/.cache/habitat_eqa/results"
GEMMA_JSONL="${OUT}/frontier_v2_gemma4_q0-19.jsonl"
LOG="${OUT}/frontier_v2_qwen3_vl_q0-19.chain.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] chain: waiting for gemma4 frontier_v2 sweep to finish (${GEMMA_JSONL})"

while true; do
    lines=0
    if [[ -f "$GEMMA_JSONL" ]]; then
        lines=$(wc -l < "$GEMMA_JSONL" | tr -d ' ')
    fi
    if [[ "$lines" -ge 20 ]] && ! pgrep -f "frontier_v2_gemma4_q0-19" >/dev/null 2>&1 && ! pgrep -f "emet-habitat run-batch.*frontier_v2_gemma4" >/dev/null 2>&1; then
        break
    fi
    sleep 60
done

echo "[$(date -Is)] gemma4 sweep done (${lines} lines). Sleeping 15s for VRAM..."
sleep 15
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true

cd "$ROOT"
export FAMILY=qwen3_vl QSTART=0 QEND=19 METHOD=graph_eqa
exec ./scripts/run_habitat_frontier_experiments.sh
