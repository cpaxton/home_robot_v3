#!/usr/bin/env bash
# Memory-confirm GE-only + regress subset gate, then optional annotated37 / paper113.
#
# Usage:
#   nohup ./scripts/run_hmeqa_memory_confirm_gate.sh >> ~/runs/emet/memory_confirm_gate/nohup.log 2>&1 &
#
# Env:
#   RUN_ANNOTATED37=1   also run annotated-37 h2h after the gate (default 1)
#   RUN_PAPER113=1      also run full 113 h2h after annotated37 (default 1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/memory_confirm_gate/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-7200}"
NEED_MIB="${NEED_MIB:-12000}"
# GE-only overnight misses + regress ids from memory-confirm work
IDS_GATE="${IDS_GATE:-4,5,6,11,13,16,18,31,49,57}"
RUN_ANNOTATED37="${RUN_ANNOTATED37:-1}"
RUN_PAPER113="${RUN_PAPER113:-1}"

{
  echo "run_id=$RUN_ID"
  echo "ids_gate=$IDS_GATE"
  git rev-parse --short HEAD
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

log "=== memory-confirm gate (dynagraph) n=$(awk -F, '{print NF}' <<<"$IDS_GATE") ==="
NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
emet_kill_stale_eval_processes
TAG="memory_confirm_gate_${RUN_ID}" IDS="$IDS_GATE" METHOD=dynagraph TIMEOUT="$TIMEOUT" \
  ./scripts/run_habitat_iter_subset.sh 2>&1 | tee "$OUT_DIR/gate_dynagraph.log"

uv run python - <<PY | tee "$OUT_DIR/GATE_SUMMARY.json"
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

tag = "memory_confirm_gate_${RUN_ID}"
p = Path.home() / ".cache/habitat_eqa/results" / f"subset_{tag}_qwen3_vl.jsonl"
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
by_q = {}
for r in rows:
    q = r["question_id"]
    prev = by_q.get(q)
    if prev is None or (episode_run_completed(r) and not episode_run_completed(prev)):
        by_q[q] = r
    elif episode_run_completed(r) == episode_run_completed(prev):
        by_q[q] = r
done = [r for r in by_q.values() if episode_run_completed(r)]
# Memory-steered should be stable; exploration-hard may flake.
stable = {4, 5, 6, 11, 13, 49}
hard = {16, 18, 31, 57}
stable_ok = sum(1 for r in done if r["question_id"] in stable and r.get("correct"))
hard_ok = sum(1 for r in done if r["question_id"] in hard and r.get("correct"))
out = {
    "tag": tag,
    "ok": sum(1 for r in done if r.get("correct")),
    "n": len(done),
    "stable_ok": stable_ok,
    "stable_n": sum(1 for r in done if r["question_id"] in stable),
    "hard_ok": hard_ok,
    "hard_n": sum(1 for r in done if r["question_id"] in hard),
    "rows": [
        {
            "question_id": r["question_id"],
            "gold": r.get("gold_answer_letter"),
            "pred": r.get("parsed_answer_letter"),
            "correct": r.get("correct"),
            "bucket": "stable" if r["question_id"] in stable else "hard",
        }
        for r in sorted(done, key=lambda x: x["question_id"])
    ],
}
print(json.dumps(out, indent=2))
PY

if [[ "$RUN_ANNOTATED37" == "1" ]]; then
  log "=== chaining annotated37 h2h ==="
  RUN_ID="ann37_after_gate_${RUN_ID}" OUT_DIR="$OUT_DIR/annotated37" \
    ./scripts/run_hmeqa_annotated37_h2h.sh 2>&1 | tee -a "$OUT_DIR/chain.log"
fi

if [[ "$RUN_PAPER113" == "1" ]]; then
  log "=== chaining paper113 h2h ==="
  RUN_ID="p113_after_gate_${RUN_ID}" OUT_DIR="$OUT_DIR/paper113" \
    ./scripts/run_hmeqa_paper113_h2h.sh 2>&1 | tee -a "$OUT_DIR/chain.log"
fi

log "DONE memory-confirm gate chain → $OUT_DIR"
