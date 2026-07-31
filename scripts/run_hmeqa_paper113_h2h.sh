#!/usr/bin/env bash
# Full GraphEQA-paper HM-EQA set (113 questions, indices 0–112) head-to-head.
# Prefer nohup overnight; one method at a time (GPU preflight).
#
# Usage:
#   nohup ./scripts/run_hmeqa_paper113_h2h.sh >> ~/runs/emet/hmeqa_paper113/nohup.log 2>&1 &
#
# Env:
#   METHODS   space-separated methods (default: "static_graph dynagraph")
#   TIMEOUT   per-batch wall timeout seconds (default 86400)
#   NEED_MIB  VRAM gate (default 12000)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_paper113/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-86400}"
NEED_MIB="${NEED_MIB:-12000}"
METHODS="${METHODS:-static_graph dynagraph}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"

{
  echo "run_id=$RUN_ID"
  echo "methods=$METHODS"
  git rev-parse --short HEAD
  git rev-parse HEAD
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

run_method() {
  local method="$1"
  local tag="paper113_${RUN_ID}_${method}"
  local jsonl="$HOME/.cache/habitat_eqa/results/subset_${tag}_${FAMILY}.jsonl"
  local logf="$OUT_DIR/${method}.log"
  log "=== paper113 method=$method tag=$tag ==="
  NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
  emet_kill_stale_eval_processes
  timeout "$TIMEOUT" "$HAB" run-batch \
    --method "$method" \
    --paper-subset \
    --max-planning-steps 20 \
    --max-movement-step 10 \
    --eqa-vl-family "$FAMILY" \
    --eqa-hf-model-id "$HF_ID" \
    --device cuda \
    --frontier-nodes \
    --frontier-keyword-weight 2 \
    --resume \
    --output "$jsonl" \
    2>&1 | tee "$logf"
  echo "$jsonl" >"$OUT_DIR/${method}_jsonl.path"
}

for method in $METHODS; do
  run_method "$method"
done

uv run python - <<PY | tee "$OUT_DIR/SUMMARY.txt"
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

run_id = "${RUN_ID}"
family = "${FAMILY}"
root = Path.home() / ".cache/habitat_eqa/results"
for method in "${METHODS}".split():
    tag = f"paper113_{run_id}_{method}"
    p = root / f"subset_{tag}_{family}.jsonl"
    if not p.exists():
        print(f"{method}: missing {p}")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by_q = {}
    for r in rows:
        q = r["question_id"]
        prev = by_q.get(q)
        if prev is None or (episode_run_completed(r) and not episode_run_completed(prev)):
            by_q[q] = r
        elif episode_run_completed(r) == episode_run_completed(prev):
            by_q[q] = r
    done = [r for r in by_q.values() if episode_run_completed(r)]
    ok = sum(1 for r in done if r.get("correct"))
    print(f"{method}: {ok}/{len(done)} correct ({p.name})")
PY

log "DONE paper113 h2h → $OUT_DIR"
