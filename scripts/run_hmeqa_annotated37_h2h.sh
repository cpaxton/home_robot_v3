#!/usr/bin/env bash
# Head-to-head HM-EQA on semantics-annotated paper indices (~37 questions).
# Fairer perception vs GraphEQA GT path than mixed mesh-only scenes.
#
# Usage:
#   nohup ./scripts/run_hmeqa_annotated37_h2h.sh >> ~/runs/emet/hmeqa_annotated37/nohup.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_annotated37/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-7200}"
NEED_MIB="${NEED_MIB:-12000}"

IDS="$(uv run python -c 'from emet.habitat.hm3d_semantics import hmeqa_annotated_question_ids as f; print(",".join(map(str, f())))')"
N="$(awk -F, '{print NF}' <<<"$IDS")"
echo "$IDS" >"$OUT_DIR/IDS.txt"
{
  echo "n=$N"
  git rev-parse --short HEAD
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

run_method() {
  local method="$1"
  local tag="annotated37_${RUN_ID}_${method}"
  log "=== annotated37 method=$method tag=$tag n=$N ==="
  NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
  emet_kill_stale_eval_processes
  TAG="$tag" IDS="$IDS" METHOD="$method" TIMEOUT="$TIMEOUT" \
    ./scripts/run_habitat_iter_subset.sh 2>&1 | tee "$OUT_DIR/${method}.log"
}

run_method graph_eqa
run_method dynagraph

uv run python - <<PY | tee "$OUT_DIR/SUMMARY.txt"
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

run_id = "${RUN_ID}"
root = Path.home() / ".cache/habitat_eqa/results"
for method in ("graph_eqa", "dynagraph"):
    tag = f"annotated37_{run_id}_{method}"
    p = root / f"subset_{tag}_qwen3_vl.jsonl"
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
    print(f"{method}: {ok}/{len(done)} correct ({p})")
PY

log "DONE annotated37 h2h → $OUT_DIR"
