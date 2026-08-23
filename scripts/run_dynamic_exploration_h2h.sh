#!/usr/bin/env bash
# Dynagraph vs static_graph (legacy graph_eqa_baseline) on the dynamic exploration
# matrix (world-change + lifelong). Run in a separate GPU session from Habitat 113.
#
# Usage:
#   nohup ./scripts/run_dynamic_exploration_h2h.sh >> ~/runs/emet/dynamic_exploration/h2h_nohup.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/dynamic_exploration/h2h_${RUN_ID}}"
mkdir -p "$OUT_DIR"
NEED_MIB="${NEED_MIB:-8000}"

{
    echo "run_id=$RUN_ID"
    git rev-parse --short HEAD
    echo "watch: $OUT_DIR/runner.log $OUT_DIR/progress.jsonl"
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

log "GPU preflight"
NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
emet_kill_stale_eval_processes

# Prefer full matrix script when present; else phased eval_dynamic_exploration.
if [[ -x "${ROOT}/scripts/run_dynamic_exploration_full.sh" ]]; then
    log "Running run_dynamic_exploration_full.sh → $OUT_DIR"
    EMET_DYNAMIC_EXPLORE_OUTPUT="$OUT_DIR" \
        ./scripts/run_dynamic_exploration_full.sh 2>&1 | tee "$OUT_DIR/full.log"
else
    log "Falling back to eval_dynamic_exploration.py phases"
    for phase in explore world-change lifelong; do
        log "phase=$phase"
        uv run python scripts/eval_dynamic_exploration.py \
            --phase "$phase" \
            --output-dir "$OUT_DIR" \
            2>&1 | tee -a "$OUT_DIR/${phase}.log"
    done
fi

log "DONE dynamic exploration h2h → $OUT_DIR"
ls -la "$OUT_DIR" | tee "$OUT_DIR/listing.txt"
