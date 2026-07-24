#!/usr/bin/env bash
# Priority Phase-1/2 paper cells (not the full large_eval matrix).
# Run only after smoke gate passes. One GPU job at a time.
#
# Usage:
#   nohup ./scripts/run_dynagraph_dynamic_paper_cells.sh [OUT_DIR] &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynamic_exploration/paper_cells_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

preflight() {
  NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
}

run_explore() {
  local episode="$1" backend="$2" k="$3" mapping="$4" tag="$5"
  preflight
  log "START $tag"
  local args=(
    --phase explore
    --episode-id "$episode"
    --backend "$backend"
    --mapping-mode "$mapping"
    --output-dir "$OUT/phase1"
  )
  if [[ "$mapping" == "explore" ]]; then
    args+=(--explore-max-iters "$k")
  fi
  uv run python scripts/eval_dynamic_exploration.py "${args[@]}" \
    2>&1 | tee -a "$OUT/phase1_${tag}.log"
  log "DONE $tag"
}

run_world_change() {
  local backend="$1"
  preflight
  log "START world-change_$backend"
  uv run python scripts/eval_dynamic_exploration.py \
    --phase world-change --episode-id robocasa_seed0_world_change \
    --backend "$backend" \
    --output-dir "$OUT/phase2" \
    2>&1 | tee -a "$OUT/phase2_${backend}.log"
  log "DONE world-change_$backend"
}

log "OUT=$OUT"
# Phase 1 priority cells (tab:dynamic_explore_phase1)
run_explore robocasa_seed0 dynagraph 8 explore robocasa_s0_dyna_k8
run_explore robocasa_seed0 dynagraph 15 explore robocasa_s0_dyna_k15
run_explore robocasa_seed0 dynagraph 0 rotate_only robocasa_s0_dyna_rotate
run_explore robocasa_seed0 graph_eqa 8 explore robocasa_s0_graph_k8
run_explore molmo_ithor0 dynagraph 15 explore molmo_i0_dyna_k15

# Phase 2 (tab:dynamic_explore_world_change)
run_world_change dynagraph
run_world_change graph_eqa

echo DONE > "$OUT/DONE"
log "All paper cells finished"
