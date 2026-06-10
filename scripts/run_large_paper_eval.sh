#!/usr/bin/env bash
# Large paper eval: full SQA3D (val+test, dynamem+dynagraph) + OVMM find-phase ladder.
#
# GPU jobs run sequentially (one VLM/isolated-episode sweep at a time).
# Logs: ~/runs/emet/large_eval/<phase>.log
#
# Usage:
#   ./scripts/run_large_paper_eval.sh              # full queue
#   ./scripts/run_large_paper_eval.sh sqa3d-val    # one phase
#   ./scripts/run_large_paper_eval.sh ovmm         # OVMM only (after SQA3D or standalone)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export MUJOCO_GL=egl
export PATH="${HOME}/.local/bin:${PATH}"

STAMP="$(date +%Y%m%d_%H%M)"
LOG_DIR="${EMET_LARGE_EVAL_LOG_DIR:-$HOME/runs/emet/large_eval}"
SQA3D_OUT="${EMET_SQA3D_OUTPUT:-$HOME/runs/emet/sqa3d}"
OVMM_OUT="${EMET_OVMM_OUTPUT_SIM:-$HOME/runs/emet/ovmm_find_phase}/large_${STAMP}"

mkdir -p "$LOG_DIR" "$SQA3D_OUT" "$OVMM_OUT"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG_DIR/master.log"; }

run_sqa3d_sweep() {
  local split="$1"
  local method="$2"
  local phase="sqa3d_${split}_${method}"
  local logfile="$LOG_DIR/${phase}.log"
  log "START $phase (split=$split method=$method)"
  {
    echo "=== $phase start $(date -Is) ==="
    uv run emet sqa3d run-real-sweep \
      --all \
      --split "$split" \
      --method "$method" \
      --no-download \
      --resume \
      --replay-mode auto \
      --isolate-episodes \
      --output-dir "$SQA3D_OUT"
    echo "=== $phase done $(date -Is) exit=0 ==="
  } >>"$logfile" 2>&1
  log "DONE $phase"
}

wait_test_scannet() {
  log "Waiting for test ScanNet download (target 67/67 mesh+sens)..."
  while true; do
    stats="$(uv run python -c "
from emet.benchmarks.sqa3d.scannet.config import collect_sqa3d_scene_ids, count_scannet_scenes_on_disk, default_scannet_root, scene_sens_present, scene_assets_present
scenes = collect_sqa3d_scene_ids('test')
root = default_scannet_root()
mesh = sum(1 for s in scenes if scene_assets_present(s, root))
sens = sum(1 for s in scenes if scene_sens_present(s, root))
print(len(scenes), mesh, sens)
")"
    read -r req mesh sens <<<"$stats"
    log "test ScanNet: mesh=${mesh}/${req} sens=${sens}/${req}"
    if [[ "${mesh:-0}" -ge "${req:-67}" && "${sens:-0}" -ge "${req:-67}" ]]; then
      break
    fi
    sleep 300
  done
  log "Test ScanNet complete"
}

aggregate_sqa3d() {
  log "Aggregating SQA3D JSONL sweeps"
  local jsonls=()
  for f in \
    "$SQA3D_OUT/dynagraph_val_q0-3261.jsonl" \
    "$SQA3D_OUT/dynamem_val_q0-3261.jsonl" \
    "$SQA3D_OUT/dynagraph_test_q0-3519.jsonl" \
    "$SQA3D_OUT/dynamem_test_q0-3519.jsonl"
  do
    [[ -f "$f" ]] && jsonls+=("$f")
  done
  if [[ ${#jsonls[@]} -eq 0 ]]; then
    log "No SQA3D JSONL files to aggregate yet"
    return 0
  fi
  uv run python scripts/aggregate_sqa3d_sweep.py \
    "${jsonls[@]}" \
    --split val \
    --output-dir "$SQA3D_OUT" \
    --csv-name aggregate_large_paper_eval.csv \
    --json-name aggregate_large_paper_eval.json \
    2>&1 | tee -a "$LOG_DIR/aggregate_sqa3d.log" || true
}

run_ovmm_ladder() {
  local logfile="$LOG_DIR/ovmm_replicates.log"
  local cpu_flag=()
  if [[ "${OVMM_CPU_ONLY:-0}" == "1" ]]; then
    cpu_flag=(--cpu-only)
    log "OVMM using --cpu-only (parallel-safe with SQA3D GPU sweeps)"
  fi
  log "START OVMM find-phase ladder (9 episodes × 4 backends × 5 seeds)"
  {
    echo "=== OVMM replicates start $(date -Is) output=$OVMM_OUT ==="
    uv run python scripts/replicate_ovmm_find_phases.py \
      --backend dynamem \
      --backend graph_eqa \
      --backend dynagraph \
      --backend ground_truth \
      --replicates 5 \
      --seed-base 0 \
      "${cpu_flag[@]}" \
      --output-dir "$OVMM_OUT"
    echo "=== OVMM replicates done $(date -Is) exit=0 ==="
  } >>"$logfile" 2>&1
  log "DONE OVMM → $OVMM_OUT/replicates/aggregate_replicates.json"
}

restart_val_dynagraph_if_partial() {
  local jsonl="$SQA3D_OUT/dynagraph_val_q0-3261.jsonl"
  local n_done=0
  if [[ -f "$jsonl" ]]; then
    n_done="$(wc -l <"$jsonl" | tr -d ' ')"
  fi
  local n_runnable
  n_runnable="$(uv run python -c "from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions; from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, filter_questions_with_scannet; print(len(filter_questions_with_scannet(load_sqa3d_questions('val'), default_scannet_root(), replay_mode='auto')))")"
  if [[ "$n_done" -lt "$n_runnable" ]]; then
    log "Restarting val dynagraph eval ($n_done/$n_runnable done; val ScanNet now complete)"
    pkill -f "emet sqa3d run-real-sweep --all --split val --method dynagraph" 2>/dev/null || true
    sleep 5
  fi
}

run_sqa3d_track() {
  restart_val_dynagraph_if_partial
  run_sqa3d_sweep val dynagraph
  run_sqa3d_sweep val dynamem
  wait_test_scannet
  run_sqa3d_sweep test dynagraph
  run_sqa3d_sweep test dynamem
  aggregate_sqa3d
}

PHASE="${1:-all}"

case "$PHASE" in
  all)
    log "=== Large paper eval START (phase=all) ==="
    log "SQA3D out=$SQA3D_OUT  OVMM out=$OVMM_OUT  logs=$LOG_DIR"
    run_sqa3d_track
    if [[ "${SKIP_OVMM:-0}" != "1" ]]; then
      run_ovmm_ladder
    fi
    log "=== Large paper eval COMPLETE ==="
    ;;
  sqa3d-val)
    restart_val_dynagraph_if_partial
    run_sqa3d_sweep val dynagraph
    run_sqa3d_sweep val dynamem
    ;;
  sqa3d-test)
    wait_test_scannet
    run_sqa3d_sweep test dynagraph
    run_sqa3d_sweep test dynamem
    aggregate_sqa3d
    ;;
  ovmm)
    run_ovmm_ladder
    ;;
  aggregate)
    aggregate_sqa3d
    ;;
  *)
    echo "Unknown phase: $PHASE (use: all | sqa3d-val | sqa3d-test | ovmm | aggregate)" >&2
    exit 1
    ;;
esac
