#!/usr/bin/env bash
# Wait for a competing emet-habitat run-batch to exit, then launch frontier experiments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAIT_PID="${WAIT_PID:-}"
FAMILY="${FAMILY:-gemma4}"
QSTART="${QSTART:-0}"
QEND="${QEND:-19}"
METHOD="${METHOD:-graph_eqa}"
OUT="${HOME}/.cache/habitat_eqa/results"
LOG="${OUT}/frontier_v2_${FAMILY}_q${QSTART}-${QEND}.launcher.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] launcher: family=${FAMILY} q=${QSTART}-${QEND} method=${METHOD}"

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] waiting for PID ${WAIT_PID} to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date -Is)] PID ${WAIT_PID} exited."
else
  echo "[$(date -Is)] scanning for emet-habitat run-batch processes..."
  while pgrep -f "emet-habitat run-batch" >/dev/null 2>&1; do
    sleep 30
  done
  echo "[$(date -Is)] no run-batch processes found."
fi

# Extra settle time for VRAM release
sleep 15
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true

cd "$ROOT"
export FAMILY QSTART QEND METHOD
exec ./scripts/run_habitat_frontier_experiments.sh
