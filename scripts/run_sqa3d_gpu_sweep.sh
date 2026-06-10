#!/usr/bin/env bash
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Preflight GPU memory and run an isolated real-VLM SQA3D sweep.
# See docs/sqa3d_compute.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MIN_FREE_MIB="${SQA3D_MIN_FREE_MIB:-14000}"
SPLIT="val"
Q_START=0
Q_END=30
REPLAY_MODE="sens"
METHOD="dynagraph"
OUTPUT_DIR="/tmp/sqa3d_gpu_sweep"
DOWNLOAD=0
WITH_SENS=0
DEVICE="cuda"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_sqa3d_gpu_sweep.sh [OPTIONS]

Preflight nvidia-smi free memory, then run:
  emet sqa3d run-real-sweep --isolate-episodes ...

Options:
  --split train|val|test     (default: val)
  --question-start N         (default: 0)
  --question-end N           (default: 30)
  --replay-mode auto|sens|mesh  (default: sens)
  --method dynamem|dynagraph (default: dynagraph)
  --output-dir PATH          (default: /tmp/sqa3d_gpu_sweep)
  --download                 Download ScanNet assets for the slice first
  --with-sens                Pass --with-sens to download (posed RGB-D)
  --device cuda|cpu          (default: cuda)
  --min-free-mib N           Abort if GPU free memory < N MiB (default: 14000)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --question-start) Q_START="$2"; shift 2 ;;
    --question-end) Q_END="$2"; shift 2 ;;
    --replay-mode) REPLAY_MODE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --download) DOWNLOAD=1; shift ;;
    --with-sens) WITH_SENS=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$DEVICE" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; use --device cpu or install NVIDIA driver." >&2
    exit 1
  fi
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "GPU: ${FREE_MIB} MiB free / ${TOTAL_MIB} MiB total (need >= ${MIN_FREE_MIB} MiB)"
  if [[ "$FREE_MIB" -lt "$MIN_FREE_MIB" ]]; then
    echo "ERROR: insufficient free GPU memory. Stop other CUDA jobs or lower SQA3D_MIN_FREE_MIB." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
  uv run emet sqa3d run-real-sweep
  --split "$SPLIT"
  --question-start "$Q_START"
  --question-end "$Q_END"
  --replay-mode "$REPLAY_MODE"
  --method "$METHOD"
  --output-dir "$OUTPUT_DIR"
  --isolate-episodes
  --device "$DEVICE"
)

if [[ "$DOWNLOAD" -eq 1 ]]; then
  CMD+=(--download)
else
  CMD+=(--no-download)
fi
if [[ "$WITH_SENS" -eq 1 ]]; then
  CMD+=(--with-sens)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
