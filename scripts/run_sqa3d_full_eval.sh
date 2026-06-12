#!/usr/bin/env bash
# Full SQA3D val/test real-VLM sweep with ScanNet download, resume, and per-episode exports.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SPLIT="${1:-val}"
METHOD="${2:-dynagraph}"
OUTPUT_DIR="${SQA3D_OUTPUT_DIR:-$HOME/runs/emet/sqa3d}"

echo "=== SQA3D full eval: split=$SPLIT method=$METHOD output=$OUTPUT_DIR ==="

echo "=== Download ScanNet (mesh + .sens) for all $SPLIT scenes ==="
uv run python scripts/download_scannet_data.py \
  --accept-tos \
  --scenes-from-sqa3d \
  --split "$SPLIT" \
  --with-sens

echo "=== Real-VLM sweep (resume + exports) ==="
uv run emet sqa3d run-real-sweep \
  --all \
  --split "$SPLIT" \
  --method "$METHOD" \
  --with-sens \
  --replay-mode auto \
  --isolate-episodes \
  --resume \
  --output-dir "$OUTPUT_DIR"

TAG="${METHOD}_${SPLIT}_q0-$(uv run python -c "from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions; print(len(load_sqa3d_questions('$SPLIT')))")"
JSONL="$OUTPUT_DIR/${TAG}.jsonl"

echo "=== Aggregate ==="
uv run python scripts/aggregate_sqa3d_sweep.py "$JSONL" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR"

echo "Done: $JSONL"
