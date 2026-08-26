#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Dynamem-only OVMM find diagnostic (no agentic/VLM loop).
# Usage:
#   OUT_DIR=~/runs/emet/ovmm_find_phase/rby1_dynamem_… \
    #     uv run emet jobs run --name ovmm-find-dynamem-diag --need-mib 8000 --gpu-exclusive \
    #     --out-dir "$OUT_DIR" -- ./scripts/run_ovmm_dynamem_diag.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_find_phase/rby1_dynamem_${RUN_ID}}"
mkdir -p "$OUT"

export EMET_SKIP_HEAD_SWEEP=1

args=(
    uv run emet ovmm find
    --episodes configs/ovmm/find_phase_episodes.yaml
    --backend dynamem
    --output-dir "$OUT"
    --mapping-rotate-steps "${MAPPING_ROTATE_STEPS:-4}"
    --episode-id "${EPISODE_ID:-default_table_rby1_s0_distinct_recep}"
)

echo "OUT=$OUT backend=dynamem episode=${EPISODE_ID:-default_table_rby1_s0_distinct_recep}"
echo "CMD: ${args[*]}"
"${args[@]}"

echo "Summary:"
uv run python scripts/summarize_ovmm_agentic_traces.py --out-dir "$OUT" || true
