#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Fast OVMM agentic-find gate (rby1; no Stretch head sweeps).
#
# Profiles (env PROFILE):
#   smoke (default)  1x rby1 S0, mapping-rotate-steps 4, max 4 rounds (~5-15 min)
#   slice            2x rby1 (S0 + RoboCasa S1), max 4 rounds (~15-40 min)
#   stretch-legacy   old Stretch 3-ep slice (hours — overnight / paper only)
#
# Usage (GPU-mutexed):
#   uv run emet jobs run --name ovmm-find-rby1-smoke --need-mib 8000 --gpu-exclusive -- ./scripts/run_ovmm_find_recep_slice.sh
#   PROFILE=slice uv run emet jobs run --name ovmm-find-rby1-slice --need-mib 8000 --gpu-exclusive -- ./scripts/run_ovmm_find_recep_slice.sh
#
# Env:
#   OUT_DIR               artifact dir
#   PROFILE               smoke | slice | stretch-legacy
#   AGENTIC_MAX_ROUNDS    default 4 (smoke/slice); stretch-legacy uses 8
#   MAPPING_ROTATE_STEPS  default 4 (smoke/slice); use 8 for voxel-coverage checks

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-smoke}"
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_find_phase/rby1_${PROFILE}_${RUN_ID}}"
mkdir -p "$OUT"

# Do not force nav teleport: default-table mapping needs stable base Z.
# Set EMET_SIM_NAV_TELEPORT=1 yourself for RoboCasa-only slices if desired.
export EMET_ALLOW_SDPA_ATTN=1
# Belt-and-suspenders with DynamemController.look_around skip for non-Stretch.
export EMET_SKIP_HEAD_SWEEP=1

MAP_ROTATE="${MAPPING_ROTATE_STEPS:-4}"
EXTRA=()

if [[ "$PROFILE" == "smoke" ]]; then
    EPISODES=(default_table_rby1_s0_distinct_recep)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "slice" ]]; then
    EPISODES=(default_table_rby1_s0_distinct_recep robocasa_rby1_pp_s1)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "stretch-legacy" ]]; then
    EPISODES=(default_table_s0_distinct_recep robocasa_pp_s1 molmo_ithor_s2_idx0)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-8}"
    EXTRA=()
    unset EMET_SKIP_HEAD_SWEEP
else
    echo "FATAL: unknown PROFILE=$PROFILE (want smoke|slice|stretch-legacy)" >&2
    exit 2
fi

args=(
    uv run emet ovmm find
    --episodes configs/ovmm/find_phase_episodes.yaml
    --backend dynagraph
    --output-dir "$OUT"
    --agentic-max-rounds "$ROUNDS"
)
args+=("${EXTRA[@]}")
for ep in "${EPISODES[@]}"; do
    args+=(--episode-id "$ep")
done

echo "PROFILE=$PROFILE OUT=$OUT ROUNDS=$ROUNDS"
echo "CMD: ${args[*]}"
"${args[@]}"

echo "Summary:"
uv run python scripts/summarize_ovmm_agentic_traces.py --out-dir "$OUT" || true
