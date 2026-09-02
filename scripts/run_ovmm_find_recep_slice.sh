#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Fast OVMM agentic-find gate (rby1 default; Stretch optional without head pans).
#
# Profiles (env PROFILE):
#   oneshot          1x rby1 S0, 4-step spin, --oneshot-localize (no AgenticEQA)
#                    ~1–3 min. Inner loop: is the object in the voxel map?
#   verify           Robocasa rby1 drive-up + YOLOE/SigLIP on current RGB (~5–15 min)
#   smoke (default)  1x rby1 S0, mapping-rotate-steps 4, max 6 rounds (~5-15 min)
#   slice            2x rby1 (S0 + RoboCasa S1), max 6 rounds (~15-40 min)
#   stretch          1x Stretch table S0, skip head pan (look_front camera)
#   stretch-kitchen  1x Stretch RoboCasa S1, skip head pan
#   stretch-legacy   old Stretch 3-ep slice WITH head pans (EMET_FORCE_HEAD_SWEEP; hours — paper only)
#
# Usage (GPU-mutexed):
#   PROFILE=verify uv run emet jobs run --name ovmm-probe-verify-rby1 --need-mib 8000 --gpu-exclusive -- \
    #     ./scripts/run_ovmm_find_recep_slice.sh
#   PROFILE=oneshot uv run emet jobs run --name ovmm-find-rby1-oneshot --need-mib 8000 --gpu-exclusive -- \
    #     ./scripts/run_ovmm_find_recep_slice.sh
#   uv run emet jobs run --name ovmm-find-rby1-smoke --need-mib 8000 --gpu-exclusive -- ./scripts/run_ovmm_find_recep_slice.sh
#   PROFILE=slice uv run emet jobs run --name ovmm-find-rby1-slice --need-mib 8000 --gpu-exclusive -- ./scripts/run_ovmm_find_recep_slice.sh
#   PROFILE=stretch uv run emet jobs run --name ovmm-find-stretch-nosweep --need-mib 8000 --gpu-exclusive -- \
    #     ./scripts/run_ovmm_find_recep_slice.sh
#
# Env:
#   OUT_DIR               artifact dir
#   PROFILE               oneshot | verify | smoke | slice | stretch | stretch-kitchen | stretch-legacy
#   AGENTIC_MAX_ROUNDS    default 6 (smoke/slice/stretch); stretch-legacy uses 8
#   MAPPING_ROTATE_STEPS  default 4 (smoke/slice/oneshot/stretch); use 8 for voxel-coverage checks

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-smoke}"
case "$PROFILE" in
    stretch|stretch-kitchen|stretch-legacy) OUT_PREFIX=stretch ;;
    *) OUT_PREFIX=rby1 ;;
esac
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_find_phase/${OUT_PREFIX}_${PROFILE}_${RUN_ID}}"
mkdir -p "$OUT"

# Do not force nav teleport: default-table mapping needs stable base Z.
# Set EMET_SIM_NAV_TELEPORT=1 yourself for RoboCasa-only slices if desired.
export EMET_ALLOW_SDPA_ATTN=1
# Head pans are off in YAML (`mapping.look_around_head_sweep: false`). Only
# PROFILE=stretch-legacy forces pans via EMET_FORCE_HEAD_SWEEP.

MAP_ROTATE="${MAPPING_ROTATE_STEPS:-4}"
EXTRA=()

if [[ "$PROFILE" == "oneshot" ]]; then
    EPISODES=(default_table_rby1_s0_distinct_recep)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-1}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE" --oneshot-localize)
elif [[ "$PROFILE" == "verify" ]]; then
    echo "PROFILE=verify OUT=$OUT"
    export EMET_SIM_NAV_TELEPORT=1
    exec uv run emet ovmm probe-verify --out "$OUT"
elif [[ "$PROFILE" == "smoke" ]]; then
    EPISODES=(default_table_rby1_s0_distinct_recep)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "slice" ]]; then
    EPISODES=(default_table_rby1_s0_distinct_recep robocasa_rby1_pp_s1)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "stretch" ]]; then
    EPISODES=(default_table_s0_distinct_recep)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "stretch-kitchen" ]]; then
    EPISODES=(robocasa_pp_s1)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-6}"
    EXTRA=(--mapping-rotate-steps "$MAP_ROTATE")
elif [[ "$PROFILE" == "stretch-legacy" ]]; then
    EPISODES=(default_table_s0_distinct_recep robocasa_pp_s1 molmo_ithor_s2_idx0)
    ROUNDS="${AGENTIC_MAX_ROUNDS:-8}"
    EXTRA=()
    export EMET_FORCE_HEAD_SWEEP=1
else
    echo "FATAL: unknown PROFILE=$PROFILE (want oneshot|verify|smoke|slice|stretch|stretch-kitchen|stretch-legacy)" >&2
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
