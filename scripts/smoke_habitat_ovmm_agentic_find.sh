#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Agentic OVMM find loop on Habitat — bounded smoke on a large HM3D scene.
#
# Runs FindObj + FindRec through the shared AgenticEQA loop (same loop as HM-EQA /
# sim OVMM find) on a whole-home HM3D scene. Requires GPU (Qwen3-VL + SigLIP + YoloE).
#
# Usage:
#   bash scripts/smoke_habitat_ovmm_agentic_find.sh [episode_id] [backend]
#   NEED_MIB=12000 uv run emet jobs run --name habitat-ovmm-agentic-find -- \
#     bash scripts/smoke_habitat_ovmm_agentic_find.sh
#
# Defaults:
#   episode: hm3d_lamp_bed_00006 (scene 00006-HkseAnWCgqk, large real home)
#   backend: dynagraph (agentic find on by default)

set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HABITAT_BIN="$REPO/.venv-habitat/bin/emet-habitat"
EPISODE_ID="${1:-hm3d_lamp_bed_00006}"
BACKEND="${2:-dynagraph}"
OUT_DIR="${EMET_OVMM_OUTPUT_HABITAT:-$HOME/runs/emet/ovmm_habitat}/agentic_smoke"

if [ ! -x "$HABITAT_BIN" ]; then
    echo "missing .venv-habitat — run ./scripts/install_habitat.sh" >&2
    exit 1
fi

if [ "${EMET_AGENTIC_FIND_SKIP_GPU_CHECK:-0}" != "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi unavailable — agentic find needs a GPU (VLM + SigLIP + YoloE)" >&2
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"
export EMET_EQA_EPISODE_DIR="$OUT_DIR"

echo "[smoke] agentic OVMM find on Habitat scene $EPISODE_ID (backend=$BACKEND)" >&2
"$HABITAT_BIN" run-ovmm-find-episode \
    --episode-id "$EPISODE_ID" \
    --backend "$BACKEND" \
    --agentic-max-rounds 4 \
    --agentic-max-nav-steps 4 \
    --output "$OUT_DIR/${EPISODE_ID}_${BACKEND}.json"
rc=$?

echo "[smoke] wrote $OUT_DIR/${EPISODE_ID}_${BACKEND}.json" >&2
exit "$rc"