#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Seeded hill-climb dev loop: EQA ~12-q + OVMM RoboCasa S1 rby1, then score.
#
# This is the FAST inner loop (dev sets), not the gate. Gate = full holdout-8 +
# balanced-32 (EQA) and Habitat HM3D nearest-v2 (OVMM), run only to promote a win.
#
# Usage (GPU-mutexed, serial — do NOT run inline):
#   DEV_SEED=0 uv run emet jobs run --name hillclimb-dev --need-mib 12000 --gpu-exclusive -- \
    #       ./scripts/run_dev_loop.sh
#
# Env:
#   DEV_SEED       seed (default 0); pass the same value for matched repeats
#   OUT_DIR        artifact dir (default ~/runs/emet/hillclimb_dev/<stamp>)
#   PHASE          eqa | ovmm | all (default all)
#   DEV_EQA_IDS    space-separated EQA question ids (default mirrors
#                  configs/benchmarks/dev_sets.yaml eqa.dev_ids)
#   DEV_OVMM_EP    OVMM episode id (default robocasa_rby1_pp_s1)
#   HABITAT_BIN    emet-habitat executable (default $ROOT/.venv-habitat/bin/emet-habitat)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${DEV_SEED:-0}"
OUT="${OUT_DIR:-$HOME/runs/emet/hillclimb_dev/$(date +%Y%m%d_%H%M%S)}"
PHASE="${PHASE:-all}"
EQA_IDS="${DEV_EQA_IDS:-2 6 12 14 15 16 25 28 31 56 65 68}"
OVMM_EP="${DEV_OVMM_EP:-robocasa_rby1_pp_s1}"
HABITAT_BIN="${HABITAT_BIN:-$ROOT/.venv-habitat/bin/emet-habitat}"
HABITAT_ENV="$(dirname "$(dirname "$HABITAT_BIN")")"
# Direct venv python (not `uv run`): frozen worktrees have no synced .venv, and
# `uv run` re-syncs and fails on missing third_party submodules. Override with
# EMET_PY to point at a sibling checkout's .venv.
PYTHON="${EMET_PY:-$ROOT/.venv/bin/python}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EMET_ALLOW_SDPA_ATTN=1 MUJOCO_GL=egl
export EMET_UV_RUN=1
export HABITAT_EQA_DATA_DIR="${HABITAT_EQA_DATA_DIR:-$HOME/.cache/habitat_eqa/data}"
export PYTHONPATH="$ROOT/src:$ROOT/packages/emet_habitat"

mkdir -p "$OUT/eqa" "$OUT/ovmm"
echo "seed=$SEED phase=$PHASE root=$ROOT sha=$(git rev-parse HEAD)" > "$OUT/manifest.txt"

# Failures never abort the loop (partial runs stay scoreable) but are listed
# per-episode below and in $OUT/failed.txt, and called out again after scoring.
FAILED=()

if [[ "$PHASE" == all || "$PHASE" == eqa ]]; then
    for q in $EQA_IDS; do
        echo "=== [eqa] q$q seed=$SEED ($(date -Is)) ==="
        rc=0
        LD_LIBRARY_PATH="$HABITAT_ENV/lib:${LD_LIBRARY_PATH:-}" \
            timeout --signal=TERM --kill-after=20s 600 \
            "$HABITAT_BIN" run-episode --question-id "$q" --method lazy_graph --query-driven-memory \
            --seed "$SEED" --max-planning-steps 20 --max-movement-step 10 \
            --no-hm3d-semantics --no-enrich-labels --output "$OUT/eqa/q$q.jsonl" || rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "!!! [eqa] q$q FAILED (exit $rc; exit 139/134 = native crash, not a scored miss)"
            FAILED+=("eqa:q$q:exit$rc")
        elif [[ ! -s "$OUT/eqa/q$q.jsonl" ]]; then
            echo "!!! [eqa] q$q exited 0 but wrote NO metrics (empty jsonl) — treat as a crash, not a miss"
            FAILED+=("eqa:q$q:empty")
        fi
        sleep 15
    done
fi

if [[ "$PHASE" == all || "$PHASE" == ovmm ]]; then
    echo "=== [ovmm] $OVMM_EP seed=$SEED ($(date -Is)) ==="
    rc=0
    "$PYTHON" -m emet.cli ovmm find --episodes configs/ovmm/find_phase_episodes.yaml \
        --backend lazy_graph --query-driven-memory --episode-id "$OVMM_EP" \
        --seed "$SEED" --mapping-rotate-steps 4 --output-dir "$OUT/ovmm" || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! [ovmm] $OVMM_EP FAILED (exit $rc)"
        FAILED+=("ovmm:$OVMM_EP:exit$rc")
    fi
fi

echo "=== scoring ==="
if [[ "$PHASE" == all || "$PHASE" == eqa ]]; then
    if ! "$PYTHON" scripts/score_eqa.py "$OUT/eqa"; then
        echo "!!! score_eqa.py produced no summary — no readable q*.jsonl rows under $OUT/eqa?"
        FAILED+=("scoring:eqa:no-rows")
    fi
    echo
fi
if [[ "$PHASE" == all || "$PHASE" == ovmm ]]; then
    if ! "$PYTHON" scripts/score_ovmm.py "$OUT/ovmm"; then
        echo "!!! score_ovmm.py produced no summary — no find-phase *.json under $OUT/ovmm?"
        FAILED+=("scoring:ovmm:no-rows")
    fi
fi

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== dev loop finished WITH ${#FAILED[@]} FAILURE(S): scores above are NOT a clean run ==="
    printf '  FAIL %s\n' "${FAILED[@]}"
    printf '%s\n' "${FAILED[@]}" > "$OUT/failed.txt"
    exit 1
else
    echo "=== dev loop finished clean: every requested episode produced metrics ==="
fi
