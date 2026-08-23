#!/usr/bin/env bash
# Launch paper dynamic-exploration + OVMM find matrix under nohup once GPU is free.
# Does not stack Habitat / full pytest in the same session.
#
# Tracks run **sequentially** (one GPU-heavy job at a time):
#   1) OVMM find backend matrix
#   2) dynamic exploration full matrix
#
# A flock prevents a second launch from overlapping an in-flight eval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${EMET_DYNAMIC_MEMORY_OUT:-$HOME/runs/emet/dynagraph_dynamic_memory}"
NEED_MIB="${NEED_MIB:-12000}"
LOCK_FILE="${EMET_DYNAMIC_MEMORY_LOCK:-$OUT_ROOT/eval.lock}"
mkdir -p "$OUT_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[launch] ERROR: another dynagraph dynamic-memory eval holds $LOCK_FILE" >&2
    echo "[launch] Wait for it to finish, or remove the lock only if you know it is stale." >&2
    exit 1
fi

echo "[preflight] waiting for >= ${NEED_MIB} MiB free GPU…"
# Do not --kill-stale here: that pkill's mujoco_server / dynagraph and can wipe
# in-flight CPU smokes. Operator may run kill-stale manually when idle.
NEED_MIB="$NEED_MIB" ./scripts/gpu_preflight.sh --wait

# Refuse to start if a previous dynagraph/EQA child is still holding VRAM.
if pgrep -af 'emet\.app\.run_dynagraph|eval_dynamic_exploration|eval_ovmm_find' >/dev/null 2>&1; then
    echo "[launch] ERROR: leftover dynagraph/eval process still running:" >&2
    pgrep -af 'emet\.app\.run_dynagraph|eval_dynamic_exploration|eval_ovmm_find' >&2 || true
    echo "[launch] Kill it (or ./scripts/gpu_preflight.sh --kill-stale) then retry." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_ROOT/run_$STAMP"
mkdir -p "$RUN_DIR"
echo "[launch] artifacts → $RUN_DIR"

# Unset ROS PYTHONPATH so a broken system cv2 cannot shadow the venv.
export -n PYTHONPATH 2>/dev/null || true
unset PYTHONPATH || true

# Track A: OVMM find backend matrix (Robocasa + Molmo). Foreground so Track B
# cannot start until this finishes (one GPU job at a time).
echo "[launch] OVMM find matrix → $RUN_DIR/ovmm_find_matrix"
set +e
env -u PYTHONPATH uv run python scripts/run_ovmm_find_backend_matrix.py \
    --backends ground_truth,dynamem,graph_eqa,dynagraph \
    --port-offset-base 200 \
    --output-dir "$RUN_DIR/ovmm_find_matrix" \
    >"$RUN_DIR/ovmm_find_matrix.log" 2>&1
OVMM_RC=$?
set -e
echo "[launch] OVMM find matrix exit=$OVMM_RC"

NEED_MIB="$NEED_MIB" ./scripts/gpu_preflight.sh --wait

# Track B: dynamic exploration full matrix (P1–P3)
export EMET_DYNAMIC_EXPLORE_OUTPUT="$RUN_DIR/dynamic_exploration"
# Shorter per-question EQA budget so a stuck nav loop cannot burn the night.
export EMET_EQA_QUESTION_TIMEOUT_S="${EMET_EQA_QUESTION_TIMEOUT_S:-900}"
# Kill hung children whose logs go silent (2× stale warn → kill).
export EMET_DYNAMIC_EXPLORE_STALE_LOG_S="${EMET_DYNAMIC_EXPLORE_STALE_LOG_S:-600}"
export EMET_DYNAMIC_EXPLORE_STALE_KILL_S="${EMET_DYNAMIC_EXPLORE_STALE_KILL_S:-1200}"
mkdir -p "$EMET_DYNAMIC_EXPLORE_OUTPUT"
echo "[launch] dynamic exploration → $EMET_DYNAMIC_EXPLORE_OUTPUT"
set +e
env -u PYTHONPATH \
    EMET_DYNAMIC_EXPLORE_OUTPUT="$EMET_DYNAMIC_EXPLORE_OUTPUT" \
    EMET_EQA_QUESTION_TIMEOUT_S="$EMET_EQA_QUESTION_TIMEOUT_S" \
    EMET_DYNAMIC_EXPLORE_STALE_LOG_S="$EMET_DYNAMIC_EXPLORE_STALE_LOG_S" \
    EMET_DYNAMIC_EXPLORE_STALE_KILL_S="$EMET_DYNAMIC_EXPLORE_STALE_KILL_S" \
    ./scripts/run_dynamic_exploration_full.sh \
    >"$RUN_DIR/dynamic_exploration_full.log" 2>&1
DYN_RC=$?
set -e
echo "[launch] dynamic exploration exit=$DYN_RC"
echo "[launch] done → $RUN_DIR (ovmm_rc=$OVMM_RC dyn_rc=$DYN_RC)"
exit $(( OVMM_RC != 0 || DYN_RC != 0 ? 1 : 0 ))
