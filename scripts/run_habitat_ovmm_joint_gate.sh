#!/usr/bin/env bash
# Combined Habitat HM-EQA + OVMM find gate on one checkout.
#
# Runs an HM-EQA regression (Habitat) then the OVMM rby1 slice (S0 table +
# Robocasa kitchen), so both stacks are validated together in one ``emet jobs
# run``. The Habitat phase defaults to the 15-qid count/clock/locate slice —
# it is the EQA-sensitive subset that exercises the same shared
# AgenticEQAExecutor (explore / proposal / verify paths) the OVMM find uses.
#
# Usage:
#   uv run emet jobs run --name habitat-ovmm-gate --need-mib 12000 --gpu-exclusive -- \
    #     ./scripts/run_habitat_ovmm_joint_gate.sh
#
# Env:
#   RUN_HABITAT   "1"/"0" (default 1) — HM-EQA regression slice
#   RUN_OVMM      "1"/"0" (default 1) — OVMM rby1 slice (PROFILE=slice)
#   PROFILE       OVMM profile override (default slice)
#   HAB_SCRIPT    HM-EQA runner (default run_hmeqa_countclock_slice.sh)
#   HAB_METHODS   HM-EQA methods (default dynagraph)
#   HAB_RESUME    "1"/"0" passed to the HM-EQA runner (default 0)
#   RUN_ID        base run id; phases get _hab / _ovmm suffixes
#
# Both phases always run. Exit is non-zero if a requested phase failed.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_HABITAT="${RUN_HABITAT:-1}"
RUN_OVMM="${RUN_OVMM:-1}"
PROFILE="${PROFILE:-slice}"
HAB_METHODS="${HAB_METHODS:-dynagraph}"
HAB_RESUME="${HAB_RESUME:-0}"
HAB_SCRIPT="${HAB_SCRIPT:-run_hmeqa_countclock_slice.sh}"
OUT_BASE="${OUT_DIR:-$HOME/runs/emet/joint_gate/${RUN_ID}}"
mkdir -p "$OUT_BASE"

echo "=== joint gate ${RUN_ID} (habitat=${RUN_HABITAT} ovmm=${RUN_OVMM}) ==="
printf '{"run_id":"%s","run_habitat":%s,"run_ovmm":%s,"profile":"%s","hab_script":"%s","hab_methods":"%s"}\n' \
    "$RUN_ID" "$RUN_HABITAT" "$RUN_OVMM" "$PROFILE" "$HAB_SCRIPT" "$HAB_METHODS" > "$OUT_BASE/meta.json"

HAB_OK="skipped"
OVMM_OK="skipped"

if [[ "$RUN_HABITAT" == "1" ]]; then
    echo "--- phase 1/2: HM-EQA ${HAB_SCRIPT} (Habitat, METHODS=${HAB_METHODS}) ---"
    set +e
    env METHODS="$HAB_METHODS" RUN_ID="${RUN_ID}_hab" RESUME="$HAB_RESUME" \
        "./scripts/${HAB_SCRIPT}"
    hab_rc=$?
    set -e
    if [[ "$hab_rc" -eq 0 ]]; then
        HAB_OK="pass"
    else
        HAB_OK="fail"
        echo "WARNING: Habitat phase exited rc=${hab_rc}; continuing to OVMM phase"
    fi
fi

if [[ "$RUN_OVMM" == "1" ]]; then
    echo "--- phase 2/2: OVMM find slice (PROFILE=${PROFILE}) ---"
    set +e
    env PROFILE="$PROFILE" RUN_ID="${RUN_ID}_ovmm" \
        ./scripts/run_ovmm_find_recep_slice.sh
    ovmm_rc=$?
    set -e
    if [[ "$ovmm_rc" -eq 0 ]]; then
        OVMM_OK="pass"
    else
        OVMM_OK="fail"
        echo "WARNING: OVMM phase exited rc=${ovmm_rc}"
    fi
fi

printf '{"habitat":"%s","ovmm":"%s"}\n' "$HAB_OK" "$OVMM_OK" | tee "$OUT_BASE/summary.json"
echo "=== joint gate ${RUN_ID} done: habitat=${HAB_OK} ovmm=${OVMM_OK} (OUT=${OUT_BASE}) ==="

# Both phases always run so a Habitat miss does not hide OVMM (and vice versa).
# The process exit is non-zero if any *requested* phase failed, so ``emet jobs``
# marks the gate red instead of treating summary.json "fail" as a green run.
gate_rc=0
if [[ "$HAB_OK" == "fail" || "$OVMM_OK" == "fail" ]]; then
    gate_rc=1
fi
exit "$gate_rc"
