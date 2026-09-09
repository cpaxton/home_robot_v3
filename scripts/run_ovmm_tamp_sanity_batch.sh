#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Small sanity batch across the OVMM + TAMP benchmarks (get-a-feel trials).
#
# Runs, serialized on the GPU lock inside one ``emet jobs`` job:
#   - OVMM find phase slice (rby1 S0 + S1 live-mapping), TRIALS_OVMM x
#   - TAMP nav_goal 15-row rerun, TRIALS_NAVGOAL x
#   - TAMP GT+MCTS battery (nori, innate_mars, rby1 x scenes 0,1), TRIALS_BATTERY x
#
# Each trial gets its own OUT subdir + log; a OUT/summary.json records per-trial
# status. Progress is reported via OUT/progress.json + `emet jobs update`.
#
# Usage:
#   uv run emet jobs run --name ovmm-tamp-sanity --need-mib 8000 --gpu-exclusive -- \
    #     ./scripts/run_ovmm_tamp_sanity_batch.sh
#
# Env:
#   OUT_DIR            base output dir (default ~/runs/emet/ovmm_tamp_sanity/<stamp>)
#   TRIALS_OVMM        default 2
#   TRIALS_NAVGOAL     default 2
#   TRIALS_BATTERY     default 2

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

STAMP="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_tamp_sanity/${STAMP}}"
export OUT
mkdir -p "$OUT"

TRIALS_OVMM="${TRIALS_OVMM:-2}"
TRIALS_NAVGOAL="${TRIALS_NAVGOAL:-2}"
TRIALS_BATTERY="${TRIALS_BATTERY:-2}"
UNITS_TOTAL=$((TRIALS_OVMM + TRIALS_NAVGOAL + TRIALS_BATTERY))
FAILED=0
DONE=0

heartbeat() {
    local phase="$1"
    local current="$2"
    uv run python -c "
from emet.eval.harness import update_eval_progress
update_eval_progress(
    r'''$OUT''',
    units_done=int('$DONE'),
    units_total=int('$UNITS_TOTAL'),
    phase='$phase',
    current_id='$current',
    units_failed=int('$FAILED'),
)
    " 2>/dev/null || true
    if [[ -n "${EMET_JOB_ID:-}" ]]; then
        uv run emet jobs update "$EMET_JOB_ID" \
            --status running \
            --units-done "$DONE" \
            --units-total "$UNITS_TOTAL" \
            --phase "$phase" \
            --current-id "$current" \
            --out-dir "$OUT" \
            >/dev/null 2>&1 || true
    fi
}

record() {
    local key="$1"
    local status="$2"
    uv run python - "$key" "$status" <<'PY'
import json, os, sys
key, status = sys.argv[1], sys.argv[2]
out = os.environ.get("OUT")
if not out:
    sys.exit(0)
summary_path = os.path.join(out, "summary.json")
summary = {}
if os.path.exists(summary_path):
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
summary[key] = status
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
PY
}

run_ovmm_trial() {
    local trial="$1"
    local key="ovmm_slice_trial${trial}"
    local trial_out="$OUT/${key}"
    heartbeat "ovmm-slice" "$key"
    echo "=== [$key] OVMM find slice (rby1 S0 + S1 live-mapping) ==="
    if env PROFILE=slice OUT_DIR="$trial_out" RUN_ID="${STAMP}_${key}" \
        ./scripts/run_ovmm_find_recep_slice.sh > "$OUT/${key}.log" 2>&1; then
        record "$key" "pass"
    else
        record "$key" "fail"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
}

NAVGOAL_IDS=(
    ithor_nav_goal_00_rby1_n8_table_2
    ithor_nav_goal_00_rby1_n8_bed_3
    ithor_nav_goal_00_rby1_n8_4
    ithor_nav_goal_01_rby1_n8_bed_2
    ithor_nav_goal_01_rby1_n8_counter_3
    ithor_nav_goal_01_rby1_n8_4
    ithor_nav_goal_00_stretch_n8_table_2
    ithor_nav_goal_00_stretch_n8_bed_3
    ithor_nav_goal_00_stretch_n8_4
    ithor_nav_goal_00_innate_mars_n8_table_2
    ithor_nav_goal_00_innate_mars_n8_bed_3
    ithor_nav_goal_00_innate_mars_n8_4
    ithor_nav_goal_00_nori_n8_table_2
    ithor_nav_goal_00_nori_n8_bed_3
    ithor_nav_goal_00_nori_n8_4
)

run_navgoal_trial() {
    local trial="$1"
    local key="navgoal_trial${trial}"
    local trial_out="$OUT/${key}"
    mkdir -p "$trial_out"
    heartbeat "tamp-navgoal" "$key"
    echo "=== [$key] TAMP nav_goal 15-row rerun ==="
    args=(uv run python scripts/eval_tamp_clutter.py
        --episodes configs/ovmm/clutter_episodes_signal.yaml
    --output-dir "$trial_out")
    for ep in "${NAVGOAL_IDS[@]}"; do
        args+=(--episode-id "$ep")
    done
    if "${args[@]}" > "$OUT/${key}.log" 2>&1; then
        record "$key" "pass"
    else
        record "$key" "fail"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
}

run_battery_trial() {
    local trial="$1"
    local key="battery_trial${trial}"
    local trial_out="$OUT/${key}"
    mkdir -p "$trial_out"
    heartbeat "tamp-battery" "$key"
    echo "=== [$key] TAMP GT+MCTS battery (nori, innate_mars, rby1 x scenes 0,1) ==="
    if uv run python scripts/eval_tamp_clutter.py \
        --test-battery \
        --battery-robots nori,innate_mars,rby1 \
        --battery-scenes 0,1 \
        --output-dir "$trial_out" > "$OUT/${key}.log" 2>&1; then
        record "$key" "pass"
    else
        record "$key" "fail"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
}

printf '{"run_id":"%s","trials_ovmm":%s,"trials_navgoal":%s,"trials_battery":%s}\n' \
    "$STAMP" "$TRIALS_OVMM" "$TRIALS_NAVGOAL" "$TRIALS_BATTERY" > "$OUT/meta.json"

for t in $(seq 1 "$TRIALS_OVMM"); do
    run_ovmm_trial "$t"
done
for t in $(seq 1 "$TRIALS_NAVGOAL"); do
    run_navgoal_trial "$t"
done
for t in $(seq 1 "$TRIALS_BATTERY"); do
    run_battery_trial "$t"
done

heartbeat "done" "summary"
echo "=== sanity batch ${STAMP} done: failed=${FAILED}/${UNITS_TOTAL} (OUT=${OUT}) ==="
cat "$OUT/summary.json" 2>/dev/null || true
exit 0
