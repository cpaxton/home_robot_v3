#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Run only through `emet jobs run --cpu-safe --gpu-exclusive` for serial sim load.
# Required: OUT_ROOT, SOURCE_SHA, NAV_ROBOT_ID, NAV_SCENE_CONFIG.
# Optional: NAV_CONFIG, NAV_FIND_QUERY, EMET_PY, NAV_SOURCE_DIR.
# Task evidence, not exit status, determines navigation acceptance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${NAV_SOURCE_DIR:-$ROOT}"
test "$(git rev-parse --short=8 HEAD)" = "${SOURCE_SHA:?}"
git diff --exit-code
git diff --cached --exit-code
robot="${NAV_ROBOT_ID:?}"
scene="${NAV_SCENE_CONFIG:?}"
config="${NAV_CONFIG:-configs/emet/query_navigation_physical_pilot.yaml}"
case "$robot" in
    stretch|rby1) ;;
    *) echo "Pilot supports stretch and rby1" >&2; exit 2 ;;
esac
case "${1:?find or explore}" in
    find) task="Find the ${NAV_FIND_QUERY:?} using find_objects. Use the tool feedback to inspect missing floor coverage and replan if necessary. Do not pick or place anything, use oracle plans, or override safety rejections. Report what you actually accomplished." ;;
    explore) task='Explore to discover navigable space. Use the tool feedback to inspect missing floor coverage and replan if necessary. Do not pick or place anything, use oracle plans, or override safety rejections. Report what you actually accomplished.' ;;
    *) exit 2 ;;
esac
out="${OUT_ROOT:?}/$robot/$1"
test ! -e "$out"
mkdir -p "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl EMET_ALLOW_SDPA_ATTN=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export EMET_CONFIG="$config" EMET_FORCE_HEAD_SWEEP=1 EMET_MOLMOSPACES_NAV_TELEPORT=0
export EMET_EQA_EPISODE_DIR="$out/evidence"
# Never inherit a different robot's GT body names or a remote model endpoint.
unset EMET_VL_ENDPOINT EMET_SIM_NAV_TELEPORT EMET_SIM_EVAL_CONFIG EMET_SIM_EVAL_TRACE
git rev-parse HEAD > "$out/source.txt"
sha256sum "$scene" configs/emet/query*pilot.yaml > "$out/config_sha256.txt"
cp "$scene" "$out/sim.yaml"
cp "${BASH_SOURCE[0]}" "$out/driver.sh"
agent=("${EMET_PY:-$PWD/.venv/bin/python}" -m emet.app.run_agent
    --config "$config" --memory-backend lazy_graph --robot "$robot" --start-sim
    --sim-config "$scene" --sim-seed 1 --headless --no-discord
    --sim-show-subprocess-output --llm qwen3-vl-eqa --eqa --debug-tools --visual-servo
-c "$task")
printf '%q ' "${agent[@]}" > "$out/command.txt"
rc=0
timeout --signal=TERM --kill-after=20s 600s "${agent[@]}" > "$out/process.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$out/process_exit.txt"
exit "$rc"
