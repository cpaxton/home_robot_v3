#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Queue HM-EQA count/clock bisect canaries (5-qid gate) at each checkpoint.
#
# Runs one job at a time via emet jobs (gpu.lock). Wait for each to finish before
# launching the next unless BISECT_PARALLEL=1.
#
# Usage:
#   ./scripts/run_hmeqa_bisect_sweep.sh
#   BISECT_SHAS="23efa534 290e54e5" ./scripts/run_hmeqa_bisect_sweep.sh
#   BISECT_INLINE=1 BISECT_SHAS="290e54e5:bisect_290e54e5_r2 c1591698" ./scripts/run_hmeqa_bisect_sweep.sh
#
# BISECT_SHAS entries may be "commit" or "commit:RUN_ID" (fresh rerun tag).
# BISECT_INLINE=1 runs canaries in-process (one emet jobs GPU lock); default
# queues each checkpoint as its own detached emet job.
#
# Checkpoints (default): peak → zoom-off → CLOSE_LOOK → frame-dedupe
# See docs/experiments/countclock_bisect.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

read -r -a ENTRIES <<< "${BISECT_SHAS:-23efa534 290e54e5 c1591698 a32271b3}"
NEED_MIB="${NEED_MIB:-12000}"

_run_canary() {
  local sha="$1"
  local run_id="$2"
  local short="${sha:0:8}"
  local name="bisect-${short}"
  if [[ -n "${run_id}" && "${run_id}" != "bisect_${short}" ]]; then
    name="${name}-rerun"
  fi
  echo "=== ${name} @ ${sha} run_id=${run_id:-bisect_${short}} ==="
  local -a env_args=(EMET_ALLOW_SDPA_ATTN=1 BISECT_SHA="${sha}" RESUME=0)
  if [[ -n "${run_id}" ]]; then
    env_args+=(RUN_ID="${run_id}")
  fi
  if [[ "${BISECT_INLINE:-0}" == "1" ]]; then
    env "${env_args[@]}" ./scripts/run_hmeqa_bisect_canary.sh
  else
    if [[ "${BISECT_PARALLEL:-0}" != "1" ]]; then
      while uv run emet jobs 2>/dev/null | rg -q 'status:\s+(running|waiting)'; do
        echo "waiting for active emet jobs..."
        sleep 60
      done
    fi
    local job_id
    job_id="$(
      uv run emet jobs run --name "${name}" --need-mib "${NEED_MIB}" -- \
        env "${env_args[@]}" ./scripts/run_hmeqa_bisect_canary.sh
    )"
    echo "queued ${job_id}"
    if [[ "${BISECT_PARALLEL:-0}" != "1" ]]; then
      while true; do
        st="$(uv run emet jobs status "${job_id}" 2>/dev/null | awk -F: '/^status:/ {print $2}' | tr -d ' ')"
        case "${st}" in
          done) echo "${name}: done"; break ;;
          failed|cancelled) echo "${name}: ${st} — see emet jobs logs ${job_id}"; exit 1 ;;
          *) sleep 120 ;;
        esac
      done
    fi
  fi
  local tag="countclock_${run_id:-bisect_${short}}_dynagraph_qwen3_vl.jsonl"
  uv run python scripts/audit_close_map_eqa_slice.py \
    --jsonl "${HOME}/.cache/habitat_eqa/results/${tag}" \
    2>/dev/null || true
}

for entry in "${ENTRIES[@]}"; do
  sha="${entry%%:*}"
  run_id=""
  if [[ "${entry}" == *:* ]]; then
    run_id="${entry#*:}"
  fi
  _run_canary "${sha}" "${run_id}"
done

echo "bisect sweep complete"
