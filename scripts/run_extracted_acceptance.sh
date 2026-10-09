#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# Invoke ONLY through emet jobs run --cpu-safe --gpu-exclusive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in eqa|room) ;; *) echo "Usage: $0 eqa|room" >&2; exit 2 ;; esac
test "$(git rev-parse HEAD)" = "${SOURCE_SHA:?freeze the exact source revision}"
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
out="${OUT_DIR:?use a new artifact directory}"
test ! -e "$out"
mkdir -p "$out"
export EMET_UV_RUN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl EMET_ALLOW_SDPA_ATTN=1
export EMET_SIM_NAV_TELEPORT=0 EMET_MOLMOSPACES_NAV_TELEPORT=0
export PYTHONPATH="$ROOT/src:$ROOT/packages/emet_habitat${PYTHONPATH:+:$PYTHONPATH}"
unset EMET_VL_ENDPOINT EMET_SIM_EVAL_CONFIG EMET_SIM_EVAL_TRACE
git rev-parse HEAD > "$out/source.txt"
cp "${BASH_SOURCE[0]}" "$out/driver.sh"
sha256sum uv.lock configs/emet/default.yaml configs/ovmm/find_phase_episodes.yaml > "$out/input_hashes.txt"
"${EMET_PY:-$ROOT/.venv/bin/python}" -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata.get("Name", "unknown") + "==" + d.version for d in distributions())))' > "$out/packages.txt"
printf 'seed=%s\nphase=%s\n' "${DEV_SEED:-0}" "$1" > "$out/run.txt"
case "${1:?eqa or room}" in
    eqa)
        unset EMET_CONFIG
        PHASE=eqa DEV_SEED="${DEV_SEED:-0}" OUT_DIR="$out" bash scripts/run_dev_loop.sh
        test ! -s "$out/failed.txt"
        ;;
    room)
        export EMET_CONFIG="$ROOT/configs/emet/shared_find_acceptance.yaml"
        cp "$EMET_CONFIG" "$out/config.yaml"
        # Each phase is bounded; exit/metrics failures are retained and stop the run.
        for episode in ${FIND_EPISODES:-default_table_s0_distinct_recep robocasa_pp_s1 molmo_ithor_s2_idx0}; do
            timeout --signal=TERM --kill-after=20s 1500s "${EMET_PY:-$ROOT/.venv/bin/python}" -m emet.cli ovmm find \
                --episodes configs/ovmm/find_phase_episodes.yaml --backend lazy_graph --query-driven-memory \
                --episode-id "$episode" --seed "${DEV_SEED:-0}" --mapping-rotate-steps 4 \
                --agentic-max-rounds 6 --agentic-max-nav-steps 3 --no-scene-cache --output-dir "$out"
            test -s "$out/${episode}_lazy_graph.json"
        done
        "${EMET_PY:-$ROOT/.venv/bin/python}" scripts/score_ovmm.py "$out"
        ;;
    *) echo "Unknown phase" >&2; exit 2 ;;
esac
