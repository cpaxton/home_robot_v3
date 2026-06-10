#!/usr/bin/env bash
# Fair-default OVMM find-phase replicate (dynamem + dynagraph, 5 seeds).
# Intended for overnight GPU runs via cron; logs to ~/runs/emet/ovmm_find_phase/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
OUT="${HOME}/runs/emet/ovmm_find_phase/fair_default_${STAMP}"
LOG="${OUT}_run.log"

mkdir -p "$(dirname "$OUT")"
{
    echo "=== OVMM fair-default replicate start $(date -Is) ==="
    echo "repo=${REPO}"
    echo "output=${OUT}"
    cd "$REPO"
    export MUJOCO_GL=egl
    export PATH="${HOME}/.local/bin:${PATH}"
    uv run python scripts/replicate_ovmm_find_phases.py \
        --episode-id default_table_s0 \
        --backend dynamem \
        --backend dynagraph \
        --replicates 5 \
        --seed-base 0 \
        --output-dir "$OUT"
    echo "=== done $(date -Is) exit=0 ==="
} >>"$LOG" 2>&1
