#!/usr/bin/env bash
# After OVMM CPU GT smoke finishes, run agent world-change find + dynamic world-change CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${EMET_CHAIN_OUT:-$HOME/runs/emet/dynagraph_phase_chain}"
mkdir -p "$OUT"

OVMM_PID="${1:-}"
if [[ -n "$OVMM_PID" ]]; then
  echo "[chain] waiting for OVMM pid $OVMM_PID…"
  while kill -0 "$OVMM_PID" 2>/dev/null; do sleep 30; done
  echo "[chain] OVMM finished"
fi

echo "[chain] agent world-change find…"
uv run python scripts/smoke_dynagraph_agent_world_change_find.py \
  --cpu-only --explore-iters 2 --port-offset 90 \
  --output "$OUT/agent_world_change_find.json" \
  >"$OUT/agent_world_change_find.log" 2>&1 || echo "[chain] agent smoke exit=$?"

echo "[chain] dynamic world-change episode…"
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph --cpu-only --port-offset 92 \
  --output-dir "$OUT/world_change" \
  >"$OUT/world_change.log" 2>&1 || echo "[chain] world-change exit=$?"

echo "[chain] done → $OUT"

# Optional: queue paper GPU sweeps once CPU chain finished (waits for free VRAM)
if [[ "${CHAIN_LAUNCH_GPU_EVAL:-0}" == "1" ]]; then
  echo "[chain] launching run_dynagraph_dynamic_memory_eval.sh…"
  nohup ./scripts/run_dynagraph_dynamic_memory_eval.sh \
    >"${EMET_DYNAMIC_MEMORY_OUT:-$HOME/runs/emet/dynagraph_dynamic_memory}/launcher.log" 2>&1 &
  echo "[chain] GPU eval launcher pid=$!"
fi
