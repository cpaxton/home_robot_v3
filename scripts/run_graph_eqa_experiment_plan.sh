#!/usr/bin/env bash
# Operational checklist for GraphEQA exploration / export / viewing / sanity checks.
# Run from repo root: ./scripts/run_graph_eqa_experiment_plan.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Phase 0: environment (uv; default groups include dev + sim) ==="
uv sync
# If import cv2 fails or cv2 has no imwrite/imshow, reinstall: uv pip install --reinstall opencv-python

echo ""
echo "Scene: default Stretch MuJoCo table (red cylinder / blue cube) — see docs/plans/TESTING_BACKENDS.md."
echo "Robocasa: Terminal 1: emet serve mujoco --use-robocasa  |  Terminal 2: emet run graph-eqa --robot-ip 127.0.0.1"
echo ""

echo "=== Live sim (manual) — interactive GraphEQA with rotate_in_place ==="
echo "  Terminal 1: MUJOCO_GL=egl PYTHONPATH=src uv run python -m emet.simulation.mujoco_server --headless"
echo "  # or: emet serve mujoco"
echo "  Terminal 2: uv run emet run graph-eqa --robot-ip 127.0.0.1"
echo "  Optional: add --save_rerun for Rerun logs under graph_eqa_log/"
echo ""

EXP_DIR="${ROOT}/exp/graph_eqa_plan_smoke"
mkdir -p "$EXP_DIR"
export EXPORT_DIR="$EXP_DIR"

echo "=== Phase 2: export smoke (mock graph; same labels as default scene GT) ==="
uv run python <<'PY'
import os

import numpy as np

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.headless_export import export_graph_eqa_dir

out = os.environ["EXPORT_DIR"]
mem = GraphEQAMemory(
    eqa_client=lambda x: "reasoning: r\nanswer: ok\nconfidence: true\naction:\nconfidence_reasoning: x",
    image_description_client=lambda x: "red cylinder, blue cube",
)
rgb = np.zeros((40, 40, 3), dtype=np.uint8)
mem.add_observation(rgb, np.array([0.08, -0.55, 0.6]), ["red cylinder"])
mem.add_observation(rgb, np.array([-0.02, -0.55, 0.6]), ["blue cube"])
export_graph_eqa_dir(mem, None, out, title="Scene graph (plan smoke)")
print("Wrote:", out)
PY

echo ""
echo "=== Phase 3: view artifacts ==="
test -f "${EXP_DIR}/scene_graph_report.txt"
echo "--- scene_graph_report.txt (first 40 lines) ---"
head -40 "${EXP_DIR}/scene_graph_report.txt"
echo ""
echo "--- emet graph-memory-show (first 40 lines) ---"
uv run emet graph-memory-show "${EXP_DIR}" 2>&1 | head -40

echo ""
echo "Optional — Rerun 3D viewer (needs display): uv run emet show-memory ${EXP_DIR}"
echo ""

echo "=== Phase 4: unit tests ==="
uv run emet test --no-sim -v \
    src/test/memory/test_graph_eqa_memory.py \
    src/test/memory/test_headless_export.py

echo ""
echo "Optional sim test (long; needs working MuJoCo server stack):"
echo "  uv run emet test -v src/test/memory/test_graph_eqa_default_scene_sim.py"
echo ""
echo "Live headless export when sim is up:"
echo "  uv run emet run graph-eqa --robot-ip 127.0.0.1 --export /path/to/out_dir"
echo "Done."
