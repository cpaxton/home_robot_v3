#!/usr/bin/env bash
# Demo script for running Stretch AI apps in simulation.
# Usage: ./scripts/demo_simulation.sh [grasp|dynamem|mapping|robocasa]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

DEMO="${1:-grasp}"

echo "=============================================="
echo "  Stretch AI Simulation Demo: $DEMO"
echo "=============================================="
echo ""
echo "Start the MuJoCo server in another terminal first:"
echo "  uv run python -m stretch.simulation.mujoco_server"
echo ""
if [ "$DEMO" = "robocasa" ] || [ "$DEMO" = "dynamem" ]; then
    echo "For $DEMO, use Robocasa:"
    echo "  uv run python -m stretch.simulation.mujoco_server --use-robocasa"
    echo ""
fi
echo "Press Enter to run the $DEMO demo..."
read -r

case "$DEMO" in
    grasp)
        echo "Running grasp_object (red cylinder)..."
        uv run python -m stretch.app.grasp_object \
            --robot_ip 127.0.0.1 \
            --target_object "red cylinder" \
            --parameter_file sim_planner.yaml \
            --show_gui
        ;;
    dynamem)
        echo "Running DynaMem with visual servoing..."
        uv run python -m stretch.app.run_dynamem \
            --robot_ip 127.0.0.1 \
            --server_ip 127.0.0.1 \
            -S \
            --visual-servo \
            --match-method class
        ;;
    mapping)
        echo "Running mapping..."
        uv run python -m stretch.app.mapping --robot_ip 127.0.0.1
        ;;
    robocasa)
        echo "Robocasa: Start the server with --use-robocasa, then run grasp or dynamem."
        echo ""
        echo "Terminal 1: uv run python -m stretch.simulation.mujoco_server --use-robocasa"
        echo "Terminal 2: uv run python -m stretch.app.run_dynamem --robot_ip 127.0.0.1 --server_ip 127.0.0.1 -S --visual-servo"
        ;;
    *)
        echo "Unknown demo: $DEMO"
        echo "Usage: $0 [grasp|dynamem|mapping|robocasa]"
        exit 1
        ;;
esac
