#!/usr/bin/env bash
# This script (c) 2024 Hello Robot under the MIT license: https://opensource.org/licenses/MIT
# Installs Stretch AI using uv (https://docs.astral.sh/uv/) - no conda required.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"

# Parse options
CPU_ONLY="false"
SKIP_ASKING="false"
NO_SAM2="false"
EXTRAS="dev"

for arg in "$@"; do
    case $arg in
        -y|--yes)
            SKIP_ASKING="true"
            ;;
        --cpu)
            CPU_ONLY="true"
            NO_SAM2="true"
            ;;
        --no-sam2)
            NO_SAM2="true"
            ;;
        --sim)
            EXTRAS="$EXTRAS,sim"
            ;;
        *)
            ;;
    esac
done

echo "=============================================="
echo "         INSTALLING STRETCH AI (uv)"
echo "=============================================="
echo "Options: CPU_ONLY=$CPU_ONLY, NO_SAM2=$NO_SAM2, EXTRAS=$EXTRAS"
echo "---------------------------------------------"

# Init required git submodules (segment-anything-2 for dynamem)
echo ""
echo "Initializing git submodules (segment-anything-2)..."
git submodule update --init --recursive third_party/segment-anything-2 2>/dev/null || true
[ -d "third_party/segment-anything-2" ] || echo "Warning: third_party/segment-anything-2 missing; dynamem extra will be skipped."

# Ensure uv is installed
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# System dependencies
echo ""
echo "Checking system dependencies..."
if [ "$SKIP_ASKING" = "true" ]; then
    sudo apt-get update
    sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1
else
    echo "Required packages: libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1"
    echo "Install with: sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1"
    read -p "Install these now? (y/n) " yn
    case $yn in
        y|Y) sudo apt-get update && sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1 ;;
        *) echo "Skipping. You may need to install these manually." ;;
    esac
fi

# Git LFS
echo ""
echo "Setting up git-lfs..."
git lfs install || { echo "Install git-lfs: sudo apt-get install git-lfs"; exit 1; }

# Create venv and install with uv (pyproject has sim=[] by default so sync works without third_party/robocasa)
echo ""
echo "Creating virtual environment and installing dependencies..."
EXTRA_ARGS="--extra dev"
if [[ "$EXTRAS" == *"sim"* ]]; then
    if [ ! -d "third_party/robocasa" ] || [ ! -d "third_party/robosuite" ]; then
        echo "Skipping sim (third_party/robocasa or robosuite missing). Run: ./scripts/install_simulation.sh  then  python scripts/enable_sim_pyproject.py  then  uv sync -e sim"
    elif ! grep -q '^robocasa = {' pyproject.toml 2>/dev/null; then
        echo "Skipping sim (not enabled in pyproject). Run: python scripts/enable_sim_pyproject.py  then  uv lock && uv sync -e sim"
    else
        EXTRA_ARGS="$EXTRA_ARGS --extra sim"
    fi
fi
[[ "$NO_SAM2" == "false" ]] && [ -d "third_party/segment-anything-2" ] && EXTRA_ARGS="$EXTRA_ARGS --extra dynamem"
uv sync $EXTRA_ARGS

# Uninstall av to avoid conflict (from old install.sh)
source .venv/bin/activate
uv pip uninstall av -y 2>/dev/null || true

echo ""
echo "=============================================="
echo "         INSTALLATION COMPLETE"
echo "=============================================="
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands with:"
echo "  uv run python -m emet.app.<module>"
echo ""
