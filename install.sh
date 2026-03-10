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
INSTALL_SIM="true"   # default: clone third_party/robocasa+robosuite and include sim extra
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
        --no-sim)
            INSTALL_SIM="false"
            ;;
        --sim)
            INSTALL_SIM="true"
            ;;
        *)
            ;;
    esac
done

[[ "$INSTALL_SIM" == "true" ]] && EXTRAS="$EXTRAS,sim"

echo "=============================================="
echo "         INSTALLING STRETCH AI (uv)"
echo "=============================================="
echo "Options: CPU_ONLY=$CPU_ONLY, NO_SAM2=$NO_SAM2, INSTALL_SIM=$INSTALL_SIM, EXTRAS=$EXTRAS"
echo "---------------------------------------------"

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

# Create venv and install with uv (uv sync creates .venv automatically; uses uv.lock if present)
echo ""
echo "Creating virtual environment and installing dependencies..."
EXTRA_ARGS="--extra dev"
[[ "$EXTRAS" == *"sim"* ]] && EXTRA_ARGS="$EXTRA_ARGS --extra sim"
[[ "$NO_SAM2" == "false" ]] && [ -d "third_party/segment-anything-2" ] && EXTRA_ARGS="$EXTRA_ARGS --extra dynamem"
uv sync $EXTRA_ARGS

# Uninstall av to avoid conflict (from old install.sh)
source .venv/bin/activate
uv pip uninstall av -y 2>/dev/null || true

# Sim: clone third_party/robosuite and robocasa, install editable, run macros and (optionally) download assets
if [ "$INSTALL_SIM" = "true" ]; then
    echo ""
    echo "Installing simulation (Robocasa + robosuite)..."
    export EMET_USE_UV=1
    SIM_SCRIPT="$ROOT_DIR/scripts/install_simulation.sh"
    if [ "$SKIP_ASKING" = "true" ]; then
        bash "$SIM_SCRIPT" -n
    else
        bash "$SIM_SCRIPT"
    fi
fi

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
