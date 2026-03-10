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
echo "         -y/--yes = non-interactive (install deps, link emet to ~/.local/bin)"
echo "Root: $ROOT_DIR"
echo "---------------------------------------------"

# Step 1: Init required git submodules (segment-anything-2 for SAM-2/dynamem).
# ok-robot is optional (docs/advanced workflows only); use: emet install submodules
echo ""
echo "[1/5] Initializing required git submodules (segment-anything-2)..."
git submodule update --init --recursive third_party/segment-anything-2
if [ ! -d "third_party/segment-anything-2" ]; then
    echo "ERROR: third_party/segment-anything-2 missing after submodule update. Check git and .gitmodules."
    exit 1
fi
echo "  -> Verified third_party/segment-anything-2 exists."

# Ensure uv is installed
echo ""
echo "[2/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# System dependencies (apt-get update may warn about other repos e.g. ROS2 keys; we continue)
echo ""
echo "[3/5] Checking system dependencies..."
if [ "$SKIP_ASKING" = "true" ]; then
    sudo apt-get update || true
    sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1
else
    echo "Required packages: libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1"
    echo "Install with: sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1"
    read -p "Install these now? (y/n) " yn
    case $yn in
        y|Y) sudo apt-get update || true; sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1 ;;
        *) echo "Skipping. You may need to install these manually." ;;
    esac
fi

# Git LFS
echo ""
echo "[4/5] Setting up git-lfs..."
git lfs install || { echo "Install git-lfs: sudo apt-get install git-lfs"; exit 1; }

# Create venv and install with uv (uv sync creates .venv automatically; uses uv.lock if present)
# Sim extra requires third_party/robocasa and third_party/robosuite (from: emet install sim)
echo ""
echo "[5/5] Creating virtual environment and installing dependencies..."
EXTRA_ARGS="--extra dev"
if [[ "$EXTRAS" == *"sim"* ]]; then
    if [ -d "third_party/robocasa" ] && [ -d "third_party/robosuite" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --extra sim"
    else
        echo "  -> Skipping sim extra (third_party/robocasa or third_party/robosuite missing)."
        echo "     After install, run: emet install sim   then  uv sync -e sim"
    fi
fi
[[ "$NO_SAM2" == "false" ]] && [ -d "third_party/segment-anything-2" ] && EXTRA_ARGS="$EXTRA_ARGS --extra dynamem"
echo "  -> Running: uv sync $EXTRA_ARGS"
uv sync $EXTRA_ARGS
echo "  -> uv sync completed."

# Uninstall av to avoid conflict (from old install.sh)
source .venv/bin/activate
uv pip uninstall av -y 2>/dev/null || true

# Quick sanity check
if ! uv run python -c "import emet; print('emet:', emet.__file__)" 2>/dev/null; then
    echo "WARNING: emet import check failed. You may need to run: uv sync $EXTRA_ARGS"
fi

# Put emet CLI in a reasonable place (~/.local/bin so it's on PATH when present)
echo ""
LINK_EMET="false"
if [ "$SKIP_ASKING" = "true" ]; then
    LINK_EMET="true"
else
    read -p "Link 'emet' to ~/.local/bin so you can run it from anywhere? (y/n) " yn
    case $yn in
        y|Y) LINK_EMET="true" ;;
        *) ;;
    esac
fi
if [ "$LINK_EMET" = "true" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$ROOT_DIR/.venv/bin/emet" "$HOME/.local/bin/emet"
    echo "  -> emet linked to $HOME/.local/bin/emet"
    if ! echo ":$PATH:" | grep -q ":${HOME}/.local/bin:"; then
        echo "  -> Add to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
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
echo "Run the CLI:  emet  (if linked above) or  uv run emet"
echo ""
echo "Optional: init all submodules (including ok-robot for advanced workflows):"
echo "  emet install submodules"
echo ""
