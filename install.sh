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
CLEAN_SIM="false"
INSTALL_MOLMOSPACES="false"

for arg in "$@"; do
    case $arg in
        -y|--yes)
            SKIP_ASKING="true"
            ;;
        --all)
            INSTALL_SIM="true"
            INSTALL_MOLMOSPACES="true"
            NO_SAM2="false"
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
        --clean)
            CLEAN_SIM="true"
            ;;
        --molmospaces)
            INSTALL_MOLMOSPACES="true"
            ;;
        *)
            ;;
    esac
done

[[ "$INSTALL_SIM" == "true" ]] && EXTRAS="$EXTRAS,sim"

echo "=============================================="
echo "         INSTALLING STRETCH AI (uv)"
echo "=============================================="
echo "Options: CPU_ONLY=$CPU_ONLY, NO_SAM2=$NO_SAM2, INSTALL_SIM=$INSTALL_SIM, EXTRAS=$EXTRAS, MOLMOSPACES=$INSTALL_MOLMOSPACES"
echo "         -y/--yes    = non-interactive (install deps, link emet to ~/.local/bin)"
echo "         --all       = install everything (sim + molmospaces + dynamem); overridable by --no-sim etc."
echo "         --sim       = install sim (Robocasa + robosuite; default). Use --no-sim to skip."
echo "         --molmospaces = create .venv-molmospaces for MolmoSpaces (scenes + rby1 robot)"
echo "         --clean     = remove and re-clone third_party/robosuite, robosuite_models, robocasa (only if needed; normally we update in place)"
echo "Root: $ROOT_DIR"
echo "---------------------------------------------"

# Optional: remove sim third_party dirs only when explicitly requested (--clean). Normally install_simulation.sh updates in place (fetch/pull).
if [ "$CLEAN_SIM" = "true" ]; then
    echo ""
    echo "Cleaning sim third_party (robosuite, robosuite_models, robocasa) before re-install..."
    for d in third_party/robosuite third_party/robosuite_models third_party/robocasa; do
        if [ -d "$d" ]; then
            rm -rf "$d"
            echo "  -> Removed $d"
        fi
    done
fi

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

# Create venv and install with uv (pyproject has sim=[] by default so sync works without third_party/robocasa)
echo ""
echo "[5/5] Creating virtual environment and installing dependencies..."
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

echo "  -> Running: uv sync $EXTRA_ARGS"
uv sync $EXTRA_ARGS
echo "  -> uv sync completed."

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
        bash "$SIM_SCRIPT" -y
    else
        bash "$SIM_SCRIPT"
    fi
fi

# MolmoSpaces: separate venv with emet-molmospaces wrapper (molmo-spaces needs mujoco 3.4 + numpy>=2.2)
# Use venv's python -m pip so we don't rely on bin/pip existing (e.g. some uv venvs).
if [ "$INSTALL_MOLMOSPACES" = "true" ]; then
    echo ""
    echo "Setting up MolmoSpaces wrapper venv (.venv-molmospaces)..."
    MLSPACES_CACHE="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
    export MLSPACES_ASSETS_DIR="$MLSPACES_CACHE"
    mkdir -p "$MLSPACES_ASSETS_DIR"
    PY_MOLMO=".venv-molmospaces/bin/python"
    if [ ! -d ".venv-molmospaces" ]; then
        uv venv .venv-molmospaces
        "$PY_MOLMO" -m pip install --upgrade pip
        # Emet (no-deps) then emet-molmospaces wrapper (pulls molmo-spaces, mujoco 3.4, numpy>=2.2)
        "$PY_MOLMO" -m pip install --no-deps -e .
        if [ -d "packages/emet_molmospaces" ]; then
            "$PY_MOLMO" -m pip install -e packages/emet_molmospaces
        else
            "$PY_MOLMO" -m pip install "molmo-spaces" "mujoco>=3.4" "numpy>=2.2"
        fi
        echo "  -> Created .venv-molmospaces with emet and emet-molmospaces wrapper"
    else
        echo "  -> .venv-molmospaces already exists"
        # Ensure wrapper is installed when package lives in repo (packages/ is not under src)
        if [ -d "packages/emet_molmospaces" ] && [ ! -x ".venv-molmospaces/bin/emet-molmospaces" ]; then
            "$PY_MOLMO" -m pip install -e packages/emet_molmospaces
            echo "  -> Installed emet-molmospaces wrapper into existing venv"
        fi
    fi
    echo "  -> MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR (set this in your shell or .env for emet molmospaces)"
    echo "  -> Run: emet molmospaces list-robots  &&  emet molmospaces serve --viewer"
fi

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
