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
echo "         -y/--yes    = non-interactive (apt, link emet); does NOT imply MolmoSpaces — pass --molmospaces or --all"
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
# Skip when DOCKER=1 (image already has deps; container often has no sudo)
echo ""
echo "[3/5] Checking system dependencies..."
if [ -n "${DOCKER:-}" ]; then
    echo "  -> Skipping apt (DOCKER=1); image has deps."
elif [ "$SKIP_ASKING" = "true" ]; then
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
# emet is not on PyPI — install with --no-deps -e . then --no-deps -e packages/emet_molmospaces, then molmo deps.
# Prefer uv pip install --python .venv-molmospaces/bin/python (uv venvs often have no pip module).
if [ "$INSTALL_MOLMOSPACES" = "true" ]; then
    echo ""
    echo "Setting up MolmoSpaces wrapper venv (.venv-molmospaces)..."
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    MLSPACES_CACHE="${MLSPACES_ASSETS_DIR:-$XDG_CACHE_HOME/molmospaces/assets}"
    export MLSPACES_ASSETS_DIR="$MLSPACES_CACHE"
    export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$XDG_CACHE_HOME/molmospaces/resource_cache}"
    mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"
    PY_MOLMO=".venv-molmospaces/bin/python"
    molmo_pip_install() {
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$PY_MOLMO" "$@"
        else
            "$PY_MOLMO" -m pip install "$@"
        fi
    }
    molmo_install_editable_chain() {
        molmo_pip_install --upgrade pip 2>/dev/null || true
        molmo_pip_install --no-deps -e .
        if [ -d "packages/emet_molmospaces" ]; then
            # Resolves molmo-spaces from GitHub + mujoco/numpy per packages/emet_molmospaces/pyproject.toml
            molmo_pip_install -e packages/emet_molmospaces
        else
            molmo_pip_install "molmo-spaces @ git+https://github.com/allenai/molmospaces.git@62b416089b2eddff339e52a32106a6bc08ed92b1" "mujoco>=3.4" "numpy>=2.2"
        fi
    }
    _molmospaces_drop_venv_if_python_too_old() {
        if [ ! -d ".venv-molmospaces" ] || [ ! -x "$PY_MOLMO" ]; then
            return 0
        fi
        if "$PY_MOLMO" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" 2>/dev/null; then
            return 0
        fi
        echo "  -> Replacing .venv-molmospaces (molmo-spaces requires Python >=3.11)..."
        rm -rf .venv-molmospaces
    }
    _molmospaces_drop_venv_if_python_too_old
    if [ ! -d ".venv-molmospaces" ]; then
        if uv venv .venv-molmospaces --python 3.11 2>/dev/null; then
            :
        elif uv venv .venv-molmospaces --python 3.12 2>/dev/null; then
            :
        else
            echo "ERROR: Need Python 3.11+ for MolmoSpaces. Install it or run: uv python install 3.11"
            exit 1
        fi
        molmo_install_editable_chain
        echo "  -> Created .venv-molmospaces with emet and emet-molmospaces wrapper"
    else
        echo "  -> .venv-molmospaces already exists"
        NEED_REPAIR=0
        if [ ! -x "$PY_MOLMO" ]; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "import emet" 2>/dev/null; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "import emet_molmospaces" 2>/dev/null; then
            NEED_REPAIR=1
        elif ! "$PY_MOLMO" -c "from molmo_spaces.molmo_spaces_constants import get_scenes; from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path" 2>/dev/null; then
            NEED_REPAIR=1
        elif [ ! -x ".venv-molmospaces/bin/emet-molmospaces" ]; then
            NEED_REPAIR=1
        fi
        if [ "$NEED_REPAIR" -eq 1 ]; then
            echo "  -> Repairing MolmoSpaces venv (editable emet + wrapper + molmo-spaces from GitHub)..."
            molmo_install_editable_chain
        fi
    fi
    if ! "$PY_MOLMO" -c "import mujoco; import emet_molmospaces; from molmo_spaces.molmo_spaces_constants import get_scenes; from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path" 2>/dev/null; then
        echo ""
        echo "ERROR: .venv-molmospaces failed verification (molmo-spaces API + mujoco + wrapper)."
        echo "  Fix: rm -rf .venv-molmospaces && ./install.sh --molmospaces -y"
        echo "  Requires Python 3.11+ (uv: uv python install 3.11)."
        exit 1
    fi
    echo "  -> Verified: molmo_spaces imports in .venv-molmospaces"
    echo "  -> MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR  MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR"
    echo "     (must differ; emet defaults cache to …/molmospaces/resource_cache next to …/assets)"
    echo "  -> Run: emet molmospaces list-robots  &&  emet molmospaces serve --viewer"
fi

# Quick sanity check
if ! uv run python -c "import emet; print('emet:', emet.__file__)" 2>/dev/null; then
    echo "WARNING: emet import check failed. You may need to run: uv sync $EXTRA_ARGS"
fi

# Put emet CLI in a reasonable place (~/.local/bin so it's on PATH when present)
# Skip when running inside Docker (PATH is set via Dockerfile instead)
echo ""
LINK_EMET="false"
if [ -n "${DOCKER:-}" ]; then
    echo "  -> Skipping link (DOCKER=1); use PATH=/app/.venv/bin or uv run emet"
elif [ "$SKIP_ASKING" = "true" ]; then
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
