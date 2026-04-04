#!/usr/bin/env bash
# Install only the MolmoSpaces wrapper venv (.venv-molmospaces). No sim, no main uv sync.
# Run from repo root:  ./scripts/install_molmospaces.sh
# Uses uv pip when available so we don't require pip inside the venv (uv venvs often have no pip).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
MLSPACES_CACHE="${MLSPACES_ASSETS_DIR:-$XDG_CACHE_HOME/molmospaces/assets}"
export MLSPACES_ASSETS_DIR="$MLSPACES_CACHE"
mkdir -p "$MLSPACES_ASSETS_DIR"

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
    echo "Replacing .venv-molmospaces (molmo-spaces requires Python >=3.11)..."
    rm -rf .venv-molmospaces
}

_molmospaces_drop_venv_if_python_too_old

if [ ! -d ".venv-molmospaces" ]; then
    echo "Creating .venv-molmospaces (Python 3.11+)..."
    if uv venv .venv-molmospaces --python 3.11 2>/dev/null; then
        :
    elif uv venv .venv-molmospaces --python 3.12 2>/dev/null; then
        :
    else
        echo "ERROR: Need Python 3.11+. Install it or run: uv python install 3.11"
        exit 1
    fi
    molmo_install_editable_chain
    echo "  -> Created .venv-molmospaces with emet and emet-molmospaces wrapper"
else
    echo ".venv-molmospaces already exists."
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
        echo "  -> Repairing MolmoSpaces venv..."
        molmo_install_editable_chain
    fi
fi
if ! "$PY_MOLMO" -c "import mujoco; import emet_molmospaces; from molmo_spaces.molmo_spaces_constants import get_scenes; from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path" 2>/dev/null; then
    echo ""
    echo "ERROR: .venv-molmospaces failed verification (molmo-spaces API + mujoco + wrapper)."
    echo "  Fix: rm -rf .venv-molmospaces && ./scripts/install_molmospaces.sh"
    exit 1
fi
echo "  -> Verified: molmo_spaces OK in .venv-molmospaces"
echo "  -> MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR"
echo "  -> Run: emet molmospaces list-robots  &&  emet molmospaces serve --viewer"
