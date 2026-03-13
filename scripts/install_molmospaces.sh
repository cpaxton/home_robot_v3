#!/usr/bin/env bash
# Install only the MolmoSpaces wrapper venv (.venv-molmospaces). No sim, no main uv sync.
# Run from repo root:  ./scripts/install_molmospaces.sh
# Uses uv pip when available so we don't require pip inside the venv (uv venvs often have no pip).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MLSPACES_CACHE="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
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

if [ ! -d ".venv-molmospaces" ]; then
    echo "Creating .venv-molmospaces..."
    uv venv .venv-molmospaces
    molmo_pip_install --upgrade pip 2>/dev/null || true
    molmo_pip_install --no-deps -e .
    if [ -d "packages/emet_molmospaces" ]; then
        molmo_pip_install -e packages/emet_molmospaces
    else
        molmo_pip_install "molmo-spaces" "mujoco>=3.4" "numpy>=2.2"
    fi
    echo "  -> Created .venv-molmospaces with emet and emet-molmospaces wrapper"
else
    echo ".venv-molmospaces already exists."
    if [ -d "packages/emet_molmospaces" ] && [ ! -x ".venv-molmospaces/bin/emet-molmospaces" ]; then
        molmo_pip_install -e packages/emet_molmospaces
        echo "  -> Installed emet-molmospaces wrapper into existing venv"
    fi
fi
echo "  -> MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR"
echo "  -> Run: emet molmospaces list-robots  &&  emet molmospaces serve --viewer"
