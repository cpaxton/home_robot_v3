#!/usr/bin/env bash
# Install Habitat EQA harness venv (.venv-habitat). Run from repo root:
#   ./scripts/install_habitat.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PY_HAB=".venv-habitat/bin/python"

habitat_pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY_HAB" "$@"
  else
    "$PY_HAB" -m pip install "$@"
  fi
}

habitat_install_editable_chain() {
  habitat_pip_install --upgrade pip 2>/dev/null || true
  habitat_pip_install --no-deps -e .
  if [ -d "packages/emet_habitat" ]; then
    habitat_pip_install -e packages/emet_habitat
  else
    echo "ERROR: packages/emet_habitat not found"
    exit 1
  fi
}

_drop_venv_if_python_wrong() {
  if [ ! -d ".venv-habitat" ] || [ ! -x "$PY_HAB" ]; then
    return 0
  fi
  if "$PY_HAB" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 11) else 1)" 2>/dev/null; then
    return 0
  fi
  echo "Replacing .venv-habitat (habitat-sim requires Python 3.9 or 3.10)..."
  rm -rf .venv-habitat
}

_drop_venv_if_python_wrong

if [ ! -d ".venv-habitat" ]; then
  echo "Creating .venv-habitat (Python 3.10 preferred for habitat-sim)..."
  if uv venv .venv-habitat --python 3.10 2>/dev/null; then
    :
  elif uv venv .venv-habitat --python 3.9 2>/dev/null; then
    :
  else
    echo "ERROR: Need Python 3.9 or 3.10. Try: uv python install 3.10"
    exit 1
  fi
  habitat_install_editable_chain
  echo "  -> Created .venv-habitat with emet + emet-habitat"
else
  echo ".venv-habitat already exists."
  NEED_REPAIR=0
  if [ ! -x "$PY_HAB" ]; then
    NEED_REPAIR=1
  elif ! "$PY_HAB" -c "import emet" 2>/dev/null; then
    NEED_REPAIR=1
  elif ! "$PY_HAB" -c "import emet_habitat" 2>/dev/null; then
    NEED_REPAIR=1
  elif [ ! -x ".venv-habitat/bin/emet-habitat" ]; then
    NEED_REPAIR=1
  fi
  if [ "$NEED_REPAIR" -eq 1 ]; then
    echo "  -> Repairing Habitat venv..."
    habitat_install_editable_chain
  fi
fi

if ! "$PY_HAB" -c "import habitat_sim; import emet_habitat" 2>/dev/null; then
  echo ""
  echo "WARNING: habitat_sim import failed. Install a platform wheel, e.g.:"
  echo "  uv pip install --python $PY_HAB habitat-sim"
  echo "See https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md"
  exit 1
fi

echo "Habitat harness ready: .venv-habitat/bin/emet-habitat info"
