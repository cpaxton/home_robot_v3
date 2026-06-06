#!/usr/bin/env bash
# Install Habitat EQA harness env (.venv-habitat) via micromamba + aihabitat-nightly.
# habitat-sim has no usable Linux wheels on PyPI; conda is required.
#
# Run from repo root:  ./scripts/install_habitat.sh
# Optional: HABITAT_HEADLESS=0 for GUI build (default: headless EGL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ENV_PREFIX="$ROOT_DIR/.venv-habitat"
MICROMAMBA_ROOT="$ROOT_DIR/.micromamba"
MICROMAMBA_BIN="$MICROMAMBA_ROOT/bin/micromamba"
PY_HAB="$ENV_PREFIX/bin/python"
HEADLESS="${HABITAT_HEADLESS:-1}"

ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN="$(command -v micromamba)"
    return 0
  fi
  if [ -x "$MICROMAMBA_BIN" ]; then
    return 0
  fi
  echo "Bootstrapping micromamba into $MICROMAMBA_ROOT ..."
  mkdir -p "$MICROMAMBA_ROOT"
  curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o /tmp/emet-micromamba.tar.bz2
  tar -xjf /tmp/emet-micromamba.tar.bz2 -C "$MICROMAMBA_ROOT" bin/micromamba
  rm -f /tmp/emet-micromamba.tar.bz2
}

mm() {
  "$MICROMAMBA_BIN" "$@"
}

habitat_pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY_HAB" "$@"
  else
    "$PY_HAB" -m pip install "$@"
  fi
}

install_habitat_sim_conda() {
  local channels=(-c conda-forge -c aihabitat-nightly)
  local variants=(withbullet)
  if [ "$HEADLESS" = "1" ]; then
    variants+=(headless)
  fi
  echo "Installing habitat-sim (${variants[*]}) into $ENV_PREFIX ..."
  # Nightly channel ships py3.10 builds; stable aihabitat is py3.9-only.
  mm install -y -p "$ENV_PREFIX" "${variants[@]}" habitat-sim "${channels[@]}"
  # Scientific stack from conda (avoid pip upgrading numpy/matplotlib under habitat-sim).
  mm install -y -p "$ENV_PREFIX" -c conda-forge matplotlib scipy pillow numpy
}

install_emet_packages() {
  echo "Installing emet + emet-habitat (editable, no upstream dep resolver) ..."
  habitat_pip_install --upgrade pip 2>/dev/null || true
  habitat_pip_install --no-deps -e .
  habitat_pip_install --no-deps -e packages/emet_habitat
  habitat_pip_install -r packages/emet_habitat/requirements-pip.txt
  # uv may skip packages it considers satisfied; verify runner import chain.
  if ! "$PY_HAB" -c "from emet_habitat.runner import run_hmeqa_episode" 2>/dev/null; then
    echo "  -> Retrying pip deps (runner import failed) ..."
    habitat_pip_install --upgrade -r packages/emet_habitat/requirements-pip.txt
  fi
}

repair_if_needed() {
  if [ ! -x "$PY_HAB" ]; then
    return 1
  fi
  if ! "$PY_HAB" -c "import habitat_sim, emet, emet_habitat" 2>/dev/null; then
    return 1
  fi
  if [ ! -x "$ENV_PREFIX/bin/emet-habitat" ]; then
    return 1
  fi
  return 0
}

ensure_micromamba

if [ ! -d "$ENV_PREFIX" ] || ! repair_if_needed; then
  if [ -d "$ENV_PREFIX" ]; then
    echo "Repairing incomplete $ENV_PREFIX ..."
  else
    echo "Creating $ENV_PREFIX (Python 3.10 + habitat-sim via micromamba) ..."
  fi
  if [ -d "$ENV_PREFIX" ]; then
    rm -rf "$ENV_PREFIX"
  fi
  mm create -y -p "$ENV_PREFIX" python=3.10 -c conda-forge
  install_habitat_sim_conda
  install_emet_packages
else
  echo "$ENV_PREFIX looks healthy."
fi

if ! "$PY_HAB" -c "
import habitat_sim
import emet_habitat
from emet.habitat.config import default_habitat_eqa_data_dir
print('habitat_sim', habitat_sim.__version__)
print('data_dir', default_habitat_eqa_data_dir())
"; then
  echo "ERROR: habitat_sim or emet.habitat failed to import." >&2
  exit 1
fi

echo ""
echo "Habitat harness ready:"
echo "  $ENV_PREFIX/bin/emet-habitat info"
echo "  uv run emet run graph-eqa-habitat --mock-llm --question-id 0"
echo ""
echo "Data: uv run python scripts/download_habitat_eqa_data.py --fetch-csv"
echo "Docs: docs/habitat/README.md"
echo "HM3D tokens: Profile → Settings → Developer Tools (docs/habitat/data.md)"
