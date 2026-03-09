#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the root directory (assuming scripts is in the root)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Initialize flags (download is part of installation by default)
DOWNLOAD_ASSETS=true
# -a = force run macro setup (overwrite if exists); otherwise we run only when macros_private is missing
SETUP_MACROS_FORCE=false

# Parse command line options
while getopts "dan" opt; do
    case ${opt} in
        d )
            DOWNLOAD_ASSETS=true
            ;;
        n )
            DOWNLOAD_ASSETS=false
            ;;
        a )
            SETUP_MACROS_FORCE=true
            ;;
        \? )
            echo "Usage: $0 [-d] [-n] [-a]"
            echo "  -d: Download kitchen assets (default: yes)"
            echo "  -n: Skip downloading kitchen assets (~5GB)"
            echo "  -a: Force setup macros (overwrite existing macros_private.py)"
            exit 1
            ;;
    esac
done

# Check if third_party directory exists
if [ ! -d "$ROOT_DIR/third_party" ]; then
    echo "Error: third_party directory does not exist in $ROOT_DIR" >&2
    exit 1
fi

# Change to the third_party directory
cd "$ROOT_DIR/third_party" || exit 1

# We use a fork of Robocasa (cpaxton/robocasa) with numpy 1.24+ compatibility so the same
# env can run robocasa and dynamem. Upstream robocasa pins numpy 1.23.x. third_party/robocasa
# is gitignored. See docs/simulation.md.

# Use the same Python as the emet CLI (e.g. venv) so we don't install into system site-packages.
PYTHON="${EMET_PYTHON:-python}"
echo "Using Python: $PYTHON"

# Prefer uv pip when available (venvs created with uv often don't have pip).
# EMET_USE_UV=1 is set by the CLI when uv is available.
pip_install_editable() {
    if [ -n "${EMET_USE_UV:-}" ] && command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    elif command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    else
        "$PYTHON" -m pip install -e .
    fi
}

# Robocasa v0.2 requires RoboSuite v1.5 (see robocasa README: "using RoboSuite v1.5 as the backend").
# robocasa_v0.1 is for older robocasa v0.1 and lacks load_composite_controller_config, PandaOmron, etc.
if [ ! -d "robosuite" ]; then
    git clone https://github.com/ARISE-Initiative/robosuite --branch v1.5.0 --single-branch --depth 1
fi
cd robosuite || exit 1
# Ensure correct version for existing clones (robocasa v0.2 needs robosuite v1.5.0)
git fetch origin --tags 2>/dev/null || true
git checkout v1.5.0 2>/dev/null || true
pip_install_editable || { echo "robosuite install failed." >&2; exit 1; }
# Create macros_private.py from macros.py if missing (silences "No private macro file" warnings).
if [ "$SETUP_MACROS_FORCE" = true ] || [ ! -f "robosuite/robosuite/macros_private.py" ]; then
    if [ "$SETUP_MACROS_FORCE" = true ]; then
        echo "y" | "$PYTHON" robosuite/scripts/setup_macros.py || true
    else
        "$PYTHON" robosuite/scripts/setup_macros.py || { echo "robosuite setup_macros failed." >&2; exit 1; }
    fi
fi
cd ..

# Clone robocasa from fork with numpy 1.24+ compat (same env as dynamem). Uses default branch.
if [ ! -d "robocasa" ]; then
    git clone git@github.com:cpaxton/robocasa.git
fi
cd robocasa || exit 1
git fetch origin 2>/dev/null || true
pip_install_editable || { echo "robocasa install failed." >&2; exit 1; }
# Create macros_private.py from macros.py if missing (silences "No private macro file" warnings).
if [ "$SETUP_MACROS_FORCE" = true ] || [ ! -f "robocasa/macros_private.py" ]; then
    if [ "$SETUP_MACROS_FORCE" = true ]; then
        echo "y" | "$PYTHON" robocasa/robocasa/scripts/setup_macros.py || true
    else
        "$PYTHON" robocasa/robocasa/scripts/setup_macros.py || { echo "robocasa setup_macros failed." >&2; exit 1; }
    fi
fi
cd ..

# Asset download is part of installation by default (~5GB); use -n to skip.
# We use the project's scripts/download_robocasa_assets.py (no robocasa import) so it
# runs with the project Python regardless of numpy version.
if [ "$DOWNLOAD_ASSETS" = true ]; then
    echo "Downloading Robocasa kitchen assets (~5GB)..."
    cd "$ROOT_DIR" || exit 1
    "$PYTHON" "$ROOT_DIR/scripts/download_robocasa_assets.py" --yes || \
        { echo "download_robocasa_assets failed." >&2; exit 1; }
    cd "$ROOT_DIR/third_party" || exit 1
fi

# Return to root directory
cd "$ROOT_DIR" || exit 1

echo "Installation complete."
