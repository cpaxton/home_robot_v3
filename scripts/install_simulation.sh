#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the root directory (assuming scripts is in the root)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Initialize flags (download is part of installation by default)
DOWNLOAD_ASSETS=true
# -a = force run macro setup (overwrite if exists); otherwise we run only when macros_private is missing
SETUP_MACROS_FORCE=false
# -y = non-interactive: skip interactive prompts (asset re-download + macro overwrite)
NONINTERACTIVE_ASSETS="false"

# Parse command line options
while getopts "dany" opt; do
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
        y )
            NONINTERACTIVE_ASSETS="true"
            ;;
        \? )
            echo "Usage: $0 [-d] [-n] [-a] [-y]"
            echo "  -d: Download kitchen assets (default: yes)"
            echo "  -n: Skip downloading kitchen assets (~10GB)"
            echo "  -a: Force setup macros (overwrite existing macros_private.py)"
            echo "  -y: Non-interactive: skip prompts (asset re-download and macro overwrite)"
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
        # Keep the main environment stable: uv sync owns dependency resolution.
        # Editable third_party installs should register package code only.
        uv pip install -e . --no-deps
    elif command -v uv >/dev/null 2>&1; then
        uv pip install -e . --no-deps
    else
        "$PYTHON" -m pip install -e . --no-deps
    fi
}

# Robocasa needs robosuite from master (get_elements in mjcf_utils; v1.5.0 lacks it).
# If already cloned, update in place (fetch + reset). Otherwise clone.
ROBOSUITE_BRANCH="master"
if [ -d "robosuite" ]; then
    echo "Updating existing third_party/robosuite (fetch + checkout $ROBOSUITE_BRANCH)..."
    cd robosuite || exit 1
    git fetch origin "$ROBOSUITE_BRANCH" 2>/dev/null || true
    git checkout "$ROBOSUITE_BRANCH" 2>/dev/null || true
    git pull --ff-only origin "$ROBOSUITE_BRANCH" 2>/dev/null || git reset --hard "origin/$ROBOSUITE_BRANCH" 2>/dev/null || true
    cd .. || exit 1
else
    git clone https://github.com/ARISE-Initiative/robosuite --branch "$ROBOSUITE_BRANCH" --single-branch --depth 1
fi
cd robosuite || exit 1
pip_install_editable || { echo "robosuite install failed." >&2; exit 1; }
# Create macros_private.py from macros.py if missing (silences "No private macro file" warnings).
if [ "$SETUP_MACROS_FORCE" = true ] || [ ! -f "robosuite/macros_private.py" ]; then
    # In non-interactive mode, auto-confirm overwrite prompts.
    if [ "$SETUP_MACROS_FORCE" = true ] || [ "$NONINTERACTIVE_ASSETS" = "true" ]; then
        echo "y" | "$PYTHON" scripts/setup_macros.py || true
    else
        "$PYTHON" scripts/setup_macros.py || { echo "robosuite setup_macros failed." >&2; exit 1; }
    fi
fi
cd ..

# Optional extra robot models for robosuite (GR1, etc.). Clone from cpaxton fork; if present, update (fetch/pull).
if [ ! -d "robosuite_models" ]; then
    git clone git@github.com:cpaxton/robosuite_models.git
fi
if [ -d "robosuite_models" ]; then
    cd robosuite_models || exit 1
    git fetch origin 2>/dev/null || true
    git pull --ff-only origin 2>/dev/null || true
    pip_install_editable || { echo "robosuite_models install failed." >&2; exit 1; }
    cd .. || exit 1
fi

# Robocasa from fork (branch feature/v1.24; numpy 1.24+ compat). If already cloned, update in place; else clone.
ROBOCASA_BRANCH="feature/v1.24"
if [ -d "robocasa" ]; then
    echo "Updating existing third_party/robocasa (fetch + checkout $ROBOCASA_BRANCH)..."
    cd robocasa || exit 1
    git fetch origin "$ROBOCASA_BRANCH" 2>/dev/null || true
    git checkout "$ROBOCASA_BRANCH" 2>/dev/null || true
    git pull --ff-only origin "$ROBOCASA_BRANCH" 2>/dev/null || git reset --hard "origin/$ROBOCASA_BRANCH" 2>/dev/null || true
    cd .. || exit 1
else
    git clone git@github.com:cpaxton/robocasa.git --branch "$ROBOCASA_BRANCH" --single-branch --depth 1
fi
cd robocasa || exit 1
pip_install_editable || { echo "robocasa install failed." >&2; exit 1; }
cd .. || exit 1

# Run robocasa setup using the installed package (python -m robocasa.scripts.setup_macros).
# macros_private.py is created at third_party/robocasa/robocasa/macros_private.py.
if [ "$SETUP_MACROS_FORCE" = true ] || [ ! -f "$ROOT_DIR/third_party/robocasa/robocasa/macros_private.py" ]; then
    cd "$ROOT_DIR" || exit 1
    if [ "$SETUP_MACROS_FORCE" = true ]; then
        echo "y" | "$PYTHON" -m robocasa.scripts.setup_macros || true
    else
        "$PYTHON" -m robocasa.scripts.setup_macros || { echo "robocasa setup_macros failed." >&2; exit 1; }
    fi
    cd "$ROOT_DIR/third_party" || exit 1
fi

# Asset download: if assets already exist, ask to re-download (default N). Use -n to skip entirely, -y to skip prompt when present.
ASSETS_DIR="$ROOT_DIR/third_party/robocasa/robocasa/models/assets"
if [ "$DOWNLOAD_ASSETS" = true ]; then
    if [ -d "$ASSETS_DIR/textures" ] && [ -n "$(ls -A "$ASSETS_DIR/textures" 2>/dev/null)" ]; then
        if [ "$NONINTERACTIVE_ASSETS" = "true" ]; then
            echo "Robocasa kitchen assets already present; skipping re-download (non-interactive)."
            DOWNLOAD_ASSETS=false
        else
            echo ""
            read -p "Robocasa kitchen assets appear to be present. Re-download? (y/N) " yn
            case "${yn:-n}" in
                y|Y) ;;
                *) echo "Skipping asset download."; DOWNLOAD_ASSETS=false ;;
            esac
        fi
    fi
fi
if [ "$DOWNLOAD_ASSETS" = true ]; then
    echo "Checking and downloading Robocasa kitchen assets as needed (~10GB max)..."
    cd "$ROOT_DIR" || exit 1
    if [ "$NONINTERACTIVE_ASSETS" = "true" ]; then
        "$PYTHON" scripts/download_robocasa_assets.py --yes || \
            { echo "download_robocasa_assets failed." >&2; exit 1; }
    else
        "$PYTHON" scripts/download_robocasa_assets.py || \
            { echo "download_robocasa_assets failed." >&2; exit 1; }
    fi
    cd "$ROOT_DIR/third_party" || exit 1
fi

# Return to root directory
cd "$ROOT_DIR" || exit 1

echo "Installation complete."
echo "For 'emet serve mujoco --use-robocasa', ensure kitchen assets were downloaded (run this script without -n if you skipped earlier)."
