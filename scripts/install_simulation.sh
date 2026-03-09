#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the root directory (assuming scripts is in the root)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Initialize flags (download is part of installation by default)
DOWNLOAD_ASSETS=true
SETUP_MACROS=false

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
            SETUP_MACROS=true
            ;;
        \? )
            echo "Usage: $0 [-d] [-n] [-a]"
            echo "  -d: Download kitchen assets (default: yes)"
            echo "  -n: Skip downloading kitchen assets (~5GB)"
            echo "  -a: Setup macros"
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

# We pin Robocasa to v0.2 for compatibility with our stack (mujoco 3.2.6, numpy<2).
# Robocasa main/v1.0 uses mujoco 3.3.1 and numpy 2.x and pulls heavier deps (e.g. torch 2.7).
# See https://github.com/robocasa/robocasa and docs/simulation.md.

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
cd ..

# Clone robocasa at v0.2 (compatible mujoco/numpy)
if [ ! -d "robocasa" ]; then
    git clone https://github.com/robocasa/robocasa --branch v0.2 --single-branch
fi
cd robocasa || exit 1
git fetch --tags origin 2>/dev/null || true
git checkout v0.2 2>/dev/null || true
pip_install_editable || { echo "robocasa install failed." >&2; exit 1; }
cd ..

# Run robocasa scripts from third_party (scripts live in robocasa/robocasa/scripts/)
# Asset download is part of installation by default (~5GB); use -n to skip.
# Robocasa's download script imports robocasa, which asserts numpy 1.23.x. This project
# often uses numpy 1.24+ (e.g. uv override for dynamem). Run the download in a temp venv
# with numpy 1.23.3 so we don't patch third_party or change the project venv.
if [ "$DOWNLOAD_ASSETS" = true ]; then
    echo "Downloading Robocasa kitchen assets (~5GB)..."
    DL_VENV="$ROOT_DIR/third_party/.robocasa_download_venv"
    rm -rf "$DL_VENV"
    # Prefer uv venv (no ensurepip needed on Debian/Ubuntu); fallback to python -m venv
    if command -v uv >/dev/null 2>&1; then
        uv venv "$DL_VENV" || { echo "Failed to create download venv with uv." >&2; exit 1; }
    else
        "$PYTHON" -m venv "$DL_VENV" || {
            echo "Failed to create download venv. On Debian/Ubuntu install: sudo apt install python3.10-venv" >&2
            echo "Or skip the download now with: emet install robocasa --no-download-assets" >&2
            exit 1
        }
    fi
    # Robocasa's install_requires includes numpy==1.23.3; use this venv so its import assert passes
    "$DL_VENV/bin/python" -m pip install -q -e robosuite -e robocasa 2>/dev/null || \
        "$DL_VENV/bin/pip" install -q -e robosuite -e robocasa 2>/dev/null || \
        { echo "Failed to install robosuite/robocasa in download venv." >&2; rm -rf "$DL_VENV"; exit 1; }
    if echo "y" | "$DL_VENV/bin/python" robocasa/robocasa/scripts/download_kitchen_assets.py; then
        rm -rf "$DL_VENV"
    else
        echo "download_kitchen_assets failed." >&2
        rm -rf "$DL_VENV"
        exit 1
    fi
fi

if [ "$SETUP_MACROS" = true ]; then
    echo "Setting up macros..."
    "$PYTHON" robocasa/robocasa/scripts/setup_macros.py || { echo "setup_macros failed." >&2; exit 1; }
fi

# Return to root directory
cd "$ROOT_DIR" || exit 1

echo "Installation complete."
