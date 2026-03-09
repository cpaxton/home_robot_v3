#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the root directory (assuming scripts is in the root)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Initialize flags
DOWNLOAD_ASSETS=false
SETUP_MACROS=false

# Parse command line options
while getopts "da" opt; do
    case ${opt} in
        d )
            DOWNLOAD_ASSETS=true
            ;;
        a )
            SETUP_MACROS=true
            ;;
        \? )
            echo "Usage: $0 [-d] [-a]"
            echo "  -d: Download kitchen assets"
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

# Clone robosuite (required by robocasa)
if [ ! -d "robosuite" ]; then
    git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
fi
cd robosuite || exit 1
pip install -e . || { echo "robosuite install failed." >&2; exit 1; }
cd ..

# Clone robocasa at v0.2 (compatible mujoco/numpy)
if [ ! -d "robocasa" ]; then
    git clone https://github.com/robocasa/robocasa --branch v0.2 --single-branch
fi
cd robocasa || exit 1
git fetch --tags origin 2>/dev/null || true
git checkout v0.2 2>/dev/null || true
pip install -e . || { echo "robocasa install failed." >&2; exit 1; }
cd ..

# Run robocasa scripts from third_party (robocasa/scripts/ in the cloned repo)
if [ "$DOWNLOAD_ASSETS" = true ]; then
    echo "Downloading kitchen assets..."
    python robocasa/scripts/download_kitchen_assets.py || { echo "download_kitchen_assets failed." >&2; exit 1; }
fi

if [ "$SETUP_MACROS" = true ]; then
    echo "Setting up macros..."
    python robocasa/scripts/setup_macros.py || { echo "setup_macros failed." >&2; exit 1; }
fi

# Return to root directory
cd "$ROOT_DIR" || exit 1

echo "Installation complete."
