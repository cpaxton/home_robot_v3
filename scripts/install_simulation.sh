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

# Note: stretch_mujoco is now merged into src/emet/simulation/ in this repo.
# Install the stretch package with sim extras: pip install -e ".[sim]"

# Clone robosuite
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
cd robosuite || exit 1
pip install -e .
cd ..

# Clone robocasa
git clone https://github.com/robocasa/robocasa
cd robocasa || exit 1
pip install -e .
cd ..

# Run robocasa scripts based on flags
# Run robocasa scripts (we're in third_party, robocasa/scripts/ in the cloned repo)
if [ "$DOWNLOAD_ASSETS" = true ]; then
    echo "Downloading kitchen assets..."
    python robocasa/scripts/download_kitchen_assets.py
fi

if [ "$SETUP_MACROS" = true ]; then
    echo "Setting up macros..."
    python robocasa/scripts/setup_macros.py
fi

# Return to root directory
cd "$ROOT_DIR" || exit 1

echo "Installation complete!"

