#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Lean emet install for NVIDIA Jetson Orin / Tegra (JetPack 5/6).
# Wrapper around ./install.sh --profile=jetson.
#
# Usage:
#   ./scripts/install_jetson.sh
#   ./scripts/install_jetson.sh -y
#
# See docs/jetson.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f /etc/nv_tegra_release ]]; then
    echo "Detected Tegra: $(tr -d '\0' </etc/nv_tegra_release | head -1)"
else
    echo "WARNING: /etc/nv_tegra_release not found — continuing anyway (aarch64 lean profile)."
fi

# Prefer existing swap file on this Orin image when present (helps sdist builds).
if [[ -f /mnt/4GB.swap ]] && ! swapon --show 2>/dev/null | grep -q .; then
    if sudo -n true 2>/dev/null; then
        echo "Enabling /mnt/4GB.swap for compiles..."
        sudo swapon /mnt/4GB.swap 2>/dev/null || true
    else
        echo "Swap file present at /mnt/4GB.swap but sudo unavailable; enable with: sudo swapon /mnt/4GB.swap"
    fi
fi

exec bash "$ROOT_DIR/install.sh" --profile=jetson "$@"
