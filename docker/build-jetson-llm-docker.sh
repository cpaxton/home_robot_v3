#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Build Tegra-CUDA LLM serve image on Jetson (JP5 / L4T R35.4).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${EMET_JETSON_LLM_IMAGE:-emet-jetson-llm:r35.4.1}"
BASE="${EMET_JETSON_LLM_BASE:-dustynv/l4t-pytorch:r35.4.1}"

if [[ ! -f /etc/nv_tegra_release ]]; then
    echo "WARNING: /etc/nv_tegra_release missing — build on a Jetson for Tegra CUDA."
fi

echo "Building $TAG from $BASE"
docker build \
    --build-arg "L4T_BASE=${BASE}" \
    -t "$TAG" \
    -f docker/Dockerfile.jetson-llm \
    .

echo "Done. Run: ./scripts/run_jetson_llm_container.sh"
