#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Run Tegra-CUDA OpenAI LLM server in Docker on Jetson Orin (JP5.1.x).
#
#   ./scripts/run_jetson_llm_container.sh
#   ./scripts/run_jetson_llm_container.sh --model Qwen/Qwen2.5-0.5B-Instruct --detach
#   ./scripts/run_jetson_llm_container.sh --model Qwen/Qwen2.5-7B-Instruct
#   ./scripts/run_jetson_llm_container.sh --build
#
# Workstation client:
#   export EMET_OPENAI_BASE_URL=http://caliban:8000/v1
#   export EMET_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${EMET_JETSON_LLM_IMAGE:-emet-jetson-llm:r35.4.1}"
NAME="${EMET_JETSON_LLM_NAME:-emet-jetson-llm}"
PORT="${EMET_LLM_SERVE_PORT:-8000}"
MODEL="${EMET_LLM_SERVE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Prefer a user-writable cache (legacy /data paths are often root-owned on this Orin).
HF_CACHE="${HF_HOME:-${HOME}/hf-cache}"
BUILD=0
DETACH=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build|-b) BUILD=1; shift ;;
    --detach|-d) DETACH=1; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --hf-cache) HF_CACHE="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$HF_CACHE"

run_docker() {
  if groups | grep -q '\bdocker\b'; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "ERROR: not a Jetson (missing /etc/nv_tegra_release)"
  exit 1
fi

if [[ "$BUILD" -eq 1 ]] || ! run_docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building image $IMAGE ..."
  EMET_JETSON_LLM_IMAGE="$IMAGE" bash "$ROOT/docker/build-jetson-llm-docker.sh"
fi

# Stop prior instance with same name
if run_docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Removing existing container $NAME"
  run_docker rm -f "$NAME" >/dev/null
fi

echo "Starting $NAME  image=$IMAGE  model=$MODEL  port=$PORT"
echo "HF cache: $HF_CACHE"
echo "Health:   curl -s http://127.0.0.1:${PORT}/health"

RUN_FLAGS=(--rm --name "$NAME" --runtime nvidia --network host --shm-size=8g)
if [[ "$DETACH" -eq 1 ]]; then
  RUN_FLAGS+=(-d)
  echo "Detach:   docker logs -f $NAME"
else
  RUN_FLAGS+=(-it)
fi

run_docker run "${RUN_FLAGS[@]}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/data/huggingface \
  -e TRANSFORMERS_CACHE=/data/huggingface \
  -e EMET_LLM_SERVE_MODEL="$MODEL" \
  -e EMET_LLM_SERVE_PORT="$PORT" \
  -e EMET_LLM_SERVE_DEVICE=cuda \
  -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro \
  -v "$HF_CACHE:/data/huggingface" \
  -v "$ROOT/docker/jetson_llm_server.py:/app/jetson_llm_server.py:ro" \
  "$IMAGE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --device cuda \
  --model "$MODEL" \
  "${EXTRA_ARGS[@]}"

if [[ "$DETACH" -eq 1 ]]; then
  echo "Container started. Wait for model load, then:"
  echo "  curl -s http://127.0.0.1:${PORT}/health"
  echo "  docker logs -f $NAME"
fi
