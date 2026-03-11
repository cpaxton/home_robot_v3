#!/bin/bash
# Build the Stretch AI CUDA 11.8 Docker image (uv-based install).
# From repo root, ensure submodules are inited: git submodule update --init --recursive third_party/segment-anything-2
# Then: ./docker/build-docker.sh [-y]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

VERSION=$(python3 src/emet/version.py)
echo "Building docker image with tag hellorobotinc/stretch-ai_cuda-11.8:$VERSION"

SKIP_ASKING="false"
for arg in "$@"; do
    case $arg in
        -y|--yes) SKIP_ASKING="true" ;;
        *) ;;
    esac
done
if [ "$SKIP_ASKING" = "false" ]; then
    read -p "Proceed? (y/n) " yn
    case $yn in
        [yY]) ;;
        *) echo "Exiting."; exit 1 ;;
    esac
fi

if [ ! -d "third_party/segment-anything-2" ]; then
    echo "Initing submodule third_party/segment-anything-2..."
    git submodule update --init --recursive third_party/segment-anything-2
fi

docker build -t "hellorobotinc/stretch-ai_cuda-11.8:$VERSION" . -f docker/Dockerfile.cuda-11.8
docker tag "hellorobotinc/stretch-ai_cuda-11.8:$VERSION" hellorobotinc/stretch-ai_cuda-11.8:latest
echo "Built. Push with: docker push hellorobotinc/stretch-ai_cuda-11.8:$VERSION && docker push hellorobotinc/stretch-ai_cuda-11.8:latest"
