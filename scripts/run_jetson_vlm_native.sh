#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Native CUDA VLM serve on JetPack 7 / L4T r39 (AGX Orin).
# The JP5 dustynv image (l4t-pytorch:r35.4.1) does not run on r39.
# PyPI aarch64 torch is CPU-only; this venv uses the cu130 index.
#
# Largest dense Qwen3-VL that fits 64 GiB unified memory *and* ~54G eMMC:
#   Qwen/Qwen3-VL-8B-Instruct  (~17G weights on disk)
# 8B fp16 ~20 GiB RAM; fp32 ~36 GiB RAM (fits 61 GiB unified, no extra disk).
# Qwen3-VL-32B bf16 (~63G) and 32B AWQ (~20G + vLLM image) need NVMe.
#
#   ./scripts/run_jetson_vlm_native.sh --setup
#   ./scripts/run_jetson_vlm_native.sh --detach
#   ./scripts/run_jetson_vlm_native.sh --detach --dtype float32
#
# Client (use the Orin LAN or Tailscale IP; Fios DNS may map hostname
# "caliban" to a stale lease):
#   export EMET_LLM_HOST=192.168.1.55
#   uv run emet llm health --host 192.168.1.55

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${EMET_JETSON_VLM_VENV:-$ROOT/.venv-vlm}"
MODEL="${EMET_LLM_SERVE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${EMET_LLM_SERVE_PORT:-8000}"
DTYPE="${EMET_LLM_SERVE_DTYPE:-float16}"
HF_CACHE="${HF_HOME:-${HOME}/hf-cache}"
SETUP=0
DETACH=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup) SETUP=1; shift ;;
        --detach|-d) DETACH=1; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --hf-cache) HF_CACHE="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,44p' "$0"
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f /etc/nv_tegra_release ]]; then
    echo "ERROR: not a Jetson (missing /etc/nv_tegra_release)" >&2
    exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"
mkdir -p "$HF_CACHE"

if [[ "$SETUP" -eq 1 ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    # cu130 wheels are cp312+; system Python on JP7 is 3.12.
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python "$VENV/bin/python" \
        "transformers>=4.55" \
        accelerate \
        pillow \
        qwen-vl-utils \
        safetensors \
        sentencepiece \
        protobuf \
        einops
    "$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA torch installed but cuda is False")
print("device", torch.cuda.get_device_name(0))
x = torch.zeros(1, device="cuda")
print("cuda tensor ok", x.device)
PY
    echo "Setup complete: $VENV"
    exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: missing $VENV — run: $0 --setup" >&2
    exit 1
fi

if [[ -x /usr/sbin/nvpmodel ]] || command -v nvpmodel >/dev/null 2>&1; then
    echo "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
fi

LOG="${HOME}/emet-vlm-serve.log"
CMD=(
    "$VENV/bin/python" "$ROOT/docker/jetson_llm_server.py"
    --vl --host 0.0.0.0 --port "$PORT" --device cuda --dtype "$DTYPE"
    --model "$MODEL"
)
CMD+=("${EXTRA_ARGS[@]}")

echo "Serving $MODEL on :$PORT  dtype=$DTYPE  venv=$VENV  HF_HOME=$HF_CACHE"
if [[ "$DETACH" -eq 1 ]]; then
    # shellcheck disable=SC2086
    setsid -f env HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE" \
        EMET_ALLOW_SDPA_ATTN=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
        "${CMD[@]}" >>"$LOG" 2>&1
    echo "Detached. Logs: tail -f $LOG"
    echo "Health: curl -s http://127.0.0.1:${PORT}/health"
    exit 0
fi
exec env HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE" \
    EMET_ALLOW_SDPA_ATTN=1 \
    "${CMD[@]}"
