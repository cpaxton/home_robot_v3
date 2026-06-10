#!/usr/bin/env bash
# Create .venv-lingbot-map with LingBot-Map + emet_lingbot_map wrapper (isolated from main .venv).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LINGBOT_REPO="${LINGBOT_MAP_REPO:-$ROOT_DIR/third_party/lingbot-map}"
VENV_DIR="$ROOT_DIR/.venv-lingbot-map"
PY="$VENV_DIR/bin/python"

clone_lingbot() {
    if [ -d "$LINGBOT_REPO/.git" ]; then
        echo "  -> lingbot-map already cloned at $LINGBOT_REPO"
        return 0
    fi
    mkdir -p "$(dirname "$LINGBOT_REPO")"
    git clone --depth 1 https://github.com/Robbyant/lingbot-map.git "$LINGBOT_REPO"
}

pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" "$@"
    else
        "$PY" -m pip install "$@"
    fi
}

echo "=============================================="
echo "  LingBot-Map venv (.venv-lingbot-map)"
echo "=============================================="

if [ ! -d "$VENV_DIR" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR" --python 3.10
    else
        python3.10 -m venv "$VENV_DIR"
    fi
    echo "  -> Created $VENV_DIR"
else
    echo "  -> $VENV_DIR already exists"
fi

clone_lingbot

echo "Installing PyTorch 2.8 (cu128)..."
pip_install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing LingBot-Map..."
pip_install -e "$LINGBOT_REPO"

echo "Installing FlashInfer (recommended for streaming)..."
pip_install flashinfer-python || echo "WARNING: flashinfer-python install failed; use --use-sdpa at runtime."

echo "Installing ninja (FlashInfer JIT compile)..."
pip_install ninja || echo "WARNING: ninja install failed; use --use-sdpa at runtime."

echo "Installing emet_lingbot_map wrapper..."
pip_install -e packages/emet_lingbot_map

echo "Optional: numpy, opencv for episode I/O..."
pip_install numpy opencv-python-headless pillow tqdm

if ! "$PY" -c "import lingbot_map; import emet_lingbot_map" 2>/dev/null; then
    echo "ERROR: verification failed (lingbot_map / emet_lingbot_map import)."
    exit 1
fi

echo ""
echo "Done. Activate: source .venv-lingbot-map/bin/activate"
echo "Set checkpoint: export LINGBOT_MAP_CHECKPOINT=/path/to/lingbot-map-long.pt"
echo "Download: https://huggingface.co/robbyant/lingbot-map"
echo "Smoke: .venv-lingbot-map/bin/python -m emet_lingbot_map infer --help"
