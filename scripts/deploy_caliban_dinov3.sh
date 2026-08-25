#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Deploy DINOv3 vits16 embedding server to a Jetson Orin (e.g. caliban) on :8002.
#
#   ./scripts/deploy_caliban_dinov3.sh --host caliban
#
# Workstation:
#   export EMET_DINOV3_ENDPOINT=http://caliban:8002

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EMET_DINOV3_HOST:-${EMET_LLM_HOST:-${EMET_CALIBAN_HOST:-}}}"
REMOTE_REPO="${EMET_CALIBAN_REPO:-~/src/home_robot_v4}"
REMOTE_HF='~/hf-cache'
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
IMAGE="${EMET_JETSON_LLM_IMAGE:-emet-jetson-llm:r35.4.1}"
MODEL_ID="${EMET_DINOV3_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
PORT="${EMET_DINOV3_SERVE_PORT:-8002}"
NAME="${EMET_JETSON_DINOV3_NAME:-emet-jetson-dinov3}"
NEED_FREE_GIB=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --model) MODEL_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "ERROR: set --host HOST or EMET_DINOV3_HOST / EMET_LLM_HOST" >&2
    exit 1
fi

HUB_DIR="models--${MODEL_ID//\//--}"
LOCAL_HUB="${HF_HUB}/${HUB_DIR}"

if [[ ! -d "$LOCAL_HUB" ]]; then
    echo "Missing local weights: $LOCAL_HUB"
    echo "Download:  uv run python -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')\""
    exit 1
fi

echo "=== deploy_caliban_dinov3  model=$MODEL_ID  port=$PORT  host=$HOST ==="

echo "[1/4] Sync DINOv3 server + runner → $HOST:$REMOTE_REPO"
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE_REPO/docker $REMOTE_REPO/scripts ~/hf-cache/hub"
rsync -az "$ROOT/docker/jetson_dinov3_server.py" "$HOST:$REMOTE_REPO/docker/jetson_dinov3_server.py"
rsync -az "$ROOT/scripts/run_jetson_dinov3_container.sh" "$HOST:$REMOTE_REPO/scripts/run_jetson_dinov3_container.sh"
ssh -o BatchMode=yes "$HOST" "chmod +x $REMOTE_REPO/scripts/run_jetson_dinov3_container.sh"

echo "[2/4] Rsync $MODEL_ID → $HOST:$REMOTE_HF/hub/$HUB_DIR"
rsync -a --info=progress2 "$LOCAL_HUB/" "$HOST:$REMOTE_HF/hub/$HUB_DIR/"

echo "[3/4] Start container name=$NAME port=$PORT"
ssh -o BatchMode=yes "$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_REPO
python3 - <<'PY'
import os
from pathlib import Path
hub = Path(os.path.expanduser("~/hf-cache/hub"))
model = hub / "$HUB_DIR"
snaps = sorted((model / "snapshots").glob("*")) if (model / "snapshots").is_dir() else []
if not snaps:
    raise SystemExit("missing snapshots under %s" % model)
snap = snaps[-1]
refs = model / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(snap.name + "\n")
print("HF ref main ->", snap.name)
PY
./scripts/run_jetson_dinov3_container.sh --detach --port $PORT --name $NAME \
    --model $MODEL_ID --hf-cache "\$HOME/hf-cache" --image $IMAGE
REMOTE

echo "[4/4] Wait for health on $HOST:$PORT"
for i in $(seq 1 60); do
    if out=$(curl -sf -m 5 "http://${HOST}:${PORT}/health" 2>/dev/null); then
        if echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)"; then
            echo "$out" | python3 -m json.tool
            echo "OK — DINOv3 on :${PORT}"
            echo "  export EMET_DINOV3_ENDPOINT=http://${HOST}:${PORT}"
            exit 0
        fi
    fi
    echo "  waiting… ($i/60)"
    sleep 5
done
echo "TIMEOUT. Check: ssh $HOST 'docker logs -f $NAME'"
exit 1
