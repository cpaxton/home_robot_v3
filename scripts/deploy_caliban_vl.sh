#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Prefer:  uv run emet deploy llm --profile unified-7b
#          uv run emet deploy llm --profile dual-2b
#
# Orin eMMC (~57G) cannot hold both Qwen2.5-7B text (~15G) and Qwen2-VL-7B (~15G)
# plus the L4T image. AGX Orin has ~64 GiB unified memory — enough for VL-7B.
#
#   dual-2b     — text CausalLM :8000 + Qwen2-VL-2B :8001
#   unified-7b  — one Qwen2-VL-7B on :8000 for text tools + captions
#
#   ./scripts/deploy_caliban_vl.sh --profile unified-7b
#   ./scripts/deploy_caliban_vl.sh --profile dual-2b
#
# Requires SSH to caliban and a local HF hub cache for the chosen model.
# After unified-7b, point Herman text + VL at the same URL:
#   agent.llm / mapping.eqa.vl_endpoint → openai@http://caliban:8000/v1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EMET_CALIBAN_HOST:-caliban}"
REMOTE_REPO="${EMET_CALIBAN_REPO:-~/src/home_robot_v3}"
REMOTE_HF='~/hf-cache'
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
IMAGE="${EMET_JETSON_LLM_IMAGE:-emet-jetson-llm:r35.4.1}"

PROFILE="dual-2b"
MODEL_ID=""
PORT=""
NAME=""
KEEP_TEXT=1
NEED_FREE_GIB=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

case "$PROFILE" in
  dual-2b|2b)
    PROFILE=dual-2b
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
    PORT="${PORT:-8001}"
    NAME="${NAME:-emet-jetson-vl}"
    KEEP_TEXT=1
    NEED_FREE_GIB=5
    ;;
  unified-7b|7b|big)
    PROFILE=unified-7b
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}"
    PORT="${PORT:-8000}"
    NAME="${NAME:-emet-jetson-llm}"
    KEEP_TEXT=0
    NEED_FREE_GIB=16
    ;;
  *)
    echo "Unknown --profile $PROFILE (use dual-2b or unified-7b)" >&2
    exit 1
    ;;
esac

HUB_DIR="models--${MODEL_ID//\//--}"
LOCAL_HUB="${HF_HUB}/${HUB_DIR}"

if [[ ! -d "$LOCAL_HUB" ]]; then
  echo "Missing local weights: $LOCAL_HUB"
  echo "Download:  uv run python -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')\""
  exit 1
fi

echo "=== deploy_caliban_vl  profile=$PROFILE  model=$MODEL_ID  port=$PORT  host=$HOST ==="

echo "[1/6] Sync jetson VL server + runner → $HOST:$REMOTE_REPO"
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE_REPO/docker $REMOTE_REPO/scripts ~/hf-cache"
rsync -az "$ROOT/docker/jetson_llm_server.py" "$HOST:$REMOTE_REPO/docker/jetson_llm_server.py"
rsync -az "$ROOT/scripts/run_jetson_llm_container.sh" "$HOST:$REMOTE_REPO/scripts/run_jetson_llm_container.sh"
ssh -o BatchMode=yes "$HOST" "chmod +x $REMOTE_REPO/scripts/run_jetson_llm_container.sh"

echo "[2/6] Stop containers that block the swap"
ssh -o BatchMode=yes "$HOST" bash -s <<REMOTE
set -euo pipefail
run_docker() {
  if groups | grep -q '\bdocker\b'; then docker "\$@"; else sudo docker "\$@"; fi
}
run_docker rm -f emet-jetson-vl 2>/dev/null || true
if [[ "$KEEP_TEXT" -eq 0 ]] || [[ "$NAME" == "emet-jetson-llm" ]]; then
  echo "Stopping emet-jetson-llm (port/name reuse or unified profile)"
  run_docker rm -f emet-jetson-llm 2>/dev/null || true
fi
REMOTE

echo "[3/6] Free disk on $HOST (root-owned HF dirs via docker as root)"
ssh -o BatchMode=yes "$HOST" \
  NEED_FREE_GIB="$NEED_FREE_GIB" PROFILE="$PROFILE" IMAGE="$IMAGE" \
  bash -s <<'REMOTE'
set -euo pipefail
run_docker() {
  if groups | grep -q '\bdocker\b'; then docker "$@"; else sudo docker "$@"; fi
}
df -h / | tail -1
rm -rf ~/src/home_robot_v3/.venv 2>/dev/null || true
rm -rf ~/.cache/uv 2>/dev/null || true

rm_hf() {
  local m="$1"
  [[ -e "$HOME/hf-cache/$m" ]] || return 0
  echo "Removing ~/hf-cache/$m"
  # Prefer existing Jetson image (already local); busybox as fallback.
  if run_docker image inspect "$IMAGE" >/dev/null 2>&1; then
    run_docker run --rm --entrypoint rm -v "$HOME/hf-cache:/data" "$IMAGE" -rf "/data/$m" || true
  fi
  if [[ -e "$HOME/hf-cache/$m" ]]; then
    run_docker run --rm -v "$HOME/hf-cache:/data" busybox:1.36 rm -rf "/data/$m" || true
  fi
  if [[ -e "$HOME/hf-cache/$m" ]]; then
    echo "WARN: still present: $m"
  fi
}

rm_hf models--Qwen--Qwen2.5-0.5B-Instruct
if [[ "$PROFILE" == "unified-7b" ]]; then
  rm_hf models--Qwen--Qwen2-VL-2B-Instruct
  rm_hf models--Qwen--Qwen2.5-7B-Instruct
elif [[ "$PROFILE" == "dual-2b" ]]; then
  : # keep text 7B + existing VL-2B unless replaced below
fi

df -h / | tail -1
avail_kb=$(df -k / | awk 'NR==2{print $4}')
need_kb=$((NEED_FREE_GIB * 1024 * 1024))
echo "Free KiB=$avail_kb need>=$need_kb (${NEED_FREE_GIB} GiB)"
if [[ "$avail_kb" -lt "$need_kb" ]]; then
  echo "ERROR: not enough free disk"
  df -h /
  du -sh ~/hf-cache/models--* 2>/dev/null || true
  exit 1
fi
REMOTE

echo "[4/6] Rsync $MODEL_ID → $HOST:$REMOTE_HF/$HUB_DIR"
rsync -a --info=progress2 "$LOCAL_HUB/" "$HOST:$REMOTE_HF/$HUB_DIR/"

echo "[5/6] Start container name=$NAME port=$PORT vl=1"
ssh -o BatchMode=yes "$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_REPO
python3 - <<'PY'
import os
from pathlib import Path
hub = Path(os.path.expanduser("~/hf-cache"))
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
./scripts/run_jetson_llm_container.sh --vl --detach --port $PORT --name $NAME \
  --model $MODEL_ID --hf-cache "\$HOME/hf-cache" --image $IMAGE
REMOTE

if [[ "$KEEP_TEXT" -eq 1 ]]; then
  echo "[5b] Ensure text CausalLM still on :8000"
  ssh -o BatchMode=yes "$HOST" bash -s <<REMOTE
set -euo pipefail
run_docker() {
  if groups | grep -q '\bdocker\b'; then docker "\$@"; else sudo docker "\$@"; fi
}
if ! run_docker ps --format '{{.Names}}' | grep -qx emet-jetson-llm; then
  echo "WARN: emet-jetson-llm not running — start manually if dual-2b text is required"
fi
REMOTE
fi

echo "[6/6] Wait for health on $HOST:$PORT"
for i in $(seq 1 90); do
  if out=$(curl -sf -m 5 "http://${HOST}:${PORT}/health" 2>/dev/null); then
    if echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)"; then
      echo "$out" | python3 -m json.tool
      if [[ "$PROFILE" == "unified-7b" ]]; then
        echo "OK — unified VL-7B on :${PORT}"
        echo "  Point BOTH Herman endpoints at openai@http://${HOST}:${PORT}/v1"
        echo "  (agent.llm + mapping.eqa.vl_endpoint), or:"
        echo "  export EMET_OPENAI_BASE_URL=http://${HOST}:${PORT}/v1"
        echo "  export EMET_VL_ENDPOINT=openai@http://${HOST}:${PORT}/v1"
      else
        echo "OK — VL on :${PORT} (text remains :8000)"
        echo "  EMET_VL_ENDPOINT=openai@http://${HOST}:${PORT}/v1"
      fi
      echo "Smoke:  uv run emet llm smoke --vl-only --vl http://${HOST}:${PORT}/v1"
      exit 0
    fi
  fi
  echo "  waiting… ($i/90)"
  sleep 10
done
echo "TIMEOUT. Check: ssh $HOST 'docker logs -f $NAME'"
exit 1
