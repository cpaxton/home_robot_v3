#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# From a workstation (olympia): free Orin disk if needed, sync VL weights + Jetson
# server script, start emet-jetson-vl on caliban:8001 beside text :8000.
#
#   ./scripts/deploy_caliban_vl.sh
#   EMET_CALIBAN_HOST=caliban ./scripts/deploy_caliban_vl.sh
#
# Requires SSH to caliban and local HF cache for Qwen/Qwen2-VL-2B-Instruct.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EMET_CALIBAN_HOST:-caliban}"
REMOTE_REPO="${EMET_CALIBAN_REPO:-~/src/home_robot_v3}"
MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
LOCAL_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen2-VL-2B-Instruct"
REMOTE_HF='~/hf-cache'

if [[ ! -d "$LOCAL_HUB" ]]; then
  echo "Missing local weights: $LOCAL_HUB"
  echo "Download first:  huggingface-cli download $MODEL_ID"
  exit 1
fi

echo "[1/5] Sync jetson VL server + runner → $HOST:$REMOTE_REPO"
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE_REPO/docker $REMOTE_REPO/scripts ~/hf-cache"
rsync -az "$ROOT/docker/jetson_llm_server.py" "$HOST:$REMOTE_REPO/docker/jetson_llm_server.py"
rsync -az "$ROOT/scripts/run_jetson_llm_container.sh" "$HOST:$REMOTE_REPO/scripts/run_jetson_llm_container.sh"
ssh -o BatchMode=yes "$HOST" "chmod +x $REMOTE_REPO/scripts/run_jetson_llm_container.sh"

echo "[2/5] Free disk on $HOST if needed (venv + smoke 0.5B)"
ssh -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -euo pipefail
df -h / | tail -1
# Host .venv is CPU-only torch — LLM serve uses Docker. Safe to drop for VL weights.
if [[ -d ~/src/home_robot_v3/.venv ]]; then
  echo "Removing ~/src/home_robot_v3/.venv to free space for VL weights"
  rm -rf ~/src/home_robot_v3/.venv
fi
if [[ -d ~/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct ]]; then
  echo "Removing unused Qwen2.5-0.5B HF cache (may need sudo if root-owned)"
  rm -rf ~/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct 2>/dev/null \
    || sudo -n rm -rf ~/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct 2>/dev/null \
    || echo "WARN: could not remove 0.5B cache (root-owned); continuing if enough free space"
fi
rm -rf ~/.cache/uv 2>/dev/null || true
avail_kb=$(df -k / | awk 'NR==2{print $4}')
echo "Free KiB on /: $avail_kb"
if [[ "$avail_kb" -lt 4500000 ]]; then
  echo "ERROR: need ~4.5 GiB free for Qwen2-VL-2B; free disk or sudo-rm root-owned hf-cache entries"
  df -h /
  exit 1
fi
df -h / | tail -1
REMOTE

echo "[3/5] Rsync $MODEL_ID weights → $HOST:$REMOTE_HF"
rsync -a --info=progress2 "$LOCAL_HUB/" "$HOST:$REMOTE_HF/models--Qwen--Qwen2-VL-2B-Instruct/"

echo "[4/5] Start VL container on :8001 (text :8000 left running)"
ssh -o BatchMode=yes "$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_REPO
# Ensure HF hub layout refs exist for from_pretrained(Qwen/Qwen2-VL-2B-Instruct)
python3 - <<'PY'
import os
from pathlib import Path
hub = Path(os.path.expanduser("~/hf-cache"))
model = hub / "models--Qwen--Qwen2-VL-2B-Instruct"
snaps = sorted((model / "snapshots").glob("*")) if (model / "snapshots").is_dir() else []
if not snaps:
    raise SystemExit("missing snapshots under %s" % model)
snap = snaps[-1]
refs = model / "refs"
refs.mkdir(parents=True, exist_ok=True)
main = refs / "main"
main.write_text(snap.name + "\n")
print("HF ref main ->", snap.name)
PY
./scripts/run_jetson_llm_container.sh --vl --detach --port 8001 --name emet-jetson-vl \
  --model $MODEL_ID --hf-cache "\$HOME/hf-cache"
REMOTE

echo "[5/5] Wait for health on $HOST:8001"
for i in $(seq 1 60); do
  if out=$(curl -sf -m 5 "http://${HOST}:8001/health" 2>/dev/null); then
    if echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)"; then
      echo "$out" | python3 -m json.tool
      echo "OK — set EMET_VL_ENDPOINT=openai@http://${HOST}:8001/v1 (Herman preset already uses caliban:8001)"
      echo "Smoke:  uv run emet llm smoke --vl-only --vl http://${HOST}:8001/v1"
      exit 0
    fi
  fi
  echo "  waiting… ($i/60)"
  sleep 10
done
echo "TIMEOUT waiting for VL ready. Check: ssh $HOST 'docker logs -f emet-jetson-vl'"
exit 1
