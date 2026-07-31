#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Smoke the OpenAI-compatible LLM on a LAN Jetson from a workstation (no SSH required).
#
#   EMET_LLM_HOST=caliban ./scripts/smoke_caliban_llm.sh
#   EMET_OPENAI_BASE_URL=http://192.168.1.55:8000/v1 ./scripts/smoke_caliban_llm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EMET_LLM_HOST:-${EMET_CALIBAN_HOST:-}}"
if [[ -n "${EMET_OPENAI_BASE_URL:-}" ]]; then
  BASE="${EMET_OPENAI_BASE_URL}"
elif [[ -n "$HOST" ]]; then
  BASE="http://${HOST}:8000/v1"
else
  echo "ERROR: set EMET_LLM_HOST or EMET_OPENAI_BASE_URL (example: EMET_LLM_HOST=caliban)" >&2
  exit 1
fi
BASE="${BASE%/}"
HEALTH="${BASE%/v1}/health"

echo "[llm] GET $HEALTH"
curl -sf -m 10 "$HEALTH" | python3 -m json.tool

echo "[llm] POST $BASE/chat/completions"
curl -sf -m 60 "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":16}' \
  | python3 -m json.tool

echo "[llm] emet OpenaiClient"
cd "$ROOT"
uv run python - <<PY
from emet.llms import get_llm_client
c = get_llm_client("openai@${BASE}", prompt="You are a concise assistant.")
print("client:", repr(c("Reply with exactly: pong", verbose=False))[:120])
PY

echo "[llm] OK"
