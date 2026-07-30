#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Smoke the OpenAI-compatible LLM on caliban from a workstation (no SSH required).
#
#   ./scripts/smoke_caliban_llm.sh
#   EMET_OPENAI_BASE_URL=http://192.168.1.55:8000/v1 ./scripts/smoke_caliban_llm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${EMET_OPENAI_BASE_URL:-http://caliban:8000/v1}"
BASE="${BASE%/}"
HEALTH="${BASE%/v1}/health"

echo "[caliban] GET $HEALTH"
curl -sf -m 10 "$HEALTH" | python3 -m json.tool

echo "[caliban] POST $BASE/chat/completions"
curl -sf -m 60 "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":16}' \
  | python3 -m json.tool

echo "[caliban] emet OpenaiClient"
cd "$ROOT"
uv run python - <<PY
from emet.llms import get_llm_client
c = get_llm_client("openai@${BASE}", prompt="You are a concise assistant.")
print("client:", repr(c("Reply with exactly: pong", verbose=False))[:120])
PY

echo "[caliban] OK"
