#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Smoke the OpenAI-compatible VL server (unified-7b: HOST:8000; dual-2b: :8001).
#
#   EMET_LLM_HOST=caliban ./scripts/smoke_caliban_vl.sh
#   EMET_VL_ENDPOINT=http://HOST:8001/v1 ./scripts/smoke_caliban_vl.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${EMET_LLM_HOST:-${EMET_CALIBAN_HOST:-}}"
if [[ -n "${EMET_VL_ENDPOINT:-}" ]]; then
  BASE="${EMET_VL_ENDPOINT}"
elif [[ -n "$HOST" ]]; then
  BASE="http://${HOST}:8000/v1"
else
  echo "ERROR: set EMET_LLM_HOST or EMET_VL_ENDPOINT (example: EMET_LLM_HOST=caliban)" >&2
  exit 1
fi
BASE="${BASE%/}"
# Allow openai@ prefix
BASE="${BASE#openai@}"

cd "$ROOT"
uv run emet llm health --vl-only --vl "$BASE"
uv run emet llm smoke --vl-only --vl "$BASE"

echo "[llm-vl] OK"
