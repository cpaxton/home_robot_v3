# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke the OpenAI-compatible VL server (caliban:8001 or local --vl serve).

  ./scripts/smoke_caliban_vl.sh
  EMET_VL_ENDPOINT=http://127.0.0.1:8001/v1 ./scripts/smoke_caliban_vl.sh
"""

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${EMET_VL_ENDPOINT:-http://caliban:8001/v1}"
BASE="${BASE%/}"
# Allow openai@ prefix
BASE="${BASE#openai@}"

cd "$ROOT"
uv run emet llm health --vl-only --vl "$BASE"
uv run emet llm smoke --vl-only --vl "$BASE"

echo "[caliban-vl] OK"
