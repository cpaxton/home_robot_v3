#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Run the routine TAMP agent-tools smoke only when an OVMM find OUT is green
# (FindObj or FindRec ≥ 1). Used to chain TAMP after agentic find without polling
# from an agent turn.
#
#   AGENTIC_OUT=~/runs/emet/ovmm_find_phase/rby1_smoke_… \
    #     ./scripts/run_tamp_gate_if_ovmm_green.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AGENTIC_OUT="${AGENTIC_OUT:?set AGENTIC_OUT to the agentic find OUT dir}"

if [[ ! -f "$AGENTIC_OUT/summary.json" ]]; then
    echo "SKIP tamp: missing $AGENTIC_OUT/summary.json"
    exit 0
fi

green="$(
    uv run python - "$AGENTIC_OUT/summary.json" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    data = json.load(f)
s = data.get("summary") or data
obj = int(s.get("find_object_success") or 0)
rec = int(s.get("find_recep_success") or 0)
print("1" if (obj >= 1 or rec >= 1) else "0")
PY
)"

if [[ "$green" != "1" ]]; then
    echo "SKIP tamp: FindObj/FindRec not green in $AGENTIC_OUT"
    exit 0
fi

echo "OVMM find green in $AGENTIC_OUT; running TAMP tools smoke"
exec "$ROOT/scripts/run_tamp_agent_tools_gate.sh"
