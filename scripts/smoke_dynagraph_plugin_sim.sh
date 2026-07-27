#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Smoke: Dynagraph memory plug-in on Robocasa + MolmoSpaces (ithor).
# Verifies --memory-backend dynagraph saves graph.json without an OV sidecar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynagraph_plugin_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "=== OUT=$OUT ===" | tee "$OUT/summary.txt"
echo "commit=$(git rev-parse --short HEAD)" | tee -a "$OUT/summary.txt"

assert_dynagraph_checkpoint() {
  local log="$1"
  local tag="$2"
  python3 - "$log" "$tag" <<'PY' | tee -a "$OUT/summary.txt"
import re, sys, time
from pathlib import Path

log_path = Path(sys.argv[1])
tag = sys.argv[2]
text = log_path.read_text(errors="replace")
print(f"--- {tag} checkpoint check ---")
# print_memory_saved_help / lifelong usually mention a path
paths = []
for m in re.finditer(r"(/[\w./-]+(?:logs|memory|saved)[\w./-]*)", text):
    p = Path(m.group(1))
    if p.is_dir():
        paths.append(p)
    elif p.parent.is_dir():
        paths.append(p.parent)
# also recent graph.json under repo logs
root = Path(".").resolve()
for g in (root / "logs").rglob("graph.json"):
    if time.time() - g.stat().st_mtime < 7200:
        paths.append(g.parent)
paths = list(dict.fromkeys(paths))
ok = False
for d in paths:
    g = d / "graph.json"
    ov = d / "open_vocab_scene_graph" / "scene_graph.json"
    if g.is_file():
        print(f"  {d}: graph=yes ov={ov.is_file()}")
        if not ov.is_file():
            print(f"PASS {tag}: dynagraph checkpoint without OV sidecar → {d}")
            ok = True
            break
if not paths:
    print(f"WARN {tag}: no checkpoint dirs found in log — inspect {log_path}")
elif not ok:
    print(f"FAIL {tag}: found graph.json but OV sidecar still present")
    sys.exit(2)
PY
}

ROBO="$OUT/robocasa"
mkdir -p "$ROBO"
echo "=== Robocasa dynagraph rotate_in_place ===" | tee -a "$OUT/summary.txt"
uv run emet run agent \
  --robot stretch \
  --start-sim --scene robocasa --headless \
  --no-llm --no-discord \
  --memory-backend dynagraph \
  --sim-seed 0 \
  --port-offset 70 \
  -c rotate_in_place -c Q \
  >"$ROBO/agent.log" 2>&1 || {
  echo "robocasa agent FAILED" | tee -a "$OUT/summary.txt"
  tail -100 "$ROBO/agent.log" | tee -a "$OUT/summary.txt"
  exit 1
}
tail -30 "$ROBO/agent.log" | tee -a "$OUT/summary.txt"
assert_dynagraph_checkpoint "$ROBO/agent.log" "robocasa"

MOLMO="$OUT/molmo"
mkdir -p "$MOLMO"
echo "=== Molmo ithor dynagraph rotate_in_place ===" | tee -a "$OUT/summary.txt"
uv run emet run agent \
  --start-sim --scene ithor --split train --index 0 --headless \
  --no-llm --no-discord \
  --memory-backend dynagraph \
  --port-offset 80 \
  -c rotate_in_place -c Q \
  >"$MOLMO/agent.log" 2>&1 || {
  echo "molmo agent FAILED" | tee -a "$OUT/summary.txt"
  tail -100 "$MOLMO/agent.log" | tee -a "$OUT/summary.txt"
  exit 1
}
tail -40 "$MOLMO/agent.log" | tee -a "$OUT/summary.txt"
assert_dynagraph_checkpoint "$MOLMO/agent.log" "molmo"

echo DONE | tee -a "$OUT/summary.txt"
