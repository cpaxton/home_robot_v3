#!/usr/bin/env bash
# After the OVMM matrix frees the GPU, run a Dynagraph Phase-1 smoke WITH EQA
# (the question-answering eval — not just graph_health).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/dynagraph_eqa_smoke}"
MATRIX_PID="${2:-}"
mkdir -p "$OUT"

if [[ -n "$MATRIX_PID" ]]; then
    echo "[eqa-smoke] waiting for matrix pid $MATRIX_PID…"
    while kill -0 "$MATRIX_PID" 2>/dev/null; do sleep 30; done
fi

echo "[eqa-smoke] waiting for GPU…"
NEED_MIB=12000 ./scripts/gpu_preflight.sh --wait

# Cap per-question EQA wall time and kill silent hung children early.
export EMET_EQA_QUESTION_TIMEOUT_S="${EMET_EQA_QUESTION_TIMEOUT_S:-600}"
export EMET_DYNAMIC_EXPLORE_STALE_LOG_S="${EMET_DYNAMIC_EXPLORE_STALE_LOG_S:-300}"
export EMET_DYNAMIC_EXPLORE_STALE_KILL_S="${EMET_DYNAMIC_EXPLORE_STALE_KILL_S:-600}"

echo "[eqa-smoke] Robocasa dynagraph explore K=3 + EQA → $OUT"
env -u PYTHONPATH \
    EMET_EQA_QUESTION_TIMEOUT_S="$EMET_EQA_QUESTION_TIMEOUT_S" \
    EMET_DYNAMIC_EXPLORE_STALE_LOG_S="$EMET_DYNAMIC_EXPLORE_STALE_LOG_S" \
    EMET_DYNAMIC_EXPLORE_STALE_KILL_S="$EMET_DYNAMIC_EXPLORE_STALE_KILL_S" \
    uv run python scripts/eval_dynamic_exploration.py \
    --smoke \
    --backend dynagraph \
    --output-dir "$OUT" \
    --port-offset-base 220 \
    >"$OUT/eqa_smoke.log" 2>&1
echo "[eqa-smoke] exit=$?"
# Print summary
python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT")
for f in sorted(out.glob("*.json")):
    d = json.loads(f.read_text())
    s = d.get("summary") or {}
    m = d.get("metrics") or {}
    eqa = m.get("eqa") or {}
    gh = m.get("graph_health") or {}
    print(f.name)
    print("  nodes=", s.get("node_count"), "health=", gh.get("failure_class"), "top=", (gh.get("top_labels") or [])[:4])
    print("  eqa_accuracy=", s.get("eqa_accuracy") or eqa.get("accuracy"))
    for q in (eqa.get("questions") or [])[:4]:
        print("  Q:", (q.get("question") or "")[:60], "|", (q.get("answer") or "")[:80])
PY
