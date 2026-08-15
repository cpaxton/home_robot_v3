#!/usr/bin/env bash
# HM-EQA count/clock slice runner (validates close-look crop on the weak classes).
#
# Runs the 15 count/clock paper-113 questions through the v3 emet-habitat harness
# (this checkout has the dense_siglip_argmax_crop -> VLM-assess 2nd-image path).
# Baseline for comparison: dynagraph pre-close-look (95/113 partial): count 23%,
# clock 20%.
#
# Env:
#   METHODS        space-separated methods (default "dynagraph")
#   QUESTION_IDS   comma-separated ids (default: 15 count/clock ids)
#   TIMEOUT        per-batch wall timeout seconds (default 7200)
#   NEED_MIB       VRAM gate (default 12000)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_countclock/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-7200}"
NEED_MIB="${NEED_MIB:-12000}"
METHODS="${METHODS:-dynagraph}"
QUESTION_IDS="${QUESTION_IDS:-12,21,28,32,33,43,47,48,51,60,78,84,86,88,93}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"

log() { echo "[$(date -Is)] $*" | tee -a "$OUT_DIR/run.log"; }

log "run_id=$RUN_ID ids=$QUESTION_IDS methods=$METHODS"
log "out=$OUT_DIR"
{
    echo "run_id=$RUN_ID"
    echo "question_ids=$QUESTION_IDS"
    echo "methods=$METHODS"
    echo "git=$(git -C "$ROOT" rev-parse --short HEAD)"
} | tee "$OUT_DIR/META.txt"

for method in $METHODS; do
    tag="countclock_${RUN_ID}_${method}"
    jsonl="$HOME/.cache/habitat_eqa/results/${tag}_${FAMILY}.jsonl"
    logf="$OUT_DIR/${method}.log"
    log "=== method=$method tag=$tag ==="
    timeout "$TIMEOUT" "$HAB" run-batch \
        --method "$method" \
        --question-ids "$QUESTION_IDS" \
        --max-planning-steps 20 \
        --max-movement-step 10 \
        --eqa-vl-family "$FAMILY" \
        --eqa-hf-model-id "$HF_ID" \
        --device cuda \
        --frontier-nodes \
        --frontier-keyword-weight 2 \
        --output "$jsonl" \
        2>&1 | tee "$logf"
    echo "$jsonl" >"$OUT_DIR/${method}_jsonl.path"
done

log "=== summary ==="
python3 - "$OUT_DIR" "$METHODS" "$FAMILY" "$RUN_ID" <<'PY' | tee "$OUT_DIR/SUMMARY.txt"
import json, os, sys
from pathlib import Path

out_dir, methods, family, run_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
root = Path.home() / ".cache/habitat_eqa/results"
for method in methods.split():
    tag = f"countclock_{run_id}_{method}"
    p = root / f"{tag}_{family}.jsonl"
    if not p.exists():
        print(f"{method}: missing {p}")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    done = [r for r in rows if r.get("correct") is not None]
    n, ok = len(done), sum(1 for r in done if r.get("correct"))
    print(f"{method}: {ok}/{n} correct ({100*ok/max(n,1):.0f}%) on count/clock ids")
    for r in sorted(done, key=lambda x: x["question_id"]):
        q = (r.get("question") or r.get("question_text") or "")[:60]
        print(f"  q{r['question_id']} {'OK ' if r.get('correct') else 'ERR'} {q}")
PY
