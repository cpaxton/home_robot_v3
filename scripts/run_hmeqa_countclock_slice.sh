#!/usr/bin/env bash
# HM-EQA count/clock slice runner (close-look crop + graph count hint on weak classes).
#
# Runs the 15 count/clock paper-113 questions through emet-habitat (dynagraph).
# Use after count-hint v2 (PR #124); baseline dynagraph pre-close-look (95/113 partial):
# count 23%, clock 20%. Aug 22 pre-fix slice: 6/15 (40%).
#
# Env:
#   METHODS        space-separated methods (default "dynagraph")
#   QUESTION_IDS   comma-separated ids (default: 15 count/clock ids)
#   TIMEOUT        per-batch wall timeout seconds (default 7200)
#   NEED_MIB       VRAM gate (default 12000); the script refuses to start with less
#   SIGLIP_EVIDENCE "1" -> EMET_EQA_AGENTIC_SIGLIP_EVIDENCE=1 (default 1)
#   CLOSE_LOOK     "0" -> EMET_EQA_AGENTIC_CLOSE_LOOK=0, pre-close-look baseline (default 1)
#   MULTIVIEW      "1" -> EMET_EQA_AGENTIC_CLOSE_LOOK_MULTIVIEW=1 (default 0)
#   RESUME         "0" -> overwrite the arm jsonl instead of resuming (default 1)
#
# Usage (prefer emet jobs — serializes on ~/runs/emet/gpu.lock):
#   uv run emet jobs run --name countclock-postfix --need-mib 12000 -- \
    #     env EMET_ALLOW_SDPA_ATTN=1 RESUME=0 ./scripts/run_hmeqa_countclock_slice.sh
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

# --- VRAM gate: fail loudly before loading anything when the GPU is contended. ---
log "vram gate: need >= ${NEED_MIB} MiB free"
PY_GATE="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY_GATE}" ]]; then
    PY_GATE="$(command -v python3)"
fi
VRAM_OK="$("${PY_GATE}" -c "
from emet.utils.gpu_preflight import check_gpu_memory
ok, msg = check_gpu_memory(int('$NEED_MIB'))
print(msg)
raise SystemExit(0 if ok else 3)
" 2>&1)" || {
    log "FATAL: $VRAM_OK"
    log "refusing to start count/clock slice; rerun via emet jobs once the GPU clears"
    exit 3
}
log "$VRAM_OK"

# CUDA VL / EQA loads need either flash-attn or the SDPA fallback. This box has no
# flash-attn wheel; always allow SDPA so the run is not a silent 0/15 failure.
export EMET_ALLOW_SDPA_ATTN=1
# A/B toggles are explicit: pin every arm to the requested feature set so the
# pre-close-look baseline is reproducible and META records exactly what ran.
export EMET_EQA_AGENTIC_SIGLIP_EVIDENCE="${SIGLIP_EVIDENCE:-1}"
export EMET_EQA_AGENTIC_CLOSE_LOOK="${CLOSE_LOOK:-1}"
if [[ "${MULTIVIEW:-0}" == "1" ]]; then
    export EMET_EQA_AGENTIC_CLOSE_LOOK_MULTIVIEW=1
else
    export EMET_EQA_AGENTIC_CLOSE_LOOK_MULTIVIEW=0
fi
RESUME="${RESUME:-1}"

log "run_id=$RUN_ID ids=$QUESTION_IDS methods=$METHODS siglip_evidence=${SIGLIP_EVIDENCE:-1} close_look=${CLOSE_LOOK:-1} multiview=${MULTIVIEW:-0} resume=$RESUME"
log "out=$OUT_DIR"
{
    echo "run_id=$RUN_ID"
    echo "question_ids=$QUESTION_IDS"
    echo "methods=$METHODS"
    echo "siglip_evidence=${SIGLIP_EVIDENCE:-1}"
    echo "close_look=${CLOSE_LOOK:-1}"
    echo "multiview=${MULTIVIEW:-0}"
    echo "resume=$RESUME"
    echo "need_mib=$NEED_MIB"
    echo "git=$(git -C "$ROOT" rev-parse --short HEAD)"
} | tee "$OUT_DIR/META.txt"

for method in $METHODS; do
    tag="countclock_${RUN_ID}_${method}"
    jsonl="$HOME/.cache/habitat_eqa/results/${tag}_${FAMILY}.jsonl"
    logf="$OUT_DIR/${method}.log"
    prog="$OUT_DIR/${method}.progress"
    log "=== method=$method tag=$tag resume=$RESUME ==="
    resume_flag=()
    if [[ "$RESUME" == "1" ]]; then
        resume_flag=(--resume)
    fi
    timeout "$TIMEOUT" "$HAB" run-batch \
        --method "$method" \
        --question-ids "$QUESTION_IDS" \
        --max-planning-steps 20 \
        --max-movement-step 10 \
        --eqa-vl-family "$FAMILY" \
        --eqa-hf-model-id "$HF_ID" \
        --device cuda \
        --no-hm3d-semantics \
        --frontier-nodes \
        --frontier-keyword-weight 2 \
        --output "$jsonl" \
        "${resume_flag[@]}" \
        2>&1 | tee "$logf"
    echo "$jsonl" >"$OUT_DIR/${method}_jsonl.path"
    # Recoverable progress: record scored/remaining qids so a later relaunch
    # (--resume) picks up where the previous arm stopped instead of restarting.
    python3 - "$jsonl" "$prog" <<'PY'
import json, sys
from pathlib import Path
p, prog = Path(sys.argv[1]), Path(sys.argv[2])
if not p.exists():
    print(f"{prog}: missing {p}", file=sys.stderr)
    raise SystemExit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
done = [r for r in rows if r.get("correct") is not None]
ids = sorted(int(r["question_id"]) for r in done)
prog.write_text("\n".join(f"done={i}" for i in ids) + ("\n" if ids else ""), encoding="utf-8")
print(f"{prog}: scored {len(ids)} qids -> {ids}", flush=True)
PY
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
