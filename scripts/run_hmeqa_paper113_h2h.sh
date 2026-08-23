#!/usr/bin/env bash
# Historical emet HM-EQA q0–112 slice (113 questions) head-to-head.
# This is not the released GraphEQA semantic-filtered 114-row selection.
# Launch through ``emet jobs``; one method at a time (GPU preflight).
#
# Usage:
#   uv run emet jobs run --name hmeqa-legacy113 --need-mib 12000 -- \
    #     ./scripts/run_hmeqa_paper113_h2h.sh
#
# Env:
#   METHODS   space-separated methods (default: "static_graph dynagraph")
#   TIMEOUT   per-batch wall timeout seconds (default 86400)
#   NEED_MIB  VRAM gate (default 12000)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_paper113/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-86400}"
NEED_MIB="${NEED_MIB:-12000}"
NEED_FREE_GB="${NEED_FREE_GB:-50}"
METHODS="${METHODS:-static_graph dynagraph}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"

# --- disk preflight: a full-113 sweep writes GB of episode debug bundles per
# method (frame PNGs / MP4 / topdown maps) under ~/.cache/habitat_eqa/episodes.
# Refuse to start when free space is below NEED_FREE_GB so we don't crash
# mid-run on a full disk (2026-08-14: /tmp filled, dill import died).
_free_kb="$(df -Pk "$HOME/.cache/habitat_eqa" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -z "${_free_kb:-}" ]]; then
    echo "[$(date -Is)] WARNING: could not read free space for ~/.cache/habitat_eqa; continuing"
else
    FREE_GB=$((_free_kb / 1024 / 1024))
    if (( FREE_GB < NEED_FREE_GB )); then
        echo "[$(date -Is)] ABORT: free disk under ~/.cache/habitat_eqa is ${FREE_GB} GB (< ${NEED_FREE_GB} GB)." >&2
        echo "  A full-113 sweep writes GB of episode bundles per method." >&2
        echo "  Free space first, e.g.:" >&2
        echo "    uv run python scripts/clean_episode_bundles.py --keep 2 --apply" >&2
        echo "  or set NEED_FREE_GB to a smaller floor to override." >&2
        exit 4
    fi
    echo "[$(date -Is)] disk preflight OK: ${FREE_GB} GB free under ~/.cache/habitat_eqa (need >= ${NEED_FREE_GB} GB)"
fi

{
    echo "run_id=$RUN_ID"
    echo "methods=$METHODS"
    git rev-parse --short HEAD
    git rev-parse HEAD
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

run_method() {
    local method="$1"
    local tag="paper113_${RUN_ID}_${method}"
    local jsonl="$HOME/.cache/habitat_eqa/results/subset_${tag}_${FAMILY}.jsonl"
    local logf="$OUT_DIR/${method}.log"
    log "=== paper113 method=$method tag=$tag ==="
    NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
    emet_kill_stale_eval_processes
    timeout "$TIMEOUT" "$HAB" run-batch \
        --method "$method" \
        --paper-subset \
        --max-planning-steps 20 \
        --max-movement-step 10 \
        --eqa-vl-family "$FAMILY" \
        --eqa-hf-model-id "$HF_ID" \
        --device cuda \
        --frontier-nodes \
        --frontier-keyword-weight 2 \
        --resume \
        --no-hm3d-semantics \
        --output "$jsonl" \
        2>&1 | tee "$logf"
    echo "$jsonl" >"$OUT_DIR/${method}_jsonl.path"
}

for method in $METHODS; do
    run_method "$method"
done

uv run python - <<PY | tee "$OUT_DIR/SUMMARY.txt"
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

run_id = "${RUN_ID}"
family = "${FAMILY}"
root = Path.home() / ".cache/habitat_eqa/results"
for method in "${METHODS}".split():
    tag = f"paper113_{run_id}_{method}"
    p = root / f"subset_{tag}_{family}.jsonl"
    if not p.exists():
        print(f"{method}: missing {p}")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by_q = {}
    for r in rows:
        q = r["question_id"]
        prev = by_q.get(q)
        if prev is None or (episode_run_completed(r) and not episode_run_completed(prev)):
            by_q[q] = r
        elif episode_run_completed(r) == episode_run_completed(prev):
            by_q[q] = r
    done = [r for r in by_q.values() if episode_run_completed(r)]
    ok = sum(1 for r in done if r.get("correct"))
    print(f"{method}: {ok}/{len(done)} correct ({p.name})")
PY

log "DONE paper113 h2h → $OUT_DIR"
