#!/usr/bin/env bash
# GraphEQA-parity baseline: our stack on the ACTUAL GraphEQA paper HM-EQA episodes,
# GT semantics on vs off (4 arms). Lets us isolate the GT-semantics perception
# effect from the Dynagraph memory delta on identical episodes.
#
# Episodes: the 114 Explore-EQA rows whose scene is in the GraphEQA enrich set
# (59 HM3D train scenes, all with .semantic.glb) — see
# emet.habitat.hmeqa_enrich_labels.grapheqa_baseline_question_ids.
#
# Arms (method x semantics):
#   dynagraph    + --use-hm3d-semantics   (GT perception, Dynagraph memory)
#   static_graph + --use-hm3d-semantics   (GT perception, GraphEQA-inspired baseline)
#   dynagraph    + --no-hm3d-semantics    (no GT, Dynagraph memory)
#   static_graph + --no-hm3d-semantics    (no GT, GraphEQA-inspired baseline)
#
# Usage (prefer emet jobs):
#   uv run emet jobs run --name hmeqa-grapheqa-baseline --need-mib 12000 -- \
#     env EMET_ALLOW_SDPA_ATTN=1 ./scripts/run_hmeqa_grapheqa_baseline.sh
#
# Env:
#   METHODS    space-separated methods (default "dynagraph static_graph")
#   SEMANTICS  space-separated semantics arms (default "on off"; "on" = GT)
#   TIMEOUT    per-arm wall timeout seconds (default 28800)
#   NEED_MIB   VRAM gate (default 12000)
#   NEED_FREE_GB  disk floor under ~/.cache/habitat_eqa (default 50)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/hmeqa_grapheqa/${RUN_ID}}"
mkdir -p "$OUT_DIR"
TIMEOUT="${TIMEOUT:-28800}"
NEED_MIB="${NEED_MIB:-12000}"
NEED_FREE_GB="${NEED_FREE_GB:-50}"
METHODS="${METHODS:-dynagraph static_graph}"
SEMANTICS="${SEMANTICS:-on off}"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"

# --- disk preflight (same guard as paper113 runner) -------------------------
_free_kb="$(df -Pk "$HOME/.cache/habitat_eqa" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -z "${_free_kb:-}" ]]; then
  echo "[$(date -Is)] WARNING: could not read free space for ~/.cache/habitat_eqa; continuing"
else
  FREE_GB=$((_free_kb / 1024 / 1024))
  if (( FREE_GB < NEED_FREE_GB )); then
    echo "[$(date -Is)] ABORT: free disk under ~/.cache/habitat_eqa is ${FREE_GB} GB (< ${NEED_FREE_GB} GB)." >&2
    echo "  Free space first, e.g.:" >&2
    echo "    uv run python scripts/clean_episode_bundles.py --keep 2 --apply" >&2
    exit 4
  fi
  echo "[$(date -Is)] disk preflight OK: ${FREE_GB} GB free under ~/.cache/habitat_eqa (need >= ${NEED_FREE_GB} GB)"
fi

# --- the ACTUAL GraphEQA paper episodes (114 rows) --------------------------
IDS="$(uv run python -c 'from emet.habitat.hmeqa_enrich_labels import grapheqa_baseline_question_ids as f; print(",".join(map(str, f())))')"
N="$(awk -F, '{print NF}' <<<"$IDS")"
echo "$IDS" >"$OUT_DIR/IDS.txt"
{
  echo "run_id=$RUN_ID"
  echo "methods=$METHODS"
  echo "semantics=$SEMANTICS"
  echo "n_episodes=$N"
  git rev-parse --short HEAD
  git rev-parse HEAD
} | tee "$OUT_DIR/META.txt"

log() { echo "[$(date -Is)] $*"; }

run_arm() {
  local method="$1" sem="$2"
  local flag tag
  if [[ "$sem" == "on" ]]; then
    flag="--use-hm3d-semantics"; tag="grapheqa_${RUN_ID}_${method}_gt"
  else
    flag="--no-hm3d-semantics"; tag="grapheqa_${RUN_ID}_${method}_nogt"
  fi
  local jsonl="$HOME/.cache/habitat_eqa/results/subset_${tag}_${FAMILY}.jsonl"
  local logf="$OUT_DIR/${method}_${sem}.log"
  log "=== grapheqa method=$method semantics=$sem tag=$tag n=$N ==="
  NEED_MIB="$NEED_MIB" "${ROOT}/scripts/gpu_preflight.sh" --wait
  emet_kill_stale_eval_processes
  timeout "$TIMEOUT" "$HAB" run-batch \
    --method "$method" \
    --question-ids "$IDS" \
    --max-planning-steps 20 \
    --max-movement-step 10 \
    --eqa-vl-family "$FAMILY" \
    --eqa-hf-model-id "$HF_ID" \
    --device cuda \
    --frontier-nodes \
    --frontier-keyword-weight 2 \
    --resume \
    "$flag" \
    --output "$jsonl" \
    2>&1 | tee "$logf"
  echo "$jsonl" >"$OUT_DIR/${method}_${sem}_jsonl.path"
}

for method in $METHODS; do
  for sem in $SEMANTICS; do
    run_arm "$method" "$sem"
  done
done

uv run python - <<PY | tee "$OUT_DIR/SUMMARY.txt"
import json
from pathlib import Path
from emet.habitat.metrics import episode_run_completed

run_id = "${RUN_ID}"
family = "${FAMILY}"
root = Path.home() / ".cache/habitat_eqa/results"
for method in "${METHODS}".split():
    for sem in "${SEMANTICS}".split():
        tag = f"grapheqa_{run_id}_{method}_{'gt' if sem=='on' else 'nogt'}"
        p = root / f"subset_{tag}_{family}.jsonl"
        if not p.exists():
            print(f"{method}/{sem}: missing {p}")
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
        print(f"{method}/{sem}: {ok}/{len(done)} correct ({p.name})")
PY

log "DONE grapheqa baseline (4 arms) → $OUT_DIR"
