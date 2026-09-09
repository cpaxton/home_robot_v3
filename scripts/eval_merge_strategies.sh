#!/usr/bin/env bash
# Compare scene-graph ingest/merge strategies on a fixed HM-EQA qid slice.
#
#   A  lazy_graph       LazyGraphController: Qwen commits only on nav-arrival close looks
#   B  dynagraph        instance nodes + deep merge policy (SigLIP appearance, label
#                       synonyms, growth cap) — configs/benchmarks/fusion_strategy_b.yaml
#   C  dynagraph        instance nodes kept OUT of the graph — fusion_strategy_c.yaml
#   D  dynagraph        current defaults (control; usually reused from prior runs)
#
# Phase 2 runs local int4 (fast, ~3 min/qid). Set EMET_VL_ENDPOINT to the caliban
# fp16 server to run the whole comparison on fp16 instead.
#
# Usage (through emet jobs, GPU-exclusive):
#   uv run emet jobs run --name merge-strategies --need-mib 12000 --gpu-exclusive -- \
    #     ./scripts/eval_merge_strategies.sh
#
# Env:
#   QIDS       comma-separated qids (default: tight slice of regressed + count/clock)
#   STRATEGIES whitespace list of a b c d (d default-control)
#   OUT_DIR    output dir (default ~/runs/emet/merge_strategies/<run_id>)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

QIDS="${QIDS:-8,14,40,43,52,57,68,76,28,47,84,93}"
STRATEGIES="${STRATEGIES:-a b c}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/merge_strategies/${RUN_ID}}"
mkdir -p "$OUT_DIR"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
# Local int4 Qwen3-VL loads need the PyTorch SDPA path (flash-attn wheel not
# installed for this torch/CUDA); the paper runner passes this on the job env.
export EMET_ALLOW_SDPA_ATTN=1

echo "=== merge-strategies ${RUN_ID}: qids=${QIDS} strategies=${STRATEGIES} ===" >&2

run_one() { # $1=fusion_cfg ('' = none) ; rest = command
    local f="$1"
    shift
    if [ -n "$f" ]; then
        EMET_GRAPH_FUSION_CONFIG="$f" "$@"
    else
        "$@"
    fi
}

run_strategy() {
    local name="$1" method="$2" fusion_cfg="${3:-}"
    local out_jsonl="$OUT_DIR/${name}.jsonl"
    echo "[$(date -Is)] === strategy=${name} method=${method} fusion=${fusion_cfg:-default} → ${out_jsonl} ===" >&2
    run_one "$fusion_cfg" "$HAB" run-batch \
        --method "$method" \
        --question-ids "$QIDS" \
        --max-planning-steps 20 \
        --max-movement-step 10 \
        --eqa-vl-family qwen3_vl \
        --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
        --device cuda \
        --frontier-nodes \
        --frontier-keyword-weight 2 \
        --no-hm3d-semantics \
        --output "$out_jsonl"
}

for s in $STRATEGIES; do
    case "$s" in
        a) run_strategy "a_lazy_graph" "lazy_graph" "" ;;
        b) run_strategy "b_deep_merge" "dynagraph" "$ROOT/configs/benchmarks/fusion_strategy_b.yaml" ;;
        c) run_strategy "c_no_instances" "dynagraph" "$ROOT/configs/benchmarks/fusion_strategy_c.yaml" ;;
        d) run_strategy "d_current" "dynagraph" "" ;;
        *) echo "unknown strategy '$s'" >&2; exit 2 ;;
    esac
done

echo "[$(date -Is)] strategies done; summarizing…" >&2
uv run python - <<PY
import json, glob, os, statistics

out = os.environ.get("OUT_DIR") or ""
qids = [int(x) for x in os.environ.get("QIDS", "").split(",") if x.strip()]
paths = sorted(glob.glob(os.path.join(out, "*.jsonl")))
rows = []
for p in paths:
    name = os.path.basename(p).replace(".jsonl", "")
    by_q = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "question_id" not in d or "correct" not in d:
            continue
        gh = d.get("graph_health", {}) or {}
        by_q[d["question_id"]] = {
            "correct": bool(d["correct"]),
            "singleton": gh.get("singleton_frac"),
            "n_object": gh.get("n_object"),
            "n_obs": gh.get("n_obs"),
        }
    rows.append((name, by_q))

print(f"{'strategy':<16} {'corr':>6} {'mean_singleton':>14} {'mean_n_obj':>11} {'mean_n_obs':>10}")
for name, by_q in rows:
    n = len(by_q)
    corr = sum(1 for v in by_q.values() if v["correct"])
    sing = [v["singleton"] for v in by_q.values() if v["singleton"] is not None]
    nobj = [v["n_object"] for v in by_q.values() if v["n_object"] is not None]
    nobs = [v["n_obs"] for v in by_q.values() if v["n_obs"] is not None]
    print(
        f"{name:<16} {corr:>3}/{n:<3} {statistics.mean(sing) if sing else float('nan'):>14.3f} "
        f"{statistics.mean(nobj) if nobj else float('nan'):>11.1f} "
        f"{statistics.mean(nobs) if nobs else float('nan'):>10.1f}"
    )
for name, by_q in rows:
    bits = " ".join(f"{q}:{'T' if by_q.get(q, {}).get('correct') else 'F'}" for q in qids if q in by_q)
    print(f"{name}: {bits}")
print("OUT=" + out)
PY
