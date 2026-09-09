#!/usr/bin/env bash
# Sweep graph-object-fusion bounds_3d_iou_merge_min on a slice of HM-EQA qids.
#
# The 2026-09-02 full-113 fp16 run (harness git bc507fbd, use_instance_graph=true)
# regressed 16 location/state qids vs the published int4 49.6% baseline: the YoloE
# instance graph floods the shared graph with 240-300 object nodes (40-53% singletons)
# at an unchanged VLM budget. This sweep varies bounds_3d_iou_merge_min (0 disables
# the bounds-IoU merge path) on a representative subset of the regressed qids and
# reports per-threshold correctness + graph singleton/observation stats, so we can
# pick the knee without hand-tuning.
#
# Usage (through emet jobs, GPU-exclusive):
#   uv run emet jobs run --name fusion-iou-sweep --need-mib 12000 --gpu-exclusive -- \
    #     env EMET_VL_ENDPOINT=openai@http://192.168.1.55:8000/v1 EMET_ALLOW_SDPA_ATTN=1 \
    #     ./scripts/sweep_fusion_iou.sh
#
# Env:
#   QIDS       comma-separated regressed qids (default: representative subset)
#   SETTINGS   whitespace-separated bounds_3d_iou_merge_min values
#   OUT_DIR    output dir (default ~/runs/emet/fusion_iou_sweep/<run_id>)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

QIDS="${QIDS:-8,14,43,52,68,76}"
SETTINGS="${SETTINGS:-0.0 0.2 0.3 0.45 0.6}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HOME/runs/emet/fusion_iou_sweep/${RUN_ID}}"
mkdir -p "$OUT_DIR"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"

echo "=== fusion IoU sweep ${RUN_ID}: qids=${QIDS} settings=${SETTINGS} ===" >&2

gen_fusion_yaml() {
    local thr="$1" f="$OUT_DIR/fusion_iou_${thr}.yaml"
    cat > "$f" <<EOF
graph_object_fusion:
  enabled: true
  spatial_merge_xy_m: 0.42
  min_centroid_dist_m: 0.55
  bounds_3d_iou_min: 0.08
  bounds_3d_iou_merge_min: ${thr}
  embedding_min_cosine: 0.62
  embedding_blend_alpha: 0.35
  require_label_match: false
  require_label_match_for_instances: true
  max_candidates: 64
  match_xy_m: 0.55
  fallback_spatial_merge_xy_m: 0.45
  instance_min_confidence: 0.12
  instance_min_mask_points: 25
EOF
}

for thr in $SETTINGS; do
    yaml_f="$OUT_DIR/fusion_iou_${thr}.yaml"
    gen_fusion_yaml "$thr"
    out_jsonl="$OUT_DIR/subset_iou${thr}.jsonl"
    echo "[$(date -Is)] === bounds_3d_iou_merge_min=${thr} → ${out_jsonl} ===" >&2
    EMET_GRAPH_FUSION_CONFIG="$yaml_f" \
        "$HAB" run-batch \
        --method dynagraph \
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
done

echo "[$(date -Is)] sweep done; summarizing…" >&2
uv run python - <<PY
import json, glob, os, statistics

out = os.environ.get("OUT_DIR") or ""
qids = [int(x) for x in os.environ.get("QIDS", "").split(",") if x.strip()]
paths = sorted(glob.glob(os.path.join(out, "subset_iou*.jsonl")))
rows = []
for p in paths:
    thr = os.path.basename(p).replace("subset_iou", "").replace(".jsonl", "")
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
            "obs": d.get("observations"),
        }
    rows.append((thr, by_q))

print(f"{'iou':>5} {'corr':>5} {'mean_singleton':>14} {'mean_n_obj':>11} {'mean_n_obs':>10}  qid:correct")
for thr, by_q in rows:
    n = len(by_q)
    corr = sum(1 for v in by_q.values() if v["correct"])
    sing = [v["singleton"] for v in by_q.values() if v["singleton"] is not None]
    nobj = [v["n_object"] for v in by_q.values() if v["n_object"] is not None]
    nobs = [v["n_obs"] for v in by_q.values() if v["n_obs"] is not None]
    bits = " ".join(f"{q}:{'T' if by_q.get(q,{}).get('correct') else 'F'}" for q in qids if q in by_q)
    print(
        f"{thr:>5} {corr:>3}/{n:<2} {statistics.mean(sing) if sing else float('nan'):>14.3f} "
        f"{statistics.mean(nobj) if nobj else float('nan'):>11.1f} "
        f"{statistics.mean(nobs) if nobs else float('nan'):>10.1f}  {bits}"
    )
print("OUT=" + out)
PY
