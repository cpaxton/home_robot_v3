#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Overnight Habitat-OVMM find-phase failure-case sweep.
#
# Runs each episode in the generated config as its own emet-habitat call (a crash
# in one episode does not abort the sweep), records per-episode JSON + progress,
# and writes a summary.json with pass/fail + env-growth metrics so runaway-map /
# empty-result failure cases are easy to grep.
#
# Usage:
#   uv run emet jobs run --name habitat-ovmm-overnight --need-mib 8000 --gpu-exclusive -- \
    #     ./scripts/run_habitat_ovmm_overnight.sh
#
# Env:
#   OUT_DIR         base output dir (default ~/runs/emet/ovmm_habitat/overnight_<stamp>)
#   EPISODES        episodes YAML (default configs/ovmm/habitat_ovmm_overnight.yaml)
#   BACKEND         emet memory backend (default dynagraph)
#   ONLY_EPISODES   optional comma-separated episode id subset (resume / debug)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gpu_preflight.sh
source "${ROOT}/scripts/gpu_preflight.sh"
emet_export_pytorch_alloc

STAMP="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_habitat/overnight_${STAMP}}"
EPISODES="${EPISODES:-$ROOT/configs/ovmm/habitat_ovmm_overnight.yaml}"
BACKEND="${BACKEND:-dynagraph}"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUT"

if [[ ! -x "$EMET_HABITAT" ]]; then
    echo "FATAL: emet-habitat not found at $EMET_HABITAT (run ./scripts/install_habitat.sh)" >&2
    exit 1
fi

FAILED=0
DONE=0

heartbeat() {
    local phase="$1"
    local current="$2"
    uv run python -c "
from emet.eval.harness import update_eval_progress
update_eval_progress(r'''$OUT''', units_done=int('$DONE'), units_total=int('$TOTAL'),
                     phase='$phase', current_id='$current', units_failed=int('$FAILED'))
    " 2>/dev/null || true
    if [[ -n "${EMET_JOB_ID:-}" ]]; then
        uv run emet jobs update "$EMET_JOB_ID" --status running --units-done "$DONE" \
            --units-total "$TOTAL" --phase "$phase" --current-id "$current" --out-dir "$OUT" \
            >/dev/null 2>&1 || true
    fi
}

record() {
    local key="$1"
    local status="$2"
    uv run python - "$key" "$status" <<'PY'
import json, os, sys
key, status = sys.argv[1], sys.argv[2]
out = os.environ.get("OUT")
if not out:
    sys.exit(0)
p = os.path.join(out, "summary.json")
s = {}
if os.path.exists(p):
    with open(p, encoding="utf-8") as fh:
        s = json.load(fh)
s[key] = status
with open(p, "w", encoding="utf-8") as fh:
    json.dump(s, fh, indent=2)
PY
}

# Resolve episode ids (respect ONLY_EPISODES subset for resume).
IDS_FILE="$OUT/episode_ids.txt"
"$ROOT/.venv-habitat/bin/python" - "$EPISODES" "${ONLY_EPISODES:-}" "$IDS_FILE" <<'PY'
import json, sys, yaml
yaml_path, subset, ids_file = sys.argv[1], (sys.argv[2] or "").strip(), sys.argv[3]
rows = (yaml.safe_load(open(yaml_path, encoding="utf-8")) or {}).get("episodes", [])
ids = [str(r["id"]) for r in rows]
if subset:
    want = {x.strip() for x in subset.split(",") if x.strip()}
    ids = [i for i in ids if i in want]
with open(ids_file, "w", encoding="utf-8") as fh:
    fh.write("\n".join(ids))
PY
mapfile -t IDS < "$IDS_FILE"
TOTAL="${#IDS[@]}"
echo "=== habitat-ovmm overnight ${STAMP}: ${TOTAL} episodes (backend=${BACKEND}) OUT=${OUT} ==="

for ep in "${IDS[@]}"; do
    heartbeat "habitat-ovmm" "$ep"
    echo "=== [$ep] ==="
    if "$EMET_HABITAT" run-ovmm-find-episode \
        --episodes "$EPISODES" --episode-id "$ep" --backend "$BACKEND" \
        --output "$OUT/${ep}_${BACKEND}.json" > "$OUT/${ep}.log" 2>&1; then
        record "$ep" "pass"
    else
        echo "[$ep] FAILED (rc=$?)"
        record "$ep" "fail"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
done

heartbeat "done" "summary"

# Aggregate env-growth + result columns from the per-episode JSON.
"$ROOT/.venv-habitat/bin/python" - "$OUT" "$BACKEND" <<'PY'
import json, os, sys
from pathlib import Path
out, backend = Path(sys.argv[1]), sys.argv[2]
rows = []
for p in sorted(out.glob(f"*_{backend}.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    rows.append(
        {
            "episode": d.get("episode_id"),
            "scene": d.get("scene"),
            "find_obj": d.get("find_object_success"),
            "find_rec": d.get("find_recep_success"),
            "partial": d.get("find_partial_success"),
            "obj_src": d.get("obj_localize_source"),
            "rec_src": d.get("recep_localize_source"),
            "wall_s": round(float(d.get("episode_wall_s") or 0), 1),
            "explored_m2": round(float(d.get("n_voxel_explored_area_m2") or 0), 1),
            "graph_nodes": d.get("n_graph_nodes"),
            "obs": d.get("n_obs"),
            "mapping_nav": d.get("mapping_n_nav"),
            "obj_nav": d.get("obj_n_nav"),
            "rec_nav": d.get("recep_n_nav"),
        }
    )
(Path(sys.argv[1]) / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"aggregate: {len(rows)} rows -> {sys.argv[1]}/aggregate.json")
PY

echo "=== habitat-ovmm overnight ${STAMP} done: failed=${FAILED}/${TOTAL} (OUT=${OUT}) ==="
cat "$OUT/summary.json" 2>/dev/null || true
exit 0
