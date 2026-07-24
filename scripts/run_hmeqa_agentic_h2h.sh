#!/usr/bin/env bash
# Habitat HM-EQA holdout: classic Dynagraph vs agentic-verify Dynagraph.
# Comparable MCQ slice (same IDs as prior paper/branch numbers).
#
# Usage:
#   nohup ./scripts/run_hmeqa_agentic_h2h.sh [OUT_DIR] >> ~/runs/emet/hmeqa_agentic_h2h_nohup.log 2>&1 &
#
# Arms:
#   classic  — EMET_EQA_AGENTIC_VERIFY=0 (Habitat planning loop; prior holdout numbers)
#   agentic  — EMET_EQA_AGENTIC_VERIFY=1 EMET_EQA_AGENTIC_ROUTER=0 (nav→SigLIP verify→answer)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/hmeqa_agentic_h2h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/figures" "$OUT/bundles"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
# Small first: holdout-4 gate; override HOLDOUT_IDS for full-8.
HOLDOUT_IDS="${HOLDOUT_IDS:-15,68,105,17}"
TIMEOUT="${TIMEOUT:-7200}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

snapshot_bundle() {
  # Copy map artifacts out of the shared Habitat cache so H2H arms do not overwrite.
  local arm="$1"
  local qid="$2"
  local ep="$3"
  local dst="$OUT/bundles/${arm}_q${qid}"
  mkdir -p "$dst"
  local src=""
  if [[ -s "$ep" ]]; then
    src="$(uv run python -c "
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
row=json.loads(p.read_text().splitlines()[0])
print(row.get('debug_bundle_dir') or '')
" "$ep" 2>/dev/null || true)"
  fi
  if [[ -z "$src" || ! -d "$src" ]]; then
    src="$HOME/.cache/habitat_eqa/episodes/h2h_${arm}_q$(printf '%04d' "$qid")/q$(printf '%04d' "$qid")_dynagraph"
  fi
  if [[ ! -d "$src" ]]; then
    log "WARN: no bundle to snapshot for $arm q$qid"
    return 0
  fi
  for f in topdown_map.png topdown_map_overlay.png topdown_gt_navmesh.png \
           explored_2d.npy obstacles_2d.npy grid_meta.json trajectory.jsonl \
           spawn_record.json metrics.json diagnostics_manifest.json floor_metrics.json \
           topdown_exploration.mp4; do
    [[ -e "$src/$f" ]] && cp -a "$src/$f" "$dst/"
  done
  [[ -d "$src/maps" ]] && rm -rf "$dst/maps" && cp -a "$src/maps" "$dst/maps"
  log "snapshot $arm q$qid → $dst"
}

if [[ ! -x "$EMET_HABITAT" ]]; then
  log "Missing emet-habitat at $EMET_HABITAT"
  exit 1
fi

# Prefer flash-attn; allow SDPA only if explicitly requested (Habitat MCQ was OK on SDPA).
if ! uv run python -c "from emet.llms.attn_impl import flash_attn_2_available; raise SystemExit(0 if flash_attn_2_available() else 1)"; then
  if [[ "${EMET_ALLOW_SDPA_ATTN:-}" != "1" ]]; then
    log "Flash-Attn 2 not installed. Set EMET_ALLOW_SDPA_ATTN=1 to run on SDPA, or wait for flash-attn install."
    exit 2
  fi
  log "WARNING: running with EMET_ALLOW_SDPA_ATTN=1 (no flash-attn)"
fi

log "OUT=$OUT HEAD=$(git rev-parse --short HEAD) ids=$HOLDOUT_IDS"
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
./scripts/gpu_preflight.sh --kill-stale || true

run_arm() {
  local name="$1"
  shift
  local out="$OUT/${name}.jsonl"
  local elog="$OUT/${name}.log"
  : >"$out"
  : >"$elog"
  log "START arm=$name ($*)"
  IFS=',' read -r -a ids <<<"$HOLDOUT_IDS"
  for qid in "${ids[@]}"; do
    qid="$(echo "$qid" | tr -d '[:space:]')"
    [[ -n "$qid" ]] || continue
    NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
    ep="$OUT/${name}_q${qid}.jsonl"
    : >"$ep"
    tag="h2h_${name}_q$(printf '%04d' "$qid")"
    log "----- $name q$qid (bundle tag=$tag) -----"
    set +e
    # shellcheck disable=SC2086
    env "$@" timeout "$TIMEOUT" "$EMET_HABITAT" run-episode \
      --question-id "$qid" \
      --method dynagraph \
      --explore-when-uncovered off \
      --no-mcq-debias \
      --memory-summary \
      --debug-run-tag "$tag" \
      --output "$ep" \
      >>"$elog" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "FAIL $name q$qid exit=$rc"
    fi
    if [[ -s "$ep" ]]; then
      cat "$ep" >>"$out"
    fi
    snapshot_bundle "$name" "$qid" "$ep"
  done
  log "DONE arm=$name"
}

run_arm classic EMET_EQA_AGENTIC_VERIFY=0
run_arm agentic \
  EMET_EQA_AGENTIC_VERIFY=1 \
  EMET_EQA_AGENTIC_ROUTER=0 \
  EMET_EQA_TRACE=1

uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT" | tee -a "$OUT/orchestrator.log"

# Side-by-side coverage maps (uses OUT/bundles/* snapshotted above).
uv run python scripts/render_hmeqa_agentic_coverage_figure.py "$OUT" \
  --question-ids "${COVERAGE_QIDS:-15,104,68}" \
  --output "$OUT/figures/hmeqa_agentic_coverage.png" \
  | tee -a "$OUT/orchestrator.log" || log "WARN: coverage figure failed"

# Also copy into paper/figs when present.
if [[ -d "$ROOT/paper/figs" && -f "$OUT/figures/hmeqa_agentic_coverage.png" ]]; then
  cp -a "$OUT/figures/hmeqa_agentic_coverage.png" "$ROOT/paper/figs/hmeqa_agentic_coverage.png" || true
  cp -a "$OUT/figures/hmeqa_agentic_h2h.png" "$ROOT/paper/figs/hmeqa_agentic_h2h.png" || true
fi

echo DONE > "$OUT/DONE"
log "All done → $OUT"
