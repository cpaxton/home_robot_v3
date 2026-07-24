#!/usr/bin/env bash
# Habitat HM-EQA holdout: classic Dynagraph vs agentic-verify Dynagraph.
# Comparable MCQ slice (same IDs as prior paper/branch numbers).
#
# Usage (prefer emet jobs so the run is managed and never blocks an agent turn):
#   uv run emet jobs run --name hmeqa-h2h --need-mib 12000 -- \
#     env EMET_ALLOW_SDPA_ATTN=1 HOLDOUT_IDS=… ./scripts/run_hmeqa_agentic_h2h.sh OUT_DIR
#   # or a dedicated terminal:
#   nohup ./scripts/run_hmeqa_agentic_h2h.sh [OUT_DIR] >> ~/runs/emet/hmeqa_agentic_h2h_nohup.log 2>&1 &
#
# NOTE: The fully-featured harness (resume, emet jobs progress/ETA, emet eval
# diagnose) lives on the eval branch (exp/agentic-hmeqa-*). This branch carries a
# crash-safety subset only — do not add emet-jobs-progress calls here unless that
# branch is merged in.
#
# Arms:
#   classic  — EMET_EQA_AGENTIC_VERIFY=0 (Habitat planning loop; prior holdout numbers)
#   agentic  — EMET_EQA_AGENTIC_VERIFY=1 EMET_EQA_AGENTIC_ROUTER=0 (nav→SigLIP verify→answer)
#
# Crash-safety env:
#   EGL_FAIL_ABORT=2      abort after this many consecutive Habitat EGL/CUDA-map
#                         failures (0 = never). Pattern: WindowlessContext /
#                         unable to find CUDA device. Empty nvidia-smi ≠ EGL OK.
#   NATIVE_CRASH_ABORT=1  stop the batch after a fatal native signal (SIGSEGV from
#                         libcuda, SIGABRT/SIGBUS/SIGILL/SIGFPE, SIGKILL/OOM).
#                         Writes native_crash_<arm>_q<ID>.log and leaves artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/hmeqa_agentic_h2h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/figures" "$OUT/bundles"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
# Small first: holdout-4 gate; override HOLDOUT_IDS for full-8.
HOLDOUT_IDS="${HOLDOUT_IDS:-15,68,105,17}"
TIMEOUT="${TIMEOUT:-7200}"
EGL_FAIL_ABORT="${EGL_FAIL_ABORT:-2}"
NATIVE_CRASH_ABORT="${NATIVE_CRASH_ABORT:-1}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

_egl_fail_streak=0
arm_log_has_egl_failure() {
  local elog="$1"
  [[ -f "$elog" ]] || return 1
  # Tail only — avoid matching older episodes in a shared arm log.
  tail -n 80 "$elog" | grep -Eqi \
    'unable to find CUDA device|WindowlessContext: Unable to create windowless context'
}

native_crash_signal() {
  # Map `timeout`/shell exit codes for fatal signals to a readable name.
  case "$1" in
    132) echo "SIGILL" ;;
    134) echo "SIGABRT" ;;
    135) echo "SIGBUS" ;;
    136) echo "SIGFPE" ;;
    137) echo "SIGKILL (possible OOM)" ;;
    139) echo "SIGSEGV (often Habitat EGL + Qwen libcuda)" ;;
    *) return 1 ;;
  esac
}

write_native_crash_capsule() {
  local arm="$1" qid="$2" rc="$3" elog="$4" signal_name="$5"
  local capsule="$OUT/native_crash_${arm}_q${qid}.log"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "arm=$arm"
    echo "question_id=$qid"
    echo "exit_code=$rc"
    echo "signal=$signal_name"
    echo "head=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "command=$EMET_HABITAT run-episode --question-id $qid"
    echo
    echo "----- episode log tail -----"
    tail -n 160 "$elog" 2>/dev/null || true
    echo
    echo "----- process snapshot -----"
    ps -eo pid,ppid,pgid,stat,etime,args | grep -E '[e]met-habitat|[h]abitat|[q]wen|[p]ython' || true
  } >"$capsule"
  log "native crash capsule → $capsule"
}

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
    env PYTHONFAULTHANDLER=1 "$@" timeout "$TIMEOUT" "$EMET_HABITAT" run-episode \
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
      if signal_name="$(native_crash_signal "$rc")"; then
        write_native_crash_capsule "$name" "$qid" "$rc" "$elog" "$signal_name"
        if [[ "$NATIVE_CRASH_ABORT" != "0" ]]; then
          log "ABORT: $signal_name in episode process. Empty ${name}_q${qid}.jsonl is a crash, not a scored miss. Inspect the capsule + journalctl -k for libcuda; retry the qid only after the GPU/driver recovers."
          exit "$rc"
        fi
      fi
      if arm_log_has_egl_failure "$elog"; then
        _egl_fail_streak=$((_egl_fail_streak + 1))
        log "EGL/CUDA-map failure streak=${_egl_fail_streak} (abort after ${EGL_FAIL_ABORT}; empty nvidia-smi ≠ EGL OK)"
        if [[ "${EGL_FAIL_ABORT}" =~ ^[0-9]+$ && "${EGL_FAIL_ABORT}" -gt 0 && "${_egl_fail_streak}" -ge "${EGL_FAIL_ABORT}" ]]; then
          log "ABORT: Habitat EGL broken (WindowlessContext / unable to find CUDA device). Fix driver/EGL or reboot; do not keep retrying in Cursor."
          exit 3
        fi
      else
        _egl_fail_streak=0
      fi
    else
      _egl_fail_streak=0
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
