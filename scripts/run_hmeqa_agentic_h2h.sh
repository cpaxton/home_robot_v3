#!/usr/bin/env bash
# Habitat HM-EQA holdout: classic Dynagraph vs agentic-verify Dynagraph.
# Comparable MCQ slice (same IDs as prior paper/branch numbers).
#
# Usage (prefer emet jobs):
#   uv run emet jobs run --name hmeqa-h2h --need-mib 12000 -- \
#     env EMET_ALLOW_SDPA_ATTN=1 HOLDOUT_IDS=… ./scripts/run_hmeqa_agentic_h2h.sh OUT_DIR
#
# Resume (skip non-empty ${arm}_q*.jsonl; rebuild aggregate jsonl):
#   RESUME=1 ARMS=classic,agentic ./scripts/run_hmeqa_agentic_h2h.sh OUT_DIR
#
# Arms:
#   classic  — EMET_EQA_AGENTIC_VERIFY=0 (Habitat planning loop; prior holdout numbers)
#   agentic  — EMET_EQA_AGENTIC_VERIFY=1 EMET_EQA_AGENTIC_ROUTER=0 (nav→SigLIP verify→answer)
#
# Env:
#   ARMS=classic,agentic   which arms to run (comma-separated)
#   RESUME=1               skip finished per-qid jsonl; do not truncate arm logs/jsonl
#   SKIP_KILL_STALE=1      skip gpu_preflight --kill-stale (keep live robot/agent jobs)
#   SKIP_GPU_WAIT=1        skip gpu_preflight --wait (e.g. NVML mismatch; Torch still sees CUDA)
#   EGL_FAIL_ABORT=2       abort after this many consecutive Habitat EGL/CUDA-map failures
#                          (0 = never). Pattern: WindowlessContext / unable to find CUDA device.
#   NATIVE_CRASH_ABORT=1   stop after a fatal native signal (default: 1). Writes
#                          native_crash_*.log and leaves prior episode artifacts intact.
#
# Recovery after an agent/session death (from *this* checkout — not a sibling tree):
#   bash scripts/status_log.sh tail
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"
OUT="${1:-$HOME/runs/emet/hmeqa_agentic_h2h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/figures" "$OUT/bundles"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
# Small first: holdout-4 gate; override HOLDOUT_IDS for full-8.
HOLDOUT_IDS="${HOLDOUT_IDS:-15,68,105,17}"
TIMEOUT="${TIMEOUT:-7200}"
ARMS="${ARMS:-classic,agentic}"
RESUME="${RESUME:-0}"
SKIP_KILL_STALE="${SKIP_KILL_STALE:-0}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
EGL_FAIL_ABORT="${EGL_FAIL_ABORT:-2}"
NATIVE_CRASH_ABORT="${NATIVE_CRASH_ABORT:-1}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

status_open "$OUT" "hmeqa-h2h"
STATUS_RESUME_CMD="uv run emet jobs run --name hmeqa-h2h-resume --need-mib 12000 -- \
env EMET_ALLOW_SDPA_ATTN=1 RESUME=1 ARMS=$ARMS HOLDOUT_IDS=$HOLDOUT_IDS \
./scripts/run_hmeqa_agentic_h2h.sh $OUT"

_egl_fail_streak=0
arm_log_has_egl_failure() {
  local elog="$1"
  [[ -f "$elog" ]] || return 1
  # Tail only — avoid matching older episodes after resume.
  tail -n 80 "$elog" | grep -Eqi \
    'unable to find CUDA device|WindowlessContext: Unable to create windowless context'
}

native_crash_signal() {
  case "$1" in
    132) echo "SIGILL" ;;
    134) echo "SIGABRT" ;;
    135) echo "SIGBUS" ;;
    136) echo "SIGFPE" ;;
    137) echo "SIGKILL (possible OOM)" ;;
    139) echo "SIGSEGV" ;;
    *) return 1 ;;
  esac
}

write_native_crash_capsule() {
  local arm="$1"
  local qid="$2"
  local rc="$3"
  local elog="$4"
  local signal_name="$5"
  local capsule="$OUT/native_crash_${arm}_q${qid}.log"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "arm=$arm"
    echo "question_id=$qid"
    echo "exit_code=$rc"
    echo "signal=$signal_name"
    echo "head=$(git rev-parse --short HEAD)"
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

gpu_wait() {
  if [[ "$SKIP_GPU_WAIT" == "1" ]]; then
    log "SKIP_GPU_WAIT=1 — not running gpu_preflight --wait"
    return 0
  fi
  NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
}

# Count IDs for progress (only requested arms).
_IFS_SAVE=$IFS
IFS=',' read -r -a _PROGRESS_IDS <<<"$HOLDOUT_IDS"
IFS=$_IFS_SAVE
PROGRESS_N_IDS=0
for _qid in "${_PROGRESS_IDS[@]}"; do
  _qid="$(echo "$_qid" | tr -d '[:space:]')"
  [[ -n "$_qid" ]] && PROGRESS_N_IDS=$((PROGRESS_N_IDS + 1))
done
_ARM_COUNT=0
IFS=',' read -r -a _PROGRESS_ARMS <<<"$ARMS"
IFS=$_IFS_SAVE
for _arm in "${_PROGRESS_ARMS[@]}"; do
  _arm="$(echo "$_arm" | tr -d '[:space:]')"
  case "$_arm" in
    classic|agentic) _ARM_COUNT=$((_ARM_COUNT + 1)) ;;
    "") ;;
    *) log "ERROR: unknown arm '$_arm' (want classic or agentic)"; exit 1 ;;
  esac
done
PROGRESS_TOTAL=$((PROGRESS_N_IDS * _ARM_COUNT))
PROGRESS_DONE=0

rebuild_arm_jsonl() {
  local name="$1"
  local out="$OUT/${name}.jsonl"
  : >"$out"
  local qid
  for qid in "${_PROGRESS_IDS[@]}"; do
    qid="$(echo "$qid" | tr -d '[:space:]')"
    [[ -n "$qid" ]] || continue
    local ep="$OUT/${name}_q${qid}.jsonl"
    if [[ -s "$ep" ]]; then
      cat "$ep" >>"$out"
    fi
  done
}

count_done_units() {
  local n=0
  local arm qid
  for arm in "${_PROGRESS_ARMS[@]}"; do
    arm="$(echo "$arm" | tr -d '[:space:]')"
    [[ -n "$arm" ]] || continue
    for qid in "${_PROGRESS_IDS[@]}"; do
      qid="$(echo "$qid" | tr -d '[:space:]')"
      [[ -n "$qid" ]] || continue
      if [[ -s "$OUT/${arm}_q${qid}.jsonl" ]]; then
        n=$((n + 1))
      fi
    done
  done
  echo "$n"
}

jobs_heartbeat() {
  # Write OUT/progress.json always; also update emet jobs if EMET_JOB_ID is set.
  local phase="$1"
  local qid="$2"
  local done="$3"
  local total="$4"
  uv run python -c "
from emet.utils.job_registry import write_progress_file
write_progress_file(
    r'''$OUT''',
    units_done=int('$done'),
    units_total=int('$total'),
    phase='$phase',
    current_id='$qid',
)
" 2>/dev/null || true
  if [[ -n "${EMET_JOB_ID:-}" ]]; then
    uv run emet jobs update "$EMET_JOB_ID" \
      --status running \
      --units-done "$done" \
      --units-total "$total" \
      --phase "$phase" \
      --current-id "$qid" \
      --out-dir "$OUT" \
      >/dev/null 2>&1 || true
  fi
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
           topdown_exploration.mp4 agentic_trace.jsonl agentic_summary.json; do
    [[ -e "$src/$f" ]] && cp -a "$src/$f" "$dst/"
  done
  [[ -d "$src/maps" ]] && rm -rf "$dst/maps" && cp -a "$src/maps" "$dst/maps"
  log "snapshot $arm q$qid → $dst"
}

if [[ ! -x "$EMET_HABITAT" ]]; then
  log "Missing emet-habitat at $EMET_HABITAT"
  status_close BLOCKED "no emet-habitat at $EMET_HABITAT" \
    "install the Habitat venv: ./scripts/install_habitat.sh"
  exit 1
fi

# Prefer flash-attn; allow SDPA only if explicitly requested (Habitat MCQ was OK on SDPA).
if ! uv run python -c "from emet.llms.attn_impl import flash_attn_2_available; raise SystemExit(0 if flash_attn_2_available() else 1)"; then
  if [[ "${EMET_ALLOW_SDPA_ATTN:-}" != "1" ]]; then
    log "Flash-Attn 2 not installed. Set EMET_ALLOW_SDPA_ATTN=1 to run on SDPA, or wait for flash-attn install."
    status_close BLOCKED "flash-attn 2 missing and EMET_ALLOW_SDPA_ATTN unset" \
      "re-launch with EMET_ALLOW_SDPA_ATTN=1, or install flash-attn first"
    exit 2
  fi
  log "WARNING: running with EMET_ALLOW_SDPA_ATTN=1 (no flash-attn)"
fi

log "OUT=$OUT HEAD=$(git rev-parse --short HEAD) ids=$HOLDOUT_IDS total_units=$PROGRESS_TOTAL arms=$ARMS resume=$RESUME"
if [[ "$RESUME" == "1" ]]; then
  PROGRESS_DONE="$(count_done_units)"
  rebuild_arm_jsonl classic
  rebuild_arm_jsonl agentic
  log "RESUME: restored aggregates; already done $PROGRESS_DONE/$PROGRESS_TOTAL units"
fi
jobs_heartbeat "init" "-" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL init"
status_note START \
  "arms=$ARMS ids=$HOLDOUT_IDS head=$(git rev-parse --short HEAD) resume=$RESUME" \
  "nothing — wait for the job. Progress: uv run emet jobs; log: bash scripts/status_log.sh tail"
gpu_wait
if [[ "$SKIP_KILL_STALE" == "1" ]]; then
  log "SKIP_KILL_STALE=1 — not running gpu_preflight --kill-stale"
else
  ./scripts/gpu_preflight.sh --kill-stale || true
fi

run_arm() {
  local name="$1"
  shift
  local out="$OUT/${name}.jsonl"
  local elog="$OUT/${name}.log"
  if [[ "$RESUME" == "1" ]]; then
    rebuild_arm_jsonl "$name"
    # Append to existing arm log rather than wiping prior episode traces.
    : >>"$elog"
  else
    : >"$out"
    : >"$elog"
  fi
  log "START arm=$name ($*)"
  jobs_heartbeat "$name" "-" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
  local qid
  for qid in "${_PROGRESS_IDS[@]}"; do
    qid="$(echo "$qid" | tr -d '[:space:]')"
    [[ -n "$qid" ]] || continue
    local ep="$OUT/${name}_q${qid}.jsonl"
    if [[ "$RESUME" == "1" && -s "$ep" ]]; then
      log "SKIP $name q$qid (resume: non-empty $ep)"
      continue
    fi
    gpu_wait
    : >"$ep"
    tag="h2h_${name}_q$(printf '%04d' "$qid")"
    log "----- $name q$qid (bundle tag=$tag) -----"
    jobs_heartbeat "$name" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
    STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL $name q$qid"
    status_note RUNNING "episode $name q$qid started (timeout ${TIMEOUT}s)" \
      "nothing — episode in flight (~4 min each). Re-check: bash scripts/status_log.sh tail"
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
          log "ABORT: $signal_name in episode process. Do not retry in this batch; inspect the crash capsule and use emet jobs status/logs after driver recovery."
          jobs_heartbeat "native-crash" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
          status_close CRASH \
            "$signal_name in $name q$qid — batch aborted at $PROGRESS_DONE/$PROGRESS_TOTAL units" \
            "1) read $OUT/native_crash_${name}_q${qid}.log  2) sudo dmesg -T | rg -i 'segfault|invalid opcode'  3) uv run emet eval diagnose  4) only then resume: $STATUS_RESUME_CMD"
          exit "$rc"
        fi
      fi
      if arm_log_has_egl_failure "$elog"; then
        _egl_fail_streak=$((_egl_fail_streak + 1))
        log "EGL/CUDA-map failure streak=${_egl_fail_streak} (abort after ${EGL_FAIL_ABORT}; empty nvidia-smi ≠ EGL OK)"
        if [[ "${EGL_FAIL_ABORT}" =~ ^[0-9]+$ && "${EGL_FAIL_ABORT}" -gt 0 && "${_egl_fail_streak}" -ge "${EGL_FAIL_ABORT}" ]]; then
          log "ABORT: Habitat EGL broken (WindowlessContext / unable to find CUDA device). Fix driver/EGL or reboot; do not keep retrying in Cursor."
          status_close EGL \
            "Habitat EGL broken (streak=${_egl_fail_streak}) — batch aborted at $PROGRESS_DONE/$PROGRESS_TOTAL units" \
            "uv run emet eval diagnose; empty nvidia-smi is NOT proof EGL works — reboot or settle the driver, then resume: $STATUS_RESUME_CMD"
          exit 3
        fi
      else
        _egl_fail_streak=0
      fi
      status_note FAIL "episode $name q$qid exit=$rc (non-fatal, continuing)" \
        "nothing yet — batch continues. Review later: tail -n 40 $OUT/${name}.log"
    else
      _egl_fail_streak=0
    fi
    if [[ -s "$ep" ]]; then
      # Re-append only this episode into aggregate (rebuild keeps order).
      rebuild_arm_jsonl "$name"
    fi
    snapshot_bundle "$name" "$qid" "$ep"
    PROGRESS_DONE=$((PROGRESS_DONE + 1))
    jobs_heartbeat "$name" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
    STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL $name q$qid"
    status_note OK "episode $name q$qid finished" \
      "nothing — job alive and advancing"
  done
  rebuild_arm_jsonl "$name"
  log "DONE arm=$name"
}

_arm_list=()
IFS=',' read -r -a _arm_list <<<"$ARMS"
for _arm in "${_arm_list[@]}"; do
  _arm="$(echo "$_arm" | tr -d '[:space:]')"
  [[ -n "$_arm" ]] || continue
  case "$_arm" in
    classic)
      run_arm classic EMET_EQA_AGENTIC_VERIFY=0
      ;;
    agentic)
      run_arm agentic \
        EMET_EQA_AGENTIC_VERIFY=1 \
        EMET_EQA_AGENTIC_ROUTER=0 \
        EMET_EQA_TRACE=1
      ;;
    *)
      log "ERROR: unknown arm '$_arm' (want classic or agentic)"
      exit 1
      ;;
  esac
done

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

jobs_heartbeat "done" "-" "$PROGRESS_TOTAL" "$PROGRESS_TOTAL"
if [[ -n "${EMET_JOB_ID:-}" ]]; then
  uv run emet jobs update "$EMET_JOB_ID" --status done \
    --units-done "$PROGRESS_TOTAL" --units-total "$PROGRESS_TOTAL" \
    --phase done --out-dir "$OUT" >/dev/null 2>&1 || true
fi
echo DONE > "$OUT/DONE"
log "All done → $OUT"
STATUS_PROGRESS="$PROGRESS_TOTAL/$PROGRESS_TOTAL done"
status_close DONE "all $PROGRESS_TOTAL units finished; summary + figures written" \
  "review $OUT/orchestrator.log and $OUT/figures/, then uv run python scripts/hmeqa_significance.py $OUT"
