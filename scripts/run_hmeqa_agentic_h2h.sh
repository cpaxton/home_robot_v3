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
#   agentic  — EMET_EQA_AGENTIC_VERIFY=1 (+ optional router / verifier / require-verified)
#
# Env:
#   ARMS=classic,agentic   which arms to run (comma-separated)
#   RESUME=1               skip finished per-qid jsonl; do not truncate arm logs/jsonl
#   SKIP_KILL_STALE=1      skip gpu_preflight --kill-stale (keep live robot/agent jobs)
#   SKIP_GPU_WAIT=1        skip gpu_preflight --wait (e.g. NVML mismatch; Torch still sees CUDA)
#   EGL_FAIL_ABORT=2       abort after this many consecutive Habitat EGL/CUDA-map failures
#                          (0 = never). Pattern: WindowlessContext / unable to find CUDA device.
#   NATIVE_CRASH_POLICY=skip  skip|abort (default skip). skip settles + retries then
#                          continues; abort stops the batch on first native crash.
#   NATIVE_CRASH_ABORT=1   deprecated alias for NATIVE_CRASH_POLICY=abort.
#   NATIVE_CRASH_RETRIES=1   retries of the same qid after a native crash (skip policy).
#   NATIVE_CRASH_SETTLE_SEC=60  sleep after native crash before retry/next.
#   NATIVE_CRASH_STREAK_ABORT=2  under skip: abort after N consecutive native crashes
#                          (early exit when harness/driver is wedged; 0 = never).
#   EMET_SKIP_CPU_AFFINITY=0  pin this process away from turbo P-cores (default on).
#                          Auto-excludes logical CPUs with cpuinfo_max_freq >=
#                          EMET_EXCLUDE_CPU_MIN_MHZ (default 6000). On the i9-14900KF
#                          that is CPUs 8-11 (both 6.0 GHz P-cores) — do NOT use
#                          taskset -c 0-7,10-31 (still leaves CPUs 10-11 online).
#   EPISODE_COOLDOWN_SEC=20  sleep + sync between episodes (0 disables). Reduces
#                          stacked Habitat/VLM teardown pressure that can hard-freeze.
#   EPISODE_GPU_WAIT=1     re-run gpu_preflight --wait between episodes (default 1).
#   EMET_EQA_AGENTIC_ROUTER=0|1  honor for agentic arm (default 0). paper-router /
#                          emet hmeqa overnight pass 1; do not hardcode over the env.
#   EMET_EQA_AGENTIC_VERIFIER / EMET_EQA_AGENTIC_REQUIRE_VERIFIED  passed through.
#   EQA_HF_MODEL_ID / EQA_VL_FAMILY / EQA_VL_QUANTIZATION  → emet-habitat
#                          --eqa-hf-model-id / --eqa-vl-family / (quant via config env).
#                          Used for larger-VLM ladder (e.g. Qwen3-VL-32B-Instruct).
#
# Prefer: uv run emet hmeqa h2h|resume  (dogfood CLI over hand-rolled env/taskset).
# Recovery after an agent/session death (from *this* checkout — not a sibling tree):
#   uv run emet status tail
#   uv run emet eval recover --need-mib 12000
#   uv run emet hmeqa resume
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
if [[ "${NATIVE_CRASH_ABORT:-}" == "1" ]]; then
  NATIVE_CRASH_POLICY="${NATIVE_CRASH_POLICY:-abort}"
else
  NATIVE_CRASH_POLICY="${NATIVE_CRASH_POLICY:-skip}"
fi
NATIVE_CRASH_RETRIES="${NATIVE_CRASH_RETRIES:-1}"
NATIVE_CRASH_SETTLE_SEC="${NATIVE_CRASH_SETTLE_SEC:-60}"
NATIVE_CRASH_STREAK_ABORT="${NATIVE_CRASH_STREAK_ABORT:-2}"
EMET_SKIP_CPU_AFFINITY="${EMET_SKIP_CPU_AFFINITY:-0}"
EMET_EXCLUDE_CPU_MIN_MHZ="${EMET_EXCLUDE_CPU_MIN_MHZ:-6000}"
EPISODE_COOLDOWN_SEC="${EPISODE_COOLDOWN_SEC:-20}"
EPISODE_GPU_WAIT="${EPISODE_GPU_WAIT:-1}"
EQA_HF_MODEL_ID="${EQA_HF_MODEL_ID:-}"
EQA_VL_FAMILY="${EQA_VL_FAMILY:-}"
EQA_VL_QUANTIZATION="${EQA_VL_QUANTIZATION:-}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

# Extra flags for emet-habitat run-episode (larger-VLM ladder).
EQA_EXTRA_ARGS=()
if [[ -n "$EQA_VL_FAMILY" ]]; then
  EQA_EXTRA_ARGS+=(--eqa-vl-family "$EQA_VL_FAMILY")
fi
if [[ -n "$EQA_HF_MODEL_ID" ]]; then
  EQA_EXTRA_ARGS+=(--eqa-hf-model-id "$EQA_HF_MODEL_ID")
fi
# Quantization is usually int4 via dynav defaults; expose for documentation / future CLI.
if [[ -n "$EQA_VL_QUANTIZATION" ]]; then
  log "NOTE: EQA_VL_QUANTIZATION=$EQA_VL_QUANTIZATION (set eqa.vl_quantization via EMET_CONFIG/--set if needed; run-episode has no dedicated flag)"
fi
if [[ ${#EQA_EXTRA_ARGS[@]} -gt 0 ]]; then
  log "EQA model overrides: ${EQA_EXTRA_ARGS[*]}"
fi

status_open "$OUT" "hmeqa-h2h"
STATUS_RESUME_CMD="uv run emet eval recover --need-mib 12000 && uv run emet hmeqa resume $OUT"

apply_eval_cpu_affinity() {
  if [[ "$EMET_SKIP_CPU_AFFINITY" == "1" ]]; then
    log "SKIP CPU affinity (EMET_SKIP_CPU_AFFINITY=1)"
    return 0
  fi
  if ! EMET_EXCLUDE_CPU_MIN_MHZ="$EMET_EXCLUDE_CPU_MIN_MHZ" \
      uv run emet eval affinity --apply --pid "$$"; then
    log "ERROR: cpu affinity failed (fail-closed)"
    status_close BLOCKED "cpu affinity could not exclude turbo cores" \
      "fix host cpufreq sysfs or set EMET_SKIP_CPU_AFFINITY=1 only if intentional; then $STATUS_RESUME_CMD"
    exit 2
  fi
  log "CPU affinity applied (exclude >=${EMET_EXCLUDE_CPU_MIN_MHZ} MHz turbo)"
}

episode_cooldown() {
  local why="${1:-between episodes}"
  if [[ "${EPISODE_COOLDOWN_SEC}" =~ ^[0-9]+$ && "${EPISODE_COOLDOWN_SEC}" -gt 0 ]]; then
    log "episode cooldown ${EPISODE_COOLDOWN_SEC}s ($why)"
    sync || true
    sleep "$EPISODE_COOLDOWN_SEC"
  fi
}

note_host_freeze_if_any() {
  # Unclean reboot mid-episode: empty current jsonl + classic.log trailing NULs.
  local progress="$OUT/progress.json"
  [[ -f "$progress" ]] || return 0
  local phase qid
  phase="$(uv run python -c "import json; print(json.load(open(r'''$progress''')).get('phase') or '')" 2>/dev/null || true)"
  qid="$(uv run python -c "import json; print(json.load(open(r'''$progress''')).get('current_id') or '')" 2>/dev/null || true)"
  [[ -n "$phase" && -n "$qid" && "$qid" != "-" ]] || return 0
  case "$phase" in
    classic|agentic) ;;
    *) return 0 ;;
  esac
  local ep="$OUT/${phase}_q${qid}.jsonl"
  [[ -e "$ep" && ! -s "$ep" ]] || return 0
  local elog="$OUT/${phase}.log"
  local has_nuls=0
  if [[ -f "$elog" ]]; then
    has_nuls="$(uv run python -c "
from pathlib import Path
b=Path(r'''$elog''').read_bytes()
print(1 if b.endswith(b'\\x00'*8) or (len(b)>64 and b[-64:].count(0)>32) else 0)
" 2>/dev/null || echo 0)"
  fi
  local capsule="$OUT/host_freeze_${phase}_q${qid}.log"
  [[ -f "$capsule" ]] && return 0
  {
    echo "timestamp=$(date -Iseconds)"
    echo "kind=host-freeze-or-hard-reboot"
    echo "arm=$phase"
    echo "question_id=$qid"
    echo "evidence=empty ${phase}_q${qid}.jsonl + progress mid-episode"
    echo "log_trailing_nuls=$has_nuls"
    echo "head=$(git rev-parse --short HEAD)"
    echo "uptime=$(uptime -p 2>/dev/null || true)"
    echo "boot=$(who -b 2>/dev/null || true)"
    echo
    echo "----- progress.json -----"
    cat "$progress" 2>/dev/null || true
    echo
    echo "----- episode log tail -----"
    if [[ -f "$elog" ]]; then
      uv run python -c "
from pathlib import Path
b=Path(r'''$elog''').read_bytes()
i=len(b)-1
while i>=0 and b[i]==0: i-=1
print(b[max(0,i-4000):i+1].decode('utf-8','replace'))
" 2>/dev/null || tail -n 80 "$elog"
    fi
  } >"$capsule"
  log "host-freeze capsule → $capsule (empty $ep; will re-run on resume)"
  status_note CRASH \
    "prior host freeze/reboot during $phase q$qid — artifacts preserved; empty jsonl will re-run" \
    "inspect $capsule; then resume with affinity baked into this script: $STATUS_RESUME_CMD"
}

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
PROGRESS_FAILED=0
_native_crash_streak=0

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
  local failed="${5:-$PROGRESS_FAILED}"
  uv run python -c "
from emet.eval.harness import update_eval_progress
update_eval_progress(
    r'''$OUT''',
    units_done=int('$done'),
    units_total=int('$total'),
    phase='$phase',
    current_id='$qid',
    units_failed=int('$failed'),
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
           floor_area.jsonl floor_area_growth.png \
           topdown_exploration.mp4 episode_rgb.mp4 agentic_trace.jsonl agentic_summary.json; do
    [[ -e "$src/$f" ]] && cp -a "$src/$f" "$dst/"
  done
  [[ -d "$src/maps" ]] && rm -rf "$dst/maps" && cp -a "$src/maps" "$dst/maps"
  [[ -d "$src/frontier_picks" ]] && rm -rf "$dst/frontier_picks" && cp -a "$src/frontier_picks" "$dst/frontier_picks"
  # Head-camera frames are large; symlink the full dir + copy a few keyframes.
  if [[ -d "$src/frames" ]]; then
    ln -sfn "$src/frames" "$dst/frames_all"
    mkdir -p "$dst/frames"
    # Evenly spaced samples for quick feh browsing without opening 360 files.
    mapfile -t _rgbs < <(ls "$src/frames"/rgb_*.png 2>/dev/null | sort)
    if ((${#_rgbs[@]} > 0)); then
      _n=${#_rgbs[@]}
      for _k in 0 1 2 3 4 5; do
        _idx=$(( _k * (_n - 1) / 5 ))
        cp -n "${_rgbs[$_idx]}" "$dst/frames/" 2>/dev/null || true
      done
    fi
  fi
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
apply_eval_cpu_affinity
if [[ "$RESUME" == "1" ]]; then
  note_host_freeze_if_any
  PROGRESS_DONE="$(count_done_units)"
  rebuild_arm_jsonl classic
  rebuild_arm_jsonl agentic
  log "RESUME: restored aggregates; already done $PROGRESS_DONE/$PROGRESS_TOTAL units"
fi
jobs_heartbeat "init" "-" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL init"
status_note START \
  "arms=$ARMS ids=$HOLDOUT_IDS head=$(git rev-parse --short HEAD) resume=$RESUME" \
  "nothing — wait for the job. Progress: uv run emet jobs; log: uv run emet status tail"
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
    if [[ "$EPISODE_GPU_WAIT" == "1" ]]; then
      gpu_wait
    fi
    local attempt=0
    local max_attempts=1
    if [[ "${NATIVE_CRASH_RETRIES}" =~ ^[0-9]+$ ]]; then
      max_attempts=$((1 + NATIVE_CRASH_RETRIES))
    fi
    local episode_ok=0
    while [[ "$attempt" -lt "$max_attempts" ]]; do
      attempt=$((attempt + 1))
      : >"$ep"
      tag="h2h_${name}_q$(printf '%04d' "$qid")"
      log "----- $name q$qid attempt=${attempt}/${max_attempts} (bundle tag=$tag) -----"
      jobs_heartbeat "$name" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
      STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL $name q$qid"
      status_note RUNNING "episode $name q$qid started (timeout ${TIMEOUT}s)" \
        "nothing — episode in flight (~4 min each). Re-check: uv run emet status tail"
      set +e
      # shellcheck disable=SC2086
      env PYTHONFAULTHANDLER=1 "$@" timeout --kill-after=30s "$TIMEOUT" "$EMET_HABITAT" run-episode \
        --question-id "$qid" \
        --method dynagraph \
        --explore-when-uncovered off \
        --no-mcq-debias \
        --memory-summary \
        --debug-run-tag "$tag" \
        --output "$ep" \
        "${EQA_EXTRA_ARGS[@]}" \
        >>"$elog" 2>&1
      rc=$?
      set -e
      if [[ "$rc" -eq 0 && -s "$ep" ]]; then
        episode_ok=1
        _egl_fail_streak=0
        _native_crash_streak=0
        break
      fi
      log "FAIL $name q$qid exit=$rc attempt=${attempt}/${max_attempts}"
      if signal_name="$(native_crash_signal "$rc")"; then
        write_native_crash_capsule "$name" "$qid" "$rc" "$elog" "$signal_name"
        uv run python -c "
from emet.eval.harness import write_crash_marker
write_crash_marker(r'''$OUT''', '$name', '$qid', returncode=int('$rc'), signal_name='$signal_name')
" 2>/dev/null || true
        _native_crash_streak=$((_native_crash_streak + 1))
        if [[ "$NATIVE_CRASH_POLICY" == "abort" ]]; then
          log "ABORT: $signal_name (NATIVE_CRASH_POLICY=abort)"
          jobs_heartbeat "native-crash" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
          status_close CRASH \
            "$signal_name in $name q$qid — batch aborted at $PROGRESS_DONE/$PROGRESS_TOTAL units" \
            "1) read $OUT/native_crash_${name}_q${qid}.log  2) uv run emet eval recover  3) $STATUS_RESUME_CMD"
          exit "$rc"
        fi
        if [[ "${NATIVE_CRASH_STREAK_ABORT}" =~ ^[0-9]+$ && "${NATIVE_CRASH_STREAK_ABORT}" -gt 0 && "${_native_crash_streak}" -ge "${NATIVE_CRASH_STREAK_ABORT}" ]]; then
          log "ABORT: native crash streak=${_native_crash_streak} (early exit; harness likely wedged)"
          jobs_heartbeat "crash-streak" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
          status_close CRASH \
            "native crash streak=${_native_crash_streak} at $name q$qid — aborted at $PROGRESS_DONE/$PROGRESS_TOTAL" \
            "uv run emet eval recover --need-mib 12000; then $STATUS_RESUME_CMD"
          exit "$rc"
        fi
        log "native crash settle ${NATIVE_CRASH_SETTLE_SEC}s (policy=skip streak=${_native_crash_streak})"
        sync || true
        sleep "${NATIVE_CRASH_SETTLE_SEC}"
        gpu_wait || true
        if [[ "$attempt" -lt "$max_attempts" ]]; then
          log "retrying $name q$qid after native crash"
          continue
        fi
        PROGRESS_FAILED=$((PROGRESS_FAILED + 1))
        status_note FAIL "episode $name q$qid $signal_name — skipped (empty jsonl; resume can retry)" \
          "batch continues. Capsule: $OUT/native_crash_${name}_q${qid}.log"
        break
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
      if [[ "$attempt" -ge "$max_attempts" ]]; then
        PROGRESS_FAILED=$((PROGRESS_FAILED + 1))
        status_note FAIL "episode $name q$qid exit=$rc (non-fatal, continuing)" \
          "nothing yet — batch continues. Review later: tail -n 40 $OUT/${name}.log"
      fi
    done
    if [[ -s "$ep" ]]; then
      rebuild_arm_jsonl "$name"
      snapshot_bundle "$name" "$qid" "$ep"
      PROGRESS_DONE=$((PROGRESS_DONE + 1))
      jobs_heartbeat "$name" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
      STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL $name q$qid"
      status_note OK "episode $name q$qid finished" \
        "nothing — job alive and advancing"
      sync || true
      episode_cooldown "post $name q$qid"
    else
      jobs_heartbeat "$name" "$qid" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
      episode_cooldown "post-fail $name q$qid"
    fi
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
      # Honor caller env (emet hmeqa --preset paper-router / overnight). Default
      # router off matches the historical holdout-8 / bal-32 scored policy.
      _router="${EMET_EQA_AGENTIC_ROUTER:-0}"
      run_arm agentic \
        EMET_EQA_AGENTIC_VERIFY=1 \
        "EMET_EQA_AGENTIC_ROUTER=${_router}" \
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

# Optional: copy figures into paper/figs (opt-in — bal-32 runs must not overwrite
# the holdout-8 paper bars/coverage panels).
if [[ "${COPY_PAPER_FIGS:-0}" == "1" && -d "$ROOT/paper/figs" ]]; then
  if [[ -f "$OUT/figures/hmeqa_agentic_coverage.png" ]]; then
    cp -a "$OUT/figures/hmeqa_agentic_coverage.png" "$ROOT/paper/figs/hmeqa_agentic_coverage.png" || true
  fi
  if [[ -f "$OUT/figures/hmeqa_agentic_h2h.png" ]]; then
    cp -a "$OUT/figures/hmeqa_agentic_h2h.png" "$ROOT/paper/figs/hmeqa_agentic_h2h.png" || true
  fi
fi

CRASHED_QIDS="$(find "$OUT" -maxdepth 1 -name '*_q*.CRASH' -printf '%f\n' 2>/dev/null | sed 's/\.CRASH$//' | paste -sd, - || true)"
if [[ -n "${CRASHED_QIDS:-}" ]]; then
  log "CRASHED_QIDS=${CRASHED_QIDS} (units_failed=${PROGRESS_FAILED}; scored=${PROGRESS_DONE}/${PROGRESS_TOTAL})"
fi

# Disk truth: skipped native crashes leave empty per-qid jsonl. Do not write
# OUT/DONE or mark the job done until every unit scored — resume retries empties.
PROGRESS_DONE="$(count_done_units)"
jobs_heartbeat "final" "-" "$PROGRESS_DONE" "$PROGRESS_TOTAL"
STATUS_PROGRESS="$PROGRESS_DONE/$PROGRESS_TOTAL final"
_incomplete=0
if [[ "$PROGRESS_DONE" -lt "$PROGRESS_TOTAL" ]]; then
  _incomplete=1
fi
_job_status="done"
_final_state="DONE"
_final_rc=0
if [[ "$_incomplete" -eq 1 ]]; then
  _job_status="failed"
  _final_state="INCOMPLETE"
  _final_rc=1
elif [[ "$PROGRESS_DONE" -eq 0 && "$PROGRESS_FAILED" -gt 0 ]]; then
  _job_status="failed"
  _final_state="FAIL"
  _final_rc=1
fi
if [[ -n "${EMET_JOB_ID:-}" ]]; then
  uv run emet jobs update "$EMET_JOB_ID" --status "$_job_status" \
    --units-done "$PROGRESS_DONE" --units-total "$PROGRESS_TOTAL" \
    --phase "${_final_state,,}" --out-dir "$OUT" >/dev/null 2>&1 || true
fi
if [[ "$_incomplete" -eq 0 && "$_final_rc" -eq 0 ]]; then
  echo DONE > "$OUT/DONE"
  log "All done → $OUT (scored ${PROGRESS_DONE}/${PROGRESS_TOTAL}, failed ${PROGRESS_FAILED})"
  status_close DONE "scored $PROGRESS_DONE/$PROGRESS_TOTAL units (failed=${PROGRESS_FAILED}); summary + figures written" \
    "review $OUT/orchestrator.log and $OUT/figures/, then uv run emet hmeqa summarize $OUT"
else
  rm -f "$OUT/DONE"
  log "INCOMPLETE → $OUT (scored ${PROGRESS_DONE}/${PROGRESS_TOTAL}, failed ${PROGRESS_FAILED}) — not writing DONE; resume to fill gaps"
  status_close "$_final_state" \
    "scored $PROGRESS_DONE/$PROGRESS_TOTAL units (failed=${PROGRESS_FAILED}); missing empty per-qid jsonl" \
    "$STATUS_RESUME_CMD"
fi
exit "$_final_rc"
