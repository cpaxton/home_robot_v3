# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# shellcheck shell=bash
#
# Tail-able experiment status log — **per checkout**, not shared across repos.
#
# Multiple agents often work in sibling trees (home_robot_v2 / v3 / v4) that all
# write under ~/runs/emet/. A single ~/runs/emet/STATUS.log would interleave
# their recovery instructions. Default path is therefore namespaced:
#
#   ~/runs/emet/status/<repo_basename>/STATUS.log
#
# Recovery (from the checkout that owns the job — never from a sibling tree):
#
#   bash scripts/status_log.sh tail
#   bash scripts/status_log.sh path
#   bash scripts/status_log.sh latest
#
# Every record ends with a "next:" line holding the literal command to run.
#
# Usage (orchestrators):
#   source scripts/status_log.sh
#   status_open "$OUT" "hmeqa-bal32"
#   STATUS_PROGRESS="6/64 classic q14"
#   STATUS_RESUME_CMD="RESUME=1 ./scripts/… "
#   status_note RUNNING "classic q14 started" "wait — job alive"
#   status_close DONE "all 64 units finished" "review $OUT/summary.txt"
#
# Env:
#   EMET_STATUS_LOG   override the per-repo mirror path
#   STATUS_PROGRESS   free-form progress shown in the header line
#   STATUS_RESUME_CMD command offered after a crash / unexpected exit

_STATUS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_REPO_ROOT="$(cd "$_STATUS_SCRIPT_DIR/.." && pwd)"
STATUS_REPO_NAME="$(basename "$STATUS_REPO_ROOT")"
STATUS_LOG_DIR="${EMET_STATUS_DIR:-$HOME/runs/emet/status/$STATUS_REPO_NAME}"
STATUS_LOG_GLOBAL="${EMET_STATUS_LOG:-$STATUS_LOG_DIR/STATUS.log}"
_STATUS_OUT=""
_STATUS_LABEL=""
_STATUS_CLOSED=0

_status_write() {
  local record="$1"
  mkdir -p "$(dirname "$STATUS_LOG_GLOBAL")" 2>/dev/null || true
  printf '%s\n' "$record" >>"$STATUS_LOG_GLOBAL" 2>/dev/null || true
  if [[ -n "$_STATUS_OUT" ]]; then
    mkdir -p "$_STATUS_OUT" 2>/dev/null || true
    printf '%s\n' "$record" >>"$_STATUS_OUT/STATUS.log" 2>/dev/null || true
  fi
  return 0
}

status_note() {
  local state="$1"
  local what="$2"
  local next="${3:-}"
  local job="${EMET_JOB_ID:-}"
  local record
  record="$(printf '=== %s  %s  %s  %s' \
    "$(date -Iseconds)" "${_STATUS_LABEL:-run}" "$state" "${STATUS_PROGRESS:--}")"
  record+=$'\n'"    repo: $STATUS_REPO_ROOT"
  record+=$'\n'"    out:  ${_STATUS_OUT:--}"
  if [[ -n "$job" ]]; then
    record+=$'\n'"    job:  $job  (cd $STATUS_REPO_ROOT && uv run emet jobs status $job)"
  fi
  record+=$'\n'"    what: $what"
  record+=$'\n'"    next: ${next:-unknown — inspect $_STATUS_OUT}"
  _status_write "$record"
  return 0
}

_status_on_exit() {
  local rc="$?"
  if [[ "$_STATUS_CLOSED" == "1" ]]; then
    return 0
  fi
  local next="${STATUS_RESUME_CMD:-}"
  if [[ -z "$next" ]]; then
    next="inspect $_STATUS_OUT (no resume command recorded)"
  fi
  local cause="orchestrator exited rc=$rc without a final status"
  if [[ -n "${_STATUS_SIGNAL:-}" ]]; then
    cause="orchestrator killed by SIG${_STATUS_SIGNAL} (rc=$rc)"
  else
    cause+=" (killed, or a bug before status_close)"
  fi
  status_note EXIT "$cause" \
    "check 'uv run emet jobs' in $STATUS_REPO_ROOT first; if the job is not running, resume with: $next"
  return 0
}

# Untrapped fatal signals bypass the EXIT trap, and SIGTERM is how
# `emet jobs cancel` stops a run — convert them into a normal exit.
_status_on_signal() {
  _STATUS_SIGNAL="$1"
  exit "$((128 + $2))"
}

status_open() {
  _STATUS_OUT="${1:?status_open needs OUT dir}"
  _STATUS_LABEL="${2:-$(basename "$_STATUS_OUT")}"
  _STATUS_CLOSED=0
  mkdir -p "$_STATUS_OUT" 2>/dev/null || true
  # Stable path so recovery does not need the timestamped run dir.
  local log_dir
  log_dir="$(dirname "$STATUS_LOG_GLOBAL")"
  mkdir -p "$log_dir" 2>/dev/null || true
  ln -sfn "$_STATUS_OUT" "$log_dir/latest" 2>/dev/null || true
  trap _status_on_exit EXIT
  trap '_status_on_signal TERM 15' TERM
  trap '_status_on_signal INT 2' INT
  trap '_status_on_signal HUP 1' HUP
  return 0
}

status_close() {
  local state="${1:-DONE}"
  local what="${2:-finished}"
  local next="${3:-}"
  status_note "$state" "$what" "$next"
  _STATUS_CLOSED=1
  return 0
}

status_tail() {
  local n="${1:-12}"
  if [[ ! -f "$STATUS_LOG_GLOBAL" ]]; then
    echo "no status log yet at $STATUS_LOG_GLOBAL (repo=$STATUS_REPO_ROOT)" >&2
    return 1
  fi
  tail -n "$n" "$STATUS_LOG_GLOBAL"
}

# CLI when executed (not sourced): path | tail [N] | latest
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-tail}" in
    path)
      echo "$STATUS_LOG_GLOBAL"
      ;;
    latest)
      if [[ -L "$STATUS_LOG_DIR/latest" || -e "$STATUS_LOG_DIR/latest" ]]; then
        readlink -f "$STATUS_LOG_DIR/latest" 2>/dev/null || ls -l "$STATUS_LOG_DIR/latest"
      else
        echo "no latest symlink at $STATUS_LOG_DIR/latest" >&2
        exit 1
      fi
      ;;
    tail)
      status_tail "${2:-12}"
      ;;
    *)
      echo "usage: $0 {path|tail [N]|latest}" >&2
      exit 2
      ;;
  esac
fi
