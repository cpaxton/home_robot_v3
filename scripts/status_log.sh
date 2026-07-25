# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
# shellcheck shell=bash
#
# Tail-able experiment status log.
#
# Long GPU runs are launched detached (emet jobs) and the launching Cursor/agent
# session frequently dies mid-run. This writes one self-contained record per
# state change so that recovery is a single command with no run-dir archaeology:
#
#   tail -n 12 ~/runs/emet/STATUS.log
#
# Every record ends with a "next:" line holding the literal command to run, so
# the last lines of the file always answer "what do I do now?".
#
# Usage:
#   source scripts/status_log.sh
#   status_open "$OUT" "hmeqa-bal32"          # writes header, arms EXIT trap
#   STATUS_PROGRESS="6/64 classic q14"        # optional, shown in the header line
#   STATUS_RESUME_CMD="RESUME=1 ./scripts/… " # used by CRASH / unexpected EXIT
#   status_note RUNNING "classic q14 started" "wait — job alive"
#   status_close DONE "all 64 units finished" "review $OUT/summary.txt"
#
# Env:
#   EMET_STATUS_LOG   global mirror path (default ~/runs/emet/STATUS.log)
#   STATUS_PROGRESS   free-form progress shown in the header line
#   STATUS_RESUME_CMD command offered after a crash / unexpected exit

STATUS_LOG_GLOBAL="${EMET_STATUS_LOG:-$HOME/runs/emet/STATUS.log}"
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
    record+=$'\n'"    out:  ${_STATUS_OUT:--}"
    if [[ -n "$job" ]]; then
        record+=$'\n'"    job:  $job  (uv run emet jobs status $job)"
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
        "check 'uv run emet jobs' first; if the job is not running, resume with: $next"
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
