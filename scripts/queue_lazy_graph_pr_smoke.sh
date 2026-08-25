#!/usr/bin/env bash
# Delayed LazyGraph validation for PR #131 (unit + agent-regression, no Habitat).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"

DELAY_SEC="${LAZY_GRAPH_SMOKE_DELAY_SEC:-21600}"
OUT="${1:-$HOME/runs/emet/lazy_graph_smoke/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
LOG="$OUT/smoke.log"

wake_at="$(date -d "+${DELAY_SEC} seconds" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date -v+"${DELAY_SEC}"S '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || echo "in ${DELAY_SEC}s")"

status_open "$OUT" "lazy-graph-pr-smoke"
STATUS_RESUME_CMD="uv run emet jobs logs lazy-graph-smoke-6h --tail 80"
status_note SCHEDULED "sleep ${DELAY_SEC}s until ${wake_at}" "wait — job registered"

{
  echo "[lazy-graph-smoke] repo=$ROOT branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD)"
  echo "[lazy-graph-smoke] sleeping ${DELAY_SEC}s until ~${wake_at}"
} | tee -a "$LOG"

sleep "$DELAY_SEC"

{
  echo "[lazy-graph-smoke] wake at $(date -Is)"
  git fetch origin
  git checkout feature/lazy-graph
  git merge --ff-only origin/feature/lazy-graph 2>/dev/null || git pull origin feature/lazy-graph || true
  echo "[lazy-graph-smoke] at commit $(git rev-parse --short HEAD)"
} | tee -a "$LOG"

status_note RUNNING "pytest lazy_graph + agent-regression" "tail -f $LOG"

set +e
uv run emet test src/test/memory/test_lazy_graph_commit.py -v 2>&1 | tee -a "$LOG"
UNIT_RC=${PIPESTATUS[0]}
uv run emet test agent-regression -q 2>&1 | tee -a "$LOG"
AGENT_RC=${PIPESTATUS[0]}
set -e

if [[ "$UNIT_RC" -eq 0 && "$AGENT_RC" -eq 0 ]]; then
  status_close DONE "lazy_graph smoke passed" "review $LOG"
  exit 0
fi

status_close FAIL "unit_rc=$UNIT_RC agent_rc=$AGENT_RC" "review $LOG"
exit 1
