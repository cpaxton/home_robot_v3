#!/usr/bin/env bash
# Real Robocasa smoke for LazyGraph (PR #131): serve sim + explore-loop + export.
# Usage:
#   ./scripts/queue_lazy_graph_robocasa_smoke.sh [OUT_DIR]
# Env:
#   ROBOT=stretch|innate_mars|…   (default stretch)
#   SEED=0
#   EXPLORE_ITERS=5
#   EXPLORE_TIMEOUT_S=3600
#   PORT_OFFSET=240
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"

ROBOT="${ROBOT:-stretch}"
SEED="${SEED:-0}"
ITERS="${EXPLORE_ITERS:-5}"
TIMEOUT_S="${EXPLORE_TIMEOUT_S:-3600}"
PORT_OFFSET="${PORT_OFFSET:-240}"
OUT="${1:-$HOME/runs/emet/lazy_graph_robocasa_smoke/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

export EMET_ZMQ_STARTUP_TIMEOUT="${EMET_ZMQ_STARTUP_TIMEOUT:-120}"
export EMET_SIM_NAV_TELEPORT="${EMET_SIM_NAV_TELEPORT:-1}"

status_open "$OUT" "lazy-graph-robocasa-smoke"
STATUS_RESUME_CMD="uv run emet jobs logs \${EMET_JOB_ID:-} --tail 80"
status_note RUNNING "serve robocasa robot=$ROBOT seed=$SEED iters=$ITERS" \
  "uv run emet jobs status \${EMET_JOB_ID:-}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  uv run emet kill-mujoco-server --all 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "[lazy-graph-robocasa] repo=$ROOT branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD)"
  echo "[lazy-graph-robocasa] OUT=$OUT robot=$ROBOT seed=$SEED iters=$ITERS timeout_s=$TIMEOUT_S port_offset=$PORT_OFFSET"
} | tee -a "$OUT/smoke.log"

uv run emet kill-mujoco-server --all 2>/dev/null || true
sleep 2

uv run emet serve robocasa --robot "$ROBOT" --headless --seed "$SEED" --port-offset "$PORT_OFFSET" \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 90); do
  if grep -qE "Server running|Server ready|RobosuiteZmqServer started|ZMQ server listening|Listening on tcp|MuJoCo Simulator is connected" "$OUT/server.log" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[lazy-graph-robocasa] server died early — see $OUT/server.log" | tee -a "$OUT/smoke.log"
    status_close FAIL "mujoco server exited before ready" "tail $OUT/server.log"
    exit 1
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "[lazy-graph-robocasa] server not ready after wait — see $OUT/server.log" | tee -a "$OUT/smoke.log"
  status_close FAIL "server startup timeout" "tail $OUT/server.log"
  exit 1
fi
sleep 8

status_note RUNNING "lazy-graph explore-loop iters=$ITERS" "tail -f $OUT/lazy_graph.log"

set +e
timeout --signal=INT "${TIMEOUT_S}" uv run emet run lazy-graph \
  --robot "$ROBOT" \
  --robot-ip 127.0.0.1 \
  --port-offset "$PORT_OFFSET" \
  --explore-loop \
  --explore-max-iters "$ITERS" \
  --explore-timeout-s "$TIMEOUT_S" \
  --headless \
  --no-rerun \
  --export "$OUT/episode" \
  >"$OUT/lazy_graph.log" 2>&1
RC=$?
set -e

{
  echo "[lazy-graph-robocasa] lazy-graph exit=$RC"
  if [[ -d "$OUT/episode" ]]; then
    echo "[lazy-graph-robocasa] export:"
    ls -la "$OUT/episode" | head -40
  else
    echo "[lazy-graph-robocasa] no episode export dir"
  fi
  # Quick signal: object nodes vs viewpoints (LazyGraph should have sparse objects).
  python3 - <<PY
from pathlib import Path
import json
out = Path("$OUT")
graph = out / "episode" / "graph.json"
if not graph.is_file():
    # common alternate names
    cands = list((out / "episode").glob("**/*graph*.json")) if (out / "episode").is_dir() else []
    print("graph.json missing; candidates:", [str(p) for p in cands[:8]])
else:
    d = json.loads(graph.read_text())
    nodes = d.get("nodes") or d.get("graph_nodes") or []
    n = len(nodes)
    print(f"graph nodes={n}")
PY
} | tee -a "$OUT/smoke.log"

if [[ "$RC" -eq 0 || "$RC" -eq 124 ]]; then
  # 124 = timeout (acceptable if export exists)
  if [[ -d "$OUT/episode" ]]; then
    status_close DONE "lazy-graph robocasa smoke ok rc=$RC" "inspect $OUT/episode ; uv run emet jobs logs \$EMET_JOB_ID --tail 40"
    exit 0
  fi
fi
status_close FAIL "lazy-graph rc=$RC" "tail $OUT/lazy_graph.log ; tail $OUT/server.log"
exit "$RC"
