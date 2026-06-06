#!/usr/bin/env bash
# Fusion on/off A/B: identical dynagraph explore, compare eval-dynagraph metrics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROBOT="${1:-innate_mars}"
SEED="${2:-0}"
ITERS="${3:-20}"
BASE="/tmp/dynagraph_fusion_ab/${ROBOT}_seed${SEED}"
export EMET_ZMQ_STARTUP_TIMEOUT="${EMET_ZMQ_STARTUP_TIMEOUT:-120}"
export EMET_SIM_NAV_TELEPORT=1

run_arm() {
  local tag="$1"
  local fusion_yaml="$2"
  local out="$BASE/$tag"
  mkdir -p "$out"
  uv run emet kill-mujoco-server --all 2>/dev/null || true
  sleep 2
  uv run emet serve mujoco --use-robocasa --robot "$ROBOT" --headless --seed "$SEED" \
    >"$out/server.log" 2>&1 &
  local spid=$!
  trap "kill $spid 2>/dev/null || true" EXIT
  for _ in $(seq 1 90); do
    if grep -qE "Server running|RobosuiteZmqServer started" "$out/server.log" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  sleep 8
  uv run emet run dynagraph \
    --robot "$ROBOT" \
    --dynav-config dynav_config.yaml \
    --graph-fusion-config "$fusion_yaml" \
    --explore-loop --explore-max-iters "$ITERS" \
    --no-rerun --cpu-only \
    --export "$out/episode" 2>&1 | tee "$out/dynagraph.log" || true
  kill "$spid" 2>/dev/null || true
  trap - EXIT
  uv run emet eval-dynagraph --episode "$out/episode" -o "$out/dynagraph_eval.json"
}

mkdir -p "$BASE"
echo "=== Fusion ON (default) ==="
run_arm fusion_on src/emet/config/agents/default_graph_object_fusion.yaml
echo "=== Fusion OFF ==="
run_arm fusion_off src/emet/config/agents/graph_object_fusion_ab_off.yaml

python3 - "$BASE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
def load(tag):
    return json.loads((base / tag / "dynagraph_eval.json").read_text())
on, off = load("fusion_on"), load("fusion_off")
gn = lambda m: m.get("graph", {}).get("node_count", 0)
sr = lambda m: (m.get("fusion", {}).get("raw") or {}).get("spatial_recall")
print(f"node_count: on={gn(on)} off={gn(off)} ratio={gn(on)/max(1,gn(off)):.2f}")
print(f"spatial_recall: on={sr(on)} off={sr(off)}")
report = {"fusion_on": on, "fusion_off": off}
(base / "fusion_ab_report.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo "Wrote $BASE/fusion_ab_report.json"
