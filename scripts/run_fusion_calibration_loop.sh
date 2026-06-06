#!/usr/bin/env bash
# Record calibration JSONL via dynagraph + tune graph_object_fusion vs live sim GT.
# Usage: ./scripts/run_fusion_calibration_loop.sh [innate_mars|stretch|all]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROBOT_ARG="${1:-all}"
export EMET_ZMQ_STARTUP_TIMEOUT="${EMET_ZMQ_STARTUP_TIMEOUT:-120}"
export EMET_STRETCH_GENERIC_ZMQ="${EMET_STRETCH_GENERIC_ZMQ:-1}"
BASE="/tmp/emet_fusion_tune"
SEED=0
LAYOUT=1
CALIBRATION_STEPS="${CALIBRATION_STEPS:-48}"
DYNAGRAPH_TIMEOUT="${DYNAGRAPH_TIMEOUT:-600}"

robots=()
case "$ROBOT_ARG" in
  innate_mars | stretch) robots=("$ROBOT_ARG") ;;
  all) robots=(innate_mars stretch) ;;
  *) echo "Usage: $0 [innate_mars|stretch|all]" >&2; exit 2 ;;
esac

wait_for_server() {
  local log="$1"
  for _ in $(seq 1 120); do
    if grep -qE "Server running|Done setting up connections|RobosuiteZmqServer started" "$log" 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Server did not become ready (see $log)" >&2
  return 1
}

count_detection_frames() {
  local frames="$1"
  python3 - "$frames" <<'PY'
import json, sys
path = sys.argv[1]
n = 0
with open(path) as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("detections"):
            n += 1
print(n)
PY
}

run_robot() {
  local robot="$1"
  local out="$BASE/$robot"
  mkdir -p "$out"
  local dynav="dynav_config.yaml"
  if [[ "$robot" == "innate_mars" ]]; then
    dynav="dynav_innate_mars.yaml"
  fi

  local server_log="$out/server.log"
  rm -f "$out/frames.jsonl"
  echo "=== $robot: free ZMQ ports ==="
  uv run emet kill-mujoco-server --all 2>/dev/null || true
  sleep 2
  echo "=== $robot: serve mujoco (background) ==="
  uv run emet serve mujoco --robot "$robot" --use-robocasa --headless --seed "$SEED" \
    >"$server_log" 2>&1 &
  local spid=$!
  cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
  trap cleanup EXIT

  wait_for_server "$server_log"
  sleep 3

  echo "=== $robot: GT from live sim_object_placements ==="
  uv run python scripts/fetch_sim_gt_from_server.py \
    --robot "$robot" \
    --seed "$SEED" \
    --layout "$LAYOUT" \
    -o "$out/gt.json"

  echo "=== $robot: dynagraph calibration export (instance-only, no explore) ==="
  set +e
  timeout "$DYNAGRAPH_TIMEOUT" uv run emet run dynagraph \
    --robot "$robot" \
    --dynav-config "$dynav" \
    --export "$out/cal" \
    --calibration-export "$out/frames.jsonl" \
    --calibration-steps "$CALIBRATION_STEPS" \
    --no-sensor-perception \
    --cpu-only \
    --no-rerun \
    -N \
    2>&1 | tee "$out/dynagraph.log"
  dg_exit=$?
  set -e
  if [[ "$dg_exit" -ne 0 ]]; then
    echo "Note: dynagraph exited $dg_exit (often harmless CUDA teardown if export finished)" >&2
  fi

  trap - EXIT
  cleanup

  local nframes ndets
  nframes=$(wc -l <"$out/frames.jsonl" 2>/dev/null || echo 0)
  ndets=$(count_detection_frames "$out/frames.jsonl" 2>/dev/null || echo 0)
  echo "=== $robot: $nframes jsonl lines, $ndets with detections ==="
  if [[ "$ndets" -lt 3 ]]; then
    echo "Too few detection frames for $robot (need >= 3)" >&2
    exit 1
  fi

  echo "=== $robot: eval-calibration (raw detections) ==="
  uv run emet eval-calibration \
    --gt "$out/gt.json" \
    --frames "$out/frames.jsonl" \
    --report "$out/calibration_eval.json"

  echo "=== $robot: tune-graph-fusion ==="
  uv run emet tune-graph-fusion \
    --gt "$out/gt.json" \
    --frames "$out/frames.jsonl" \
    --report "$out/fusion_tune_report.json" \
    --write-config "$out/graph_object_fusion_tuned.yaml" \
    --min-recall 0.2

  python3 - "$out/calibration_eval.json" <<'PY'
import json, sys
path = sys.argv[1]
raw = json.load(open(path))["raw"]
spatial = float(raw.get("spatial_recall", 0))
label = float(raw.get("label_recall", 0))
print(f"spatial_recall={spatial:.3f} label_recall={label:.3f}")
if spatial < 0.6:
    print(f"WARN: spatial_recall {spatial:.3f} < 0.6 (viewpoint or detection coverage)", file=sys.stderr)
PY

  echo "=== $robot: eval-calibration (after tuned fusion) ==="
  uv run emet eval-calibration \
    --gt "$out/gt.json" \
    --frames "$out/frames.jsonl" \
    --fusion-config "$out/graph_object_fusion_tuned.yaml" \
    --report "$out/calibration_eval_fused.json"

  local cfg_repo="$ROOT/src/emet/config/agents/graph_object_fusion_${robot}.yaml"
  cp "$out/graph_object_fusion_tuned.yaml" "$cfg_repo"
  echo "=== $robot: copied tuned config -> $cfg_repo ==="
}

for r in "${robots[@]}"; do
  run_robot "$r"
done

echo "All robots finished under $BASE"
