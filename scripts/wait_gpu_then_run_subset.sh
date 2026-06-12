#!/usr/bin/env bash
# Wait for stable free VRAM, then run the HM-EQA subset; retry with --resume until all
# target questions are done (or max wall time). Robust to a bursty co-tenant GPU job that
# triggers the OOM-killer mid-episode: each attempt resumes from the existing results file.
set -u

NEED_MIB="${NEED_MIB:-16000}"    # require this much free VRAM before launching (margin for bursts)
STABLE="${STABLE:-3}"             # consecutive passing checks
INTERVAL="${INTERVAL:-30}"        # seconds between checks
MAX_WAIT_MIN="${MAX_WAIT_MIN:-360}"
TAG="${TAG:-iter11_explore}"
IDS="${IDS:-3,14,17,28,31,35,81,94}"
TIMEOUT="${TIMEOUT:-2400}"
RESULTS="$HOME/.cache/habitat_eqa/results/subset_${TAG}_qwen2_5_vl.jsonl"

n_target=$(awk -F, '{print NF}' <<<"$IDS")
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))

count_done() {
  [ -f "$RESULTS" ] || { echo 0; return; }
  uv run python - <<'PY' "$RESULTS"
import json, sys
from pathlib import Path
from emet.habitat.metrics import episode_run_completed
p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(sum(1 for r in rows if episode_run_completed(r)))
PY
}

wait_for_gpu() {
  local ok=0 free now
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    now=$(date +%H:%M:%S)
    if [ "${free:-0}" -ge "$NEED_MIB" ]; then ok=$((ok+1)); else ok=0; fi
    echo "[$now] free=${free}MiB need=${NEED_MIB} stable=${ok}/${STABLE} done=$(count_done)/${n_target}"
    [ "$ok" -ge "$STABLE" ] && return 0
    [ "$(date +%s)" -ge "$deadline" ] && return 2
    sleep "$INTERVAL"
  done
}

cd /home/cpaxton/src/home_robot_v4
while [ "$(count_done)" -lt "$n_target" ]; do
  if ! wait_for_gpu; then
    echo "gave up after ${MAX_WAIT_MIN} min; done=$(count_done)/${n_target}"
    exit 2
  fi
  echo "=== launching attempt (done=$(count_done)/${n_target}, TAG=$TAG, METHOD=${METHOD:-graph_eqa}) ==="
  TAG="$TAG" IDS="$IDS" TIMEOUT="$TIMEOUT" METHOD="${METHOD:-graph_eqa}" ./scripts/run_habitat_iter_subset.sh || true
  echo "=== attempt ended; done=$(count_done)/${n_target} ==="
  sleep 5
done
echo "ALL DONE: $(count_done)/${n_target} questions in $RESULTS"
