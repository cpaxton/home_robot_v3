#!/usr/bin/env bash
# Shared GPU preflight helpers for overnight eval / smoke scripts.
#
# Usage:
#   source "$(dirname "$0")/gpu_preflight.sh"
#   emet_export_pytorch_alloc
#   emet_kill_stale_eval_processes
#   emet_wait_gpu_stable 12000
#
# One-shot check (exit 1 if insufficient VRAM):
#   NEED_MIB=14000 ./scripts/gpu_preflight.sh --check
#
# Env (defaults):
#   NEED_MIB=12000          Minimum free VRAM (MiB)
#   GPU_STABLE_CHECKS=3     Consecutive passing nvidia-smi reads
#   GPU_WAIT_INTERVAL=30    Seconds between checks
#   GPU_SETTLE_SEC=15       Sleep after killing stale jobs
#   GPU_KILL_STALE=1        Set 0 to skip process cleanup
set -u

EMET_GPU_PREFLIGHT_SOURCED=1

EMET_NEED_MIB="${NEED_MIB:-12000}"
EMET_GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
EMET_GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"
EMET_GPU_SETTLE_SEC="${GPU_SETTLE_SEC:-15}"

# Paths that load MuJoCo/Habitat sim even with ``pytest -m "not sim"``.
EMET_PYTEST_NO_SIM_IGNORES=(
  src/test/simulation
  src/test/molmospaces
  src/test/memory/test_graph_eqa_default_scene_sim.py
  src/test/memory/test_dynagraph_ground_truth_sim.py
  src/test/memory/test_dynagraph_ground_truth_molmospaces_sim.py
  src/test/agent/test_agent_find_object_sim.py
  src/test/agent/test_agent_memory_query_sim.py
  src/test/scene_graph/test_scene_graph_in_sim.py
  src/test/scene_graph/test_scene_graph_robocasa.py
  src/test/mapping/test_red_cylinder_in_sim.py
  src/test/mapping/test_innate_mars_da3_sim.py
  src/test/mapping/test_innate_mars_lingbot_sim.py
)

emet_export_pytorch_alloc() {
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
}

emet_gpu_free_mib() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' '
}

emet_gpu_log_compute_apps() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
  fi
}

emet_kill_stale_eval_processes() {
  local kill_gpu="${1:-1}"
  pkill -f 'emet\.simulation\.mujoco_server' 2>/dev/null || true
  pkill -f 'eval_dynamic_exploration\.py' 2>/dev/null || true
  pkill -f 'emet run dynagraph' 2>/dev/null || true
  pkill -f 'emet run graph-eqa' 2>/dev/null || true
  pkill -f 'emet run dynamem' 2>/dev/null || true
  pkill -f 'emet sqa3d run' 2>/dev/null || true
  pkill -f 'emet-habitat' 2>/dev/null || true
  pkill -f 'run_agent' 2>/dev/null || true
  sleep "$EMET_GPU_SETTLE_SEC"
  if [ "$kill_gpu" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    local pid cmd
    while read -r pid; do
      [ -z "$pid" ] && continue
      cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
      if [[ "$cmd" == *home_robot* ]] || [[ "$cmd" == *emet* ]]; then
        echo "[gpu] kill pid=$pid: ${cmd:0:100}"
        kill "$pid" 2>/dev/null || true
      fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    sleep 5
  fi
}

emet_wait_gpu_stable() {
  local need_mib="${1:-$EMET_NEED_MIB}"
  local ok=0 free
  while [ "$ok" -lt "$EMET_GPU_STABLE_CHECKS" ]; do
    free=$(emet_gpu_free_mib)
    if [ "${free:-0}" -ge "$need_mib" ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    echo "[gpu] free=${free}MiB need=${need_mib} stable=${ok}/${EMET_GPU_STABLE_CHECKS}"
    [ "$ok" -ge "$EMET_GPU_STABLE_CHECKS" ] && return 0
    sleep "$EMET_GPU_WAIT_INTERVAL"
  done
  return 1
}

emet_gpu_preflight_check() {
  local need_mib="${1:-$EMET_NEED_MIB}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found" >&2
    return 1
  fi
  local free total
  free=$(emet_gpu_free_mib)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "GPU: ${free} MiB free / ${total} MiB total (need >= ${need_mib} MiB)"
  if [ "${free:-0}" -lt "$need_mib" ]; then
    echo "ERROR: insufficient free GPU memory" >&2
    emet_gpu_log_compute_apps >&2
    return 1
  fi
  return 0
}

emet_pytest_no_sim_ignore_args() {
  local p
  for p in "${EMET_PYTEST_NO_SIM_IGNORES[@]}"; do
    printf '%s\n' "--ignore=$p"
  done
}

emet_gpu_between_steps() {
  local need_mib="${1:-$EMET_NEED_MIB}"
  if [ "${GPU_KILL_STALE:-1}" != "0" ]; then
    emet_kill_stale_eval_processes
  fi
  emet_wait_gpu_stable "$need_mib" || {
    echo "WARNING: GPU wait timed out (free < ${need_mib} MiB); continuing anyway" >&2
    return 0
  }
}

# Allow ``./scripts/gpu_preflight.sh --check`` without sourcing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  case "${1:-}" in
    --check)
      emet_gpu_preflight_check "${NEED_MIB:-12000}"
      ;;
    --kill-stale)
      emet_kill_stale_eval_processes
      ;;
    --wait)
      emet_wait_gpu_stable "${NEED_MIB:-12000}"
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      ;;
    *)
      echo "Usage: $0 --check | --kill-stale | --wait" >&2
      exit 1
      ;;
  esac
fi
