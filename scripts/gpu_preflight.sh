#!/usr/bin/env bash
# Shared GPU preflight helpers for overnight eval / smoke scripts.
#
# Preferred interactive CLI (canonical implementation):
#   uv run emet eval status
#   uv run emet eval check --need-mib 12000
#   uv run emet eval wait --need-mib 12000
#   uv run emet eval kill-stale
#
# Usage from bash orchestrators:
#   source "$(dirname "$0")/gpu_preflight.sh"
#   emet_export_pytorch_alloc
#   emet_kill_stale_eval_processes
#   emet_wait_gpu_stable 12000
#
# One-shot (delegates to ``emet eval`` when available):
#   NEED_MIB=14000 ./scripts/gpu_preflight.sh --check
#
# Env (defaults):
#   NEED_MIB=12000          Minimum free VRAM (MiB)
#   GPU_STABLE_CHECKS=3     Consecutive passing nvidia-smi reads
#   GPU_WAIT_INTERVAL=30    Seconds between checks
#   GPU_SETTLE_SEC=15       Sleep after killing stale jobs
#   GPU_KILL_STALE=1        Set 0 to skip process cleanup
#   EMET_GPU_PROTECT_PIDS   Extra PIDs never killed by kill-stale
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
    src/test/motion/test_rby1_mujoco_arm_ik.py
)

_emet_gpu_preflight_root() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$here/.." && pwd
}

# Invoke canonical ``emet eval …`` (venv → PATH → uv run).
_emet_eval_cli() {
    local root
    root="$(_emet_gpu_preflight_root)"
    if [ -x "$root/.venv/bin/emet" ]; then
        "$root/.venv/bin/emet" eval "$@"
        return $?
    fi
    if command -v emet >/dev/null 2>&1; then
        emet eval "$@"
        return $?
    fi
    if command -v uv >/dev/null 2>&1; then
        (cd "$root" && uv run emet eval "$@")
        return $?
    fi
    echo "ERROR: emet not found (install with uv sync); cannot run: emet eval $*" >&2
    return 127
}

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
    local args=(kill-stale)
    if [ "$kill_gpu" = "0" ]; then
        args+=(--no-gpu)
    fi
    if [ -n "${EMET_GPU_SETTLE_SEC:-}" ]; then
        args+=(--settle-sec "$EMET_GPU_SETTLE_SEC")
    fi
    _emet_eval_cli "${args[@]}"
}

emet_wait_gpu_stable() {
    local need_mib="${1:-$EMET_NEED_MIB}"
    NEED_MIB="$need_mib" _emet_eval_cli wait --need-mib "$need_mib"
}

emet_gpu_preflight_check() {
    local need_mib="${1:-$EMET_NEED_MIB}"
    NEED_MIB="$need_mib" _emet_eval_cli check --need-mib "$need_mib"
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
            sed -n '2,28p' "$0"
            ;;
        *)
            echo "Usage: $0 --check | --kill-stale | --wait" >&2
            echo "Prefer: uv run emet eval {status,check,wait,kill-stale}" >&2
            exit 1
            ;;
    esac
fi
