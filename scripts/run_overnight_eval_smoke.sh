#!/usr/bin/env bash
# Unified overnight eval smoke: HM-EQA + OVMM Habitat + SQA3D with diagnostics artifacts.
#
# Usage:
#   ./scripts/run_overnight_eval_smoke.sh
#   MOCK_LLM=1 ./scripts/run_overnight_eval_smoke.sh   # layout check without VLM
#   SKIP_SQA3D=1 ./scripts/run_overnight_eval_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-eval_smoke_$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-$RUN_ID}"
if [ "$TAG" != "$RUN_ID" ]; then
  echo "WARNING: TAG=$TAG != RUN_ID=$RUN_ID; figure pack uses RUN_ID. Set RUN_ID=$TAG or pass --artifact-tag $TAG." >&2
fi
NEED_MIB="${NEED_MIB:-14000}"
TIMEOUT_HMEQA="${TIMEOUT_HMEQA:-3600}"
TIMEOUT_OVMM="${TIMEOUT_OVMM:-2400}"
TIMEOUT_SQA3D="${TIMEOUT_SQA3D:-3600}"
MOCK_LLM="${MOCK_LLM:-0}"
SKIP_SQA3D="${SKIP_SQA3D:-0}"

HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT_HAB="${HOME}/.cache/habitat_eqa"
LOG_DIR="${OUT_HAB}/overnight/${RUN_ID}"
mkdir -p "$LOG_DIR" "${OUT_HAB}/results"

export EMET_EVAL_EXPORT_MAP=1
export EMET_EVAL_EXPORT_VIDEO=1
export EMET_EVAL_EXPORT_FRAMES=1

HMEQA_IDS="${HMEQA_IDS:-3,14,17}"
FAMILY="${FAMILY:-qwen3_vl}"
HF_ID="${HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"

wait_gpu() {
  local need="${1:-$NEED_MIB}" ok=0 free
  while [ "$ok" -lt 3 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
    if [ "${free:-0}" -ge "$need" ]; then ok=$((ok+1)); else ok=0; fi
    echo "[gpu] free=${free}MiB need=${need} stable=${ok}/3"
    [ "$ok" -ge 3 ] && return 0
    sleep 30
  done
  return 1
}

run_hmeqa_method() {
  local method="$1"
  local jsonl="${OUT_HAB}/results/${TAG}_hmeqa_${method}.jsonl"
  local log="${LOG_DIR}/hmeqa_${method}.log"
  echo "=== HM-EQA method=${method} ids=${HMEQA_IDS} ==="
  local mock_flag=()
  [ "$MOCK_LLM" = "1" ] && mock_flag=(--mock-llm)
  timeout "$TIMEOUT_HMEQA" "$HAB" run-batch \
    --method "$method" \
    --question-ids "$HMEQA_IDS" \
    --paper-subset \
    "${mock_flag[@]}" \
    --eqa-vl-family "$FAMILY" \
    --eqa-hf-model-id "$HF_ID" \
    --device cuda \
    --export-map \
    --export-video \
    --resume \
    --output "$jsonl" \
    2>&1 | tee "$log"
}

run_ovmm_backend() {
  local backend="$1"
  local out="${HOME}/runs/emet/ovmm_habitat/${TAG}_${backend}"
  mkdir -p "$out"
  local log="${LOG_DIR}/ovmm_${backend}.log"
  echo "=== OVMM Habitat backend=${backend} ==="
  timeout "$TIMEOUT_OVMM" "$HAB" run-ovmm-find-batch \
    --backend "$backend" \
    --output-dir "$out" \
    --run-tag "${TAG}_ovmm" \
    --export-map \
    --export-video \
    2>&1 | tee "$log"
}

run_sqa3d_method() {
  local method="$1"
  local out="${HOME}/runs/emet/sqa3d/${TAG}_${method}"
  mkdir -p "$out"
  local log="${LOG_DIR}/sqa3d_${method}.log"
  echo "=== SQA3D method=${method} val q0-2 ==="
  local extra=()
  [ "$MOCK_LLM" = "1" ] && extra=(--mock-llm)
  timeout "$TIMEOUT_SQA3D" uv run emet sqa3d run-real-sweep \
    --split val \
    --question-start 0 \
    --question-end 2 \
    --method "$method" \
    --replay-mode sens \
    --no-download \
    --export-root "${OUT_HAB}/episodes/${TAG}_sqa3d" \
    --output-dir "$out" \
    "${extra[@]}" \
    2>&1 | tee "$log" || true
}

echo "RUN_ID=$RUN_ID LOG_DIR=$LOG_DIR"

if [ ! -x "$HAB" ]; then
  echo "Missing $HAB — run ./scripts/install_habitat.sh"
  exit 1
fi

if [ "$MOCK_LLM" != "1" ]; then
  wait_gpu || { echo "GPU wait failed"; exit 2; }
fi

for m in graph_eqa dynagraph; do
  run_hmeqa_method "$m" || true
  [ "$MOCK_LLM" != "1" ] && wait_gpu || true
done

for b in dynamem graph_eqa dynagraph; do
  run_ovmm_backend "$b" || true
  [ "$MOCK_LLM" != "1" ] && wait_gpu || true
done

if [ "$SKIP_SQA3D" != "1" ]; then
  if uv run emet sqa3d verify 2>&1 | tee "${LOG_DIR}/sqa3d_verify.log"; then
    for m in dynagraph dynamem; do
      run_sqa3d_method "$m"
      [ "$MOCK_LLM" != "1" ] && wait_gpu || true
    done
  else
    echo "SKIP SQA3D (verify failed — see ${LOG_DIR}/sqa3d_verify.log)"
  fi
fi

FIG_DIR="${HOME}/runs/emet/eval_smoke/${RUN_ID}/figures"
FIG_ARGS=(--run-id "$RUN_ID" --output-dir "$FIG_DIR")
if [ "$TAG" != "$RUN_ID" ]; then
  FIG_ARGS+=(--artifact-tag "$TAG")
fi
uv run python scripts/build_eval_figure_pack.py "${FIG_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/figure_pack.log" || true

echo "Done. Logs: $LOG_DIR  Figures: $FIG_DIR"
echo "Summarize: uv run python scripts/build_eval_figure_pack.py --run-id $RUN_ID --summary-only"
