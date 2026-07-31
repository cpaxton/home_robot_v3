#!/usr/bin/env bash
# Sequential HM-EQA VLM comparison on Q0-19 (graph_eqa + frontier v2, int4).
# Requires .venv-habitat and an exclusive GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HAB="${ROOT}/.venv-habitat/bin/emet-habitat"
OUT="${HOME}/.cache/habitat_eqa/results"
mkdir -p "$OUT"

QSTART="${QSTART:-0}"
QEND="${QEND:-19}"
MAX_PLANNING="${MAX_PLANNING:-20}"
MAX_MOVEMENT="${MAX_MOVEMENT:-10}"
EXPECTED=$((QEND - QSTART + 1))
MASTER_LOG="${OUT}/vlm_sweep_q${QSTART}-${QEND}.log"

# name|family|hf_model_id (empty = family default / gemma VRAM auto-tier)
CONFIGS=(
  "gemma4_e4b_auto|gemma4|"
  "gemma4_e4b|gemma4|google/gemma-4-E4B-it"
  "gemma4_e2b|gemma4|google/gemma-4-e2b-it"
  "gemma3_4b|gemma4|google/gemma-3-4b-it"
  "qwen3_vl_4b|qwen3_vl|"
  "qwen25_vl_3b|qwen2_5_vl|"
)

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG"; }

log "VLM sweep Q${QSTART}-${QEND} planning=${MAX_PLANNING} movement=${MAX_MOVEMENT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | tee -a "$MASTER_LOG" || true

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r slug family hf_id <<<"$entry"
  TAG="vlm_sweep_${slug}_q${QSTART}-${QEND}"
  JSONL="${OUT}/${TAG}.jsonl"
  RUN_LOG="${OUT}/${TAG}.log"

  if [[ -f "$JSONL" ]]; then
    n=$(wc -l <"$JSONL" | tr -d ' ')
    if [[ "$n" -ge "$EXPECTED" ]]; then
      log "SKIP ${slug}: ${JSONL} already has ${n}/${EXPECTED} episodes"
      continue
    fi
    log "RESUME ${slug}: ${n}/${EXPECTED} episodes in ${JSONL}"
  else
    log "START ${slug} family=${family} hf_id=${hf_id:-auto}"
  fi

  extra=()
  if [[ -n "$hf_id" ]]; then
    extra+=(--eqa-hf-model-id "$hf_id")
  fi

  if ! "$HAB" run-batch \
    --method static_graph \
    --question-start "$QSTART" \
    --question-end "$QEND" \
    --paper-subset \
    --max-planning-steps "$MAX_PLANNING" \
    --max-movement-step "$MAX_MOVEMENT" \
    --eqa-vl-family "$family" \
    --device cuda \
    --resume \
    --output "$JSONL" \
    "${extra[@]}" \
    2>&1 | tee -a "$RUN_LOG"; then
    log "FAILED ${slug} (see ${RUN_LOG})"
    continue
  fi

  n=$(wc -l <"$JSONL" | tr -d ' ')
  log "DONE ${slug}: ${n}/${EXPECTED} -> ${JSONL}"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "$MASTER_LOG" || true
done

log "Sweep finished. Summarize with:"
log "  python3 ${ROOT}/scripts/summarize_vlm_sweep.py --q-start ${QSTART} --q-end ${QEND}"
