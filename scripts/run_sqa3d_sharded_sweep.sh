#!/usr/bin/env bash
# Multi-GPU SQA3D real-VLM sweep: shard question range across GPUs, merge JSONL, aggregate.
#
# Example (4 GPUs, full val dynagraph):
#   ./scripts/run_sqa3d_sharded_sweep.sh --split val --method dynagraph --all --gpus 0,1,2,3
#
# See docs/sqa3d_compute.md and docs/paper_benchmarks.md (Large paper eval queue).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL=egl

SPLIT="val"
METHOD="dynagraph"
Q_START=0
Q_END=30
RUN_ALL=0
GPUS="0"
OUTPUT_DIR="${EMET_SQA3D_OUTPUT:-$HOME/runs/emet/sqa3d}"
REPLAY_MODE="auto"
ISOLATE=1
RESUME=1
DOWNLOAD=0
WITH_SENS=0
LOG_DIR="${EMET_LARGE_EVAL_LOG_DIR:-$HOME/runs/emet/large_eval}/sqa3d_shards"

usage() {
    cat <<'EOF'
Usage: ./scripts/run_sqa3d_sharded_sweep.sh [OPTIONS]

Shard a real-VLM SQA3D sweep across one or more GPUs (linear speedup when each GPU is exclusive).

Options:
  --split train|val|test       (default: val)
  --method dynamem|dynagraph   (default: dynagraph)
  --question-start N           (default: 0)
  --question-end N             (default: 30; ignored with --all)
  --all                        Full split size
  --gpus 0,1,2,3               Comma-separated CUDA device ids (default: 0)
  --output-dir PATH
  --replay-mode auto|sens|mesh (default: auto)
  --no-isolate-episodes        Faster in-process batch (OOM risk on long runs)
  --no-resume
  --download                   Download ScanNet for slice before sweep
  --with-sens                  With --download, fetch .sens too
  --log-dir PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --question-start) Q_START="$2"; shift 2 ;;
        --question-end) Q_END="$2"; shift 2 ;;
        --all) RUN_ALL=1; shift ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --replay-mode) REPLAY_MODE="$2"; shift 2 ;;
        --no-isolate-episodes) ISOLATE=0; shift ;;
        --no-resume) RESUME=0; shift ;;
        --download) DOWNLOAD=1; shift ;;
        --with-sens) WITH_SENS=1; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

IFS=',' read -r -a GPU_ARR <<<"$GPUS"
NGPU="${#GPU_ARR[@]}"
if [[ "$NGPU" -lt 1 ]]; then
    echo "ERROR: need at least one GPU id in --gpus" >&2
    exit 1
fi

if [[ "$RUN_ALL" -eq 1 ]]; then
    Q_END="$(uv run python -c "from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions; print(len(load_sqa3d_questions('$SPLIT')))")"
fi

TOTAL=$((Q_END - Q_START))
if [[ "$TOTAL" -le 0 ]]; then
    echo "ERROR: empty question range [$Q_START, $Q_END)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Build shard ranges (last shard takes remainder).
SHARD_SPECS=()
for ((i = 0; i < NGPU; i++)); do
    s=$((Q_START + (TOTAL * i) / NGPU))
    e=$((Q_START + (TOTAL * (i + 1)) / NGPU))
    if [[ "$e" -gt "$s" ]]; then
        SHARD_SPECS+=("${GPU_ARR[$i]}:${s}:${e}")
    fi
done

echo "SQA3D sharded sweep: split=$SPLIT method=$METHOD range=[$Q_START,$Q_END) shards=${#SHARD_SPECS[@]} output=$OUTPUT_DIR"

PIDS=()
for spec in "${SHARD_SPECS[@]}"; do
    gpu="${spec%%:*}"
    rest="${spec#*:}"
    s="${rest%%:*}"
    e="${rest##*:}"
    tag="${METHOD}_${SPLIT}_q${s}-${e}"
    log="$LOG_DIR/${tag}_gpu${gpu}.log"
    echo "  GPU $gpu → questions [$s, $e) log=$log"

    isolate_flag=(--isolate-episodes)
    [[ "$ISOLATE" -eq 0 ]] && isolate_flag=(--no-isolate-episodes)
    resume_flag=(--resume)
    [[ "$RESUME" -eq 0 ]] && resume_flag=(--no-resume)
    dl_flag=(--no-download)
    [[ "$DOWNLOAD" -eq 1 ]] && dl_flag=(--download)
    sens_flag=()
    [[ "$WITH_SENS" -eq 1 ]] && sens_flag=(--with-sens)

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        uv run emet sqa3d run-real-sweep \
            --split "$SPLIT" \
            --method "$METHOD" \
            --question-start "$s" \
            --question-end "$e" \
            --replay-mode "$REPLAY_MODE" \
            --output-dir "$OUTPUT_DIR" \
            "${isolate_flag[@]}" \
            "${resume_flag[@]}" \
            "${dl_flag[@]}" \
            "${sens_flag[@]}"
    ) >"$log" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done
if [[ "$FAIL" -ne 0 ]]; then
    echo "ERROR: one or more shards failed (see $LOG_DIR)" >&2
    exit 1
fi

JSONLS=()
for spec in "${SHARD_SPECS[@]}"; do
    rest="${spec#*:}"
    s="${rest%%:*}"
    e="${rest##*:}"
    j="$OUTPUT_DIR/${METHOD}_${SPLIT}_q${s}-${e}.jsonl"
    if [[ -f "$j" ]]; then
        JSONLS+=("$j")
    else
        echo "WARN: missing expected JSONL $j" >&2
    fi
done

if [[ ${#JSONLS[@]} -eq 0 ]]; then
    echo "ERROR: no shard JSONL outputs found" >&2
    exit 1
fi

MERGED_TAG="${METHOD}_${SPLIT}_q${Q_START}-${Q_END}_merged"
uv run python scripts/aggregate_sqa3d_sweep.py \
    "${JSONLS[@]}" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --csv-name "${MERGED_TAG}.csv" \
    --json-name "${MERGED_TAG}.json"

echo "Done. Shards: ${JSONLS[*]}"
echo "Aggregate: $OUTPUT_DIR/${MERGED_TAG}.csv"
