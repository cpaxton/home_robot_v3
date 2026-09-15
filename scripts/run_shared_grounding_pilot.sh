#!/usr/bin/env bash
# Bounded diagnostic battery, not a sweep or a task-success aggregator.
# Run from a frozen checkout through emet jobs --cpu-safe --gpu-exclusive.
# Required: OUT and executables for the selected phase. Optional SAM2_SOURCE for environments
# without an installed SAM2 package. See docs/experiments/shared_grounding_pilot.md.
# SIM_AGENT_CONFIG, SIM_CONFIG, SIM_COMMAND and SIM_EVAL_CONFIG select explicit simulator cases only;
# Habitat/EQA rows and the default simulator control remain unchanged.
set -euo pipefail
: "${OUT:?fresh output directory required}"
PHASE="${PHASE:-all}"
case "$PHASE" in
    all|habitat|eqa) : "${HABITAT_BIN:?emet-habitat executable required}" ;;
    sim|manipulation) ;;
    *) echo "Unknown PHASE: $PHASE (all, habitat, eqa, sim, manipulation)" >&2; exit 2 ;;
esac
if [[ "$PHASE" == all || "$PHASE" == sim || "$PHASE" == manipulation ]]; then
    : "${AGENT_PY:?shared agent Python executable required}"
    SIM_AGENT_CONFIG="${SIM_AGENT_CONFIG:-$PWD/configs/emet/query_detector_segmented_pilot.yaml}"
    SIM_CONFIG="${SIM_CONFIG:-$PWD/configs/sim/default_table_stretch.yaml}"
    SIM_EVAL_CONFIG="${SIM_EVAL_CONFIG:-$PWD/configs/benchmarks/tabletop_physical_eval.json}"
    SIM_COMMAND="${SIM_COMMAND:-Use pick_place to put the red cylinder on the blue cube. Report any failure; do not use oracle scene tasks or plans.}"
    for config in "$SIM_AGENT_CONFIG" "$SIM_CONFIG" "$SIM_EVAL_CONFIG"; do
        if [[ ! -f "$config" ]]; then
            echo "Missing simulator configuration: $config" >&2
            exit 2
        fi
    done
fi
if [[ -e "$OUT" ]]; then
    echo "Refusing to reuse output directory: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EMET_ALLOW_SDPA_ATTN=1 MUJOCO_GL=egl
export PYTHONPATH="$PWD/src:$PWD/packages/emet_habitat${SAM2_SOURCE:+:$SAM2_SOURCE}"
unset EMET_VL_ENDPOINT EMET_SIM_NAV_TELEPORT
# The MolmoSpaces server has a separate legacy teleport default. Physical
# acceptance must use wheel drive even when the caller enables that shortcut.
export EMET_MOLMOSPACES_NAV_TELEPORT=0
git rev-parse HEAD > "$OUT/source.txt"
git diff --exit-code
git diff --cached --exit-code
sha256sum configs/emet/query*pilot.yaml > "$OUT/config_sha256.txt"
cp configs/emet/query*pilot.yaml "$OUT/"
cp "${BASH_SOURCE[0]}" "$OUT/driver.sh"
printf '%s\n' "$PHASE" > "$OUT/phase.txt"
printf 'case\texit_code\tseconds\n' > "$OUT/process_status.tsv"
overall=0
run_case() {
    local name="$1" limit="$2" started=$SECONDS rc=0
    shift 2
    mkdir -p "$OUT/$name"
    printf '%s\n' "$EMET_CONFIG" > "$OUT/$name/config_path.txt"
    printf '%q ' "$@" > "$OUT/$name/command.txt"
    printf '\n' >> "$OUT/$name/command.txt"
    echo "Starting $name ($(date -Is))"
    EMET_EQA_EPISODE_DIR="$OUT/$name/evidence" \
        timeout --signal=TERM --kill-after=20s "$limit" "$@" \
        > "$OUT/$name/process.log" 2>&1 || rc=$?
    printf '%s\t%d\t%d\n' "$name" "$rc" "$((SECONDS-started))" | tee -a "$OUT/process_status.tsv"
    if [[ "$rc" != 0 ]]; then overall=1; fi
    # Stop after timeouts: inspect child-process cleanup before another heavy job.
    if [[ "$rc" == 124 || "$rc" == 137 ]]; then exit "$rc"; fi
}
if [[ "$PHASE" == all || "$PHASE" == habitat || "$PHASE" == eqa ]]; then
    habitat_env="$(dirname "$(dirname "$HABITAT_BIN")")"
    LD_LIBRARY_PATH="$habitat_env/lib:${LD_LIBRARY_PATH:-}" \
        "$habitat_env/bin/python" scripts/check_habitat_runtime.py > "$OUT/habitat_render_preflight.log" 2>&1
    LD_LIBRARY_PATH="$habitat_env/lib:${LD_LIBRARY_PATH:-}" \
        "$habitat_env/bin/python" scripts/check_sam2_runtime.py > "$OUT/habitat_preflight.log" 2>&1
    for variant in hybrid qwen_box; do
        if [[ "$variant" == hybrid ]]; then
            export EMET_CONFIG="$PWD/configs/emet/query_detector_segmented_pilot.yaml"
        else
            export EMET_CONFIG="$PWD/configs/emet/query_segmented_support_pilot.yaml"
        fi
        for scene in 00006 00025; do
            if [[ "$PHASE" == eqa ]]; then break; fi
            name="${variant}_ovmm_${scene}"
            run_case "$name" 600s "$HABITAT_BIN" run-ovmm-find-episode \
                --episodes configs/ovmm/habitat_find_phase_nearest_v2.yaml \
                --episode-id "hm3d_lamp_bed_${scene}_nearest_v2" \
                --backend lazy_graph --query-driven-memory --seed 0 --agentic-find \
                --agentic-max-rounds 12 --agentic-max-nav-steps 8 --output "$OUT/$name/result.json"
        done
        for q in 15 16 25; do
            name="${variant}_eqa_${q}"
            run_case "$name" 600s "$HABITAT_BIN" run-episode --question-id "$q" \
                --method lazy_graph --query-driven-memory --max-planning-steps 20 \
                --max-movement-step 10 --no-hm3d-semantics --no-enrich-labels \
                --export-map --export-video --debug-run-tag "$(basename "$OUT")-$name" \
                --output "$OUT/$name/result.jsonl"
        done
    done
fi
if [[ "$PHASE" == all || "$PHASE" == sim || "$PHASE" == manipulation ]]; then
    "$AGENT_PY" scripts/check_sam2_runtime.py > "$OUT/agent_preflight.log" 2>&1
    export EMET_CONFIG="$SIM_AGENT_CONFIG"
    sha256sum "$EMET_CONFIG" "$SIM_CONFIG" "$SIM_EVAL_CONFIG" > "$OUT/sim_config_sha256.txt"
    cp "$EMET_CONFIG" "$OUT/sim_agent_config.yaml"
    export EMET_FORCE_HEAD_SWEEP=1
    agent=("$AGENT_PY" -m emet.app.run_agent --config "$EMET_CONFIG"
        --memory-backend lazy_graph --robot stretch --start-sim
        --sim-config "$SIM_CONFIG" --headless --no-discord
    --sim-show-subprocess-output --llm qwen3-vl-eqa --eqa --debug-tools)
    if [[ -n "${SIM_SEED:-}" ]]; then agent+=(--sim-seed "$SIM_SEED"); fi
    if [[ "$PHASE" != manipulation ]]; then
        run_case hybrid_absent_find 360s "${agent[@]}" \
            -c 'Use find_objects once to locate a yellow banana. Do not pick or place anything. Report failure if it cannot be located.'
    fi
    export EMET_SIM_EVAL_CONFIG="$SIM_EVAL_CONFIG"
    export EMET_SIM_EVAL_TRACE="$OUT/hybrid_learned_pick_place/physical_trace.jsonl"
    cp "$EMET_SIM_EVAL_CONFIG" "$OUT/physical_eval_config.json"
    cp "$SIM_CONFIG" "$OUT/sim_config.yaml"
    run_case hybrid_learned_pick_place 600s "${agent[@]}" --visual-servo \
        -c "$SIM_COMMAND"
    # Process completion is not task success. Private GT stays on disk and is
    # scored only after the agent exits; no evaluator labels enter observations.
    "$AGENT_PY" -m emet.eval.manipulation_trace "$EMET_SIM_EVAL_TRACE" \
        --output "$OUT/hybrid_learned_pick_place/physical_result.json" || exit 1
fi
exit "$overall"
