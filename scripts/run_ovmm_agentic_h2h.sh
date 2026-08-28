#!/usr/bin/env bash
# OVMM find head-to-head: two arms on identical episodes, one deliberate axis.
#
# Mirrors the HM-EQA A/B gate structure (scripts/run_hmeqa_agentic_h2h.sh):
# frozen manifest + resume, per-episode crash/cooldown hygiene, per-arm summary.
# Use it to validate an OVMM change the same way we validate HM-EQA arms.
#
# Default axis = unified agentic mapping (this PR):
#   arm "rotate"  — default mapping (episode explore_steps / scene cache),
#                   rotate-only for kitchen legs.
#   arm "unified" — live agentic explore mapping (--explore-steps 8
#                   --no-scene-cache), the new coverage loop.
#
# Usage (prefer emet jobs):
#   uv run emet jobs run --name ovmm-ab --need-mib 12000 --gpu-exclusive -- \
    #     ./scripts/run_ovmm_agentic_h2h.sh
#
# Env:
#   ARMS=rotate,unified   arms to run (comma-separated)
#   EPISODES=...          comma-separated episode ids (default: rby1 S0 + kitchen)
#   RESUME=1              skip validated COMPLETE markers; rebuild aggregates
#   EXPLORE_STEPS=8       explore_steps for the unified arm
#   BACKEND=dynagraph     memory backend for both arms
#   ROUNDS=6              --agentic-max-rounds
#   CRASH_RETRIES=1       retries of one episode after a native/EGL failure
#   COOLDOWN_SEC=20       sleep between episodes (reduces stacked teardown pressure)
#   OUT_DIR=...           output root (default ~/runs/emet/ovmm_agentic_h2h/<RUN_ID>)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/status_log.sh
source "$ROOT/scripts/status_log.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT_DIR:-$HOME/runs/emet/ovmm_agentic_h2h/${RUN_ID}}"
mkdir -p "$OUT"
ARMS="${ARMS:-rotate,unified}"
EPISODES="${EPISODES:-default_table_rby1_s0_distinct_recep,robocasa_rby1_pp_s1}"
RESUME="${RESUME:-0}"
EXPLORE_STEPS="${EXPLORE_STEPS:-8}"
BACKEND="${BACKEND:-dynagraph}"
ROUNDS="${ROUNDS:-6}"
CRASH_RETRIES="${CRASH_RETRIES:-1}"
COOLDOWN_SEC="${COOLDOWN_SEC:-20}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a EP_LIST <<< "$EPISODES"

# Frozen manifest (commit + arms + episodes + key env). Resume refuses to
# silently change an arm axis on a partially-complete run.
MANIFEST="$OUT/run_manifest.json"
COMMIT="$(git rev-parse HEAD)"
if [[ -f "$MANIFEST" && "$RESUME" != "1" ]]; then
    log "ERROR: $MANIFEST exists; set RESUME=1 to continue it"
    exit 2
fi
if [[ -f "$MANIFEST" ]]; then
    PREV_COMMIT="$(uv run python -c "import json;print(json.load(open('$MANIFEST'))['commit'])")"
    if [[ "$PREV_COMMIT" != "$COMMIT" ]]; then
        log "ERROR: resume would mix commits ($PREV_COMMIT != $COMMIT); aborting"
        exit 2
    fi
else
    uv run python - "$MANIFEST" "$ARMS" "$EPISODES" "$BACKEND" "$ROUNDS" "$EXPLORE_STEPS" "$COMMIT" <<'PY'
import json, sys
manifest = {
    "run_id": sys.argv[1].split("/")[-1],
    "commit": sys.argv[7],
    "arms": sys.argv[2].split(","),
    "episodes": sys.argv[3].split(","),
    "backend": sys.argv[4],
    "rounds": int(sys.argv[5]),
    "explore_steps": int(sys.argv[6]),
    "axis": "mapping: rotate/rotate+cache vs live agentic explore (--explore-steps)",
}
json.dump(manifest, open(sys.argv[1], "w"), indent=2)
PY
    log "manifest written: $MANIFEST"
fi

arm_args() {
    # Per-arm emet ovmm find args (one deliberate axis).
    local arm="$1"
    local ep="$2"
    echo --episodes configs/ovmm/find_phase_episodes.yaml \
        --backend "$BACKEND" \
        --agentic-max-rounds "$ROUNDS" \
        --mapping-rotate-steps 4 \
        --episode-id "$ep"
}

status_open "$OUT" "ovmm-ab"

arm_results=()
for arm in "${ARM_LIST[@]}"; do
    arm_out="$OUT/$arm"
    mkdir -p "$arm_out"
    arm_ok=0
    for ep in "${EP_LIST[@]}"; do
        marker="$arm_out/${ep}.complete"
        if [[ -f "$marker" && "$RESUME" == "1" ]]; then
            log "[$arm] $ep already COMPLETE; skipping"
            arm_ok=$((arm_ok + 1))
            continue
        fi
        ep_out="$arm_out/$ep"
        mkdir -p "$ep_out"
        rm -f "$marker"
        args=("$(arm_args "$arm" "$ep")")
        if [[ "$arm" == "unified" ]]; then
            args+=(--explore-steps "$EXPLORE_STEPS" --no-scene-cache)
        fi
        rc=0
        for attempt in $(seq 0 "$CRASH_RETRIES"); do
            log "[$arm] $ep attempt=$attempt: emet ovmm find ${args[*]}"
            set +e
            uv run emet ovmm find "${args[@]}" --output-dir "$ep_out"
            rc=$?
            set -e
            if [[ "$rc" -eq 0 ]]; then
                break
            fi
            log "[$arm] $ep attempt=$attempt rc=$rc (retry policy)"
            sleep "$COOLDOWN_SEC"
        done
        if [[ "$rc" -ne 0 ]]; then
            log "[$arm] $ep FAILED (rc=$rc)"
            continue
        fi
        echo "complete" > "$marker"
        arm_ok=$((arm_ok + 1))
        sleep "$COOLDOWN_SEC"
    done
    arm_results+=("$arm=$arm_ok/${#EP_LIST[@]}")
done

echo "=== OVMM A/B summary ===" | tee -a "$OUT/orchestrator.log"
for arm in "${ARM_LIST[@]}"; do
    n_done=""
    for res in "${arm_results[@]}"; do
        case "$res" in
            "${arm}="*) n_done="${res#*=}" ;;
        esac
    done
    echo "arm=$arm episodes_complete=${n_done:-0/${#EP_LIST[@]}}"
done

# Per-arm FindObj/FindRec from per-episode JSON metrics.
uv run python - "$OUT" "${ARM_LIST[@]}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
arms = sys.argv[2:]
print("OVMM find A/B")
for arm in arms:
    rows = []
    for jf in sorted((out / arm).glob("*/*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if "find_object_success" not in d:
            continue
        rows.append(
            {
                "episode": d.get("episode_id"),
                "find_object": d.get("find_object_success"),
                "find_recep": d.get("find_recep_success"),
                "partial": d.get("find_partial_success"),
                "obj_err": d.get("localization_err_obj_m"),
                "obj_source": d.get("obj_localize_source"),
                "mapping_n_explore": d.get("mapping_n_explore"),
                "wall_s": d.get("episode_wall_s"),
            }
        )
    n = len(rows)
    n_obj = sum(1 for r in rows if r["find_object"])
    n_rec = sum(1 for r in rows if r["find_recep"])
    print(f"arm={arm} episodes={n} FindObj={n_obj}/{n} FindRec={n_rec}/{n}")
    for r in rows:
        print("  ", r)
    summary = {"arm": arm, "episodes": rows, "find_object": n_obj, "find_recep": n_rec, "n": n}
    (out / f"{arm}_summary.json").write_text(json.dumps(summary, indent=2))
PY

status_close ok "ovmm AB arms: ${arm_results[*]}" "OUT=$OUT"
