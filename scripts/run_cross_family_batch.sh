#!/usr/bin/env bash
# Cross-family eval batch: HM-EQA + OVMM find (Robocasa / MolmoSpaces / table).
#
# Reuses the existing per-family runners (emet hmeqa h2h, emet ovmm find), one
# experiment per unit, and writes a unified results.csv + per-unit jsonl so dumb
# errors surface fast. Wrap with `emet jobs run` for GPU mutex + affinity:
#
#   uv run emet jobs run --name batch-42 --need-mib 12000 -- \
#     SEED=42 ./scripts/run_cross_family_batch.sh OUT_DIR
#
# Env:
#   SEED            random seed (default bash $RANDOM; fixed -> reproducible)
#   N_EXPERIMENTS   batch size (default 5)
#   HMEQA_IDS       comma list of HM-EQA qids to draw from (default bal-32)
#   EMET_SIM_NAV_TELEPORT  set to 1 for fast sim base nav (recommended)
#   SKIP_GPU_WAIT   set to 1 to skip gpu_preflight --wait
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/status_log.sh" 2>/dev/null || true

OUT="${1:?usage: run_cross_family_batch.sh OUT_DIR [SEED]}"
SEED="${2:-${SEED:-$RANDOM}}"
N="${N_EXPERIMENTS:-5}"
mkdir -p "$OUT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

# --- pools ---------------------------------------------------------------
HMEQA_POOL=(${HMEQA_IDS:-})
if [[ ${#HMEQA_POOL[@]} -eq 0 ]]; then
    HMEQA_POOL=(2 6 8 11 12 14 15 16 17 18 21 25 27 28 29 31 32 33 34 38 39 40 41 43 44 47 48 49 57 76 80 84)
fi
OVMM_POOL=(default_table_s0 default_table_s0_blue_cube default_table_s0_distinct_recep)
ROBOCASA_POOL=(robocasa_pp_s1 robocasa_pp_s1_explore5)
MOLMO_POOL=(molmo_ithor_s2_idx0 molmo_ithor_s2_idx0_explore15 molmo_ithor_s2_idx1 molmo_ithor_s2_idx2)

# --- deterministic pick: one per family, then fill randomly ---------------
POOLS_JSON="$(uv run python -c '
import json, sys
hmeqa = [int(x) for x in sys.argv[1].split()]
print(json.dumps({
    "hmeqa": hmeqa,
    "ovmm": sys.argv[2].split(),
    "robocasa": sys.argv[3].split(),
    "molmo": sys.argv[4].split(),
}))' "${HMEQA_POOL[*]}" "${OVMM_POOL[*]}" "${ROBOCASA_POOL[*]}" "${MOLMO_POOL[*]}")"
# shellcheck disable=SC2207
PICKED=()
mapfile -t PICKED < <(uv run python - "$SEED" "$N" "$POOLS_JSON" <<PY
import json, random, sys
seed, n = int(sys.argv[1]), int(sys.argv[2])
families = json.loads(sys.argv[3])
out = []
fam_order = list(families)
rng = random.Random(seed)
rng.shuffle(fam_order)
for fam in fam_order:
    if len(out) >= n:
        break
    if families[fam]:
        out.append((fam, rng.choice(families[fam])))
rest = [(fam, it) for fam, items in families.items() for it in items if (fam, it) not in out]
rng.shuffle(rest)
for fam, it in rest:
    if len(out) >= n:
        break
    out.append((fam, it))
for fam, item in out:
    print(f"{fam} {item}")
PY
)
log "seed=$SEED picked=${#PICKED[@]} experiments"
: > "$OUT/pick.txt"
for p in "${PICKED[@]}"; do
    log "  pick: $p"
    echo "$p" >> "$OUT/pick.txt"
done

# --- runners -------------------------------------------------------------
run_hmeqa() {
    local qid="$1" dir="$OUT/hmeqa_q${qid}"
    mkdir -p "$dir"
    # Single-qid agentic arm via the standard H2H runner (writes agentic_q{qid}.jsonl).
    uv run emet hmeqa h2h "$dir" --arms agentic --ids "$qid" --preset paper-router --foreground
}

run_ovmm() {
    local ep="$1"
    local dir="$OUT/ovmm_${ep}"
    mkdir -p "$dir"
    uv run emet ovmm find \
        --episodes configs/ovmm/find_phase_episodes.yaml \
        --episode-id "$ep" \
        --backend dynagraph \
        --output-dir "$dir"
}

# --- GPU preflight -------------------------------------------------------
if [[ "${SKIP_GPU_WAIT:-0}" != "1" ]]; then
    NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait || true
fi

# --- run -----------------------------------------------------------------
echo "family,item,outcome,note" > "$OUT/results.csv"
done=0
for p in "${PICKED[@]}"; do
    read -r fam item <<<"$p"
    log "START $fam $item ($((done+1))/${#PICKED[@]})"
    outcome="UNKNOWN"
    note=""
    if [[ "$fam" == "hmeqa" ]]; then
        if run_hmeqa "$item" >"$OUT/hmeqa_q${item}.log" 2>&1; then
            outcome="DONE"
            res="$OUT/hmeqa_q${item}/agentic_q${item}.jsonl"
            if [[ -s "$res" ]]; then
                note="$(uv run python -c "import json;d=json.loads(open(r'''$res''').read());print('correct='+str(d.get('correct'))+' pred='+str(d.get('predicted_answer'))+' gold='+str(d.get('gold_answer_letter')))" 2>/dev/null || echo jsonl)" || note=""
            else
                note="no result jsonl"
            fi
        else
            outcome="FAIL"
        fi
    else
        if run_ovmm "$item" >"$OUT/ovmm_${item}.log" 2>&1; then
            outcome="DONE"
            res="$OUT/ovmm_${item}/${item}_dynagraph.json"
            if [[ -s "$res" ]]; then
                note="$(uv run python -c "import json;d=json.load(open(r'''$res'''));print('obj='+str(d.get('find_object_success'))+' recep='+str(d.get('find_recep_success')))" 2>/dev/null || echo json)" || note=""
            else
                note="no result json"
            fi
        else
            outcome="FAIL"
        fi
    fi
    echo "$fam,$item,$outcome,$note" >> "$OUT/results.csv"
    done=$((done+1))
    log "END $fam $item -> $outcome ($note)"
done
log "batch complete: seed=$SEED done=$done/${#PICKED[@]} -> $OUT/results.csv"
cat "$OUT/results.csv"
