#!/usr/bin/env bash
# Comparable preliminary H2H: Habitat HM-EQA holdout-8, static_graph vs dynagraph.
# This is the slice used in prior paper/branch numbers — not Robocasa freeform answer-only.
#
# Usage:
#   nohup ./scripts/run_hmeqa_holdout_h2h.sh [OUT_DIR] >> ~/runs/emet/hmeqa_holdout_h2h_nohup.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/runs/emet/hmeqa_holdout_h2h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/figures"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
HOLDOUT_IDS="${HOLDOUT_IDS:-15,56,65,68,79,88,104,105}"
TIMEOUT="${TIMEOUT:-7200}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/orchestrator.log"; }

if [[ ! -x "$EMET_HABITAT" ]]; then
    log "Missing emet-habitat at $EMET_HABITAT"
    exit 1
fi

log "OUT=$OUT HEAD=$(git rev-parse --short HEAD) ids=$HOLDOUT_IDS"
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
./scripts/gpu_preflight.sh --kill-stale || true

run_method() {
    local method="$1"
    local out="$OUT/${method}.jsonl"
    local elog="$OUT/${method}.log"
    : >"$out"
    : >"$elog"
    log "START method=$method"
    IFS=',' read -r -a ids <<<"$HOLDOUT_IDS"
    for qid in "${ids[@]}"; do
        qid="$(echo "$qid" | tr -d '[:space:]')"
        [[ -n "$qid" ]] || continue
        NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait
        ep="$OUT/${method}_q${qid}.jsonl"
        : >"$ep"
        log "----- $method q$qid -----"
        set +e
        timeout "$TIMEOUT" "$EMET_HABITAT" run-episode \
            --question-id "$qid" \
            --method "$method" \
            --explore-when-uncovered off \
            --no-mcq-debias \
            --memory-summary \
            --output "$ep" \
            >>"$elog" 2>&1
        rc=$?
        set -e
        if [[ "$rc" -ne 0 ]]; then
            log "FAIL $method q$qid exit=$rc"
        fi
        if [[ -s "$ep" ]]; then
            cat "$ep" >>"$out"
        fi
    done
    log "DONE method=$method"
}

run_method static_graph
run_method dynagraph

uv run python - <<PY | tee -a "$OUT/orchestrator.log"
import json
from pathlib import Path

out = Path("$OUT")
summary = {}
for m in ("static_graph", "dynagraph"):
    p = out / f"{m}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    ok = sum(1 for r in rows if r.get("correct"))
    walls = [r.get("wall_s") or r.get("episode_wall_s") for r in rows]
    walls = [w for w in walls if isinstance(w, (int, float))]
    summary[m] = {
        "n": len(rows),
        "correct": ok,
        "accuracy": (ok / len(rows)) if rows else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "per": [
            {
                "q": r.get("question_id"),
                "correct": r.get("correct"),
                "pred": r.get("predicted_answer"),
                "gold": r.get("gold_answer_letter"),
                "wall": r.get("wall_s") or r.get("episode_wall_s"),
            }
            for r in rows
        ],
    }
    print(
        f"{m}: {ok}/{len(rows)} acc={summary[m]['accuracy']} "
        f"mean_wall={summary[m]['mean_wall_s']}"
    )
(out / "h2h_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    names = list(summary.keys())
    accs = [summary[n]["accuracy"] or 0 for n in names]
    walls = [summary[n]["mean_wall_s"] or 0 for n in names]
    ax[0].bar(names, accs, color=["#4C78A8", "#F58518"])
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("HM-EQA holdout-8")
    ax[1].bar(names, walls, color=["#4C78A8", "#F58518"])
    ax[1].set_ylabel("mean wall (s)")
    ax[1].set_title("Cost")
    fig.suptitle("static_graph vs Dynagraph")
    fig.tight_layout()
    fig.savefig(out / "figures" / "hmeqa_holdout_h2h.png", dpi=140)
    print(f"wrote {out / 'figures' / 'hmeqa_holdout_h2h.png'}")
except Exception as e:
    print(f"figure skip: {e}")
PY

echo DONE > "$OUT/DONE"
log "All done → $OUT"
