#!/usr/bin/env bash
# Q17 ×3 gate under default HM-EQA Dynagraph harness (merge-on unified_eqa).
#
# Pass criterion: ≥2/3 correct (gold D) with fingerprint-verified merge 0.45.
# Does NOT pass --explore-when-uncovered / merge CLI overrides — yaml defaults only.
#
# Usage:
#   ./scripts/run_q17_merge_on_gate.sh
#   RUN_DIR=~/runs/emet/branch_verify_20260711 ./scripts/run_q17_merge_on_gate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-$HOME/runs/emet/branch_verify_20260711}"
GATE_ROOT="${GATE_ROOT:-$RUN_DIR/q17_gate_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$GATE_ROOT"
EMET_HABITAT="${EMET_HABITAT:-$ROOT/.venv-habitat/bin/emet-habitat}"
SUMMARY="$GATE_ROOT/SUMMARY.json"
LOG="$GATE_ROOT/gate.log"
: >"$LOG"

if [[ ! -x "$EMET_HABITAT" ]]; then
  echo "Missing emet-habitat at $EMET_HABITAT (run ./scripts/install_habitat.sh)" | tee -a "$LOG"
  exit 1
fi

echo "===== Q17 merge-on gate $(date -Is) =====" | tee -a "$LOG"
echo "GATE_ROOT=$GATE_ROOT" | tee -a "$LOG"

./scripts/gpu_preflight.sh --kill-stale >>"$LOG" 2>&1 || true
NEED_MIB="${NEED_MIB:-12000}" ./scripts/gpu_preflight.sh --wait >>"$LOG" 2>&1

TRIALS=3
for i in $(seq 1 "$TRIALS"); do
  tag="q17_gate_t${i}"
  ep_out="$GATE_ROOT/${tag}.jsonl"
  ep_log="$GATE_ROOT/${tag}.log"
  : >"$ep_out"
  echo "===== trial $i/$TRIALS tag=$tag $(date -Is) =====" | tee -a "$LOG"
  "$EMET_HABITAT" run-episode \
    --question-id 17 \
    --method dynagraph \
    --output "$ep_out" \
    >>"$ep_log" 2>&1 || echo "trial $i failed exit=$?" | tee -a "$LOG"
  cat "$ep_log" >>"$LOG"
done

python3 - "$GATE_ROOT" "$SUMMARY" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

gate = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for i in range(1, 4):
    p = gate / f"q17_gate_t{i}.jsonl"
    if not p.is_file() or not p.stat().st_size:
        rows.append({"trial": i, "ok": False, "error": "missing_jsonl"})
        continue
    line = next((ln for ln in p.read_text().splitlines() if ln.strip()), "")
    if not line:
        rows.append({"trial": i, "ok": False, "error": "empty_jsonl"})
        continue
    r = json.loads(line)
    harness = r.get("harness") or {}
    rows.append(
        {
            "trial": i,
            "ok": True,
            "correct": bool(r.get("correct")),
            "predicted_answer": r.get("predicted_answer"),
            "gold_answer_letter": r.get("gold_answer_letter"),
            "graph_nodes": r.get("graph_nodes") or (r.get("graph") or {}).get("num_nodes"),
            "harness": harness,
            "output": str(p),
        }
    )

correct = sum(1 for r in rows if r.get("correct"))
merge_vals = [
    float(r["harness"]["dynagraph_merge_xy_m"])
    for r in rows
    if r.get("harness") and r["harness"].get("dynagraph_merge_xy_m") is not None
]
fallback_vals = [
    float(r["harness"]["fallback_spatial_merge_xy_m"])
    for r in rows
    if r.get("harness") and r["harness"].get("fallback_spatial_merge_xy_m") is not None
]
passed = correct >= 2
fingerprint_ok = (
    len(merge_vals) == len([r for r in rows if r.get("ok")])
    and all(abs(v - 0.45) < 1e-6 for v in merge_vals)
    and all(abs(v - 0.45) < 1e-6 for v in fallback_vals)
)
out = {
    "gate": "q17_merge_on",
    "correct": correct,
    "trials": len(rows),
    "passed": passed,
    "pass_criterion": ">=2/3 correct (D)",
    "fingerprint_ok": fingerprint_ok,
    "expected_merge_xy_m": 0.45,
    "rows": rows,
}
summary_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"Q17 gate: {correct}/{len(rows)} correct  passed={passed}  fingerprint_ok={fingerprint_ok}")
for r in rows:
    if not r.get("ok"):
        print(f"  t{r['trial']}: ERROR {r.get('error')}")
        continue
    h = r.get("harness") or {}
    print(
        f"  t{r['trial']}: {'OK' if r.get('correct') else 'FAIL'} "
        f"pred={r.get('predicted_answer')} gold={r.get('gold_answer_letter')} "
        f"nodes={r.get('graph_nodes')} "
        f"merge={h.get('dynagraph_merge_xy_m')} "
        f"fallback={h.get('fallback_spatial_merge_xy_m')} "
        f"explore={h.get('explore_when_uncovered')} "
        f"profile={h.get('profile')}"
    )
print(f"SUMMARY={summary_path}")
sys.exit(0 if passed else 1)
PY
exit_code=$?
echo "===== DONE $(date -Is) exit=$exit_code =====" | tee -a "$LOG"
exit "$exit_code"
