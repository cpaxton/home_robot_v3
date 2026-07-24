# Paper data (minimal)

Checked-in **summaries only** — not full episode traces, RGB dumps, or voxel grids.
Use these JSON files to rebuild tables/figures and to re-run the Habitat H2H.

## Classic vs agentic-verify (`hmeqa_agentic_h2h/`)

| File | What it is |
|------|------------|
| `manifest.json` | Slice IDs, harness flags, git commit, reproduce entrypoints |
| `holdout8_summary.json` | Scored holdout-8 letters / steps (paper Table) |
| `holdout4_summary.json` | Earlier gate (n=4) |
| `balanced32_summary.json` | Wave 1 balanced-32 H2H (may be `status: in_progress` until agentic DONE) |
| `failset104105_summary.json` | q104/q105 infra regression (empty pred / `n_object=0`) before vs after fix |
| `coverage_panel_metrics.json` | Per-question metrics for `figs/hmeqa_agentic_coverage.png` |

**Not stored here:** `explored_2d.npy`, trajectories, `eqa_history`, VLM traces, mp4s. Those live under `~/runs/emet/hmeqa_agentic_h2h8_*` / `~/.cache/habitat_eqa/episodes/` when you re-run.

### Rebuild figures from checked-in summaries

```bash
# Bars (accuracy + mean steps) from holdout-8 summary
uv run python scripts/summarize_hmeqa_agentic_h2h.py \
  --from-summary paper/data/hmeqa_agentic_h2h/holdout8_summary.json \
  --output paper/figs/hmeqa_agentic_h2h.png
```

If `--from-summary` is unavailable, copy a run dir that already has `classic.jsonl` / `agentic.jsonl`, or regenerate (below).

Coverage maps need fresh episode bundles (maps):

```bash
# Full H2H (writes OUT/bundles + figures). Prefer emet jobs; one GPU job.
OUT=~/runs/emet/hmeqa_agentic_h2h8_repro
uv run emet jobs run --name hmeqa-h2h8-repro --need-mib 12000 -- \
  env EMET_ALLOW_SDPA_ATTN=1 HOLDOUT_IDS=15,56,65,68,79,88,104,105 \
  COVERAGE_QIDS=15,104,68 \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT"

# After DONE: compare letters/steps to paper/data/.../holdout8_summary.json
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
diff -u paper/data/hmeqa_agentic_h2h/holdout8_summary.json \
  "$OUT/h2h_summary.json" || true
```

### Expected headline numbers (holdout-8)

| Arm | Correct | Accuracy | Mean planning steps |
|-----|---------|----------|---------------------|
| Classic | 5/8 | 62.5% | 60.25 |
| Agentic | 8/8 | 100% | 18.25 |

### Balanced-32 (Wave 1)

See `hmeqa_agentic_h2h/balanced32_summary.json` and `hmeqa_agentic_h2h/README.md`. Classic-only snapshot at first land: **12/31 (~39%)**, mean steps ~52; agentic pending resume job.

VLM: `Qwen/Qwen3-VL-8B-Instruct`. Harness: Dynagraph, `explore_when_uncovered=off`, no MCQ debias, memory-summary on, agentic router off.

More context: [docs/experiments/habitat_eqa_results.md](../../docs/experiments/habitat_eqa_results.md) (Classic vs agentic-verify section), [docs/experiments/agentic_scale.md](../../docs/experiments/agentic_scale.md).
