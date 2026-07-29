# Paper data (minimal)

Checked-in **summaries only** — not full episode traces, RGB dumps, or voxel grids.
Use these JSON files to rebuild tables/figures and to re-run the Habitat H2H.

## Classic vs agentic-verify (`hmeqa_agentic_h2h/`)

| File | What it is |
|------|------------|
| `manifest.json` | Slice IDs, harness flags, git commit, reproduce entrypoints |
| `holdout8_summary.json` | Scored holdout-8 letters / steps (paper Table) |
| `holdout4_summary.json` | Earlier gate (n=4) |
| `balanced32_summary.json` | Composite of-record: classic 9/32 (router-off) + agentic **16/32** (paper-router+explore) |
| `balanced32_router_off_agentic_archive.json` | Prior matched H2H agentic 11/32 (router off) |
| `balanced32_overnight_replicate.json` | Independent overnight bal-32 (classic 10/32, agentic 12/32; router off) |
| `failset104105_summary.json` | q104/q105 infra regression (empty pred / `n_object=0`) before vs after fix |
| `coverage_panel_metrics.json` | Per-question metrics for `figs/hmeqa_agentic_coverage.png` |

**Not stored here:** `explored_2d.npy`, trajectories, `eqa_history`, VLM traces, mp4s. Those live under `~/runs/emet/hmeqa_agentic_h2h8_*` / `~/.cache/habitat_eqa/episodes/` when you re-run.

### Rebuild figures from checked-in summaries

```bash
# Bars (accuracy + mean steps) from holdout-8 summary
uv run python scripts/summarize_hmeqa_agentic_h2h.py \
  --from-summary paper/data/hmeqa_agentic_h2h/holdout8_summary.json \
  --output paper/figs/hmeqa_agentic_h2h.png
# Balanced-32 bars (do not overwrite holdout-8 fig)
uv run python scripts/summarize_hmeqa_agentic_h2h.py \
  --from-summary paper/data/hmeqa_agentic_h2h/balanced32_summary.json \
  --output paper/figs/hmeqa_agentic_h2h_bal32.png
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

### Balanced-32 (composite of-record)

See `hmeqa_agentic_h2h/balanced32_summary.json` and `hmeqa_agentic_h2h/README.md`.

| Arm | Correct | Accuracy | Mean planning steps | Policy |
|-----|---------|----------|---------------------|--------|
| Classic | 9/32 | 28.1% | 48.7 | router-off matched H2H |
| Agentic | **16/32** | **50.0%** | **32.7** | paper-router on + explore (2026-07-27) |

Cross-policy McNemar p≈0.09 (n.s. at α=0.05); Wilcoxon steps p≈1.8e-5. Prior matched agentic (router off): 11/32 @ 17.8 steps (archived). Overnight replicate under router-off: classic 10/32, agentic 12/32. Holdout paper-router variance: agentic 5/8 (docs only; headline remains 8/8).

VLM: `Qwen/Qwen3-VL-8B-Instruct`. Harness: Dynagraph, `explore_when_uncovered=off`, no MCQ debias, memory-summary on, owlv2 proposals, allow-unverified.

More context: [docs/experiments/habitat_eqa_results.md](../../docs/experiments/habitat_eqa_results.md) (Classic vs agentic-verify section), [docs/experiments/agentic_scale.md](../../docs/experiments/agentic_scale.md).
