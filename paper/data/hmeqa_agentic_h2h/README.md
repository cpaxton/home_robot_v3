# Classic vs agentic-verify (HM-EQA)

Minimal checked-in summaries only (no RGB / voxels / traces).

| File | Slice | Status |
|------|-------|--------|
| `holdout4_summary.json` | n=4 gate | complete |
| `holdout8_summary.json` | n=8 paper table | complete |
| `balanced32_summary.json` | Wave 1 scale (n=32) | complete — agentic 9/32 < classic 12/32 |
| `failset104105_summary.json` | q104/q105 infra regression | failfix5 fixed `n_object`/empty-pred; letters still wrong |
| `coverage_panel_metrics.json` | coverage figure metrics | holdout-8 |
| `manifest.json` | IDs, harness, run dirs | |

## Balanced-32 (live)

Run dir: `~/runs/emet/hmeqa_agentic_bal32_20260723_212307`  
Job: `emet jobs status 20260724_005904_6db975`

When `OUT/DONE` appears:

```bash
OUT=~/runs/emet/hmeqa_agentic_bal32_20260723_212307
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
# Merge status fields if desired, then:
cp "$OUT/h2h_summary.json" paper/data/hmeqa_agentic_h2h/balanced32_summary.json
```

## Holdout-8 headline

| Arm | Correct | Accuracy | Mean planning steps |
|-----|---------|----------|---------------------|
| Classic | 5/8 | 62.5% | 60.25 |
| Agentic | 8/8 | 100% | 18.25 |

## Failset q104/q105 (infra)

| Run | Agentic | q104 | q105 | `n_object` |
|-----|---------|------|------|------------|
| failfix4 (broken) | 0/2 empty preds | `""` / D | `""` / A | 0 / 0 |
| failfix5 (infra fixed) | 0/2 wrong letters | B / D | B / A | 8 / 8 |

Paper holdout-8 agentic reference remains D/A correct — do not overwrite `holdout8_summary.json` with this short failset.

See parent [`paper/data/README.md`](../README.md) and [`docs/experiments/agentic_scale.md`](../../../docs/experiments/agentic_scale.md).
