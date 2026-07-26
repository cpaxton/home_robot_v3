# Classic vs agentic-verify (HM-EQA)

Minimal checked-in summaries only (no RGB / voxels / traces).

| File | Slice | Status |
|------|-------|--------|
| `holdout4_summary.json` | n=4 gate | complete |
| `holdout8_summary.json` | n=8 paper table | complete — classic 5/8, agentic **8/8** |
| `balanced32_summary.json` | n=32 of record | complete — classic **9/32**, agentic **11/32**; router off; steps win |
| `balanced32_overnight_replicate.json` | n=32 replicate | complete — classic **10/32**, agentic **12/32** |
| `failset104105_summary.json` | q104/q105 infra regression | failfix5 fixed `n_object`/empty-pred; letters still wrong |
| `coverage_panel_metrics.json` | coverage figure metrics | holdout-8 |
| `manifest.json` | IDs, harness, run dirs | |

## Balanced-32 (2026-07-26)

Run of record: `~/runs/emet/hmeqa_agentic_bal32r2_20260726_105946`  
Job: `20260726_105950_99aa96` @ commit `8c9a388`

| Arm | Correct | Accuracy | Mean planning steps |
|-----|---------|----------|---------------------|
| Classic | 9/32 | 28.1% | 48.7 |
| Agentic | 11/32 | 34.4% | 17.8 |

McNemar p≈0.73 (not significant). Wilcoxon on steps p≈4e-7 (agentic cheaper).  
Independent overnight replicate (`hmeqa_overnight_20260726_022227/bal32`): classic 10/32, agentic 12/32.

Policy actually scored: owlv2 proposals + allow-unverified + **`EMET_EQA_AGENTIC_ROUTER=0`** (H2H previously hardcoded router off; fixed to honor env for future runs).

Historical salvage-bug Wave 1 (`hmeqa_agentic_bal32_20260723_212307`): classic 12/32, agentic 9/32 — do not cite as current method.

```bash
OUT=~/runs/emet/hmeqa_agentic_bal32r2_20260726_105946
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
uv run python scripts/hmeqa_significance.py "$OUT"
cp "$OUT/h2h_summary.json" paper/data/hmeqa_agentic_h2h/balanced32_summary.json
# Holdout paper figs only (opt-in):
COPY_PAPER_FIGS=1 …  # or rebuild from holdout8_summary.json
```

## Holdout-8 headline (paper table)

| Arm | Correct | Accuracy | Mean planning steps |
|-----|---------|----------|---------------------|
| Classic | 5/8 | 62.5% | 60.25 |
| Agentic | 8/8 | 100% | 18.25 |

Paper figures `figs/hmeqa_agentic_h2h.png` / `hmeqa_agentic_coverage.png` are **holdout-8** (Q15/Q104/Q68 panels). Do not overwrite from bal-32 runs.

## Failset q104/q105 (infra)

| Run | Agentic | q104 | q105 | `n_object` |
|-----|---------|------|------|------------|
| failfix4 (broken) | 0/2 empty preds | `""` / D | `""` / A | 0 / 0 |
| failfix5 (infra fixed) | 0/2 wrong letters | B / D | B / A | 8 / 8 |

Paper holdout-8 agentic reference remains D/A correct — do not overwrite `holdout8_summary.json` with this short failset.

See parent [`paper/data/README.md`](../README.md) and [`docs/experiments/agentic_scale.md`](../../../docs/experiments/agentic_scale.md).
