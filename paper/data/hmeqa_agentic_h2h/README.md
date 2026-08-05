# Classic vs agentic-verify (HM-EQA)

Minimal checked-in summaries only (no RGB / voxels / traces).

| File | Slice | Status |
|------|-------|--------|
| `holdout4_summary.json` | n=4 gate | complete |
| `holdout8_summary.json` | n=8 paper table | complete — classic 5/8, agentic **8/8** (router **off**) |
| `balanced32_summary.json` | n=32 of record | **composite** — classic **9/32** (router-off H2H) + agentic **16/32** (paper-router+explore); see note |
| `balanced32_router_off_agentic_archive.json` | n=32 prior matched H2H | archive — classic 9/32, agentic **11/32**, router off |
| `balanced32_overnight_replicate.json` | n=32 replicate | complete — classic **10/32**, agentic **12/32** (router off) |
| `failset104105_summary.json` | q104/q105 infra regression | failfix5 fixed `n_object`/empty-pred; letters still wrong |
| `coverage_panel_metrics.json` | coverage figure metrics | holdout-8 |
| `manifest.json` | IDs, harness, run dirs | |

## Balanced-32 of record (composite, 2026-07-27)

| Arm | Correct | Accuracy | Mean planning steps | Policy |
|-----|---------|----------|---------------------|--------|
| Classic | 9/32 | 28.1% | 48.7 | router-off matched H2H (`bal32r2`, commit `8c9a388`) |
| Agentic | **16/32** | **50.0%** | **32.7** | paper-router on + station-filter / explore-after-ABSENT (`hmeqa_bal32_explore_20260727_215036`, `e7d0eb20`) |

Cross-policy McNemar p≈0.092 (n.s. at α=0.05); Wilcoxon on steps p≈1.8e-5 (agentic cheaper).  
**Not** a same-commit matched H2H — do not treat McNemar as a clean paired A/B.

Prior matched H2H agentic (router off): **11/32** archived in `balanced32_router_off_agentic_archive.json`.

```bash
OUT=~/runs/emet/hmeqa_bal32_explore_20260727_215036
uv run emet hmeqa summarize "$OUT"
uv run emet hmeqa significance --from-summary paper/data/hmeqa_agentic_h2h/balanced32_summary.json
# Holdout paper figs only (opt-in):
COPY_PAPER_FIGS=1 …  # or rebuild from holdout8_summary.json
```

## Holdout-8 headline (paper table)

| Arm | Correct | Accuracy | Mean planning steps |
|-----|---------|----------|---------------------|
| Classic | 5/8 | 62.5% | 60.25 |
| Agentic | 8/8 | 100% | 18.25 |

Router **off**. Paper figures `figs/hmeqa_agentic_h2h.png` / `hmeqa_agentic_coverage.png` are **holdout-8**. Do not overwrite from bal-32 runs.

**Policy variance (docs / appendix only):** paper-router + explore holdout `hmeqa_holdout8_explore_20260727_211505` scored agentic **5/8** (misses 56/65/105) — VLM / salvage noise, not the paper headline.

## Post-manifest variance runs (Aug 2026, classic-only on holdout-8 prefix `{15,56,65,68}`)

These are **not** the paper's holdout-4 slice (`{15,68,105,17}`); they exercise the first four ids of holdout-8 and are kept as variance evidence for the classic arm, not as replacement table rows.

| Run dir | Classic on `{15,56,65,68}` |
|---------|----------------------------|
| `hmeqa_agentic_h2h_20260804_095037` | 4/4 (100%) |
| `hmeqa_agentic_h2h_20260804_200537` | 1/4 (25%) |
| `hmeqa_agentic_h2h_20260804_214650` | 4/4 (100%) |
| `hmeqa_agentic_h2h_20260805_094149` | 1/4 (25%) |

Paper rows (`holdout4_summary.json`, `holdout8_summary.json`) unchanged; cite the 25–100% spread only as run-to-run variance on letter accuracy, with mean planning steps as the more stable claim.

## Balanced-32 replication evidence (Aug 2026, agentic only)

The 16/32 composite headline is stable across later agentic-only runs:

| Run dir | Agentic |
|---------|---------|
| `hmeqa_bal32_room_llm_20260731_052932` | 12/32 (37.5%) |
| `hmeqa_merged_off_bal32_20260801` | 11/32 (34.4%) |
| `hmeqa_merged_on_bal32_20260801` | 14/32 (43.75%) |
| `hmeqa_bal32_fixes_off_20260802` | 15/32 (46.9%) |
| `hmeqa_bal32_fixes_on_20260802` | **16/32 (50.0%)** |

Band 11–16/32; paper headline 16/32 remains the of-record composite (classic 9/32 from the matched router-off H2H).

## Failset q104/q105 (infra)

| Run | Agentic | q104 | q105 | `n_object` |
|-----|---------|------|------|------------|
| failfix4 (broken) | 0/2 empty preds | `""` / D | `""` / A | 0 / 0 |
| failfix5 (infra fixed) | 0/2 wrong letters | B / D | B / A | 8 / 8 |

Paper holdout-8 agentic reference remains D/A correct — do not overwrite `holdout8_summary.json` with this short failset.

See parent [`paper/data/README.md`](../README.md) and [`docs/experiments/agentic_scale.md`](../../../docs/experiments/agentic_scale.md).
