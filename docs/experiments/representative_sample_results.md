# Representative cross-benchmark sample

**Run ID:** `rep_sample_20260706`  
**Habitat tuning matrix:** `dynagraph_tune_20260706_110513`  
**VLM:** Qwen3-VL-8B-Instruct (`qwen3_vl`)  

## Habitat HM-EQA — dynagraph ablation (tuning matrix)

| Arm | holdout-8 | canonical-8 | Notes |
|-----|-----------|-------------|-------|
| `baseline` | 5/5 | 7/8 | tuned default: debias off, memory on, conservative explore |
| `no_debias` | 6/8 | 6/8 | debias on |
| `no_memory` | 5/5 | 7/8 | memory summary off |
| `no_explore` | 8/8 | 7/8 | explore off |
| `graph_eqa_like` | 7/8 | 6/8 | debias off, memory off, explore off |

## Habitat HM-EQA — method comparison (representative run + reference)

| Method | slice | n | accuracy |
|--------|-------|---|----------|
| `dynagraph holdout8` | — | 7 | 5/7 (71.4%) |
| `dynagraph holdout8 ref` | — | 8 | 3/8 (37.5%) |
| `graph eqa holdout8` | — | 8 | 7/8 (87.5%) |
| `graph eqa holdout8 ref` | — | 8 | 7/8 (87.5%) |

## OVMM find-phase (sim)

| Tier | Backend | n | FindObj | FindRec | Partial |
|------|---------|---|---------|---------|---------|
| molmo | `dynagraph` | 1 | 0.0% | 0.0% | 0.0% |
| robocasa | `dynagraph` | 1 | 0.0% | 0.0% | 0.0% |
| s0 | `dynamem` | 1 | 100.0% | 100.0% | 100.0% |

## Figures

Artifacts: `~/runs/emet/representative_sample/rep_sample_20260706/figures/`

| Figure | File |
|--------|------|
| HM-EQA ablation (holdout-8) | `hmeqa_ablation_holdout8.png` |
| Top-down maps | `paper_maps_tuning/q*/topdown_map.png` |
| Graph retrieval panels | `retrieval_panels/retrieval_q*.png` |
| OVMM backend bars | `ovmm_findobj_findrec.png` (after OVMM leg) |
| LaTeX table snippet | `../tables/representative_sample.tex` |

Monitor run: `tail -f ~/runs/emet/representative_sample/nohup.log`
