# Habitat HM-EQA — results vs prior art

Recorded numbers for the Dynagraph paper and parity appendix. **JSONL paths** live under `~/.cache/habitat_eqa/results/` unless noted.

Maintainer: update this file and `paper/sections/05_results.tex` / `paper/sections/appendix/05_habitat_eqa_parity.tex` together after each sweep.

## JSONL tagging (pre- vs post-nav-fix)

**July 2026 nav stack** (Image-N viewpoint/standoff waypoints, navmesh trajectory follow, `already_at_goal` blocking, frontier distance sort, recent-goal penalty). Tag new runs e.g. `_postfix_nav202607` in the output filename.

JSONL written **before** this stack (June 2026 phrase/location MCQ fixes only) are **not directly comparable** on search-style questions that depended on spin-in-place nav. When updating paper tables, prefer post-nav-fix sweeps or note the stack in the table caption.

### July 2026 nav stack (landed + first evals complete)

| Fix | Effect |
|-----|--------|
| `_navigation_waypoint_for_obs` | VLM `action: Image N` → capture viewpoint or standoff (`eqa.image_nav_min_approach_m`, default 0.35 m) |
| `habitat_navmesh_navigate` | Early `already_at_goal`; `execute_trajectory` on navmesh waypoints |
| Frontier pick | Sort by distance + deprioritize recent goals |
| `--mock-llm-explore` | Movement/diagnostics smoke without real VLM |
| Eval diagnostics | Overlay stride maps, `topdown_exploration.mp4`, Habitat substep RGB |

See [habitat/usage.md](../habitat/usage.md#navigation-habitat-only) and [plans/2026-06-03_habitat_eqa_exploration_improvements.md](../plans/2026-06-03_habitat_eqa_exploration_improvements.md).

## Prior art (GraphEQA paper, full 113 questions)

Reproduced from GraphEQA Table 1 (HM-EQA column). All use Habitat-Sim, 20 VLM planning iterations, API or strong VLMs, and (for GraphEQA) **full HM3D GT semantics** → Hydra 3DSGs.

| Method | VLM | n | Accuracy | Notes |
|--------|-----|---|----------|-------|
| Explore-EQA | (paper stack) | 113 | **51.7%** | Baseline exploration + EQA |
| GraphEQA | GPT-4o | 113 | **63.5%** | Hydra 3DSG + enriched frontiers |
| GraphEQA | Gemini-2.5 Pro | 113 | **67.0%** | Best published sim result |
| GraphEQA | Llama4-Mav | 113 | 57.7% | Strong open model in ref. stack |

Mean planning steps in the paper are **3–5** on *successful* trials (API VLMs often stop early). Trajectory length 3.6–12.6 m on successes.

## Our harness (`emet-habitat`, local VLMs)

Same HM-EQA CSV (113 questions, indices 0–112), 20/10 step budget, RTX 4090. Stack differs: `GraphEQAMemory` / DynaMem voxels, optional graph frontier nodes, navmesh nav — see [appendix 05](../../paper/sections/appendix/05_habitat_eqa_parity.tex).

**Comparability gaps:** local 3–8B VLMs (not GPT-4o/Gemini); **partial** HM3D semantics (~37/113 questions had `.semantic.glb` at last audit); different scene graph and frontier planner.

### Full benchmark (largest n)

| Method | VLM | n | Accuracy | JSONL tag |
|--------|-----|---|----------|-----------|
| graph_eqa repro | gemma-3-4b-it | **113** | **41.6%** (47/113) | `graph_eqa_gemma3_paper_q0-112.jsonl` |

This is our best **full-benchmark** number — still **~20–25 pp below** GraphEQA API baselines, but above random (25%) and in the ballpark of Explore-EQA if semantics + VLM tier were matched.

### Letter-balanced and canonical slices (Qwen2.5-VL-3B era)

| Method | Slice | n | Accuracy | JSONL tag |
|--------|-------|---|----------|-----------|
| dynagraph | balanced-32 | 32 | **34.4%** (11/32) | `subset_bal32_dynagraph_qwen2_5_vl.jsonl` |
| graph_eqa | balanced-32 | 32 | 31.2% (10/32) | `subset_bal32_graph_eqa_qwen2_5_vl.jsonl` |
| dynagraph | canonical-8 | 8 | 50.0% (4/8) | `subset_cmp_dynagraph_qwen2_5_vl.jsonl` |
| graph_eqa | canonical-8 | 8 | 37.5% (3/8) | `subset_cmp_graph_eqa_qwen2_5_vl.jsonl` |
| dynagraph + MCQ debias | canonical-8 | 8 | **62.5%** (5/8) | `fable5_dg_debias_c8` (overnight 2026-06-11) |

Balanced-32 ids: `2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84` (8 per gold letter A–D).

Canonical-8 ids: `3,14,17,28,31,35,81,94`.

### Post-fix eval (Qwen3-VL-8B, June 2026 — pre–July nav stack)

Engineering: navmesh nav + compass fix, phrase-level CONFIRMED_MEMORY, SigLIP retention through EQA, location-MCQ prompts, MCQ debias. **Does not include** July Image-N waypoint / nav-success rules — re-run held-out and full 113 after nav stack.

| Slice | n | Accuracy | JSONL tag |
|-------|---|----------|-----------|
| Smoke tuning {3,14,17} | 3 | 66.7% (2/3) | `postfix_smoke_q3_14_17_dynagraph.jsonl` |
| Q17 alone (phrase + location MCQ) | 1 | 100% (1/1) | `q0017_phrase_prompt_fix.jsonl` |
| **Held-out random-8** (seed 20260627, excludes 3,14,17) | 8 | **50.0%** (4/8) | `subset_holdout8_postfix_20260627_qwen3_vl.jsonl` |

Held-out ids: `15,56,65,68,79,88,104,105`. Correct: Q15, Q79, Q88, Q104. Mean planning steps: 50.5.

### Post-nav-fix eval (Qwen3-VL-8B, July 2026 nav stack)

Run id **`postfix_nav20260705_larger`** (2026-07-05 18:19 → 2026-07-06 00:20, ~6 h). Stack: navmesh nav, Image-N waypoints, frontier nodes (`--frontier-nodes --frontier-keyword-weight 2`), 20/10 step budget, HM-EQA prompts. Overnight log: `~/.cache/habitat_eqa/overnight/postfix_nav20260705_larger/summary.txt`.

**Smoke (canonical Q3, Q14, Q17)** — both methods 100% before the larger sweep:

| Method | n | Accuracy | JSONL tag |
|--------|---|----------|-----------|
| graph_eqa | 3 | **100%** (3/3) | `full_nav_fix_20260705_172102_hmeqa_graph_eqa.jsonl` |
| dynagraph | 3 | **100%** (3/3) | `full_nav_fix_20260705_172102_hmeqa_dynagraph.jsonl` |

**Larger sweep** (`postfix_nav20260705_larger`):

| Method | Slice | n | Accuracy | JSONL tag |
|--------|-------|---|----------|-----------|
| graph_eqa | held-out random-8 | 8 | **87.5%** (7/8) | `subset_postfix_nav20260705_larger_holdout8_graph_eqa_qwen3_vl.jsonl` |
| dynagraph | held-out random-8 | 8 | 37.5% (3/8) | `subset_postfix_nav20260705_larger_holdout8_dynagraph_qwen3_vl.jsonl` |
| graph_eqa | balanced-32 | 32 | 37.5% (12/32) | `subset_overnight_postfix_nav20260705_larger_balanced32_graph_eqa_qwen3_vl.jsonl` |
| dynagraph | balanced-32 | 32 | **40.6%** (13/32) | `subset_overnight_postfix_nav20260705_larger_balanced32_dynagraph_qwen3_vl.jsonl` |
| graph_eqa | paper Q0–19 | 20 | 45.0% (9/20) | `subset_overnight_postfix_nav20260705_larger_paper20_graph_eqa_qwen3_vl.jsonl` |
| dynagraph | paper Q0–19 | 20 | **50.0%** (10/20) | `subset_overnight_postfix_nav20260705_larger_paper20_dynagraph_qwen3_vl.jsonl` |

Held-out ids (same seed as June): `15,56,65,68,79,88,104,105`. **graph_eqa** correct: all except Q104. **dynagraph** correct: Q15, Q65, Q88 only. Canonical-8 skipped in this run (`SKIP_PHASES=canonical8_*`).

**Head-to-head (same questions, both methods):**

| Slice | graph_eqa | dynagraph | both correct | graph only | dyna only |
|-------|-----------|-----------|--------------|------------|-----------|
| held-out-8 | 7/8 | 3/8 | 3 | 4 | 0 |
| balanced-32 | 12/32 | 13/32 | 8 | 4 | 5 |
| paper Q0–19 | 9/20 | 10/20 | 8 | 1 | 2 |

**Takeaways:** July nav fixes strongly help **graph_eqa** on held-out search questions (7/8 vs 4/8 pre-nav dynagraph baseline on the same ids). On balanced-32 and paper-20, methods are **roughly tied** (dynagraph +1 on each). Holdout dynagraph regression (3/8) traces to dynagraph-only extras (MCQ debias flipping correct letters, CONFIRMED_MEMORY anchoring wrong objects, extra frontier exploration when graph coverage is incomplete) — not missing nav wiring (same `GraphEQAController` base). MCQ debias on dynagraph: balanced-32 +5/−4, paper-20 +3/−3 (net neutral).

### Dynagraph vs graph_eqa on same harness

On balanced-32 and paper slices at 3B/8B, **dynagraph ≥ graph_eqa** (+1–3 pp) when June fix stack (debias, CONFIRMED_MEMORY) helps. **Exception:** post-nav held-out-8 where graph_eqa leads 7/8 vs 3/8 — tune dynagraph HM-EQA extras on a follow-up branch. Mock-LLM smoke: both methods 100% on Q0–5 (grading sanity).

## Interpretation

| Comparison | Gap | Likely cause |
|------------|-----|--------------|
| Us (41.6% / 113, gemma-3-4b) vs GraphEQA GPT-4o (63.5%) | ~−22 pp | Weaker local VLM + partial semantics + different graph stack |
| Us (87.5%, hold-out-8 graph_eqa, Qwen3-VL-8B + July nav) vs random (25%) | +62.5 pp | Nav stack works on search slice; small n |
| Us (50%, hold-out-8 dynagraph pre-nav) vs post-nav graph_eqa (87.5%) | +37.5 pp | Nav + baseline graph path >> dynagraph extras on holdout |
| Us (40–50%, bal-32 / paper-20 @ 8B) vs Explore-EQA (51.7%) | ~parity | Need full 113 for meaningful comparison |
| dynagraph vs graph_eqa (bal-32 @ 8B post-nav) | +3 pp (13/32 vs 12/32) | Debias/memory help on average; hurt on some holdout eps |

We are **not** yet competitive with published GraphEQA on the full benchmark. We **are** above chance; post-nav balanced-32 favored dynagraph slightly (+1 pp) while holdout favored graph_eqa (+4 pp) before harness tuning.

## Dynagraph cross-environment tuning (2026-07)

Unified harness config: [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml) + `apply_dynagraph_harness()`. **Tuned `habitat_eqa` dynagraph defaults** (Qwen3-VL-8B target):

| Flag | Tuned default | Rationale |
|------|---------------|-----------|
| `memory_summary` | on | SigLIP CONFIRMED_MEMORY helps search questions |
| `mcq_debias` | **off** | 8B model: debias flipped correct answers on holdout (e.g. Q68 B→A) |
| `explore_when_uncovered` | **conservative** | Keep query-time frontier override; disable habitat-only uncovered hijack |

Ablation matrix: `./scripts/run_dynagraph_tuning_matrix.sh` (arms: `baseline,with_debias,no_memory,no_explore,graph_eqa_like`). Paper battery after tuning converges: `./scripts/run_dynagraph_tuned_paper_battery.sh`. Results land under `~/runs/emet/dynagraph_tuning/<RUN_ID>/`.

**Eval pending** on tuned harness — update tables below after first `tuned_paper_*` or tuning-matrix run completes.

## Planned experiments (priority)

**Done (2026-07-05/06, `postfix_nav20260705_larger`):** held-out-8 both methods; balanced-32 both methods @ Qwen3-VL-8B; paper Q0–19 both methods; smoke Q3/14/17 @ 100%.

**In progress (`feature/dynagraph-tuning`):**

1. **Harness landed** — per-env flags in `dynagraph.yaml`; Habitat CLI ablations; figure scripts.
2. **Tuning matrix** — holdout-8 + canonical-8 ablations @ Qwen3-VL-8B.
3. **Paper battery** — seven-track smoke + Habitat/OVMM/SQA3D/Robocasa/Molmo with tuned config.
4. **Full 113-question** sweep after holdout ≥ graph_eqa baseline.
5. **Semantics / frontier ablations** — unchanged from prior plan.

### Commands

```bash
# Held-out random-8 (repro)
TAG=holdout8_postfix_20260627 IDS=15,56,65,68,79,88,104,105 METHOD=dynagraph TIMEOUT=7200 \
  ./scripts/run_habitat_iter_subset.sh

# Tuning matrix + tuned paper battery
ARMS=baseline,with_debias,graph_eqa_like ./scripts/run_dynagraph_tuning_matrix.sh
SKIP_SMOKE=1 ./scripts/run_dynagraph_tuned_paper_battery.sh

# Habitat HM-EQA with tuned harness (default via apply_dynagraph_harness)
.venv-habitat/bin/emet-habitat run-batch \
  --method dynagraph \
  --paper-subset \
  --question-start 0 --question-end 112 \
  --eqa-vl-family qwen3_vl \
  --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda --resume \
  --output ~/.cache/habitat_eqa/results/full113_dynagraph_qwen3_vl_postfix.jsonl

# Balanced-32
IDS=2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84 \
  TAG=bal32_qwen3_vl_postfix METHOD=dynagraph TIMEOUT=7200 \
  ./scripts/run_habitat_iter_subset.sh
```

Overnight orchestrators: [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh), [`scripts/run_overnight_cross_track_smoke.sh`](../scripts/run_overnight_cross_track_smoke.sh), `scripts/run_overnight_eval_smoke.sh`, `scripts/run_overnight_habitat_eval.sh`, `scripts/run_fable5_overnight.sh`, `scripts/run_extensive_habitat_eval.sh`.

## Related docs

- [habitat_eqa.md](habitat_eqa.md) — entrypoint and metrics
- [../plans/fable5-dynagraph-habitat.md](../plans/fable5-dynagraph-habitat.md) — debias / bake-off history
- [../plans/2026-06-03_habitat_eqa_exploration_improvements.md](../plans/2026-06-03_habitat_eqa_exploration_improvements.md) — exploration fix log
