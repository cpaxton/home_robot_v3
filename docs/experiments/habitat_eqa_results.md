# Habitat HM-EQA — results vs prior art

Recorded numbers for the Dynagraph paper and parity appendix. **JSONL paths** live under `~/.cache/habitat_eqa/results/` unless noted.

Maintainer: update this file and `paper/sections/05_results.tex` / `paper/sections/appendix/05_habitat_eqa_parity.tex` together after each sweep.

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

### Post-fix eval (Qwen3-VL-8B, June 2026)

Engineering: navmesh nav + compass fix, phrase-level CONFIRMED_MEMORY, SigLIP retention through EQA, location-MCQ prompts, MCQ debias.

| Slice | n | Accuracy | JSONL tag |
|-------|---|----------|-----------|
| Smoke tuning {3,14,17} | 3 | 66.7% (2/3) | `postfix_smoke_q3_14_17_dynagraph.jsonl` |
| Q17 alone (phrase + location MCQ) | 1 | 100% (1/1) | `q0017_phrase_prompt_fix.jsonl` |
| **Held-out random-8** (seed 20260627, excludes 3,14,17) | 8 | **50.0%** (4/8) | `subset_holdout8_postfix_20260627_qwen3_vl.jsonl` |

Held-out ids: `15,56,65,68,79,88,104,105`. Correct: Q15, Q79, Q88, Q104. Mean planning steps: 50.5.

### Dynagraph vs graph_eqa on same harness

On every comparable slice, **dynagraph ≥ graph_eqa** (+1–3 pp on balanced-32; debias and CONFIRMED_MEMORY widen the gap on search-style questions). Mock-LLM smoke: both methods 100% on Q0–5 (grading sanity).

## Interpretation

| Comparison | Gap | Likely cause |
|------------|-----|--------------|
| Us (41.6% / 113, gemma-3-4b) vs GraphEQA GPT-4o (63.5%) | ~−22 pp | Weaker local VLM + partial semantics + different graph stack |
| Us (50%, hold-out-8, Qwen3-VL-8B) vs random (25%) | +25 pp | Harness works; small n |
| Us (50%, hold-out-8) vs Explore-EQA (51.7%) | ~parity on tiny slice | Not statistically meaningful until full 113 |
| dynagraph vs graph_eqa (same VLM) | +3 pp (bal-32) | SigLIP frontiers, CONFIRMED_MEMORY, debias |

We are **not** yet competitive with published GraphEQA on the full benchmark. We **are** above chance and Dynagraph consistently beats our GraphEQA baseline on the same code path.

## Planned experiments (priority)

1. **Full 113-question dynagraph** with `Qwen/Qwen3-VL-8B-Instruct` + current fix stack (phrase memory, location MCQ, nav, debias). Headline number for paper vs Table 1 prior art.
2. **Matched baseline:** same 113 with `graph_eqa` + same VLM (isolate Dynagraph gains at 8B).
3. **Semantics coverage:** download remaining HM3D train `.semantic.glb`; re-run ablation with/without GT semantics on Q0–19.
4. **Frontier ablation** (Q0–19, Qwen2.5-VL-3B): fluid only vs fluid+kw vs graph frontier nodes (`run_habitat_frontier_ablation.sh`).
5. **Balanced-32 rerun** at Qwen3-VL-8B (compare to 11/32 @ 3B).
6. **Optional upper bound:** API VLM on canonical-8 or balanced-32 through same harness (isolates VLM tier vs stack).
7. **Efficiency:** early-stop when SigLIP CONFIRMED_MEMORY + graph cover all phrases (reduce mean steps from ~50 toward paper’s ~3–5 on successes).

### Commands

```bash
# Held-out random-8 (repro)
TAG=holdout8_postfix_20260627 IDS=15,56,65,68,79,88,104,105 METHOD=dynagraph TIMEOUT=7200 \
  ./scripts/run_habitat_iter_subset.sh

# Full 113 (planned)
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
