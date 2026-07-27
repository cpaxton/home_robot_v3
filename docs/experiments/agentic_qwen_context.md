# Copyright (c) Chris Paxton 2026

"""Classic vs agentic Qwen context surfaces (HM-EQA letter accuracy).

## Working theory

Agentic letter misses often come from **thinner or mis-scored context**, not only
frontier search. Classic Dynagraph answers with one multi-image ``query_answer``;
agentic mid-loop decisions use a text-only router and a single-view assess, then
a final ``query_answer`` whose diversified image selection previously **ignored**
the verified obs. Habitat scoring reads ``graph_memory.last_eqa_raw`` — so a
correct agentic ``vlm_suggested`` letter could lose to truncated ``[salvage]``.

## Surfaces

| Call | Images | Graph / memory text |
|------|--------|---------------------|
| Classic ``query_answer`` | up to ``eqa_max_images`` (4) via ``_select_relevant_obs_ids`` | ``SCENE_GRAPH`` (~48 nodes) or spatial REGION blocks + Dynagraph ``CONFIRMED_MEMORY`` + HISTORY |
| Agentic router ``build_state_message`` | none | counts + hypotheses; with spatial RAG, compact REGION text |
| Agentic ``vlm_assess`` | 1 full-frame RGB | ≤12 inventory labels (no full SCENE_GRAPH) |
| Agentic ``submit_answer`` → ``query_answer`` | verified obs forced as Image 1 when ``force_obs_ids`` set; fill remaining | same as classic |

## Spatial RAG (prompt retrieval)

When ``eqa.spatial_rag: true`` or ``EMET_EQA_SPATIAL_RAG=1``,
[`spatial_rag.py`](../../src/emet/memory/graph_eqa/spatial_rag.py) seeds on
question keywords / preferred obs ids, expands planar neighbors
(``spatial_rag_radius_m``, default 2.5 m), clusters into up to
``spatial_rag_max_regions`` (6) regions, and emits:

```text
SCENE_GRAPH (spatial regions):
REGION 1 (near Image 20): sofa, red pillow, side table
  anchor (1.10, -2.39); images: 20, 18
```

Default is **off** in config until the context-fix fail-set is scored; force on
with the env var for experiments. Fallback: ranked flat top-K ``SCENE_GRAPH``.

## Logging gaps

Episode bundles store ``raw_eqa.txt``, ``agentic_trace.jsonl``, ``last_eqa_obs_ids`` /
``prompt_obs_count``, but **not** the full multimodal prompt or Image 1..N PNGs.
``EMET_EQA_TRACE=1`` is required for assess/submit attribution. Submit traces may
include ``force_obs_ids``, ``spatial_rag``, and ``sync_scored_answer``.

## Offline attribution

```bash
uv run emet hmeqa failures ~/runs/emet/hmeqa_agentic_bal32r2_20260726_105946
uv run emet hmeqa failures ~/runs/emet/hmeqa_failset_contextfix_…
```

## bal-32r2 classic_only (pre-fix evidence)

| qid | Classic | Agentic scored | Agentic assess/submit | Notes |
|-----|---------|----------------|-----------------------|-------|
| 11 | D | empty | ABSENT streak on ``silver trash can`` | explore/context; empty submit |
| 28 | D | A via ``[salvage]`` | assess+submit **D** | scored_vs_submit_mismatch |
| 39 | C | B via ``[salvage]`` | assess+submit **C** | scored_vs_submit_mismatch |

Fixes landed: ``query_answer(force_obs_ids=…)``,
``_sync_scored_answer_to_graph_memory``, richer router state text, spatial RAG module.

## Fail-set validation (contextfix)

Job ``hmeqa-failset-contextfix`` →
``~/runs/emet/hmeqa_failset_contextfix_20260726_172419``
(ids ``15,56,65,68,79,88,104,105,11,14,28,39,47``, paper-router).
Completed **26/26** (``DONE`` 2026-07-26 19:19); Cursor V8 deaths during the run
did **not** kill the detached ``emet jobs`` Habitat process.

Pass bar: holdout-8 agentic ≥7/8; recover ≥4/5 of ``{11,14,28,39,47}``;
q28/q39 should no longer be ``scored_vs_submit_mismatch``.

### Results (scored)

| Metric | Value |
|--------|-------|
| Paired acc | classic **8/13**, agentic **8/13** (Δacc=0, McNemar p=1) |
| Steps | classic mean 44.7 → agentic **18.0** (Wilcoxon p≈1e-4) |
| Holdout-8 agentic | **4/8** (below ≥7/8 bar) |
| Recover ``{11,14,28,39,47}`` | **4/5** (pass): 14,28,39,47 OK; **q11** still empty |
| q28 / q39 | scored **D/C** matching gold — **no** ``scored_vs_submit_mismatch`` |
| Remaining classic_only | **q56, q104** → ``context_thin_assess`` (agentic B vs classic C/D) |

Buckets: ``ok`` 8, ``context_thin_assess`` 3, ``empty_or_abstain`` 1 (q11), ``other`` 1.

### Spatial RAG follow-up

Job ``hmeqa-failset-spatialrag`` / ``20260726_224324_c4a6f4`` →
``~/runs/emet/hmeqa_failset_spatialrag_20260726_224321``
(same 13 ids, paper-router, ``EMET_EQA_SPATIAL_RAG=1``; **26/26 DONE**).

| Metric | contextfix | +spatial RAG |
|--------|-------------|--------------|
| Holdout-8 agentic | 4/8 | **6/8** (still <7/8) |
| Recover ``{11,14,28,39,47}`` | 4/5 | **2/5** (q14 empty, q47 wrong) |
| Agentic paired acc | 8/13 | 8/13 |
| Classic paired acc | 8/13 | 9/13 |
| q11 | empty | still empty |
| Wins | — | q56, q65 flipped wrong→right |
| Regressions | — | q14 empty, q47 A→B |

RAG helped holdout thin-assess (56/65) but did not fix q11 abstain and regressed two recovery-set letters. Keep default ``spatial_rag: false`` until abstain/q11 and regression investigated; do not treat as merge-ready win.

## Router vs deterministic policy (holdout regression)

Paper / overnight holdout **8/8** used ``EMET_EQA_AGENTIC_ROUTER=0``. Failsets with
``--preset paper-router`` (``ROUTER=1``) dropped to 4–6/8. Traces showed the VLM
router looping ``navigate_to_obs`` on one hyp while ``capture_and_update`` returned
``NO_NEW_OBS`` — no verify, no ``_tried`` update, no planner feedback.

Fixes in ``agentic_eqa`` (keep spatial RAG off):

- ``navigate_to_obs``: ``look_around`` on ``NO_NEW_OBS`` (parity with explore).
- Always ``verify_siglip`` after a fresh capture (router and fallback).
- Stall path: one forced verify on the current view, mark ``STALLED_NAV_LOOP``,
  block re-nav to that id, surface ``NAV_LOOP`` in router state.
- Fresh graph obs → cheap hypothesis refresh for the planner.
## Overnight holdout re-baseline

**Goal reminder:** one ``emet`` agent that does useful work across mobile-manipulation
settings (sim + real: Robocasa/MolmoSpaces, Habitat EQA, Discord/chat). HM-EQA
holdout is a **letter/context stress test** of the shared agentic loop
(nav → capture → graph update → verify → decide), not the end product.

### Fix1 (nav-loop / candidate-refresh) — DONE 6/8

- Job ``hmeqa-holdout8-fix`` / ``20260727_011709_cbb3a0``
- OUT ``~/runs/emet/hmeqa_holdout8_fix_20260727_011706``
- paper-router, spatial RAG **off**: agentic **6/8**, classic **5/8**, 0 ``nav_loop``
- Misses: **q56** (early CANDIDATE+answerable → A; gold C), **q105** (ABSENT streak /
  bad SigLIP phrases → budget B)
- Overnight ROUTER=0 still **8/8** at ``~/runs/emet/hmeqa_overnight_20260726_022227/holdout8``

### Fix2 — SigLIP phrase + defer weak CANDIDATE — DONE **5/8** (regressed)

- Job ``hmeqa-holdout8-fix2`` / ``20260727_023233_0a27b8``
- OUT ``~/runs/emet/hmeqa_holdout8_fix2_20260727_023229`` (agentic-only, paper-router)
- **q56 fixed** (A→C), but **q65** and **q104** regressed vs fix1; **q105** still wrong
- Misses: q65 C≠A (budget/query submit), q104 B≠D (salvage kitchen), q105 B≠A
- Pass bar still unmet (≥7/8). Do not relaunch until shared-loop diagnosis of the
  three misses; overnight ROUTER=0 remains 8/8.

### Diagnosis (2026-07-27) — three distinct failure modes

1. **q65 (state MCQ) — deferral + bad salvage**
   - First ``vlm_assess``: ``present=True``, VLM wanted **A**, SigLIP ``CANDIDATE``.
     ``deferred_weak_candidate`` forced ``answerable=False`` and **dropped**
     ``suggested_answer``.
   - After that: router re-nav / explore; verifies stuck on ``SKIPPED_SAME_VIEW``;
     **no second ``vlm_assess``**. ``budget_hit`` → Unknown.
   - ``final_location_salvage`` then invented **C**. Classifier bug:
     ``question_is_attribute_state`` misses “leave … on?”;
     ``choices_are_location_mcq`` treats “No, it is off / Yes, it is on” as places
     (``choices_are_attribute_state`` also misses those phrases).
   - Fix1 unlocked the same early A and scored correctly.

2. **q104 — never verified; coordinate dump → salvage lottery**
   - Fix1: late assess ``present=True suggested=D`` with SigLIP ``ABSENT`` → unlock → D.
   - Fix2: all assesses absent; budget submit is XYZ prose; post-hoc ``[salvage]`` → B.
     Overnight also submitted XYZ but salvage lucked into D.
   - **Do not hard-block unlock on SigLIP ABSENT** — that path is how fix1/overnight
     win when VLM sees the object (comment in ``vlm_assess`` already says this).

3. **q105 — coverage under router, not the deferral gate**
   - Overnight: late assess finds bowl on island → A (also with SigLIP ABSENT).
   - Fix1+fix2: ABSENT streak only; never ``present=True``; budget →
     ``final_location_salvage`` / query → B. ``deferred_weak_candidate`` never fired.

### Second random holdout-8b (queued behind fix3)

Paper holdout-8 is a **fixed gate** (docs: seed 20260627, excludes tuning ``3,14,17``), not a
strong population sample: only **7 scenes** (q104/q105 share one), location-heavy (4/8),
n=8. Overnight 8/8 can overfit that slice; **bal-32** is the real scale check.

Second independent draw (seed ``20260727``, paper-113 minus smoke + paper holdout):

- IDs ``4,19,36,45,57,75,84,90`` (8 scenes; bal-32 overlap ``57,84`` only)
- Job ``hmeqa-holdout8b`` / ``20260727_120126_0be117`` (waits on fix3)
- OUT ``~/runs/emet/hmeqa_holdout8b_20260727_120123``
- Same policy as fix3: paper-router, agentic-only, no ABSENT unlock gates
- Result: **2/8** (weak on this draw) — reinforces that paper holdout alone is not enough

### Fix3 result + VLM frontier explore (code)

- Fix3 paper holdout: **5/8** (miss q56/q104/q105) — premature assess + utility-only explore
- **Code:** agentic ``explore_frontier`` now prefers ``_vlm_frontier_choice`` (utility-rank
  reachable frontier RGBs ≤6, VLM picks image). ``agentic_max_nav_steps`` **5→8**.
- Classic coverage path still gated by ``EMET_VLM_FRONTIER_SCORING`` (default off).
