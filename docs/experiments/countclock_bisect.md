# HM-EQA count/clock slice regression bisect (2026-08-26)

Frozen 15-qid set: `12,21,28,32,33,43,47,48,51,60,78,84,86,88,93` via
[`scripts/run_hmeqa_countclock_slice.sh`](../../scripts/run_hmeqa_countclock_slice.sh).

## Saved run ladder

| Run tag | Commit | Score | `confident_wrong` | Notes |
|---------|--------|-------|-------------------|-------|
| `countclock_regression_20260825_175843` | `543fb8c0` | **6/15** | 4 | Pre action-contract |
| `countclock_20260825_201933` | `23efa534` | **7/15** | 2 | **Peak** — action contract; `resume=1` |
| `countclock_20260826_084546` | `c1591698` | **5/15** | 5 | +`CLOSE_LOOK_STATUS`, lean profile |
| `countclock_close_look_ctx_*` | `a32271b3` | **3/15** | 7 | +frame dedupe + `eqa_decisions/` |

Per-qid grid (OK = correct):

| qid | 6/15 | 7/15 | 5/15 | 3/15 |
|-----|------|------|------|------|
| 12 | MISS | **OK** | MISS | MISS |
| 47 | OK | **OK** | MISS | MISS |
| 48 | MISS | **OK** | MISS | MISS |
| 86 | MISS | MISS | **OK** | MISS |
| 93 | OK | **OK** | OK | MISS |

**7/15 → 3/15 regressions (lost OK):** q12, q47, q48, q93.

## Suspect commits (after peak `23efa534`)

1. **`0ebdabb5`** — lean eval output profile (unlikely score impact; fewer dumps).
2. **`290e54e5`** — center-zoom **default OFF**; prefer unattached FIND views.
3. **`3d782803`** — follow-unspent-find guard order (agentic path; classic unaffected).
4. **`c1591698`** — `CLOSE_LOOK_STATUS` prompt block + close-map audit script. Observed **7→5** cliff.
5. **`a32271b3`** — frame dedupe in `query_answer`, `eqa_decisions/` export. Observed **5→3** cliff.

## CPU postmortem findings

### Confident wrong correlates with low `eqa_iterations`

At 3/15, seven episodes ended `model_confident=true` and wrong. Several used only
1–2 EQA iterations (q48, q51, q86) vs 18–20 on the 7/15 peak for the same qids.

### `CLOSE_LOOK_STATUS` + empty HISTORY on regressions

On 3/15 bundles with `eqa_decisions/` (q12, q47, q48 final iterations):

- `CLOSE_LOOK_STATUS` present on every prompt.
- **HISTORY empty** on final iteration (budget trim dropped all prior lines, including
  iter-2 “living room, not bedroom” on q12).
- `GRAPH_COUNT` present but often pins wrong-room obs (q12: `obs1` only).
- No `FIND_QUEUE` line on final iter for q12/q47/q48.

7/15 bundles predate `eqa_decisions/` — compare via `eqa_history.json` + jsonl instead.

### q12 failure mode (3/15)

- 5 EQA iters, conf=true, answer “None” (gold: Two).
- Attached: living-room sofa (obs1) + doorway glimpse (obs58, `aimed=false` at 0.41m).
- VLM conflated white sofa with “white bedding”; never entered bedroom.

### Hypothesis

Regression is **not** one commit — it is a stack:

1. **Center-zoom off** (`290e54e5`) may hurt clock/detail reads (q47 lost at 5/15).
2. **`CLOSE_LOOK_STATUS` block** adds prompt noise; model may answer “None” confidently
   when all views show `resolved=no` (q12 at 5/15 and 3/15).
3. **Frame dedupe** (`a32271b3`) may drop diverse attached views (q93 lost at 3/15).
4. **HISTORY budget trim** drops room-mismatch lessons before final submit.

## Bisect protocol

**Canary qids:** `12,47,48,86,93` (5-qid gate before full 15-qid confirm).

```bash
uv run emet habitat safe-start   # wait for probe done
uv run emet jobs run --name bisect-<shortsha> --need-mib 12000 -- \
  env EMET_ALLOW_SDPA_ATTN=1 RESUME=0 OUTPUT_PROFILE=lean \
  QUESTION_IDS=12,47,48,86,93 RUN_ID=bisect_<shortsha> \
  ./scripts/run_hmeqa_countclock_slice.sh
```

Checkpoints: `23efa534` → `290e54e5` → `c1591698` → `a32271b3`.

Audit:

```bash
uv run python scripts/audit_close_map_eqa_slice.py \
  --jsonl ~/.cache/habitat_eqa/results/bisect_<shortsha>_dynagraph_qwen3_vl.jsonl
```

## Split plan (`feat/eqa-prompt-context`)

Do **not** dump `CLOSE_LOOK_STATUS` into classic `query_answer` prompts — that block
correlated with the 7→5 cliff (confident “None” when every view said `resolved=no`).
Close-look geometry lives on the voxel [close-map](../close_map.md) and on
`inspect_graph` catalog rows, not in the HM-EQA system prompt.

| Slice | Status | What |
|-------|--------|------|
| Infra | Done on this branch | `eqa_decisions/` export, close-map **audit** columns, inspection pack, HISTORY `obs=` ids. No frame dedupe. No confidence gate. |
| Classic SCENE_GRAPH | This change | `ATTACHED_INDEX` (Image 1..K ↔ obs id). SCENE_GRAPH / GRAPH_COUNT tag `[graph obs N]`, not `[Image {obs_id}]`. `QUESTION_ROOM` + pinned HISTORY. No `FIND_QUEUE` duplicate of GRAPH_COUNT. No `CLOSE_LOOK_STATUS`. |
| Agentic find | This change | `inspect_graph` is the query for “where do you see X, and how well does voxel/SigLIP match” (`yoloe_hit`, `siglip_sim`, compact `close_map`). Close-look **required** from the question (VLM extract **OR** count/clock/state keywords). Stay via close-map on `investigate`. Same tool names. |
| Defer | Until canary confirms | Frame dedupe, default center-zoom off, `close_look_confidence_gate`. |

Live method: [dynagraph.md](../dynagraph.md) (LLM picks tools; Python supplies `localize_text` + close-map stay).

## Tuning ladder (one knob at a time)

CPU first, then the 5-qid canary (`12,47,48,86,93`), then the 15-qid slice. Do **not**
retune on full-113. Peak to beat: **7/15** at `23efa534`.

| Knob | Default | What it changes | Gate |
|------|---------|-----------------|------|
| `EMET_CLOSE_MAP_R_M` | 0.55 | Aimed range that counts as resolved | red-cylinder + countclock canary (too tight → never leave; too loose → doorway “resolved”) |
| `EMET_CLOSE_MAP_AIM_DEG` | 25 | On-axis cone (q12 doorway: aimed=false at ~0.41 m) | same |
| `EMET_CLOSE_MAP_QUERY_RADIUS_M` | 0.35 | Neighborhood around card XY | OVMM FindObj err vs 0.3 m |
| `EMET_CLOSE_MAP_ESCAPE_ATTEMPTS` | 4 | Approaches before escape | agentic loop length |
| Voxel SigLIP gate | 0.21 (localize) / image 0.12 | Proposal vs miss | `test_voxel_localize` + oneshot S0 |
| First-hit pin vs `refresh=True` | pin | Explore must not erase mapping hit | OVMM FindObj 1/1 regression |
| `EMET_EQA_AGENTIC_CLOSE_LOOK` | on | Task close-look classifier (VLM **OR** keywords) | count/clock vs location MCQ explore |
| `EMET_EQA_HYP_RECALL_K` | 6 | Catalog size | `inspect_graph` `n_detections` |

**CPU pack (every code change):**

```bash
uv run emet test --no-sim src/test/llms/test_hmeqa_prompt_budget.py \
  src/test/memory/test_ovmm_agentic_routing.py src/test/memory/test_ovmm_agentic_find.py \
  src/test/mapping/test_voxel_localize.py src/test/mapping/test_close_map.py \
  src/test/agent/test_skill_packs.py src/test/memory/test_graph_eqa_memory.py -q
```

**GPU** (jobs only, after CPU green): `emet habitat safe-start`, then the canary via
[`scripts/run_hmeqa_countclock_slice.sh`](../../scripts/run_hmeqa_countclock_slice.sh).
Audit with [`scripts/audit_close_map_eqa_slice.py`](../../scripts/audit_close_map_eqa_slice.py)
(geometry vs score — not a prompt dump).
