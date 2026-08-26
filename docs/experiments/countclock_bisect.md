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

## Split plan

- **Infra PR** (`feat/eqa-decision-traces`): `eqa_decisions/` export, audit columns,
  inspection pack, HISTORY `obs=` ids — **no** dedupe, **no** CLOSE_LOOK behavior.
- **Prompt-first PR** (on peak + infra): `QUESTION_ROOM`, pinned HISTORY, CLOSE_LOOK
  header docs — **no** server-side confidence gates.
- **Defer / revert** until bisect confirms: frame dedupe, default center-zoom off,
  `close_look_confidence_gate`.
