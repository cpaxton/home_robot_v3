# Agentic HM-EQA scale experiments

Results branch: **`exp/agentic-hmeqa-bal32-results`** (off `main` after scale-doc PR #79).
Live Wave 1 run: `~/runs/emet/hmeqa_agentic_bal32_20260723_212307`.

Goal: test whether classic vs agentic-verify Dynagraph gains hold past holdout-8.

## Ladder

| Wave | What | Status |
|------|------|--------|
| 0 | Flash-Attn in `.venv-habitat`; bundle-tag smoke; scale doc (#79) | done (SDPA fallback) |
| 1 | **Balanced-32** classic vs agentic H2H (primary) | **DONE — agentic miss** (12/32 classic vs 9/32 agentic) |
| 1b | Fail-set ablation after answer-token fix (holdout-8 + classic_only) | next |
| 2 | Paper-20 **or** annotated-semantics H2H (one night) | blocked until 1b recovers |
| 3 | Full 113 classic dynagraph + `graph_eqa`; agentic 113 only if Wave 1–2 support claim | later |

## Harness (all Habitat H2Hs)

- Method: `dynagraph`
- VLM: `Qwen/Qwen3-VL-8B-Instruct`
- `explore_when_uncovered=off`, `--no-mcq-debias`, `--memory-summary`
- Agentic: `EMET_EQA_AGENTIC_VERIFY=1`, `EMET_EQA_AGENTIC_ROUTER=0`
- Classic: `EMET_EQA_AGENTIC_VERIFY=0`
- Always use distinct `--debug-run-tag` / `OUT/bundles/{arm}_qN` (see `scripts/run_hmeqa_agentic_h2h.sh`)

## Balanced-32 IDs

Same letter-balanced set as overnight scripts:

```text
2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84
```

## Commands

```bash
# Wave 1 (prefer emet jobs; one GPU job)
OUT=~/runs/emet/hmeqa_agentic_bal32_$(date +%Y%m%d_%H%M%S)
uv run emet jobs run --name hmeqa-agentic-bal32 --need-mib 12000 -- \
  env EMET_ALLOW_SDPA_ATTN=1 \
  HOLDOUT_IDS=2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84 \
  COVERAGE_QIDS=15,104,68 \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT"

uv run emet jobs                 # progress / ETA
uv run emet jobs logs JOB_ID --tail 40

# Resume after crash (skip non-empty per-qid jsonl; finish missing + agentic):
uv run emet jobs run --name hmeqa-bal32-finish --need-mib 12000 --out-dir "$OUT" -- \
  env EMET_ALLOW_SDPA_ATTN=1 RESUME=1 ARMS=classic,agentic SKIP_KILL_STALE=1 \
  HOLDOUT_IDS=2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84 \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT"

# After DONE: minimal paper data
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
cp "$OUT/h2h_summary.json" paper/data/hmeqa_agentic_h2h/balanced32_summary.json
```

## Go / no-go (after Wave 1)

- **Result (2026-07-24):** classic **12/32**, agentic **9/32**, mean steps 52.5 vs 18.0.
- Root cause for agentic miss: submit path forced `EMET_EQA_ANSWER_MAX_NEW_TOKENS=64` → **32/32 `[salvage]`** answers; classic used 256. Classic-only losses: **11,14,28,39,47**.
- If agentic **accuracy ≤ classic** on bal-32 → stop; diagnose before any 113 agentic. **Triggered — tune before Wave 2.**
- If agentic **wins or ties** with clearly lower mean planning steps → Wave 2 or Wave 3 classic 113.

## Fail-set ablation (Wave 1b)

After removing the 64-token clamp (+ traces in episode bundles):

```bash
OUT=~/runs/emet/hmeqa_agentic_failset_$(date +%Y%m%d_%H%M%S)
# Holdout-8 regression + classic_only recovery IDs
IDS=15,56,65,68,79,88,104,105,11,14,28,39,47
uv run emet jobs run --name hmeqa-failset-a1 --out-dir "$OUT" -- \
  env EMET_ALLOW_SDPA_ATTN=1 EMET_EQA_TRACE=1 SKIP_KILL_STALE=1 SKIP_GPU_WAIT=1 \
  ARMS=classic,agentic HOLDOUT_IDS="$IDS" \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT"
```

Pass: agentic recovers ≥4/5 classic_only and holdout-8 ≥7/8.

### Fail-set regressions found (2026-07-24)

Holdout-8 overnight was **agentic 8/8 vs classic 5/8**. Re-running agentic after removing the 64-token clamp regressed until:

1. **NL choice text → false abstain → wrong `[memory-location]`** (q56): VLM answered “The room with the blue curtains” (=C) but letter parse only looked for bare `A–D`, so nearest-furniture memory overwrote to **A**. Fix: use `extract_mcq_letter(answer, choices)` before override; normalize matched choice text to a letter (`[choice-text]`).
2. **`Answer: Unknown` skipped salvage** (q65): non-empty Unknown blocked the empty-answer salvage path that night’s 64-token truncations used. Fix: treat Unknown/none as emptyish for salvage.
3. **`libcuda` SIGSEGV on scene `yogvKWUrdnw`** (q104/q105): Habitat EGL + Qwen3-VL vision generate mid-episode. Mitigation: `torch.cuda.synchronize()` before multimodal generate in `qwen3_vl_client`.
4. **Empty letters + `n_object=0` on q104/q105 (failfix4):** an experiment tied `EMET_EQA_AGENTIC_VERIFY=1` to skipping per-frame VLM graph label extract. On scenes without HM3D semantics that left only `["object"]` → nav samples/frontiers; answer prompts attached black 8×8 frontier placeholders; location MCQ abstain → `pred=""`. **failfix5** (`20260724_144029_f7ace2`): `n_object=8/8`, non-empty letters, 33× `max_new=128` vision extracts — but letters still wrong (B/B vs gold D/A). Checked-in: `paper/data/hmeqa_agentic_h2h/failset104105_summary.json`. Guards: never answer off frontier placeholders; phrase ranking for SigLIP; Action Image N via graph `obs_id`; agentic verify must not auto-disable label VLM.
5. **failfix5 wrong B letters (grounding):** truncated VLM (no `answer:`) → empty → `[memory-location]` invented B; agentic full MCQ string made SigLIP phrases like `table sunroom answer` instead of `fruit bowl`. Fix: `question_stem_for_keywords()` strips `A)…Answer:` before phrase/object heuristics; location MCQ empty/Unknown skips memory-location and salvage-location invent (normalize empty → `Unknown`).
6. **failfix6 round waste (fallback policy):** with the router off, `_fallback_tool` allowed only one `explore_frontier` ever, so once nav hypotheses were consumed it re-ran an identical ABSENT `verify_siglip` for 5/8 rounds and the final submit came from the `budget_hit` path, which skipped the Action:N follow entirely. The default verify phrase also ranked MCQ-option nouns (`kitchen island`) above stem phrases (`fruit bowl`) because VLM keyword extract feeds option nouns into `_relevant_objects`. Fixes: fallback explores while nav budget remains (stop when motion happened and frontiers are gone), submits as soon as budget is spent and a verify is on record; `budget_hit` submit honors one Action:N/unknown-explore follow-up; verify phrase ranking prefers phrases occurring in the question stem. Guards in `src/test/eval/test_agentic_eqa_verification.py` (`test_fallback_*`, `test_verify_phrase_prefers_question_stem_over_mcq_option`).

Keep the 64-token clamp **removed** (bal-32 salvage bug); do not reintroduce `setdefault("64")`. Do **not** reintroduce agentic-verify auto-skip of `SensorGraphBuilder` VLM extract.

## Wave 0 notes (2026-07-23)

- Bundle-tag smoke: `h2h_smoke_classic_q0015` → `~/.cache/habitat_eqa/episodes/h2h_smoke_classic_q0015/q0015_dynagraph/` (Q15 classic correct, 60 steps).
- **Flash-Attn in `.venv-habitat`:** blocked — system CUDA toolkit is 12.4 while Habitat torch is `2.12.0+cu130`. Main `.venv` already has flash-attn 2.8.3. Habitat H2Hs use `EMET_ALLOW_SDPA_ATTN=1` until toolkit/torch align (or a prebuilt cu130 wheel is available).

## Related

- Holdout results: [habitat_eqa_results.md](habitat_eqa_results.md) (Classic vs agentic-verify)
- Minimal checked-in summaries: [`paper/data/hmeqa_agentic_h2h/`](../../paper/data/hmeqa_agentic_h2h/), [`paper/data/README.md`](../../paper/data/README.md)
- Do not chain Robocasa + full pytest + Habitat VLM in one session.
