# Agentic HM-EQA scale experiments

Branch: **`exp/agentic-hmeqa-scale`** (cut from `main` after PR #77 merge).

Goal: test whether classic vs agentic-verify Dynagraph gains hold past holdout-8.

## Ladder

| Wave | What | Status |
|------|------|--------|
| 0 | Flash-Attn in `.venv-habitat`; bundle-tag smoke; this doc | done (SDPA fallback) |
| 1 | **Balanced-32** classic vs agentic H2H (primary) | running |
| 2 | Paper-20 **or** annotated-semantics H2H (one night) | after go/no-go |
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
# Wave 1 (prefer nohup; one GPU job)
OUT=~/runs/emet/hmeqa_agentic_bal32_$(date +%Y%m%d_%H%M%S)
nohup env EMET_ALLOW_SDPA_ATTN=1 \
  HOLDOUT_IDS=2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84 \
  COVERAGE_QIDS=15,104,68 \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT" \
  >> ~/runs/emet/hmeqa_agentic_bal32_nohup.log 2>&1 &

# After DONE: minimal paper data
uv run python scripts/summarize_hmeqa_agentic_h2h.py "$OUT"
cp "$OUT/h2h_summary.json" paper/data/hmeqa_agentic_h2h/balanced32_summary.json
```

## Go / no-go (after Wave 1)

- If agentic **accuracy ≤ classic** on bal-32 → stop; diagnose before any 113 agentic.
- If agentic **wins or ties** with clearly lower mean planning steps → Wave 2 or Wave 3 classic 113.

## Wave 0 notes (2026-07-23)

- Bundle-tag smoke: `h2h_smoke_classic_q0015` → `~/.cache/habitat_eqa/episodes/h2h_smoke_classic_q0015/q0015_dynagraph/` (Q15 classic correct, 60 steps).
- **Flash-Attn in `.venv-habitat`:** blocked — system CUDA toolkit is 12.4 while Habitat torch is `2.12.0+cu130`. Main `.venv` already has flash-attn 2.8.3. Habitat H2Hs use `EMET_ALLOW_SDPA_ATTN=1` until toolkit/torch align (or a prebuilt cu130 wheel is available).

## Related

- Holdout results: [habitat_eqa_results.md](habitat_eqa_results.md) (Classic vs agentic-verify)
- Minimal checked-in summaries: [`paper/data/hmeqa_agentic_h2h/`](../../paper/data/hmeqa_agentic_h2h/), [`paper/data/README.md`](../../paper/data/README.md)
- Do not chain Robocasa + full pytest + Habitat VLM in one session.
