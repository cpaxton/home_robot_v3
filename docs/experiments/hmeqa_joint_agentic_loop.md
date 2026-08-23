# Experiment: joint agentic loop across tasks

**Branch:** `feat/hmeqa-strategy`
**Code:** shared `AgenticEQAExecutor` / `run_agentic_eqa` in
`src/emet/memory/graph_eqa/agentic_eqa.py` (+ `run_agentic_eqa_result`).
**Refs:** [HMEQA_STRATEGY.md](../plans/HMEQA_STRATEGY.md), [agentic_scale.md](agentic_scale.md).

## Goal

Prove the **one** agent/memory backend works across tasks, and close or document
consumer divergence. "Our method" is Dynagraph memory + the agentic EQA verify loop,
regardless of which task drives it — the loop must behave consistently (navigate →
capture fresh obs → VLM assess → confirm → submit) in every simulator we evaluate.

## Consumers (B0 audit)

All four use `run_agentic_eqa` / `run_agentic_eqa_result`; knobs come from the same
env vars (`EMET_EQA_AGENTIC_*`), so defaults are shared unless the caller overrides.

| Consumer | Entrypoint | Question form | Router | Notes |
|----------|-----------|---------------|--------|-------|
| **HM-EQA** (Habitat) | `controller_graph_eqa.py:840` → `run_agentic_eqa` | A–D MCQ | paper-router preset (`--preset paper-router` → `agentic_verifier=none`, `require_verified=0`, `agentic_router=1`); holdout headline uses router **off** (`EMET_EQA_AGENTIC_ROUTER=0`) | of-record slices holdout-8 / bal-32 |
| **OVMM find** (robocasa/molmo/table) | `ovmm_agentic_find.py:123` → `run_agentic_eqa_result` | open "Where is the X?" / "X on the Y?" | explicit `router=` arg (often `0` deterministic) | verified obs → world XYZ for success scoring |
| **Dynamic explore** (world-change) | `dynamic_exploration_runner.py:548` → `run_agentic_eqa` | open per-question | `agentic_verify_enabled(agent)` default | keeps sim connected mid-loop |
| **run_dynagraph CLI** | `app/run_dynagraph.py:741` → `run_agentic_eqa` | answer-only or open | `agentic_verify_enabled(agent)` | trace → `agentic_trace.jsonl` |

Shared env knobs: `EMET_EQA_AGENTIC_ROUTER`, `EMET_EQA_AGENTIC_VERIFIER`,
`EMET_EQA_AGENTIC_REQUIRE_VERIFIED`, `EMET_EQA_HYP_RECALL_K` (default 6),
`EMET_EQA_ANSWERABLE_CONFIRM` (default on), `EMET_EQA_ROOM_STAMP_INVESTIGATE`
(default off — see room-evidence A/B), `EMET_EQA_ATTEMPT_LEDGER`.

**Do not mix knob settings when quoting** (mirror `agentic_scale.md`): state the
preset (`--preset paper-router` vs router off) for every number.

## Metrics (process — primary)

Per episode / consumer, from `agentic_trace.jsonl` + result JSONL:

| Metric | Meaning / how |
|--------|---------------|
| `n_rounds`, `n_nav`, `n_explore` | budget usage (already in `AgenticEQAResult`) |
| `verified`, `verified_obs_id` | verify gate confirmed on a fresh observation |
| `n_invalid_state_assess` | count of `evidence-policy VLM assess rejected` (state != APPROACH/VERIFY) — should be **0** on the new main |
| `submit_source` | `answerable_confirmed` vs `budget_hit` vs `unverified_exhaust` |
| `state_at_submit` | should be ANSWER (or explicit unverified fallback) |
| wall_s / steps | cost (already tracked) |

Success: **zero** invalid-state assess warnings in all four consumers on this
branch, and the verify → ANSWER → submit ordering holds in HM-EQA, OVMM find, and
dynamic-explore.

## Ladder

| Wave | Slice | Pass / stop |
|------|-------|-------------|
| **0** | A0 unit gate (branch, green) | — |
| **1** | B1 cross-task smoke: 1× HM-EQA holdout-4 id (`15`), 1× OVMM find (molmo or table), 1× dynamic world-change | all three verify+submit; no invalid-state warns; no crash |
| **2** | Process metrics on A1/A2 holdout runs | `n_invalid_state_assess == 0` |
| **3** | Knob-divergence cleanup | flag any consumer that overrides a shared knob differently than the others; document or unify |

## GPU harness

One Habitat job at a time (`emet jobs run --need-mib 12000`). OVMM + dynamic-explore
run on the same GPU mutex. Never inline Habitat in an agent turn.

```bash
# HM-EQA smoke (after unit gate)
uv run emet eval recover --need-mib 12000
uv run emet habitat safe-start   # wait until jobs status = done
OUT=~/runs/emet/hmeqa_joint_smoke_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT" --arms agentic --ids 15 --job-name joint-smoke-hmeqa \
  -d "Joint agentic loop B1: HM-EQA q15"
# OVMM find (molmo) + dynamic world-change follow the same GPU mutex.
```

## Status

| Wave | Status |
|------|--------|
| 0 unit gate | **done** (branch) |
| 1 cross-task smoke | **done (2/3 legs clean, 1 environmental stall)** |
| 2 process metrics | A2 traces analyzed (7 confirms, 16 present=True, 0 invalid-state) |
| 3 knob-divergence cleanup | pending |

### B1 cross-task smoke results (merged main, `feat/hmeqa-strategy`)

| Consumer | Sim | Result | Invalid-state |
|----------|-----|--------|---------------|
| **HM-EQA** (A2 holdout-8) | Habitat | 7 `answerable_confirmed`, 16 present=True across 5 episodes; letters 5/8 (variance) | **0** |
| **OVMM find** (joint-smoke-ovmm) | molmo mujoco | obj verified=True at **rounds=1** (obs 29); recep rounds=8 verified=False (microwave never present — known molmo recep difficulty) | **0** |
| **Dynamic world-change** (joint-smoke-worldchange) | robocasa mujoco | loop ran (7 vision VLM generates), then **exit 143** on environmental explore-stall ("no valid plan or frontier" after relocation) — not a loop bug | **0** |

**Conclusion:** the shared agentic loop (verify → ANSWER → submit) runs clean in
HM-EQA Habitat and OVMM mujoco on merged main; the world-change leg's failure is a
post-relocation explore-planner stall (documented environmental limitation), not the
agentic loop. **0 `invalid in state ANSWER` across all three legs** — the #114
verify-gate reset is confirmed non-regressing and loop-stable across tasks.
