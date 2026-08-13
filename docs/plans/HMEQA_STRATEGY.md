# HM-EQA strategy (post-#114 verify-gate merge)

Consolidated plan for the next HM-EQA milestones on top of `main` after PR #114
(cross-family batch, world-frame nav fixes, perception throttling, and the
**verify-gate reset** in `_begin_policy_approach`).

**Branch:** `feat/hmeqa-strategy` (plan + test fixes + run ladder; no room-timeline code).
**Room timeline A/B** continues on `feature/graph-room-evidence` (separate; coordinate GPU).
**Refs:** [habitat_eqa_results.md](../experiments/habitat_eqa_results.md) (numbers),
[agentic_scale.md](../experiments/agentic_scale.md) (of-record ladder),
[graph_room_evidence.md](../experiments/graph_room_evidence.md) (rooms A/B),
[joint agentic loop](../experiments/hmeqa_joint_agentic_loop.md) (cross-task loop).

## Why this now

1. PR #114 merged a **verify-gate behavior change** into the *shared* agentic loop:
   `_begin_policy_approach` resets the evidence policy from ANSWER → SEARCH when a
   **new** hypothesis is re-investigated. That code is used by HM-EQA Habitat
   episodes, OVMM find, dynamic-exploration world-change, and the `run_dynagraph`
   CLI. It was validated on OVMM find (molmo: rounds 8 → 1) but **not re-validated
   on HM-EQA**, whose of-record numbers predate the change.
2. The **wave-2b room-evidence control** (no stamp/ledger) scored **3/10** on the
   rooms probe vs **7/11** prior — a drop on *untreated* ids on merged main. Either
   run variance or the merged loop changed HM-EQA letters. Must be checked before
   quoting any new number.
3. Pre-existing agentic-loop **unit tests were broken on main** (stale vs NavOutcome
   `#111` and single-view present-confirm `#102`). Fixed on this branch so the loop
   has a green gate.
4. Larger VLM (Qwen3-VL-32B int4) smoke passed (peak 20.4/24.5 GB) — the biggest
   accuracy lever per `agentic_scale.md`, still unrun.

## Track A — Regression check on HM-EQA (gate first)

**Goal:** confirm the #114 verify-gate reset did not regress HM-EQA agentic; fix if
it did. The reset only fires when switching to a *different* hypothesis while the
policy is in ANSWER — rare in HM-EQA (single-hypothesis / same-obs re-verify), so
most episodes are unaffected, but it must be shown empirically.

| Wave | Slice | Command | Pass / stop |
|------|-------|---------|-------------|
| A0 | Unit + CPU | `uv run emet test src/test/eval/test_agentic_eqa_verification.py src/test/memory/test_attempt_ledger.py src/test/eval/test_hmeqa_launch.py src/test/memory/test_room_policy.py src/test/memory/test_graph_eqa_memory.py -q` | all green (done on branch) |
| A1 | holdout-4 gate `{15,68,105,17}` | agentic router-off `emet hmeqa h2h --arms agentic --ids 15,68,105,17` | ≥3/4, steps ≲20 vs 19.3 baseline |
| A2 | holdout-8 `{15,56,65,68,79,88,104,105}` | agentic router-off (matches 8/8 headline) | ≥6/8, steps ≲20 vs 18.3 baseline |
| A3 | bal-32 `(32 ids)` | agentic paper-router composite compare | only if A2 healthy; 16/32 baseline |

If A2 letters regress: inspect whether the reset fires on HM-EQA episodes
(single-hypothesis questions shouldn't trigger it); if it does, gate the reset to
OVMM-style multi-hypothesis runs or revert for the HM-EQA path.

## Track B — Joint agentic loop across tasks (primary goal)

**Goal:** prove the one agent/memory backend (`AgenticEQAExecutor` +
`run_agentic_eqa`) works across tasks, and close/document consumer divergence.
Experiment doc: [hmeqa_joint_agentic_loop.md](../experiments/hmeqa_joint_agentic_loop.md).

1. **B0 audit** — consumer table (done on branch; see experiment doc): HM-EQA,
   OVMM find, dynamic-exploration world-change, `run_dynagraph` CLI.
2. **B1 cross-task smoke** — one HM-EQA holdout-4 id + one OVMM find ep (molmo or
   table) + one dynamic-exploration world-change ep, all agentic on this branch.
   Success: verify → ANSWER → submit works in all three sims; zero
   `invalid in state ANSWER` warnings.
3. **B2 process metrics** — per-consumer `_verified` / ANSWER / submit trace stats
   from `agentic_trace.jsonl`.
4. **B3 write-up** — what "our method" is as one loop across tasks, with the knob
   table (mirror `agentic_scale.md` "do not mix knobs" rule).

## Track C — Room-evidence ladder (coordinate, don't conflict)

- Continues on `feature/graph-room-evidence`; **one Habitat job at a time**.
- Wave-2b control (untreated rooms probe) finished 10/11 with **3/10** on merged main —
  record it; treat as the new untreated baseline pending Track A regression result.
- Wave 3 wrong-room focus `{6,11,48,2}` treatment (stamp opt-in via
  `EMET_EQA_ROOM_STAMP_INVESTIGATE=1`) only after Track A tells us whether the
  merged loop changed letters.

## Track D — Accuracy levers (after A/B pass)

Ordered by paper priority, all gated:

1. **Full-113 dynagraph @ 8B + nav stack** (paper §04 item #1) — **DONE 2026-08-13**: dynagraph **44.2%** (50/113, steps 50.1) vs static_graph **37.2%** (42/113) — **+7 pp memory gain at scale**. Job `hmeqa-paper113-d1`, OUT `~/runs/emet/hmeqa_paper113/20260813_104004`. Required the `run-batch --debug-run-tag` fix (committed `8fb7c1a5`) + `EMET_ALLOW_SDPA_ATTN=1`.
2. **32B int4** holdout-4 agentic → holdout-8 if healthy
   (`--eqa-hf-model-id Qwen/Qwen3-VL-32B-Instruct`); abort on OOM/EGL streak.
3. **Matched static_graph** on the same 113 (isolate Dynagraph memory gains) — runs after dynagraph in the same job.
4. Optional: semantics on/off on Q0–19, frontier ablation, API-VLM canonical-8
   upper bound.

## GPU coordination

- One Habitat job at a time (`emet jobs run --need-mib 12000`; never inline
  Habitat in an agent turn).
- Check `uv run emet jobs` + `uv run emet eval status` before launching A2/B1.
- After a crash: `sudo dmesg -T | rg 'segfault|libcuda'` · `emet eval diagnose` ·
  `emet jobs` (see [known_issues.md](../known_issues.md)).

## Status

| Item | Status |
|------|--------|
| A0 unit gate | **done** (branch: fixed stale tests + empty-pose guard) |
| B0 consumer audit | done (see joint-loop doc) |
| A1 holdout-4 | **draw 1: 2/4, draw 2: 1/4** (combined 3/8). q105→B, q17→A identical wrong letters in both draws. **0 invalid-in-state ANSWER, 0 assess-rejected** — verify gate runs clean. q15/q68 stable-varies. Prior 4/4 was also `confident=False` (search-limited, not clean confirms). **Conclusion: #114 reset safe for HM-EQA; letter drop = documented search-failure variance** (target never surfaces in graph → final MCQ on incomplete evidence) |
| A2 holdout-8 | **done: 5/8** (q15, q56, q79, q88, q104 ✓; q65, q68, q105 ✗) vs 8/8 headline. Gate works: **7 `answerable_confirmed`, 16 present=True assesses** across holdout; 5 episodes confirmed, 3 (q79/q88/q105) never saw target (search misses). **0 invalid-in-state / 0 assess-rejected** — #114 reset confirmed non-regressing. 5/8 matches documented paper-router variance (docs: 5/8 on this slice; classic swings 25–100%) |
| B1 cross-task smoke | **done: 2/3 clean + 1 environmental stall** — HM-EQA (Habitat) 7 confirms/0 errors; OVMM find (molmo) verified round 1 / 0 errors; dynamic world-change ran 7 vision assesses / 0 errors then exit 143 on post-relocation explore-stall. **0 `invalid in state ANSWER` across all legs** |
| Wave-2b control result | 3/10 on merged main (v4, terminated at 10/11) |