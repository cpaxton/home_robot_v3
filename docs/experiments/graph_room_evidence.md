# Experiment plan: manifest-locked graph room evidence A/B

**Branch:** `feature/graph-room-evidence`  
**Code:** explicit rollout axes + versioned `run_manifest.json`; room timeline on
`GraphEQAMemory`; `AttemptRecord.room`; grounded agent state
**Refs:** [attempt_ledger.md](../attempt_ledger.md#room-timeline-graph-history),
[agentic_scale.md](agentic_scale.md),
[agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)

> **Execution hold (2026-08-13):** do not launch this experiment while job
> `20260813_103856_788d34` (`hmeqa-paper113-d1`, checkout `home_robot_v2`) is
> running. The reported `emet-habitat` process is its descendant, not a second
> job. Do not call `emet eval kill-stale`, cancel it, or start a competing
> Habitat/VLM job.

## Question and hypothesis

Test whether stable graph evidence and room-scoped history, when made visible to
the grounded decision policy, reduce wrong-room dwell, repeated inspections, and
budget exhaustion.

The experiment must first re-establish a reproducible legacy control. Existing
results do not isolate room stamps:

| Run | Result | Interpretation |
|-----|--------|----------------|
| July paper-router rooms probe | **7/11**, 34.1 mean planning steps | Historical reference only |
| Aug 12 bundled stamp+ledger treatment | **2/11**, 32.6 steps | Hard regression, but several axes changed together |
| Aug 13 plain-control partial | **3/10**, 37.8 steps; stopped before q84 | Legacy control also drifted; q6/q11/q12/q47 flipped from correct in July to wrong |

The old OUTs predate the new manifest boundary. Keep them as diagnostic
evidence; do not resume or combine them with new rows.

Primary hypothesis: variant A2 below lowers median wrong-room dwell and repeated
attempts relative to A0 on the same IDs. Letter accuracy is a safety gate, not
the optimization target on small slices.

Non-goals:

- Claiming a new paper number from a 2-, 6-, or 11-question slice
- Reintroducing room-mismatch `ESCAPE_MIN_TRAVEL_M` latches
- Changing model, budgets, IDs, verifier, router preset, or navigation stack
  between paired variants
- Testing all rollout axes at once

## Frozen variants

Every command sets every experimental axis explicitly. `--preset paper-router`
may set verifier/router defaults, but it must not change these axes.

| ID | Decision | Graph evidence | Room history | Room policy | Target hints | Investigate stamp | Ledger | Purpose |
|----|----------|----------------|--------------|-------------|--------------|-------------------|--------|---------|
| **A0** `gre-a0-legacy-r1` | `legacy` | `off` | `off` | `canonical` | on | off | `off` | Baseline recovery |
| **A1** `gre-a1-shadow-r1` | `legacy` | `shadow` | `shadow` | `canonical` | on | off | `shadow` | Collection/no-leakage control |
| **A2** `gre-a2-grounded-r1` | `grounded_v2` | `agent` | `agent` | `canonical` | on | off | `agent` | Agent-visible grounded state |
| **A3** `gre-a3-stamp-r1` | `grounded_v2` | `agent` | `agent` | `canonical` | on | **on** | `agent` | Isolate the known stamp risk; smoke only first |

Defer `room_policy=llm` and `--no-room-target-hints` until A2 passes. Each is a
separate one-axis ablation; neither belongs in the first treatment.

Fairness contract:

- Use one clean commit for all paired variants. The manifest freezes the full
  commit and dirty-tree digest; any later edit intentionally makes resume fail.
- Hold IDs, order, Qwen model/family/quantization, answer and movement budgets,
  paper-router preset, and operational crash policy constant.
- Run one GPU job at a time and preserve each OUT independently.
- Audit each manifest before comparing outputs. Config digests should differ
  only because the declared variant axes differ.

## Metrics and invariants

Collect per episode from `run_manifest.json`,
`bundles/agentic_qN/agentic_trace.jsonl`, `world_evidence.json`,
`attempt_ledger.json`, and `room_events.json`.

| Metric | Definition / gate |
|--------|-------------------|
| Manifest parity | Same git state, IDs/order, model, and budgets across paired OUTs |
| Shadow no-leakage | A1 collects evidence/history/attempts, but policy-facing prompt/state contains none of it |
| `room_event_coverage` | Fraction of target-room episodes with at least one qualified room event; A2 target **≥70%** |
| `wrong_room_dwell_rounds` | Non-target-room rounds spent on investigate/look/verify before leaving |
| `path_m_before_first_target_room` | Planar path before first target-room entry; report missing entry separately |
| Repeat attempts | Same place/tool/outcome repeated without new evidence; derive from the ledger |
| Budget behavior | Decision rounds, nav count, `budget_hit`, and forced-answer provenance |
| Escape behavior | `escape_source` histogram; no new sticky room latch is allowed |
| Secondary outcome | Letter, confidence/provenance, observations, planning steps, wall time |

Use paired per-ID deltas. Do not treat a single small-slice letter difference as
statistically meaningful. A2 passes the process gate only if median
`wrong_room_dwell_rounds` falls versus A0, repeat attempts do not increase, and
letter accuracy is no more than one answer worse on the paired six.

## Staged ladder

| Wave | Slice / variants | Pass / stop |
|------|------------------|-------------|
| **0 — CPU + freeze** | Final clean commit; targeted unit/config tests; help audit | No GPU until tests pass and the current paper113 job is gone |
| **1 — paired six** | IDs `2,6,11,12,47,76`; run A0, then A1, then A2 | A0 must recover at least **3/4** on regression IDs `6,11,12,47`; otherwise stop before attributing anything to grounded state |
| **2 — stamp isolation** | A3 on `2,76` only | Confirm stamps are actually recorded; stop on routing/pathology regression |
| **3 — rooms probe** | A0 vs A2 on `6,8,11,12,21,28,39,47,48,80,84` | Process gate above; letters are a safety gate |
| **4 — wrong-room focus** | Start `2,29,76`, then add high-dwell IDs selected from Wave 3 A0 | Dwell/path-to-target improve on paired rows |
| **5 — scale** | Holdout-8 or bal-32 | Only after Waves 1–4; write a new OUT and never overwrite paper artifacts |

If A0 scores below 3/4 on the regression IDs, repeat A0 once before declaring a
baseline failure. If both runs fail, investigate policy/provenance drift and do
not spend GPU on A1–A3.

q104 is intentionally deferred to holdout scale: it is a known native-crash hot
scene and is not needed for the baseline-recovery gate.

## Prepared harness (do not run during the execution hold)

The following commands were checked against current CLI help, but intentionally
not launched while paper113 is active.

Readiness:

```bash
uv run emet jobs
# Continue only when there are no intentional Habitat/VLM jobs or descendants.
git status --short
uv run emet habitat safe-start --need-mib 12000 \
  --job-name habitat-egl-probe-grounded-graph
# Wait for the returned probe job to be done, then inspect its logs.
```

Prefer a clean experiment commit. A dirty launch is reproducible only while the
working-tree digest remains byte-identical.

Paired-six A0:

```bash
IDS=2,6,11,12,47,76
OUT_A0=~/runs/emet/gre_a0_legacy_r1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT_A0" --preset paper-router --arms agentic --ids "$IDS" \
  --decision-policy legacy --graph-evidence-mode off --room-history-mode off \
  --room-policy canonical --room-target-hints --no-investigate-stamp \
  --attempt-ledger-mode off --variant-id gre-a0-legacy-r1 \
  --job-name gre-a0-legacy-r1 -d "Graph-room evidence paired-six A0 legacy control"
```

Paired-six A1:

```bash
OUT_A1=~/runs/emet/gre_a1_shadow_r1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT_A1" --preset paper-router --arms agentic --ids "$IDS" \
  --decision-policy legacy --graph-evidence-mode shadow --room-history-mode shadow \
  --room-policy canonical --room-target-hints --no-investigate-stamp \
  --attempt-ledger-mode shadow --variant-id gre-a1-shadow-r1 \
  --job-name gre-a1-shadow-r1 -d "Graph-room evidence paired-six A1 shadow"
```

Paired-six A2:

```bash
OUT_A2=~/runs/emet/gre_a2_grounded_r1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT_A2" --preset paper-router --arms agentic --ids "$IDS" \
  --decision-policy grounded_v2 --graph-evidence-mode agent --room-history-mode agent \
  --room-policy canonical --room-target-hints --no-investigate-stamp \
  --attempt-ledger-mode agent --variant-id gre-a2-grounded-r1 \
  --job-name gre-a2-grounded-r1 -d "Graph-room evidence paired-six A2 grounded"
```

Launch one command only after the previous job is terminal. Record commit, job
ID, OUT, and the exact next command before each launch. Resume only the same OUT:

```bash
uv run emet hmeqa resume "$OUT_A0" --job-name gre-a0-legacy-r1-resume
```

Omitted frozen flags come from the manifest; operational job metadata may
change. A commit, dirty-tree, or frozen-config mismatch must fail closed.

## Analysis recipe

Before interpreting an OUT:

1. Validate `run_manifest.json` and record its variant ID/config digest.
2. Confirm paired manifests have identical git state, IDs/order, model, and
   budgets.
3. Confirm A1 shadow rows exist in storage but do not appear in policy-facing
   state or prompts.
4. Produce a per-ID CSV with the metrics above and paired deltas A1−A0 and
   A2−A0.
5. Qualitatively inspect q2 plus the two largest dwell improvements/regressions.

A CPU-only summarizer is a prerequisite before Wave 3; add
`scripts/summarize_graph_room_evidence.py` rather than relying on ad-hoc grep
counts. It should accept multiple OUTs, fail on manifest parity violations, and
write both JSON and CSV.

## Decision rules

| Result | Action |
|--------|--------|
| A0 baseline does not recover after one repeat | Stop; diagnose policy/provenance drift before any treatment |
| A1 data missing | Fix collectors/writers before A2 |
| A1 evidence leaks into policy state | Fix shadow semantics; invalidate that run |
| A2 history present but dwell unchanged | Improve state-card salience, still without an escape latch |
| A2 dwell/repeats improve and letters are within the safety gate | Promote to paired rooms-11 |
| A3 regresses | Keep investigate stamps off; retain only diagnostic writers |
| Letters regress hard | Keep timeline for diagnostics only; do not enable stamp+ledger on paper of-record until understood |
| Temptation to add sticky min-travel from room mismatch | **Reject** — out of scope; reopen only as a separate ablation labeled as a latch |

## Status

| Wave | Status | OUT / notes |
|------|--------|-------------|
| Prior q2 pilot | archived | `hmeqa_room_evidence_quick_20260812_122709`; useful wiring evidence only |
| Prior bundled treatment | archived | `hmeqa_room_evidence_w2_20260812_231348`; 2/11, confounded |
| Prior control partial | archived | `hmeqa_room_evidence_ctrl_w2_20260813_001608`; 3/10, stopped before q84, no new manifest |
| Execution hold | **active** | Wait for paper113 job `20260813_103856_788d34`; do not interfere |
| Wave 0 | prepared | Phase-one CLI/manifest targeted CPU/config suite: 171 passed on 2026-08-13; rerun on final clean commit |
| Wave 1 A0 | queued, not launched | Paired six after hold clears |
| Waves 1 A1/A2 | blocked on A0 gate | |
| Waves 2–5 | blocked | |

## Related

- Abandoned latch PR: [#107](https://github.com/cpaxton/home_robot_v3/pull/107) (escape floor on room mismatch)
- Prior rooms+verify probe: [agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)
- Scale ladder: [agentic_scale.md](agentic_scale.md)
