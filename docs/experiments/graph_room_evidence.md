# Experiment plan: manifest-locked graph room evidence A/B

**Branch:** `feature/graph-room-evidence`
**Code:** explicit rollout axes + versioned `run_manifest.json`; room timeline on
`GraphEQAMemory`; `AttemptRecord.room`; grounded agent state
**Refs:** [attempt_ledger.md](../attempt_ledger.md#room-timeline-graph-history),
[agentic_scale.md](agentic_scale.md),
[agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)

> **Status (2026-08-23): no-go for scale.** The strongest clean pair still has
> A2 grounded at 3/6 versus A1 shadow at 4/6, with more planning steps and
> budget hits. The final `b91059e5` integrity canary then produced two
> native-clean q11 attempts; both failed to see the trash can directly,
> exhausted all eight rounds, and answered A instead of D. The run also exposed
> an over-strict best-effort object-crop completion check, now covered by a
> CPU regression. The capability gate remains failed, so no A0/A1/A2 scale work
> should start.

## Question and hypothesis

Test whether stable graph evidence and room-scoped history, when made visible to
the grounded decision policy, reduce wrong-room dwell, repeated inspections, and
budget exhaustion.

The experiment re-established a reproducible legacy control. Earlier results
remain diagnostic-only because they did not isolate room stamps:

| Run | Result | Interpretation |
|-----|--------|----------------|
| July paper-router rooms probe | **7/11**, 34.1 mean planning steps | Historical reference only |
| Aug 12 bundled stamp+ledger treatment | **2/11**, 32.6 steps | Hard regression, but several axes changed together |
| Aug 13 plain-control partial | **3/10**, 37.8 steps; stopped before q84 | Legacy control also drifted; q6/q11/q12/q47 flipped from correct in July to wrong |

The old OUTs predate the new manifest boundary. Keep them as diagnostic
evidence; do not resume or combine them with new rows.

The primary hypothesis was that A2 would lower wrong-room dwell and repeated
attempts without a material letter regression. The completed paired-six did not
support that hypothesis.

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
  paper-router preset, and operational crash policy constant. New manifests also
  hash the question/init-pose CSVs, freeze the HM3D root path, apply the declared
  quantization/tool/nav budgets at runtime, and reject unfrozen `EMET_EQA_*`
  policy overrides.
- Run one GPU job at a time and preserve each OUT independently.
- Audit each manifest before comparing outputs. Config digests should differ
  only because the declared variant axes differ.

## Completed evidence

The strongest locally controlled diagnostic comparison is the clean same-commit
`fae4b89c` pair:

| Variant / OUT | Result | Process |
|---------------|--------|---------|
| A1 shadow — `~/runs/emet/gre_a1_shadow_fae4b89c_20260814_131815` | **4/6**: q6, q11, q12, q76 | 29.2 mean planning steps; 2 budget hits; 621 evidence views, 55 ledger rows, 11 room events; all 40 policy-visible event lists empty |
| A2 grounded — `~/runs/emet/gre_a2_grounded_fae4b89c_20260814_160721` | **3/6**: q6, q11, q47 | 34.2 mean planning steps; 3 budget hits; room events on 3/6 |

A2 gained q47 but lost q12 and q76. It did not demonstrate lower dwell or fewer
repeats, and the planned CPU summarizer was not implemented. This is a no-go for
promotion. The pair measures the entire grounded treatment—decision policy plus
graph/history/ledger visibility—not graph evidence in isolation.

Other manifest-era results:

| OUT | Job | Result / use |
|-----|-----|--------------|
| `gre_a0_resolver_e85cda20_20260813_232912` | `20260813_232917_c9edba` | 4/6; A0 regression gate 3/4 |
| `gre_a1_shadow_e85cda20_20260814_000315` | `20260814_000321_dd2556` | 3/6; valid outcomes, but shadow artifacts were not persisted |
| `gre_a2_grounded_imgfirst_20260814_171000` | `20260814_171004_86674a` | cancelled after two IDs; invalid comparison |
| `gre_a2_grounded_imgfirst_postmain_20260815_011514` | `20260815_011843_f470c3` | dirty diagnostic, 3/6; no matched control |
| `gre_q11_a2_canary_d3ee32e8_20260822_175238` | `20260822_175646_219c08` | dirty GT-free diagnostic, **1/1**; verified semantic D from direct obs 76, no salvage/native failure |
| `gre_q11_a2_clean_81a2c689_20260823_003735` | `20260823_004358_462498` | clean GT-free repeat, **0/1**; no native failure, but all checks were absent/not-answerable and budget fallback produced A instead of D |

The clean run's first registered launcher (`20260823_004111_23a7d7`) exited
before Habitat because the inner script rejected the manifest just created by
the outer CLI. The recorded job is the official same-OUT `emet hmeqa resume`;
it preserved the clean manifest and produced the only episode row. The
CLI-to-script prepared-manifest handoff is fixed in the follow-up commit.

There is no clean same-commit A0/A1/A2 triplet. The principal A1/A2 manifests
also predate explicit freezing of HM3D semantics and per-question enrich labels,
so they do not establish a GT-free perception result.

Per-question findings:

- **q2:** target parsing admitted bathroom and bedroom, suppressing mismatch
  handling; the agent could not bind rug color to the shower and hit budget.
- **q6:** robust after answer-resolution fixes; useful visual reasoning despite
  noisy graph-room labeling, not a demonstrated room-history gain.
- **q11:** prior correct letters were plausibility guesses, and three complete
  GT-free pre-fix follow-ups scored 0/3. The post-fix dirty canary reached the
  kitchen, investigated obs 76, and directly returned
  `Next to the refrigerator` from a present+answerable view. That 1/1 validates
  the corrected path once, but it needs a clean repeat before promotion.
- **q12:** the graph undercounted nightstands while images supported two; A2
  trusted the graph and regressed. The dirty image-first diagnostic recovered
  the answer, supporting evidence precedence.
- **q47:** A2 eventually captured a clear wall-clock view and fixed
  reason/answer consistency; this was a visual-evidence gain, not room routing.
- **q76:** A2 exhausted six frontier explores without entering a bedroom. A1's
  correct fallback was not evidence of successful target-room search.

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

## Conservative next ladder

| Wave | Slice / variants | Pass / stop |
|------|------------------|-------------|
| **0 — CPU + freeze** | Correctness tests, compact round-trip, CLI/help/doc audit; clean commit with HM3D semantics and enrich labels explicitly off | No simulator or VLM until all CPU gates pass |
| **1 — q11 canary** | A2 on `11`; repeat once only after a pass | No exception/ID rewrite; reaches kitchen; direct evidence supports the relation; no plausibility or budget fallback |
| **2 — baseline gate** | A0 on `6,11,12,47` | ≥3/4; repeat once if below, then stop all treatment work on a second failure |
| **3 — shadow gate** | A1 on the same four | All artifacts present; every policy-visible event list empty; no severe letter regression |
| **4 — grounded gate** | A2 on the same four | At most one letter below A1; budget hits and mean steps no worse; grounded q11 and q12/q47 reason-letter consistency |
| **5 — hard-room pair** | A1 then A2 on `2,76`; repeat the pair once | q2 reaches bathroom and binds rug/shower color; q76 reaches bedroom; dwell/repeats improve |
| **6 — formal paired six** | Clean A0/A1/A2 on `2,6,11,12,47,76` | Manifest parity plus process gates; otherwise stop |
| **7 — later work** | A3 `2,76`, rooms-11, then holdout/scale | Only after all prior gates pass |

q104 is intentionally deferred to holdout scale: it is a known native-crash hot
scene and is not needed for the baseline-recovery gate.

### PR #115 integrity revalidation

The next GPU action is not a scale run. After the two CPU integrity commits are
clean, run one fresh q11 A2 canary behind a detached 12-GiB EGL safe-start. The
canary passes only when all of these conditions hold:

1. The episode has a schema-v3 manifest plus a validated atomic
   `COMPLETE.json`; aggregate/DONE counts come from completion markers rather
   than JSONL file existence.
2. The staged diagnostic snapshot passes the frozen artifact profile and its
   hashes/config digest agree with the completion marker.
3. The grounded trace has stable question/session IDs, fresh current-pose room
   state after motion, durable fused evidence references, and only chooses an
   investigate/frontier action that was rendered in that router call.
4. q11 reaches the kitchen and direct present/answerable evidence supports the
   requested relation. Budget fallback, plausibility, or merely producing a
   syntactically valid wrong-answer row is a capability failure.

Repeat q11 once only after that first pass. Then run A0 on `6,11,12,47`; A1 and
A2 remain blocked until the baseline gate recovers. The `2,76` hard-room pair,
A3, rooms-11, holdout, and bal-32 remain blocked by the existing ladder.

## Managed harness

Run a fresh detached probe before every experimental job. Do not queue the next
probe or experiment until the current job is terminal. Every experiment must
remain under the host-wide `emet jobs run --gpu-exclusive` lock (the lock is
automatic for GPU-like commands, but do not opt out with `--no-gpu-exclusive`).
Managed H2H launches keep `SKIP_KILL_STALE=1`; stale-process cleanup belongs in
preflight, never inside the live orchestrator holding that lock.

Readiness:

```bash
uv run emet jobs
git status --short
uv run emet habitat safe-start --need-mib 12000 \
  --job-name habitat-egl-probe-grounded-graph
uv run emet jobs status JOB_ID
uv run emet jobs logs JOB_ID --tail 40
# Continue only after status=done and logs contain "Habitat EGL OK".
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
  --attempt-ledger-mode off --no-hm3d-semantics --no-enrich-labels \
  --variant-id gre-a0-legacy-r1 \
  --job-name gre-a0-legacy-r1 -d "Graph-room evidence paired-six A0 legacy control"
```

Paired-six A1:

```bash
OUT_A1=~/runs/emet/gre_a1_shadow_r1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT_A1" --preset paper-router --arms agentic --ids "$IDS" \
  --decision-policy legacy --graph-evidence-mode shadow --room-history-mode shadow \
  --room-policy canonical --room-target-hints --no-investigate-stamp \
  --attempt-ledger-mode shadow --no-hm3d-semantics --no-enrich-labels \
  --variant-id gre-a1-shadow-r1 \
  --job-name gre-a1-shadow-r1 -d "Graph-room evidence paired-six A1 shadow"
```

Paired-six A2:

```bash
OUT_A2=~/runs/emet/gre_a2_grounded_r1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT_A2" --preset paper-router --arms agentic --ids "$IDS" \
  --decision-policy grounded_v2 --graph-evidence-mode agent --room-history-mode agent \
  --room-policy canonical --room-target-hints --no-investigate-stamp \
  --attempt-ledger-mode agent --no-hm3d-semantics --no-enrich-labels \
  --variant-id gre-a2-grounded-r1 \
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
| A0 resolver | complete | `gre_a0_resolver_e85cda20_20260813_232912`: 4/6 |
| A1 shadow | complete | `gre_a1_shadow_fae4b89c_20260814_131815`: 4/6; collection/no-leakage passed |
| A2 grounded | **no-go** | `gre_a2_grounded_fae4b89c_20260814_160721`: 3/6, slower, more budget hits |
| GT-free q11 pre-fix sequence | complete, failed capability gate | 0/3 complete runs; infrastructure improved |
| Post-fix q11 diagnostic | passed once, dirty | `gre_q11_a2_canary_d3ee32e8_20260822_175238`: 1/1, direct verified evidence, no salvage |
| Clean q11 repeat | complete, failed capability gate | `gre_q11_a2_clean_81a2c689_20260823_003735`: 0/1; native-clean, but eight-round budget fallback after no present/answerable view |
| PR #115 integrity q11 | complete, failed capability gate | job `20260823_081229_66d6e3`, OUT `gre_q11_a2_integrity_b91059e5_20260823_081204`: two native-clean attempts, both A vs D with no direct evidence; completion correctly stayed incomplete, but first exposed a best-effort object-crop policy bug |
| Recovery correctness | CPU integrity gate complete | Managed lifecycle/completion and grounded-state/replay contracts pass 141 lifecycle, 182 agent-regression, 108 graph-memory/evidence, and 11 focused state/metadata tests; the separate full no-sim baseline still has three known unchanged map-rendering failures |
| Next GPU work | blocked | Diagnose why grounded q11 repeatedly misses the kitchen trash-can evidence; do not launch the wider matrix until a CPU-testable search/state fix has a clean direct-evidence canary |

## Related

- Abandoned latch PR: [#107](https://github.com/cpaxton/home_robot_v3/pull/107) (escape floor on room mismatch)
- Prior rooms+verify probe: [agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)
- Scale ladder: [agentic_scale.md](agentic_scale.md)
