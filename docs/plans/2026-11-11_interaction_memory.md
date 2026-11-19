# Interaction-grounded memory and frontier-model evaluation

Date: 2026-11-11. Status: **proposed; no runtime changes or new results**.
This extends [embodied agent planning](2026-08-08_embodied_agent_planning.md)
phases 1–5, rather than replacing the existing memory backend.

## Motivation and related work

[MessyMem](https://messymem.github.io/), Banwasi et al., CoRL 2026
([paper v2](https://arxiv.org/html/2609.15976v2), sections 3–5 and appendices A–C),
combines a persistent spatial graph, interaction-derived properties, and linked
visual keyframes. Its central relevance is reuse across tasks: remember what an
action revealed, then retrieve the supporting visual detail when a later goal
needs it. A record that a tool failed is not yet a grounded belief about why.

The paper reports **80.0% task progress**, not 80% complete manipulation success,
on its 25-task continuous simulation. The long-horizon task mix includes finding,
matching, memory queries, and destination selection without object transport.
Its real platform uses demonstrated/retargeted primitives and fiducial-supported
localization. These are useful memory results, not a GPT-only control benchmark
or directly comparable to Emet's HM-EQA accuracy or observed manipulation gates.
Consult the paper for full setup and baseline definitions; no published score is
an acceptance threshold for Emet.

Do not claim novelty for simply combining scene graphs, interaction summaries,
and images. Our **hypothesis to test** is that provenance, uncertain interaction
beliefs, evidence lifetime, and measured execution feedback make that combination
more useful under changing scenes and imperfect robot skills.

## Existing foundations and remaining gaps

Paths below are relative to the repository root; baseline surveyed at main
`cd169673ba69c5e8f63dbda4c0bcab25d373e339`.

| Foundation | Existing seam | Proposed extension / missing evidence |
| --- | --- | --- |
| Typed attempts and temporary retry suppression | `src/emet/memory/graph_eqa/attempt_ledger.py`, `action_history.py`; [reference](../attempt_ledger.md) | Link attempts to supporting observations and uncertain properties; static-world suppression is not a lifelong scheduler. |
| Stable identities, immutable RGB views, append-only evidence | `src/emet/memory/graph_eqa/world_evidence.py` | Reuse these identities and exports for interaction windows; prove cross-task and process-restart retention without ID collisions. |
| Bounded assessment history | `src/emet/memory/graph_eqa/agentic/view_history.py`, `src/emet/memory/view_selection.py` | Task-relevant long-term interaction retrieval, not just recent views; keep historical evidence separate from current geometry. |
| Task-relevant graph/image recall | `src/emet/memory/graph_eqa/eqa/eqa_views.py`, `spatial/spatial_rag.py` | Entity + interaction + temporal retrieval with explicit budgets and source attribution. |
| Measured TAMP effects and guarded plans | [TAMP overview](../tamp.md), [task API](../api/tamp_task_plans.md) | Preserve command/effect authority: a remembered VLM claim must not certify motion, contact, support, or successful execution. |

## Proposed information contract

The following are design fields, **not existing APIs or configuration options**.
Use additive versioned records alongside existing attempt/evidence exports;
retain legacy imports and default-off behavior. Avoid a second competing graph.

An interaction episode references an attempt ID, stable entity/place IDs,
world/session/task IDs, action parameters, controller result, measured effects,
and bounded pre/during/post view IDs. Capture time, clock domain, original
acquisition age, camera pose/calibration and motion epoch remain attached to
sensor evidence. Missing images/telemetry are explicit, not fabricated frames.

A property claim references that episode and supporting views, its predicate and
value, source (measured versus inferred), uncertainty, applicability conditions,
validity interval, and superseded/contradicting evidence IDs. Model confidence is
not a calibrated probability. The immutable event log retains disagreements;
a materialized current belief can change without deleting prior evidence.

Example: an opening attempt with no observed fixture motion records that outcome.
A secure handle grasp plus a pull may support **possibly locked**; a missed or
occluded grasp supports **grasp uncertain**, not locked. Neither permits a
permanent inaccessible label. Without disambiguating evidence, return unknown
and offer an inspection or alternate approach within the existing motion rules.

Historical evidence may guide search and explain an earlier failure. It cannot
renew hidden geometry, establish free space, override current contradictions,
rebind a stale plan, or unlock an execution capability. Task boundaries clear
working context while retaining explicitly selected world memory; world resets
must not inherit old physical state. Cross-session reuse requires a verified
world identity and relocalization, otherwise quarantine it as historical only.

## Independently landable implementation slices

All runtime slices begin opt-in and preserve pinned benchmark configs and EQA
tool names. Each slice needs tests at its seam plus a recorded live gate before
any default promotion. These are acceptance requirements, not completed work.

### M1 — Interaction evidence capture and persistence

Extend the existing ledger/world-evidence seams with interaction episode links
and bounded keyframe capture. Preserve failure/partial/abort outcomes as well as
success. Deduplicate by capture identity while retaining distinct before/after
states. Define per-episode and total byte/count limits; retain metadata/tombstones
when images are evicted. Never leave a retriever believing evicted RGB exists.

Acceptance: serialization/restart round trips preserve IDs and evidence ages;
task transitions retain selected episodes; world resets isolate them; duplicate
callbacks do not duplicate attempts; missing/evicted frames and interrupted
writes are handled explicitly. Old exports still load. No planner changes.

### M2 — Shadow interaction analyzer and belief updates

Analyze a bounded action window plus controller/effect feedback, writing
schema-validated **candidate** claims. Reuse existing outcomes rather than
inventing a parallel success classifier. Unknown entity IDs and unsupported
claims are rejected; untrusted scene text remains evidence, never instructions.
The analyzer does not control retry eligibility in this slice.

Acceptance: independently labeled held-out windows distinguish grasp failure,
no-motion after contact, occlusion, successful opening, and transport errors.
Report confusion matrix, abstentions, unsupported-property rate and disagreement
with measured effects. Include locked-to-unlocked and moved-object reversals.
A VLM-inferred lock cannot become measured truth or overwrite raw evidence.

### M3 — Bounded persistent retrieval and planner exposure

Pool candidates from entity/place links, interaction labels and temporal context;
cap each source, deduplicate, then optionally rerank thumbnails. Trace candidate
IDs, exclusions, selected image hashes, ordering, source times, token/image caps,
latency and cost. Keep a deterministic retrieval control. Missing evidence must
be visible to the planner; no silent replacement with a different frame.

Acceptance: retrieve old evidence amid recent distractors; distinguish same-label
instances by visual details; surface contradictions and missing images; never
mix worlds. Measure evidence recall at K and unsupported answers as well as task
success. Match image/token budgets against existing view selection. No claim that
more history is automatically better; existing context regressions remain relevant.

### M4 — Dynamic belief invalidation and retry scheduling

Only after M1–M3: connect confirmed target changes, contradictory observations,
qualified relocalization and explicit validity expiry to a scheduler. Expiry
makes a belief stale/unknown; it does not prove that a cabinet opened. Reopen
an action for new relevant evidence without letting the failure event itself
count as progress. Keep scheduling separate from model-visible history.

Acceptance: unchanged failure cannot loop indefinitely; new relevant evidence
reopens work; irrelevant graph churn does not; world changes invalidate stale
claims; old physical plans still fail freshness checks. Use shadow decisions
before a separate enforce comparison. Existing static HM-EQA enforcement remains
scoped as documented in [the ledger reference](../attempt_ledger.md).

## Evaluation: what does GPT already supply?

Freeze exact provider/model IDs, reasoning settings, prompts, seeds, source SHA,
perception/controller versions, tool schemas, task ordering and budgets before
running. Model labels alone do not define a reproducible baseline. Audit the
OpenAI client for the selected model's supported interface before comparison;
do not assume its historical GPT-4o choices qualify a newer model.

First run two separate fixed-packet diagnostics: (1) planner model comparison
with identical supplied memory and (2) analyzer comparison on identical labeled
interaction windows. Hold the other role fixed, report schema failures and
uncertainty, and keep these diagnostics separate from end-to-end gains.

Then run the following planner-model × memory matrix, initially with the same
frozen analyzer for every interaction-memory cell:

| Memory treatment | Existing local Qwen baseline | Selected frontier GPT |
| --- | --- | --- |
| Existing graph and view selection | control | model-only change |
| Graph + persistent keyframes, no inferred properties | visual-memory isolation | same |
| Graph + interaction properties, no additional historical images | interaction-memory isolation | same |
| Graph + properties + linked keyframes | combined treatment | combined treatment |

Run off/shadow/visible checks before live comparisons to detect hidden prompt
leakage. Hold retry enforcement fixed and off for the primary matrix; evaluate
M4 separately. Also include a budget-matched recent-history control and a
memory-cleared-between-tasks control to test whether persistence adds value.
The graph may retain spatial identity in that control, but it must not leak
prior interaction claims or images; record the exact reset policy.

Use paired **complete sequences** with identical initial scenes/task order/seeds;
reset worlds between arms, retain memory within sequences. Separate development
fixtures from held-out layouts, object appearances and change schedules. Begin
with 12 held-out sequences per cell as a pilot, report paired sequence-level
uncertainty and failure denominators, and set a larger sample size from pilot
variance before confirmatory runs. Do not treat correlated tasks as independent
trials or tune on the held-out sequence. Report compute/API failures separately
and include them in attempted-task denominators.

### Task families and assistance tiers

- Reuse an earlier drawer inspection for a later object query.
- Distinguish failed grasp from a possibly locked fixture; later unlock it and
  test whether evidence updates reopen the search.
- Find a particular same-label object using an old label/pattern image.
- Choose a reachable object/destination from layout evidence, then separately
  test actual transfer when the observed execution stack is qualified.
- Move an object externally between tasks and test contradiction/staleness.
- Retrieve an old relevant event after many distractors and a process restart.

Use recorded interaction replays first, then privileged simulator controls,
then qualified observed-input simulation, then hardware. Replays can test memory
and decisions but cannot establish contact/control success. A fixture unsupported
by current articulation or observed TAMP remains blocked, not counted as a
completed capability. Record assistance, oracle inputs and controller mode per
trial; never pool tiers into one headline.

### Required outcomes and promotion gate

Report complete-task success and task progress **separately**, along with exact
subgoal definitions, repeated failed actions, useful retries after changes,
false persistent claims, evidence recall, abstention, action/planner calls,
wall-clock time, retrieval p50/p95, storage growth and API cost. Keep model
judgments separate from independently measured final effects. Record full prompt
packets/image hashes so answer versus acquisition failures can be attributed.

No default promotion from this planning PR or a component test count. A proposed
runtime promotion must pass existing HM-EQA/OVMM and tool-contract regressions,
show a positive paired sequence-level effect on the preregistered primary
complete-task measure, and report its 95% interval. Before confirmatory runs,
freeze acceptable latency/cost and existing-task regression margins. Any
unsupported execution-success claim or violation of geometry/plan freshness
blocks promotion, regardless of average score. Negative results stay recorded.

## Next deliverable

Land M1 alone with a versioned schema, persistence fixtures and a frozen replay
manifest. Follow with M2 shadow labels and M3 retrieval diagnostics before live
model comparisons. This document does not launch paid inference, simulation,
training or hardware motion. The existing manipulation qualification gates remain
prerequisites for execution experiments.
