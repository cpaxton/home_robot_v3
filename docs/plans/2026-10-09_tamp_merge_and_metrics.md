# TAMP integration review and metric improvement plan

Review date: 2026-10-09. Placement head `1cb803bc`, parent `046a4eae`,
foundation `2a0c5ec1`, main `a9a5f1d1`. This is a scoped review of integration,
failure handling, documentation and acceptance evidence, not a complete audit
of every changed line. No feature PR is approved for merge by this document.

## Implementation update

The review findings below remain as the original audit. Implemented follow-ups:

- #178 now targets the #177 foundation branch and has an API/execution title.
- #179 has isolated commits for local RNG propagation (`d3235ed3`), accurate
  truncation/stable reasons (`37025e12`), oriented support patches (`d60d53f4`),
  exact replay snapshots (`4d96ec66`) and measured carried geometry (`d1c755f7`).
- Draft #186 (`feat/tamp-placement-recovery`) stops on uncertain execution,
  retains unattempted objects in the denominator, preserves placement reason
  codes and exports operation-scoped search/release evidence. A returned detach
  helper is not verified completion; held state remains unknown. Automatic
  recovery and observation-based attachment verification remain outstanding.
- Draft #187 (`feat/tamp-paired-metrics`) adds a comparator over existing ledgers,
  planner/runtime provenance, paired gains/losses, scene-cluster uncertainty and
  PDF/SVG export. See [usage](../experiments/tamp_paired_metrics.md). It rejects
  unpaired/nonterminal runs and does not automatically promote candidates.
- Combined placement/recovery source `7e20a9b3`: **235 tests pass**. Metrics:
  **16 tests pass**, including actual vector export. Foundation/integration
  command, navigation, evaluator and tool tests: **109 pass, 2 skipped**.
- Diagnostic gate `20261009_121359_46c309` at `6ea3f780` finished failed: CHAT
  failed, 0/3 admissions, no repeats. It confirmed stop-on-failure and support
  availability, and exposed stale payload bounds in the exact live snapshot.
- Corrected full gate `20261009_122159_920b79` at `7e20a9b3` is running with
  fresh snapshots. Exact pre-fix replay `20261009_122347_8b00fe` is running with
  CPU affinity. Their terminal results must be recorded before acceptance.

No PR has been merged into main. Full live placement/composition acceptance,
explicit recovery/held-state observation, native crash attribution, required CI
configuration and cross-stack navigation integration remain merge work; passing
unit suites and creating PRs do not complete those gates.

## State and evidence

| Layer | Evidence | Readiness |
| --- | --- | --- |
| Physical motion/benchmark foundation, #177 | Measured motion contracts, fixture admission, historical experiments | Review independently; historical assisted results are not physical or learned-agent acceptance |
| Agent execution/API, #178 | JSON tools, scoped single-use handles, snapshot checks and measured evidence | Useful contracts, but latest CHAT task fails at placement |
| Placement, #179 | 182 contract/unit tests; GT component boxes, observed-volume API, free support regions, multiple approach/IK/path candidates | Not ready: live placement, support coverage and offline native crash unresolved |
| TAMP recovery/composition | Partial results exist; placement details remain executor-local | New stacked PR needed for held/released state and recovery |

The scoped 182-test placement/TAMP/motion suite was rerun during this review:
**182 passed in 14.82 s**. Documentation whitespace checks passed.

The latest full gate is `20261009_081750_e342cc` at `1335d5d5`:
CHAT failed, admission **0/3**, repeats **not run**. Its authoritative summary is
`~/runs/emet/placement-fair-search-20261009/results/readiness.json`.
The [experiment record](../experiments/placement_planning_20261009.md) details
the failures. Historical 21/25 was conditional GT/MCTS performance on an admitted
subset with different execution assumptions; it is not today's acceptance score.

## Review findings and disposition

1. **Placement acceptance blocker:** live CHAT search reaches RRT but returns no
   complete path after 20 iteration-limit failures. Scene00 exhausts 96 pose-IK
   calls. Capture exact measured state, geometry and RNG state for each failed
   approach. Test reachability, goal validity and path connectivity separately.
   Fixed orientation and approach locations are hypotheses to test, not proven
   causes. Do not increase budgets blindly or infer a need for rearrangement.
2. **Support coverage blocker:** scene12 supplies no accepted support patches;
   the adapter supports axis-aligned collision-box tops only. Add a grounded
   support provider for the actual fixture (and observed surfaces), or explicitly
   report unsupported geometry. Do not substitute visual AABB tops or easier goals.
3. **Composition blocker:** `clutter_chain.py` continues to another object after
   failed grasp/place without verifying held state. `task_search.py` records only
   completed operations and selected measurements. Stop on uncertain state until
   recovery is explicit; test failure before release, after release, and during
   retreat. Treat downstream failures in existing runs as potentially dependent.
4. **Agent diagnostic gap:** `api.failure_code` collapses new placement reasons
   into `operation_failed`; `plan_data` reports `base_endpoint_only` even when
   internal placement did additional sampled arm/payload checks. Propagate safe,
   stage-specific JSON evidence and scope; never claim stronger coverage for the
   whole task than each executed stage supports.
5. **Reproducibility gap:** placement's local RNG seeds IK, while configuration
   space sampling uses global NumPy RNG and RRT also uses Python random. Thread
   explicit RNGs through search, or control and record all RNGs in isolated runs.
   A `seed` argument alone currently does not reproduce the complete search.
6. **Budget reporting gap:** `free_surface_centers` flags region truncation but
   does not flag truncation at `max_centers`; base candidates are also sliced.
   Return explicit truncation reasons. Normalize RRT failures into a stable code
   with node counts as data: the current rejection keys embed changing node counts.
7. **Native diagnostic blocker:** two approximate offline reconstructions crashed;
   the crash site is in pose IK, but root cause is unknown. Minimize in a subprocess
   with an exact snapshot and archive runtime/library versions. The live gate's
   ordinary search failure is separate evidence; do not conflate the two.
8. **Tracking remains open:** scene02 fails before placement. Use paired clear-space
   versus contact tests and measured command/state/contact evidence before changing
   planning or tracking tolerances. Test count alone cannot resolve these failures.

## Atomic merge sequence

1. Review #177 as the foundation. Its current head is an ancestor of #178.
   Retarget #178 to #177's branch during review so its own diff is visible
   (40 files versus 91 against current main). Rename #178 to reflect its full
   API/execution scope. Do not rewrite shared history merely to tidy commits.
2. Review and merge #177 only with its declared foundation tests and accurate
   limitations. Then retarget/update #178 onto resulting main and rerun tests.
   Preserve ancestry where practical; squash requires reconciling descendant
   branches to avoid reintroducing the foundation diff.
3. Keep #179 based on #178. Fix each placement cause in a separate commit with
   its regression. Require the original CHAT success plus 3/3 admissions and
   9/9 fresh repeats before declaring the placement integration ready.
4. Create a separate composition PR atop #179 for findings 3–4, with documented
   schema changes and failure-state tests. It may be developed before placement
   acceptance, but must not be presented as end-to-end validated until its own
   multi-action gate passes.
5. Merge #178, #179 and composition in dependency order after their respective
   gates. Test the final combined candidate against the actual main head; green
   tests on old individual branch heads do not establish integrated correctness.

GitHub reported no status checks on #177–179 at review time. Make CPU contracts
and schema tests required checks; attach immutable managed simulation artifacts
as a separate acceptance gate. Geometric mergeability is not test evidence.

Parallel stack #180–185 is owned separately and was inspected only for metadata
and file overlap. #182 overlaps navigation results, robot footprints, voxel maps,
A*, navigation sweeps and agent tools. #169 also overlaps navigation results and
grasp handoff behavior. Before either stack lands, agree which implementation owns
each shared contract, resolve in an isolated integration branch, and run both
navigation and TAMP regressions. Do not blindly merge overlapping implementations
or include unrelated experiment #183 as a dependency. No changes to those PRs
are authorized or performed by this review.

## Documentation acceptance

- [Agent API](../apis/tamp.md): canonical tool inputs/JSON outputs, discovery →
  plan → execute example, errors/recovery, handle lifetime, partial effects,
  assistance and evidence limits. Current tools are `scene_tasks`,
  `plan_pick_place`, `execute_pick_place_plan`, and `pick_place`.
- [Placement API](../apis/placement.md): Python planning interface (not a new
  CHAT tool), frames/units, support/payload/voxel contracts, defaults and budgets,
  pure planning versus execution, scope and unsupported geometry.
- Composition PR must update schema examples and tool metadata together with
  tests that parse actual outputs. Document state inspection and recovery only
  once callable implementations exist; do not promise a missing recovery tool.
- Add runner/metric documentation: exact invocation, registry and asset hashes,
  seeds, assistance, artifact locations, timeouts, denominator rules and figures.
  Link these from the agent guide; keep historical results separate from current
  readiness. Correct stale queued status when terminal evidence arrives.

## Repeatable metric improvement protocol

### Suites and immutable inputs

Keep four separately reported tracks: pure placement geometry/search; measured
simulated placement; multi-action GT TAMP; learned-agent tool use. Within each,
separate GT/observed geometry, robot, scene family and assistance mode. Latch,
teleport and physical execution scores must never be pooled.

Use the current three fixtures as regression canaries, not as evidence of broad
generalization. Freeze a larger development set stratified by payload size and
offset, clutter density, support geometry, reach, and observed-space coverage.
Use an independent reference to establish constructive solvability; retain
admission failures in coverage reporting. Freeze held-out scenes/layouts/object
instances before tuning. Re-admitting separately per candidate can change the
denominator and must not be used for paired method comparisons.

### Required ledger and scorecard

One JSON record per scheduled trial: schema version, case/fixture hashes, source
commit, environment/assets/config hashes, robot, geometry and execution mode,
all seeds, search budgets, timeout, terminal status, first failed stage, stable
reason code, completed stages, verified held/released state, timings, and paths
to measurement/contact logs. Missing, crashed or timed-out runs stay visible.

Report:

| Metric | Denominator / interpretation |
| --- | --- |
| Admission coverage | Independently admitted / all proposed cases |
| End-to-end task success (primary) | Verified complete successes / all scheduled fixed-set trials |
| Placement success | Verified placements / all scheduled placement trials; separately conditional on reaching placement |
| Stage outcomes | Counts plus explicit number reaching each stage; separate first failures from cascades |
| Search effectiveness | Valid alternatives, IK/RRT calls, truncation, normalized rejection counts |
| Cost | Median/p95 planning and execution time, timeout/crash rate; failed trials included with censoring stated |
| Execution quality | Position/orientation error, attachment drift, collisions and release/retreat outcomes |
| Agent performance | Full success, tool calls, replans, invalid calls and recovery success, on its own track |

### One improvement at a time

1. Reproduce the largest first-failure class with a saved exact state. Form a
   single testable hypothesis and add a regression at the lowest useful layer.
2. Run baseline and candidate on identical cases and all fixed seeds in fresh
   processes. Alternate run order to reduce environment drift. Keep geometry,
   tolerance, timeout and computational budget fixed; budget changes are explicit
   cost/success ablations, not free improvements.
3. Require zero regression-canary failures and no new collision/false-success
   violations. Select on development full-task success first, then cost at equal
   success. Report paired gains/losses and uncertainty; a tiny apparent gain is
   inconclusive, not automatically a promotion.
4. Validate selected candidates on untouched holdout at milestones, not after
   every tweak. Report scene-clustered uncertainty so repeated seeds do not
   masquerade as independent scenes. Predeclare the repeat count before running.
5. Promote only terminal, attributable runs; retain the previous champion and
   all failed candidates. Never tune thresholds or remove fixtures to improve
   the score. Expand observed-geometry and robot coverage after canaries pass.

Reuse `run_tamp_experiments.py` and terminal ledger checks in
`plot_tamp_benchmark.py`; extend their schema rather than create a competing
score pipeline. Paper figures should show coverage beside success, first-failure
breakdowns, paired baseline/candidate changes, success versus planning time, and
GT-versus-observed ablations. Export PDF/SVG plus source data and a provenance
manifest. Current placement failures belong in diagnostics, not a success figure.

## Immediate order of work

1. Preserve this terminal baseline and prevent unresolved held-state continuation.
2. Isolate the native crash and capture exact live placement failure snapshots.
3. Repair grounded support coverage; diagnose IK versus goal/path connectivity
   with orientation and approach ablations under fixed budgets.
4. Repair seed/budget/reason reporting so paired trials are interpretable.
5. Rerun placement canaries, then composition and broader held-out evaluation.
   Open-ended MCTS rearrangement follows a working placement/recovery interface.
