# Placement and TAMP acceptance

## Scope and PR order

1. **Placement primitive — PR #179**, based on `feat/tamp-agent-execution`.
   Own geometry, support candidates, approach/IK/path search, measured execution,
   release/retreat evidence, and placement regressions. Keep fixes generic across
   objects and both ground-truth and observed geometry.
2. **TAMP composition — separate stacked PR when implementation starts**, based
   on the accepted placement code. Own JSON failure/evidence propagation,
   post-action state, recovery and sequential plan tests. Do not fold these API
   changes into the placement PR.
3. **Rearrangement search — subsequent scope**, only after ordinary placement
   works and failures distinguish occupied space from incomplete motion search.
   A bounded search failure does not establish that clutter must be moved.

Keep commits atomic: one diagnosed cause, its regression, and relevant docs.
Keep immutable run source IDs; docs-only changes do not require replacing a run.

## Placement completion criteria

- Reproduce the original failed CHAT task without changing its object, receptacle
  or success scorer. Record pickup, transport, preplace, place, release and retreat
  separately. Assisted snap/release must remain explicit in reported results.
- Pass the original three-fixture admission and three-repeat readiness gate;
  retain failed admissions in the denominator. The current queued job is
  `20261009_081750_e342cc`, source `1335d5d5`.
- Cover a clear support, an off-grid clutter gap, a blocked support, an unreachable
  first approach with a reachable alternative, payload detours, and geometry
  changes before execution. Test occupied and unknown observed volumes as well
  as ground truth. Unit cases do not replace simulator execution evidence.
- Attribute rejection to surface filtering, base validity, IK, goal collision,
  edge/path search, budget exhaustion, or measured execution. Preserve crashes
  and timeouts as failed diagnostics, not geometric infeasibility.
- Require full arm and payload checks; do not relax collision or tracking
  thresholds to obtain success. Base endpoint checks do not certify its route.

## TAMP composition completion criteria

- Agent-facing results stay JSON and preserve machine-readable placement failure
  classes, search-budget status, geometry source and collision scope without
  leaking simulator identities or arbitrary internal exception messages.
- Distinguish failure before release (object still held) from release followed by
  failed verification or retreat. Confirm state before retry; never replay grasp
  or placement blindly after partial execution.
- Exercise pick/place followed by another pick/place, placement retry from held
  state, stale-scene replanning, and post-release failure recovery. Confirm that
  subsequent actions use refreshed state and that completed operations are not
  repeated.
- Preserve current plan-handle and scene-version contracts. A motion alternative
  is conditional on its scene and attachment; it is not a reusable unconditional
  action or a proof of task feasibility.

## Current evidence and next actions

Placement has 182 passing contract/unit tests at `1335d5d5`; full simulation
acceptance is pending the shared GPU. See
[experiment history](../experiments/placement_planning_20261009.md) for earlier
failures and exact run provenance.

Code inspection found that `task_search` wraps placement failure in a string and
`api.failure_code` maps several new placement reasons to `operation_failed`.
The executor's `last_placement_search` and `last_surface_search` are not sufficient
as an agent-facing recovery contract. Address this in the TAMP stack, with tests
for state after partial execution, rather than broadening the placement PR now.

The saved-scene offline reconstruction is approximate, not a live replay. Its
latest retry exited with SIGSEGV before a final result. A second run with Python
faulthandler also crashed, with the stack at `solve_pose_ik`'s `model.joint(name)`
lookup (`mujoco_arm_ik.py:122`), called by `plan_placement_paths`. This identifies
the crash site, not its cause. An isolated 96-call pose-IK loop completed without
crashing (all targets rejected), so do not attribute this to ordinary IK failure.
Logs: `/tmp/placement_fair_search_fault_1009.txt` and
`/tmp/placement_ik_isolation_1009.txt`. Isolate the integrated native failure before
interpreting a missing result as a slow or unsuccessful search. Continue to use
the unchanged queued live gate as the end-to-end acceptance test.
