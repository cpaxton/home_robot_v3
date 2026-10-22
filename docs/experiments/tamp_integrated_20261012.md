# Integrated TAMP follow-up, October 12

PRs #179, #200, #201, and #195 are merged. #195 adds native input contracts;
observed-scene providers and native execution adapters remain unwired. The
following experiment uses simulator assistance, not native contact execution.

## Full public-tool diagnostic

Job `20261012_182257_5480fe`, source `2b76b0375b6fe9ecb4bb532af96c747a40275f93`
(the implementation integrated in #195; later merge changed docs/install files).
Artifacts: `/home/cpaxton/runs/emet/tamp-integrated-20261012/`.
Sequence: scene_tasks → open → plan_pick_place → execute_pick_place_plan.
Task admission selects the first bowl task supporting the requested open action.
All requested cases remain in the denominator. This changed admission policy is
not a paired comparison or a rescore of the historical 3/7.

| Case | Result | Failure stage |
| --- | --- | --- |
| RBY1 kitchen 0 | Passed assisted tool/proximity checks | — |
| Stretch kitchen 0 | base_pose_validation_unavailable | Planning; missing backend endpoint |
| Sourccey kitchen 0 | ik_failed | Grasp after approach |
| RBY1 kitchen 2 | collision | Planning after successful opening |

**1/4 passed; no case timed out.** RBY1 kitchen 0 issued 580 acknowledged
commands (maximum acknowledgment latency 0.904 s). A teardown command failure
occurred after task completion. These checks do not establish sustained support,
contact stability, native manipulation, or real-robot readiness.

## Dynamic placement metadata correction

A new MuJoCo regression reproduced stale geometry on the merged implementation:
a free object fell about 0.198 m and translated about 0.100 m, but session metadata
still reported its original position. Explicit simulator commands refreshed
geometry; subsequent free physics did not necessarily do so. This qualifies the
placement verification evidence above: cached metadata can miss post-release
motion. Do not reinterpret that pass as a verified settled placement.

The focused fix tracks measured free-joint poses and refreshes the entire semantic
assembly when they change. Both robosuite and Stretch publication use it; Stretch
uses its physics-process snapshot. Cached topology and last-published poses avoid
geometry scans for stationary bodies and preserve cumulative small movement.
Metadata remains privileged simulator input. Publication does not rerun dynamics.

Validation: the new physics-motion regression fails before the fix and passes
afterwards. The expanded targeted run had 132 passes and one unrelated fixture
failure (`test_fast_state_does_not_refresh_scene_geometry`: mock spec missing
`actuator_names`), reproduced on the integrated baseline. No skips. Ruff passes.
A fresh RBY1 assisted replay must assess behavior with current geometry.

## Next work

1. Validate release against fresh measurements over a settling interval, with
   explicit support/retention evidence rather than proximity alone.
2. Implement Stretch base validation and native adapter/provider contracts without
   advertising unsupported capabilities or treating simulator metadata as sensors.
3. Replay Sourccey grasp IK with measured base/arm state and alternate approaches.
4. Diagnose kitchen-2 collision and admit reference-solvable tasks before scoring
   agent performance; never remove failures silently.

Kitchen-1 reconstruction artifacts are in `kitchen1_diagnosis/` under the same
artifact directory, including `blocked_centers.png`, `.pdf`, and numeric inputs.
For the saved fixed-orientation payload and conservative boxes, the broad top
support is obstructed overhead and the interior admissible centers are covered
by expanded obstacle footprints. Shortening the approach alone does not solve
this geometry. This is not proof that physical placement is impossible.
