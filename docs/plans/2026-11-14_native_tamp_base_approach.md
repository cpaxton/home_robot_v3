# Native TAMP: checked base approach before grasp

November 15 scope freeze: this feature belongs on
[`feat/tamp-native-base-approach`](https://github.com/cpaxton/home_robot_v3/tree/feat/tamp-native-base-approach),
stacked on #234. Finalize #234 for review as the bounded refinement/evidence
change; its failed full-task gate remains recorded and moves into this
follow-up's acceptance work. Review readiness is not task qualification.

## Evidence and scope

The November 14 contact-relative pilot passes the familiar control and shift15,
but fails swap-right25. Face-aligned candidate ordering also preserves those
controls but fails the swap after lift. The fixed-base planner rejects more
aligned pregrasp poses by IK before selecting a rotated pinch. These remain
failed tasks under the original retention bound, not controller qualification.

A private captured-state IK sensitivity probe fails the aligned pregrasp at the
captured base pose and succeeds with +25/+40 mm world-X base offsets, using the
same 3 mm / 0.1 rad IK tolerances and joint margins. This numerical experiment
does not prove global infeasibility at the original pose, collision freedom,
navigability or task success. It motivates testing an actual checked approach.
The probe changes private qpos only; that is never an execution mechanism.

The next implementation should add pre-grasp base positioning through the same
native primitive and multi-step tool interfaces. It remains explicitly
privileged simulated Stretch development. Loaded base transport, observed
navigation/exploration, other robot bindings and hardware qualification are
separate gates. No sensor or controller-training capability is inferred.

## Planning contract

1. Keep planning on private data. Generate bounded base alternatives from robot
   kinematics and object geometry, without object-name/case-specific offsets.
   Retain the current base pose as an alternative. Prefer candidates that permit
   face-aligned contact geometry and interior joint targets.
2. Plan any necessary unloaded arm preparation, the base approach, and the full
   grasp/lift/carry/release/retreat sequence. Validate the base sweep with robot
   geometry and scene obstacles, including intermediate poses and any extended
   arm. Endpoint clearance alone cannot authorize base motion. Keep intended
   wheel/floor contact distinct from forbidden contact; document resolution.
3. Preserve one common planning deadline and IK budget across alternatives.
   Do not grant each base pose a fresh 256-call/120-second allowance. Unknown or
   unavailable geometry rejects before actuation.
4. Store immutable approach/path targets with the model/session/scene binding.
   Preserve object goals, whole-footprint constraints, original retention
   reference semantics and supported-settle criteria. Placement feasibility
   must be checked before choosing an attractive pick pose.

## Execution contract

1. Validate the stored scene, robot state and complete approach again before
   commands. Execute checked arm preparation and base motion through ordinary
   client controllers, without root-pose writes or navigation teleport.
2. Require fresh measured base convergence and stillness. SDK acknowledgement
   alone is not arrival. Monitor the current swept segment, state freshness,
   unexpected contact and deadlines. Confirm a stop/hold before advancing;
   uncertain command outcomes remain partial and forbid automatic continuation.
3. Replan the grasp from the actual arrived state before closing. The existing
   arm executor then retains its within-segment base-drift guard. Do not adjust
   observations to commanded pose or move a loaded base implicitly.
4. Keep the existing public tool names. Publish the changed motion scope and
   distinguish approach/preflight/motion/verification failures in semantic JSON.
   Known pre-command rejection has `execution_started=false`; possible dispatch
   is partial. Consume failed handles and retain original task accounting.

## Verification and rollout

- Real MuJoCo tests: private planning does not mutate live data or actuate;
  shifted bases admit aligned IK with unchanged bounds; unknown geometry and
  colliding intermediate base poses reject even with a clear endpoint.
- Controller contracts: false ACK, stale feedback, missed arrival, interrupted
  motion and unknown command outcome cannot lead to a grasp command. Verify
  measured arrival, quiet handoff and unchanged arm/retention checks.
- Freeze the original control, shift15 and swap-right25 development cases for
  the first live approach pilot. Preserve fixed-base failures and independent
  reset provenance. Kinematic probe success cannot replace this gate.
- After that gate passes, replay the entire original eight-task/18-transfer
  population and four no-command interface boundaries on one frozen source.
  Keep all planning, execution and infrastructure failures in their populations.
- Repeat the original four actual `run agent` cases using the existing remote
  VLM, with independent final-goal scoring and the separate-user-turn recovery
  scenario. Do not rescore historical native 4/8 or Caliban 3/4 results.
- Add fresh held-out layouts after development; observed occupancy/contact and
  exploration remain separately unqualified. No merge/readiness promotion from
  a numerical probe, individual primitive or smaller successful subset.

Preserve #234's evidence and failed outcomes. Base approach is a separate
cohesive feature; it must not be presented as already implemented in #234.

## November 16 implementation and development replay

Review: [PR #242](https://github.com/cpaxton/home_robot_v3/pull/242), stacked on #234.

The follow-up implements this contract behind `bounded_approach`, including
sampled full-robot sweep checks, measured arrival/stop, fresh grasp planning and
loaded placement refinement inside the original region and unused budgets.
Initial failures, an intermediate placement-candidate regression and the agent
startup-policy mismatch remain recorded. The corrected pilot passes 3/3 tasks;
the original full native population passes 8/8 tasks, 18/18 transfers and 4/4
interface boundaries. The actual `run agent` cohort passes 4/4 tasks and 10/10
transfers with the existing remote VLM. Historical scores are not rescored.

See [the source-bound results](../experiments/tamp_native_base_approach_20261116.md).
The next gates are fresh held-out layouts, explicit execution failures/recovery,
then observed self-occupancy/contact admission and exploration. Successful
development replays do not qualify those capabilities or hardware sensors.
