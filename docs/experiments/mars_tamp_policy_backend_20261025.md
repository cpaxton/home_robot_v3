# Fixed-checkpoint Mars TAMP execution, 2026-10-25

The public TAMP simulator can now select a fixed learned actor for motion
execution. Earlier public-tool pilots always used the reference controller;
their IK failures were upstream planning failures, not failures of an invoked
learned actor. This change closes the actor handoff, not learned-task readiness.

Follow-up: the [startup audit](mars_policy_startup_audit_20261025.md) shows that
this initial checkpoint run failed before any arm command, during a settling
hold. It fixes idle inference, reaches the first actual pregrasp target, and
separately diagnoses the remaining learned tracking failure. The original raw
reports below remain intact; their zero-execution-IK observation does not
describe the later execution-time geometric replanning.

## Execution boundary

The existing `scene_tasks`, `plan_pick_place`, and `execute_pick_place_plan`
tools are unchanged. `MarsSimRobot.from_checkpoint` selects the policy backend.
TAMP keeps object/grasp selection, private collision/support screening, joint
witnesses, task sequencing, and support-confirmed release. Arm witnesses are
projected with FK to world pose targets anchored at the commanded base pose;
whole-body transport uses the timed pose schedule. Each healthy control tick
calls the deterministic actor with only normal history and task-pose preview.
There is no reference-controller fallback. Malformed/out-of-range actions,
inference exceptions and explicit inference/cycle deadline overruns latch a
stop. Existing speed, acceleration, grip, force, retention and wheel checks stay
in effect; partial execution requires measured stop confirmation.

This first adapter retains the arm executor's joint convergence and fixed-base
certificate (1 cm translation / 0.02 rad rotation during arm segments). It does
not certify arbitrary alternate arm configurations just because an end-effector
pose matches. Planning still uses IK, including measured-state replanning where
needed. The actor does not solve IK or receive its joint witness as an input.
Removing planning IK requires a replacement path/collision feasibility check;
a learned policy alone is not that check.

## Compatibility admission

The tested bundle is `tracking-retention-v3-gripper-head/seed-46/step-10240`,
with policy zip SHA-256
`a6be629f113a6b7f0a7a64d479f0eca675bff55d2b36c84074a37793cd0d6fed`.
The gripper-v3 training archive binds its exact manifest and weights to training
sources. A re-exported `last` bundle has a different zip checksum and is
intentionally rejected by this evidence binding.

The normal whole-scene policy check remains intact. Admission reconstructs the
training model and checks its original fingerprint, pins robot/model/asset
sources and runtime versions from the weight-bound training archive, and
compares the compiled robot with the live fixture. The identity covers robot
geometry, mesh vertices/topology/transforms, inertias, joints, actuators,
couplings, explicit collision overrides and solver options. Global mesh BVH
allocation addresses are excluded because adding an external fixed geom shifts
that cache allocation; mesh content remains bound. Robot indices must stay
unchanged. Extra actuators or mesh payload assets reject conservatively.

Only primitive scene payloads and external support changes are admitted by this
path. Each new world still undergoes normal TAMP geometry screening and runtime
monitoring. Robot compatibility is not a safety or infeasibility certificate.

## Matched development cases

The inventory is
`configs/benchmarks/mars_tamp_policy_development_v1.json`: three selected frozen
fixtures, two layouts, 20/50 g masses, delivery followed by return. Each
controller starts a fresh matched episode; there are no stage resets, object
attachments, state teleportation, fault clearing or threshold changes.

| Frozen case | Reference completed stages | Checkpoint completed stages | Actual policy calls | Checkpoint result |
| --- | ---: | ---: | ---: | --- |
| `left:20g:0` | 1/2 | 0/2 | 14 | Approach base-drift stop |
| `left:50g:0` | 1/2 | 0/2 | 14 | Approach base-drift stop |
| `right:20g:0` | 1/2 | 0/2 | 14 | Approach base-drift stop |

Complete-program success is 0/3 for each controller. Reference delivery succeeds
in all three, followed by bounded return-planning rejection. The checkpoint
admits in all three, reaches actual public execution, then drifts beyond the arm
segment's base certificate before picking up the object. All three partial
executions confirm stopping. Neither controller trips the wheel-lift or 30 N
force gate in these runs; peak recorded policy contact load is 4.38 N. This does
not establish compliant contact or grasp retention: the actor never reaches a
verified loaded carry. Keep planned-but-unexecuted stages in the denominator.

The public replies identify `mars_checkpoint_policy` and expose admission and
actor-call counts. Before/after tool checkpoints include physics and commanded
targets for debugging; they are not complete resumable simulator saves.
Reports retain all planning rejections, independent predicates and measured
traces. Aggregate motion metrics include failed segments and stop transients;
do not compare a subsecond failed approach's jerk with an entire successful
delivery as a matched task-quality estimate.

![Measured checkpoint approach errors and contact loads](../../paper/data/whole_body_control/tamp_policy_approach_v1.png)

Hand position error also grows to about 15 mm during the failed approach. The
problem is broader than crossing a stricter base threshold: this checkpoint
moves away from its near-initial target in the new execution distribution.
Loosening the drift guard would not establish useful tracking.

## Real-agent handoff

The existing Caliban service advertised `Qwen/Qwen3-VL-8B-Instruct`. On the
same `left:20g:0` program, the real model selected discovery, planning and
execution through the normal CHAT prompt/parser/dispatcher in three turns.
Planning succeeded with nine IK calls before motion. The actor then made 14
calls, with zero execution IK calls before the same approach base-drift stop;
the partial public receipt confirmed stopping and the agent ended the attempt.
Tool replies correctly identify the checkpoint controller and its bound weight
checksum. No teacher action was substituted. Complete-program success remains
0/1 with 0/2 verified transfers for this selected agent attempt.
The agent and scripted checkpoint run have exactly matching initial physics,
recorded normalized actions and recorded base states; model latency does not
advance this independently stepped simulator.

The server's exact weight revision was not reverified and its sampler used
server defaults. Inputs are explicitly privileged; physics pauses during model
inference. This exercises the real agent/controller connection without claiming
observed grounding, live inference concurrency or successful manipulation. Raw
prompts, replies, physical traces and the driver source are archived in
`paper/data/whole_body_control/tamp_policy_agent_v1.json.gz`; its adjacent JSON
summary binds the archive checksum. The matched scripted comparisons are in
`tamp_policy_development_v1.json.gz` and its adjacent summary.

## Validation

All 660 source-bound regression tests pass, including the 34 focused policy,
public-adapter and sequence tests. The receipt
`paper/data/whole_body_control/tamp-policy-regressions-v2-20261025.receipt.json`
and adjacent JUnit XML were verified against the current control/test sources.
Tests reject changed robot mass, fingertip geometry, mesh vertices, actuator
gear, gravity, solver settings, couplings, weights and training provenance.
They also exercise actual public-tool actor calls without teacher fallback,
malformed actions and an inference deadline stop. Runtime-package parity,
relevant lint and the paper build pass. Passing software regressions does not
change the negative learned-task results above; PR 224 remains a draft.

## Next training and integration work

Use these actual certified approach/grasp/lift/transport/place/retreat command
streams and matched reset states as the training/evaluation distribution.
The existing checkpoint was trained with a 5 cm base goal tolerance, whereas
arm-path execution permits only 1 cm drift. That mismatch is a concrete
distribution/objective concern, not proof of the sole cause. Preserve tight
hand accuracy and teach phase-appropriate base stability rather than globally
loosening collision certificates or tightening every free-space base target.
Continue to balance loaded tasks against tracking, quiet hold, smoothness and
disturbance recovery. Keep the failed return/regrasp planner as a separate
regression; more PPO does not repair an upstream planner rejection.

Broader household objects/scenes, normal-observation grounding, live asynchronous
retargeting, successful learned pick/place and Herman remain open. No training,
hardware connection, driver deployment or hardware motion is performed by this
integration experiment. Reproduction options are documented in the
[runbook](../mars_reactive_rl.md#fixed-checkpoint-tamp-execution-backend).
