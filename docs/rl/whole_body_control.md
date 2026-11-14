# Whole-body control philosophy

The aim is a learned controller that executes base and hand pose goals through
physical motion, including changing goals, contact and carrying. Accurate hand
tracking, a quiet base, retained objects and reliable stopping all matter.
TAMP supplies useful paths and demanding acceptance cases; ordinary base-only,
hand-only and combined goals also belong in the controller's task distribution.

This page describes design intent and lessons from the current experiments.
Individual implementations and simulator results have narrower scope; follow
the linked reports for their exact contracts and limitations.

## Keep planning, learning and execution responsibilities explicit

| Layer | Responsibility |
| --- | --- |
| Planner / reference producer | Supply reachable pose goals, masks, timestamps and gripper intent; use FK/IK witnesses for geometric admission where needed. |
| Learned controller | Use measured history and announced pose preview to request bounded base/arm motion. |
| Executor and monitors | Enforce command ownership, masks, native limits, contact/support rules, release authority, deadlines and measured stopping. |
| Physics / robot feedback | Establish the motion, contact and settling that actually occurred. |

The common task description is a masked world-frame SE(2) base target and SE(3)
hand targets with gripper intent. Robot-specific adapters define feasible action
coordinates: Mars has forward/yaw base motion and five arm joints. A sideways
base velocity is not an available control direction.

Specify mask meaning in the versioned runtime contract. An unconstrained base
can be needed to reach a hand target; a constrained base is a tracking objective,
not an instruction to zero wheel requests. Actuator authorization and hold rules
remain the executor's responsibility.

The actor may see announced future goals. It must not depend on future measured
states, private joint-path witnesses, case IDs or simulator-only task answers.
Geometry used to construct or admit a reference is not automatically an actor
input. See the [mastery contract](../plans/mars_tracking_mastery.md) and
[public-tool integration](../plans/mars_controller_integration.md).

## Use the teacher to expose gaps and establish comparisons

The analytic task-pose teacher provides a reproducible comparator and a check
that the selected physical problem can be executed under the declared limits.
Compare from the same reset and reference. Keep teacher failures visible;
failure of one solver does not establish geometric infeasibility or an upper
bound on what a learned controller could achieve.

A learned runtime should be evaluated as the declared learned actor. The legacy
feedback experiments emit complete motion requests without calling a teacher
at inference. A two-policy command replacement is a diagnostic intervention;
it must not be reported as a qualified, exportable single actor.

Training decomposition can still help. Initialize from a competent component,
freeze selected functions while adapting others, and export the declared actor
with its matching normalization. A frozen output row alone does not preserve
its function when shared actor context can change. Even an identical function
can receive different observations after another part of the robot moves, so
physical nonregression must still be measured.

Keep one actor at inference while testing the learning machinery. Recurrent
history encoders, structured feedback heads and separate critics are
architectural choices to evaluate. Critic-only simulator information must stay
outside actor inputs and exported inference. The separate parent-branch
[GRU / multi-critic prototype](https://github.com/cpaxton/home_robot_v3/blob/feat/mars-base-isolation/docs/plans/mars_whole_body_v2.md)
tests that separation; the optional legacy-MLP features in this branch are an
independent experiment. Neither architecture is established by software tests
alone.

## Represent motion and geometry deliberately

Moving references require anticipation as well as error correction. The analytic
teacher uses reference feedforward plus pose feedback. The legacy actor already
has a timestamped preview; explicit reference velocity is an opt-in way to make
that information accessible to its gain head. Such a feature must use the
correct frame and units, respect masks and executor progress, and vanish for
paused or completed stationary references. Reference corners remain part of
the evaluation problem.

Orientation error also needs an explicit convention. The prototype's optional
principal axis-angle error uses the logarithm of a relative rotation; angular
reference velocity uses a relative rotation over elapsed time. Rotation-column
residuals are another representation, so adding axis-angle features does not by
itself establish a tracking improvement. Test non-identity frames, angle
wrapping, half turns, gradients and transfer compatibility.

Preserve useful equilibria: at a stationary target, zero pose error and zero
measured velocity should permit zero mean motion. Audit observation scaling and
clipping before attributing failure to network capacity. Changing normalization
changes the actor's input function even when its weights are unchanged. The
[feature prototype](../experiments/mars_reference_velocity_prototype.md) and
[feedback-unit audit](../experiments/mars_feedback_units_20261113.md) document
the concrete implementations and limitations.

## Learn coordination while preserving useful skills

The base and arm are physically coupled. Arm tracking can induce wheel motion;
base corrections can disturb hand pose, contact or object retention. Separate
base and arm scores therefore need a combined evaluation too.

Preferring arm motion when it can supply the required task direction is a
useful allocation hypothesis. Base motion must remain available when required
by reachability or an active base goal. Diagnose unnecessary base correction
with matched rollouts before imposing a permanent yaw mask or authority rule.

The preview-velocity experiment illustrates this distinction. Six continuations
largely retained arm tracking while losing full passes through base-linear
jerk; explicit velocity did not prevent the regression. On four selected
training cases per variant, restoring initial base commands recovered 4/4 full
passes, while restoring initial arm commands recovered 0/4. That supports a
bounded frozen-base/context training test. It does not establish generalization
or prove that base freezing is the final WBC architecture. See the
[complete results and pilot protocol](../experiments/mars_reference_velocity_prototype.md).

Build difficulty around observed gaps: directional translations, rotations,
postures, coupled motion and loaded transitions. Count actual control-step
exposure, since early failures can defeat an apparently balanced episode mix.
Keep ordinary goals alongside TAMP-shaped paths. Tightening orientation or
position precision separately is a possible curriculum change to test, not an
established result of the velocity experiment.

## Treat smoothness and stopping as measured outcomes

Tracking and settling success must be reported alongside base-linear, base-yaw
and joint jerk, contact, support and retention outcomes. Millimetre hand error
does not imply that base jerk passed. A good average cannot hide a failed
direction, posture or transition.

Reward shaping can improve learning signals, but it does not redefine the
evaluation contract. Diagnose which motion channel fails before changing its
weight. Preserve native limits and paired teacher budgets; distinguish those
experimental comparator bounds from calibrated hardware limits.

For a fault stop, inspect the first violated condition and measured stop
confirmation. Do not infer that the policy chose to stop early from a short
rollout alone. In the inspected pregrasp archives, the recorded stops are
Cartesian tracking-envelope faults. The
[fault evidence](../experiments/mars_reference_velocity_prototype.md#fault-stop-evidence)
does not justify weakening stops or adding a premature-stop penalty.

## Make experiments useful even when the candidate fails

Declare the source bundle, changed factor, cases, seeds, transition budget and
fixed evaluation checkpoints before training. Verify initialized behavior and
normalization transfer. Require fresh source-bound software and teacher gates
appropriate to the selected experiment, then preserve every failure and the
initial/final reset and reference identities.

Report admission, tracking/settling, full-contract success and public task
execution separately, with complete denominators. Compare each final with its
own measured initial anchor. Selected-case interventions diagnose mechanisms;
held-out and broader physical evaluations test whether the change transfers.
Do not quietly extend a failed run or choose a successful checkpoint afterward.

Software review and policy adoption are different decisions. Correct, tested,
opt-in research code and negative results can be useful to merge. Changing
runtime defaults or promoting a checkpoint requires the declared behavioral
nonregression and acceptance evidence. Simulator success alone does not qualify
public-tool integration, normal-observation agent execution or hardware motion.

Respect the compute host's operating rules. On Olympia, the current native
arrangement is one CPU14 training/preflight worker and at most one CPU15 test
worker, one numerical thread each, CUDA hidden and ordinary rendering disabled.
Honor reservations and verify host-visible child exit before handoff. Caliban
and alternative physics backends require their own complete portability/parity
gates before qualification. See the
[resource and portability plan](../plans/mars_parallel_simulation.md).
