# Whole-body control philosophy

The aim is a learned controller that executes base and hand pose goals through
physical motion, including changing goals, contact and carrying. Accurate hand
tracking, a quiet base, retained objects and reliable stopping all matter.
TAMP supplies useful paths and demanding acceptance cases; ordinary base-only,
hand-only and combined goals also belong in the controller's task distribution.

RL is meant to make execution adapt when the world, the robot or the reference
changes. A controller should keep useful motion when it can, recover from small
disturbances, and recognize when a commanded pose is blocked. Tracking error,
contact severity, support, grip and the quality of the stop all matter to that
decision. Evaluate these behaviors as well as nominal tracking.

This page describes design intent and lessons from the current experiments.
Individual implementations and simulator results have narrower scope; follow
the linked reports for their exact contracts and limitations.

## Coupled control across embodiments

Coupled whole-body control is the intended solution for both Stretch and Mars.
A hand's SE(3) workspace includes authorized base motion; an arm-only
reachability failure is not evidence that general whole-body control is
infeasible. A stationary base is a requested constraint for selected cases,
not a universal prerequisite for hand tracking.

Keep the task interface independent of arm count and joint count: one optional
component-masked SE(2) base target, one SE(3) target per supported hand, and the
native coordinates of each gripper/hand. Base regions and partial constraints
leave coordination freedom; arbitrary independently sampled full base/hand
constraints need not be jointly reachable. Runtime support for regions or
additional hands must be implemented and tested explicitly before claiming it.
Robot adapters retain their own joints, actuator limits, base kinematics and
support/contact checks. A holonomic base and a differential-drive base must not
share an assumed lateral-velocity action. This common interface does not yet
require shared policy weights across robots.

Generate reachable references from complete robot configurations and paths,
including base motion, and admit the requested trajectory under native timing,
collision and support limits. Cover hand-only goals with free base targets,
fixed-world-hand/base motion, jointly feasible constrained targets, mid-motion
retargeting and settling. Keep arm-only cases as diagnostics or explicit
fixed-base tasks, while reporting coupled performance separately.

A hardware limitation requires evidence: distinguish a missing kinematic
solution, infeasible dynamics/contact or reference timing, a simulator/model
error, a teacher/solver failure, and learned-policy tracking failure. Neither
counting arm joints nor one solver's rejection settles that diagnosis. See the
[cross-embodiment comparison plan](../plans/general_whole_body_tracking.md#coupled-control-and-embodiment-comparison).

## Keep planning, learning and execution responsibilities explicit

| Layer | Responsibility |
| --- | --- |
| Planner / reference producer | Send pose goals with frames, masks, timestamps and gripper intent. Use FK/IK witnesses when the planner can establish feasibility. |
| Learned controller | Use measured history and announced pose preview to request bounded base/arm motion and adapt when progress is blocked. |
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

Define training primitive boundaries in the executor/harness, while keeping
the actor's pose-tracking interface consistent across stages. An unloaded
grasp-pose boundary ends before closure, with measured arrival and stopping;
closure, support-reserved lift and loaded retention require their own evidence.
Stage-specific teacher admission must cover the requested motion. The
[unloaded-prefix readout](../experiments/mars_unloaded_grasp_admission_20261116.md)
shows why execution boundaries and planning admission both need explicit scope.

WBC should be the sole source of normal base and arm motion commands. Operator,
teleoperation and planner inputs share the same waypoint lifecycle. They can
send and replace timestamped pose schedules and receive command receipts with
blocked outcomes and measured stop status. The robot-side driver owns the
watchdog and protective stop independently of WBC. Start with
simulator submission and cancellation, then verify the robot-side actuator
adapter and watchdog before enabling hardware submission. See
[controller deployment](CONTROLLER_DEPLOYMENT.md).

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
[GRU / multi-critic prototype](../plans/mars_whole_body_v2.md)
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

## Handle blocked goals and contact during execution

The controller will receive goals that require self-collision, penetration of
an obstacle, or motion past a joint limit. A feasible path can also become
blocked after an obstacle moves or a user replaces the goal mid-motion.
Use reachability and collision estimates to choose and shape an approach.
Treat uncertain proxy overlap as a reason to slow or monitor more closely
within the validated operating envelope. Preserve hard bounds, actuator
ownership, feedback validity and protective stops.

When a safe combination of base and arm motion can make progress, the
controller should use it. Brief, low-energy contact can be compatible with
the task. If further tracking would require pushing, reduce drive into the
contact and report `blocked` to the planner or console. A bounded retreat is
appropriate when clearance, support and grip allow it. Confirm unloading as
well as settling. A stationary arm can still be pressing on an obstacle.
Forceful contact, sustained drive without progress, wheel unloading and
actuator faults need an independent protective response.

Start with a recovery expert for coupled base-and-hand commands. Include
reachable controls, benign contact, infeasible endpoints, moving obstacles
and mid-motion waypoint replacements in the training and held-out evaluation
distributions. Vary initial pose and speed, obstacle geometry and compliance,
and sensing and actuation delay. Reward useful tracking progress and timely
unloading so the expert has a reason to yield before a guard fires. The
current actor sees measured joint motion, tracking error and previous commands.
Use simulated contact truth for rewards, labels and critics. Evaluate whether
adding source-timed motor effort improves detection once that signal is
calibrated on hardware. Qualify the expert on normal tracking as well as
recovery before applying the training to the other experts. Keep one actuator
command owner throughout any recovery or retarget.

Score benign-contact recovery and blocked-command handling separately. Report
task progress, false stops, peak force or effort, impulse, time spent pushing,
time to unload and confirmed hold. On Herman, increase speed within an envelope
measured on hardware across poses and speeds. Measure feedback age, detection
and actuator delays, stopping travel and load decay. Keep the fast contact
response and command-loss watchdog beside the driver, independent of actor
inference, video and the browser. The
[Herman blocked-waypoint baseline](../experiments/herman_wbc_blocked_waypoints_20261122.md)
records why stop confirmation alone is insufficient.

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
