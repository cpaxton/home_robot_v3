# Mars WBC: stabilize the base, define smoothness, then train

This is the ordered continuation after [PR #238](https://github.com/cpaxton/home_robot_v3/pull/238).
The goal is one reactive actor that follows base/hand references, settles and
satisfies physical smoothness, support, force and retention contracts. A better
hand endpoint is useful progress; it is not the full objective.

## Latest decision after actual-reset validation

The [completed reset/baseline study](../experiments/mars_current_feedback_reset_validation_20261115.md)
passes all 24 loop stop checks and 1,299 fresh software regressions. Current-COM
feedback is validated through initialization, forward/reverse/yaw and finite
disturbances. Retained legacy gives 151/164 general and 12/12 on the common C1
cases, versus GRU candidate 8/12. Original budgets and outcomes remain available.

Actual TAMP execution remains the limiting result: retained legacy 0/6 versus
teacher 5/6. A further 24 fixed episodes find no learned success from extra
settling, the frozen velocity final or the GRU candidate. The difficult teacher
fixture changes from an execution lift-stage failure to a planning rejection
after settling; it is not silently admitted under the old exception.

The [four-episode input diagnostic](../experiments/mars_tamp_input_diagnostic_20261115.md)
now verifies exact teacher recording parity and exact offline actor replay.
Targets reach the learned actor, but its small commands lag; open-jaw feedback
then leaves the narrow training normalization range. Its final matched error is
3.01 mm/0.02966 rad versus teacher 1.11 mm/0.00108 rad. The separate teacher
rejection is measured-pose lift replanning with `arm_path_static_support_margin`.
The public failure now preserves `planning_failed` and semantic support-margin
evidence. Keep the legacy actor as the stronger general baseline. Independently
admit measured-state references; then test
actual pregrasp/near-contact training mixed with general tracking, collecting
normalization statistics from that distribution. Keep real grip feedback and
all safety checks. PR #240's direction/audit manifests remain preparation,
not native admission or training evidence. More time on C1-only training is
not the selected next action. No default promotion or hardware qualification
follows from these results; retain the staged validation.

## Why this order

| Evidence | What it supports | What it does not establish |
| --- | --- | --- |
| Reduced base exploration improves arm tracking/settling 1/10→7/10; full arm stays 1/10. | Keep the option available for controlled experiments; examine the remaining base motion. | A qualified actor, a coupled-task improvement or a reason to change defaults. |
| Nine candidate linear-jerk peaks occur during ordinary running. | Investigate motion before faults and stopping. | That the scalar gate alone measures physically harsh motion. |
| Constant-acceleration reversals spike unsigned-speed jerk; saved candidate signed-vector peaks are larger too. | Make the measurement definition explicit and preserve both diagnostics. | Permission to relabel the candidate's failures. |
| Matched native stationary anchors show high instantaneous-velocity proxy values under both zero actor and analytic teacher. | Investigate the plant and sampling independently of learned commands. | A policy-only or reset-only explanation. |
| The companion 2-ms substep trace shows approximately 50-Hz chassis/wheel-command motion under zero requested base velocity. | Isolate the inner feedback loop and actuator/contact response before increasing RL penalties. | Proof of which feedback source or gain is responsible. |
| Current-COM feedback with the same frozen actor raises full success 2/12→8/12 and arm smoothness 1/10→10/10 in matched-reset replay. | Prioritize correctly timed current-velocity feedback; the nine arm jerk failures disappear without PPO or a new gate. | Qualification of the complete production/reset path or proof that COM is best for every load/task. |

The first three results are archived in the
[completed exploration report](../experiments/mars_reference_base_exploration_20261115.md).
The native results come from the companion measurement study in
[PR #239](https://github.com/cpaxton/home_robot_v3/pull/239), with separate
producer identities. Its
[completed report](https://github.com/cpaxton/home_robot_v3/blob/feat/wbc-jerk-measurements/docs/experiments/mars_aligned_jerk_measurements_20261115.md)
preserves all 64 valid episodes and the invalid first twelve-episode feedback
attempt separately. The selected runtime source `77714a30` passes 1,284 fresh
regressions. This does not qualify the broader/reset/loaded behavior or change
the original PR #238 experiment producer.

## 1. Isolate and repair the low-level base loop

**Change to investigate first:** replace ambiguous/stale feedback with a
correctly timed velocity at an explicit physical point. Verify the wheel
transmission, feedback units, saturation and integration timing as part of
that change. The actor should not have to cancel inner-loop oscillation.
The companion implementation exposes this as the Mars-only
`wheel_feedback_current_velocity` option, initially default off, preserving
gains, limits and output shape. Correct cache timing is the first patch to
validate; COM versus axle remains an explicit physical-point decision.

The companion study owns the completed **four feedback variants × three
teacher cases = twelve episodes**: frozen feedback, feedback gain off,
current-qvel COM feedback and current-qvel axle feedback. Hold, forward and
quarter-pitch cases separate quietness, base motion and arm coupling; yaw still
needs its own validation. These are causal diagnostic
interventions, not unchanged-policy qualification replays.

The selected next implementation is the minimal **current-velocity feedback**
option, retaining gains and limits, rather than disabling feedback. In the
completed three-case teacher probe, disabling feedback raises forward linear
jerk from approximately 1.24 to 13.80 m/s³. Current COM and axle feedback both
remove the running stationary chatter and retain all three frozen task gates.
The companion 24-episode learned replay (`77714a30`) then gives the same
candidate weights 8/12 full passes with current COM versus 2/12 cached; arm
tracking stays 7/10 while all ten arm smoothness checks pass. This isolates a
plant improvement, not a learned-policy update or a reason to raise jerk costs.

Record requested motion, applied motion, wheel targets/forces, body/axle/COM
velocities and actual timestamps at every physics step. Report oscillation
frequency/amplitude, command/force variation, drift, task error, stopping and
every original gate. Disabling feedback is a diagnostic, not a preferred fix
merely because a stationary trace becomes quieter.

| Diagnostic result | Next implementation |
| --- | --- |
| Current velocity at the same physical point removes the oscillation while retaining motion. | Correct feedback timing/cache use; retain the physical reference point. |
| Changing the point matters after timing is matched. | Specify and test the wheel rolling/axle reference and rigid-body transformations. |
| Only reducing/removing feedback helps. | Identify loop gain, saturation, delay and inner velocity-actuator interaction; select one bounded controller change. |
| None of these explains the oscillation. | Investigate actuator/contact/solver response with fixed commands before another policy experiment. |

After these completed readouts, any further causal diagnostic is one separately
declared batch of at most **24 additional native episodes**, with no PPO, before
another explicit plan decision. Broad admission below has its own declared
population. Do not launch an open gain,
friction, damping or solver sweep. Use actuator/model units and measured loop
response to choose values; do not tune contact physics solely to pass a policy.

MuJoCo computes derived fields before integration, so feedback timing matters.
Use a valid control-callback/stage arrangement or a demonstrated consistent
equivalent; do not insert an untracked `mj_forward` into a live replay. Confirm
the installed integrator before considering split stepping, because its
compatibility is integrator-dependent.
[MuJoCo simulation-loop documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop).
Wheel control values and actuator-space force must be interpreted through the
actual transmission, not assumed to be interchangeable with joint torque.
[MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).

**Exit:** a selected loop change has a causal explanation, improves the
stationary oscillation, retains forward/yaw tracking and measured stopping,
and introduces no support/force fault. Then freeze its source and run fresh
full checks, all 164 general teacher cases and six pregrasps, keeping the
known object-motion challenge in the denominator and reporting its new measured
outcome. Replay both frozen learned actors on all twelve
cases to measure the plant change without retraining. Publish failures too.
A changed plant is a new producer, not a renamed old qualification receipt.
Also rebaseline the strongest retained legacy actor on the 164 general cases
and six actual pregrasp primitives before choosing more training. Historical
negative results under the old loop remain valid history; they do not show
which frozen controller is strongest on the corrected plant. Keep weights
unchanged, declare the environment-option difference and report compatibility
and producer identity explicitly instead of silently retagging a bundle.
Matched-reset interventions do not qualify initialization: also validate the
option enabled through the actual reset/conditioning path, motion transitions,
hold and stopping. New reset states must keep their own hashes rather than be
reported as identical to the earlier plant.
Separate current-source teacher admission from the frozen budget anchor used
for comparison. Report both the archived-budget diagnostic scores (including
8/12) and fresh-source calibrated budgets. A quieter teacher can tighten a
relative jerk budget; the diagnostic gain is not a promise of the same count
after recalibration. Preserve the anchor's real source/hash rather than
renaming it as current-source admission.

## 2. Lock measurement semantics and budget identity

**Immediate change:** standardize timestamped reporting and publish separate
admission and acceptance-contract records: source, metric version, point/frame,
cadence, reset convention, budget matrix and teacher-anchor hash. Retain the
existing gate formula for the first curriculum comparison. A new reward or
gate definition is not a prerequisite for teaching the missing directions.

Record signed world-XY velocity of a named point, its interval average over the
25 physics substeps per control tick, instantaneous samples and reset state.
Compare the integral with measured displacement. Distinguish angular-Z velocity
from Euler heading rate under tilt. Use consistent timestamps and derivative
initialization; report startup, running and stop events separately without
dropping startup samples from the original score.

A 20-Hz endpoint sample cannot characterize the observed 50-Hz behavior on its
own. Any control-band measurement needs predeclared anti-aliasing/aggregation
**before** decimation, with latency, initialization and bandwidth documented.
Keep the raw high-rate motion, force and support evidence. Do not apply a
post-hoc output filter or pick a cutoff because one checkpoint starts passing.
If the plant has meaningful high-frequency motion, diagnose that motion too.

**Exit before curriculum training:** analytic controls and native traces
establish what each existing channel measures, and both variants share one
predeclared contract/budget identity with every original score retained.
Keep the reward unchanged in that comparison.

**Conditional future change:** if smoothness remains limiting, implement one
shared motion channel for the reward and proposed contract, validated on
constant velocity, acceleration/reversal and turning. If a metric revision is
justified, version it and reevaluate teacher and actors
on it while retaining every old score. No pass is granted by definition alone.
Teacher-relative budgets and absolute limits must be explicit for the new
producer; a degraded teacher must not silently make the task easier.
Freeze and admit the separate audit set in step 4 before starting the learning
comparison. A filter proposed for the actual servo belongs in the versioned
plant change, with its delay/frequency response evaluated in both training and
execution; it is not a post-processing shortcut for reported smoothness.

## 3. Train missing directions, then change the objective only if needed

The existing C1 training pool is intentionally a diagnostic: base hold and two
quarter-pitch motions. Forward, roll and supported-contact failures are not
evidence that those skills were taught and failed. With the loop intervention,
the remaining four failures are base-forward envelope and roll-left,
roll-right and full supported-up orientation timeouts. Keep that distinction
visible before choosing the next learning change.

Choose the strongest validated single actor for task progress; the GRU must
earn its complexity against the retained source. If the corrected-plant
baselines still justify GRU learning, the **first learning change** is a
predeclared direction-balanced general curriculum
on the corrected plant. Commit an exact case manifest covering five base
primitives, both signs of all five arm-joint directions and two coupled
primitives: seventeen FK-reachable general cases. Use existing admitted
families where possible, with case/target identities frozen before training.
The FK/joint witness remains private; the actor still receives pose goals.
Use a separately versioned curriculum manifest/verification path; do not weaken
the existing three-case C1 specification or exploration-pair guard to admit it.
Allocate candidate control ticks across base/arm/coupled stages at 30/50/20 and
retain the control's declared C1 scheduler. Record both manifests and per-case
exposure, not just episode counts. Supported contact has a
separate later curriculum, rather than being claimed from this general basis.

Compare the three-case diagnostic curriculum against this seventeen-case
curriculum in **one seed-90 pair, 16,384 decisions per variant, 32,768 total**.
Both use the same corrected/admitted plant, unchanged reward and frozen gate,
shared advantages, exploration scale 0.1, LR 3e-4, five epochs, batch128 and
rollout256. Initialize identical zero-head actors, critics, optimizers and RNG;
do not silently continue an earlier checkpoint. Curriculum is the only changed
factor. Bind common initial state/settings and both declared case manifests;
rollout trajectories may differ because task scheduling intentionally differs.
This is a new larger-case study, not an extension of the finished C1 budget.
The candidate's allocation gives approximately 800–1,600 ticks per case; the
same total budget in the control avoids conflating curriculum with more PPO.

**Primary outcome:** full-contract gains on the twelve fixed cases and the
separately frozen audit in step 4, without loss of the control's tracking/base
capability or independent safety. Publish every physical failure, per-axis jerk,
orientation, drift, actual case/tick exposure and gradient/advantage diagnostics.
No intermediate selection, automatic extension or weaker gate. Keep defaults
and the retained production source until adoption has its own evidence.

**If orientation remains limiting after actual directional exposure:** test the
implemented axis-angle idea separately against rotation-column feedback, with
explicit frame/scale/initialization semantics. Do not reinterpret trained gain
columns or combine representation changes with the curriculum comparison.

**If jerk remains limiting after the plant fix:** first use the shared measured
channel from step 2, then test teacher-calibrated per-axis dense excess,
incremental peak-excess cost and full-contract terminal credit. The current v2
terminal credit omits smoothness, and the measured jerk penalty is small next
to tracking return; those are concrete objective changes to test, not grounds
for an immediate coefficient increase. Preserve physical-tick discounting,
the three groups and actor/critic separation. Teacher budgets stay training-only.
Use a separately declared one-factor 4,096+4,096-decision reward pair; this
conditional study does not run just because the curriculum experiment finishes.
Changed reward arrays must not be misrepresented as an identical first rollout.

**If no full-success improvement:** retain the negative result and diagnose the
remaining signal/exposure/gradient constraint before another budget. If the
curriculum succeeds on fixed and audit cases, separately register the same
paired study on seeds91/92 (65,536 additional decisions at the same per-run
budget). Replication and broad evaluation precede any general capability claim.
Force, support, retention and measured-stop guards remain independent; do not
forbid fault stops or extend deadlines to make trajectories finish.

## 4. Establish broader capability before expanding the task

Replay the retained source, analytic comparator and selected candidate on the
complete existing 164-case general population, reporting each stage/split and
contact stratum. Do not combine overlapping 12/122/164 populations into a new
denominator. Preserve coupled and orientation regressions even if pitch improves.
Freeze a separate 24-case audit (eight base, eight arm, eight coupled) with
disjoint resets and declared task families before the learning comparison;
obtain its teacher admission before optimization. It is not a fresh holdout
if cases are selected after looking at the candidate.

Only then advance actual pregrasp, closure/static retention, small lift,
loaded transport/braking and placement/release in separate bounded stages.
Each needs its own admitted cases, rehearsal budget by actual control ticks,
full prior-task readout and public-tool evaluation. Supported open-jaw tracking
does not qualify an object grasp. The retained actor's existing pregrasp/task
failures remain visible. Use the
[safety/task curriculum](2026-11-13_whole_body_safety_curriculum.md) for those
stage contracts rather than inferring loaded readiness from free-space success.

## 5. Scale only a coherent problem

Keep actuator/sensor delay, payload/friction variation, perturbations and held-out
goals as explicit robustness studies. A differentiable world model may follow
stable plant and measurement definitions, with actuator/executor state,
whole-trajectory holdouts and native policy validation. Compare total collection,
fitting, training and evaluation cost. Model-predicted success is not a native
qualification result or a reason to call this state of the art.
Use executed actuator commands and physical timestamps, not only raw policy
requests, so the model represents the actual servo/executor behavior.

For a competitive capability claim, compare the analytic pose controller,
retained legacy actor and new actor under the same plant, goals, limits and
evaluation code. Report full task success, all physical failures, latency and
total compute across the declared training seeds and fresh audits. Published
humanoid/mobile-manipulation methods motivate designs; different robots/tasks
do not provide a matched Mars benchmark by themselves.

Caliban remains conditional on its existing frozen-fixture/native parity gate.
Olympia stays at one CPU14 native training/replay worker and at most one CPU15
test worker, one numerical thread, CUDA hidden and ordinary rendering disabled.
No local GPU training or hardware motion is part of this plan. Hardware requires
its own measured feedback, limits/watchdog/stop verification and authorization.

## Delivery and review

- PR #238 owns exploration, review fixes and this overarching plan. Original
  experiment producer `3ac722e1`, outcomes and hashes remain unchanged.
- The companion measurement branch owns native/substep instrumentation and
  loop-isolation evidence. Publish it separately, then use its completed result
  for the step-1 decision. Do not duplicate its probes or mix producer identities.
- A loop fix, a measurement/contract revision and a reward experiment are
  separate reviewable changes. Every runtime source needs matching fresh gates.
- Research code can merge with documented negative results. Changing defaults,
  promoting a checkpoint and demonstrating hardware readiness are separate
  decisions; none follows automatically from passing software tests.

Track completion in [TODO.md](../../TODO.md#current-wbc-follow-through-2026-11-15).
This document declares decisions and conditional budgets, not completed work or
permission to skip the preceding evidence gate.
