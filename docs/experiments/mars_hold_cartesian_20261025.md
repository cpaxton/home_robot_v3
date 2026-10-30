# Mars hold dynamics and Cartesian TAMP execution, 2026-10-25

This follows the [hold-repair experiments](mars_hold_repair_20261025.md).
The work isolates a low-level feedback problem, aligns training with the runtime
stop contract, and introduces an explicit simulator Cartesian arm execution mode.
Task, force, retention and wheel thresholds remain unchanged. No hardware is used.

## Hold dynamics

Matched ten-second runs start from identical frozen pregrasp states and compare
zero velocity, explicit actuator hold, reference and learned control. All six
base velocities are recorded from raw pre-step actor observations. The original
extra wheel velocity feedback gain is 2.0. A zero request still produces about
6.6 mm/s forward RMS velocity in the first fixture; explicit hold reduces it to
about 0.55 mm/s. Continuous quiet increases from 0.15 s to 9.9 s. This isolates
a controller issue without changing mass, friction, geometry or solver settings.
Most failed quiet samples are forward-velocity violations; pitch rocking also
appears, but is not the dominant failure in these two fixtures.

A separate six-second gain ablation uses the same original learned weights:

| Extra wheel feedback gain | Longest quiet hold, left / right (s) | Peak hand error, left / right (mm) |
|---|---|---|
| 0 | 5.85 / 6.00 | 0.60 / 0.44 |
| 0.25 | 6.00 / 5.80 | 0.45 / 1.21 |
| 0.5 | 6.00 / 5.85 | 0.48 / 0.99 |
| 1.0 | 0.10 / 0.30 | 1.08 / 0.68 |

The new research configuration uses gain 0.5. It is stored in the environment
manifest, validated, restored on checkpoint attachment and checked on exact
resume. Legacy defaults remain 2.0. This is a simulator development choice;
loaded turning, disturbance rejection and real motor calibration still need
validation. The underlying native wheel velocity actuators, speed/acceleration
limits and independent force/wheel monitors remain active.

## Training corrections

The earlier training success predicate checked planar base motion, while runtime
stopping checked all six floating-base velocities. `whole_body_settling` aligns
the new training configuration with the strict runtime predicate. Tests isolate
pitch motion that the old success predicate accepted. The existing jerk, effort,
contact and bounded all-six-DOF body-motion costs remain present.

The tracking-majority mixture still selects 70% reach, 10% pregrasp, 12% carry,
5% placement and 3% disturbance. Within reach, 20% are stationary holds and 30%
are nearby FK-certified pregrasp targets; the remainder preserve the broader
tracking distribution. Thus expected total reset shares are 14% holds, 21%
nearby tracking and 35% broader tracking, with 30% manipulation. Half the nearby
cases isolate reachable wrist rotations of 0.08–0.20 rad; the rest vary several
arm joints. They retain fixed base goals, collision/static-support screening,
derivative-bounded trajectories and normal open-gripper posture. Near-object
examples cannot be labeled as certified empty grasps.

Angular feedback errors use a declared 0.15 rad scale rather than 1 rad. Explicit
training transfer compensates both context/critic inputs and the learned gain
columns so the deterministic actor and critic functions match before training.
A function-level test verifies this migration. Exact resume rejects a changed
scale or wheel gain; incompatible bundles are not silently relabeled.

One CPU worker completed 16,384 PPO refinement steps, followed by a 1,024-step
continuation after the explicit gripper-hold handoff repair. Both use offline
scalar-only W&B. The earlier refinement snapshot retains its original source
binding; the continuation records the final control implementation. These runs
are research diagnostics, not carry qualification.

## Explicit Cartesian execution contract

`MarsSimRobot(..., arm_execution="cartesian")` requires an explicit policy.
The public replay CLI exposes `--arm-execution cartesian`; joint execution
remains the default. The agent still uses the public discovery, planning and
execution tools. There is no teacher fallback or new hardware path.

Planning continues to produce a collision/support-screened joint witness. The
Cartesian executor converts its whole path to FK poses on private data, with
minimum-jerk joint interpolation and existing speed/acceleration/jerk bounds.
The actor sees a timed Cartesian schedule with preview, rather than a succession
of isolated joint waypoints. TAMP's original requested pose is the endpoint;
allowed IK residual cannot add to the execution tolerance. A smooth final pose
correction bridges the witness's bounded IK residual without executing IK.

Execution checks the measured configuration and support margin before and after
every 50 ms control cycle, plus the original 1 cm / 0.02 rad full base-drift
envelope, payload retention and continuous force/wheel monitoring. Measured hand
lag retimes the schedule under a fixed total deadline. Large path tracking error
stops execution. This is sampled online collision checking, not a continuous
configuration-space certificate for the learned path. Benign contacts retain the
existing checker's exemptions; no force or retention limit is relaxed.

Endpoints use the existing 3 mm / 0.02 rad pose limits and 0.3 s settling window,
with all-six-DOF quiet and the existing 0.002 rad joint stillness bound. An
explicit terminal actuator hold preserves the grip. The executor can then
command bounded closure while the arm/base remain held; actor outputs cannot
move the held body, clear faults or authorize release. New tests cover this
handoff, nonunique joint solutions, exact Cartesian goal enforcement, stale
paths, measured collision, base drift and a zero actor's failure on a displaced
goal. Failures preserve public TAMP error categories and measured stop receipts.

## Evaluation and next gates

Final matched controller/teacher cases, public joint-versus-Cartesian replay and
fresh source-bound regression results are recorded with the completed run below.
No model is promoted on the basis of implementation or a short hold alone.

The completed first comparison after these repairs has five learned stationary
holds with continuous quiet streaks of 6.00/5.75 seconds (six-second runs) and
9.85/10.00/9.75 seconds (ten-second runs). Peak hand error stays below 1.8 mm.
Nearby motion still fails: one fixture reaches 4.61 mm endpoint error but crosses
the 1 cm base boundary; the other stops on an inference deadline. Wrist rotation
reduces its 0.12 rad error only to about 0.101 rad before base yaw exceeds 0.02 rad.
The matched Cartesian teacher succeeds on both nearby and rotation fixtures.

Both public arm execution modes fail all three programs at pregrasp. The joint
mode makes 101 policy calls per case; Cartesian mode makes 371/391/371. The latter
now rejects actual pose progress/endpoint tracking rather than demanding a joint
witness. This does not establish executable manipulation. Broad five-category
validation also remains negative, including loaded tasks; the paired gate fails.

The next diagnosed mismatch is the training heading tolerance: 0.2 rad permits
base rotation ten times the fixed-base execution envelope. Fixed-base training
needs matching hard boundaries and stronger anchored-base feedback/reward.
Cartesian execution also needs its private measured-state snapshot refreshed
each cycle for retention checks and event timestamps. These follow-up corrections
must get fresh tests and training evidence; they are not counted as solved by
this first snapshot.

The first complete source-bound regression snapshot passes **719 tests**.
Evidence: `hold_dynamics_v3`, `hold_rotation_training_v3`, and
`hold_cartesian_validation_v3` summaries and checksummed raw archives in
`paper/data/whole_body_control/`. The adjacent regression receipt identifies
this source snapshot; later changes require a new receipt.

## Completed boundary/progress refinement

The additional 8,192-step run uses matching fixed-base boundaries, a 0.01 rad
heading settling bound on anchored trials, stronger anchored-base reward and
same-target hand progress. Total new PPO work in this iteration is **25,600
steps** across three runs, all CPU-only with offline scalar W&B.

| Learned development check | Result |
|---|---|
| Five stationary holds | All exceed two seconds continuous six-DOF quiet; peak hand error below 1.7 mm |
| Two nearby FK targets at six seconds | 3.20 / 3.47 mm position error; 0.0455 / 0.0479 rad angle error; both quiet at endpoint |
| Continuous nearby hold | Longest quiet pose streaks 1.50 / 0.65 s; neither reaches two seconds |
| Two reachable 0.12 rad wrist rotations | Both stop on the unchanged base envelope; 0.090 / 0.093 rad remains |
| Three frozen public programs, learned Cartesian mode | 0/3 complete; all stop at pregrasp with confirmed stops |

The nearby endpoint checks use 5 mm / 0.05 rad. They do **not** pass TAMP's
3 mm / 0.02 rad arm endpoint contract. General five-category validation and
matched teacher results remain archived; one carry diagnostic succeeds, but
the paired no-regression gate is false. One episode per category is not a
success-rate estimate. No learned checkpoint is promoted.

Compatible UI sessions now restore the bundle's wheel gain and retention
observation suffix, with matching observation-space shape. Switching back to
the reference restores its original settings without moving/resetting physics.
Model compatibility checks remain enforced. Completed-training evidence now
retains the original recorded source manifest and environment, including
historical omitted defaults; later UI/reporting edits cannot relabel training.

## Integrated analytic teacher and terminal braking

An explicit Cartesian analytic teacher runs the same public discovery, planning
and execution tools on frozen case `left:20g:0`, using two documented rotation
metrics (1.0 and 0.15 m/rad). It uses no learned weights or fallback. Both versions
complete pregrasp, grasp and lift, proving this prefix is executable through the
new Cartesian mode. Their first transport runs then trigger the wheel-unloading
guard during the transition to position hold: maximum wheel gaps are about
0.16–0.19 mm, against the unchanged 0.10 mm unloading criterion. A result-path
bug overwrote that first safety cause with `transport_tracking_timeout`.
The fix preserves the first cause in the robot state, result and public tool
code. Analytic callable counters are explicitly labeled as reference calls,
rather than checkpoint inference. The original misleading reports remain
archived alongside the corrected results.

The Cartesian transport mode now clears drag compensation and waits for measured
whole-body quiet while the pose loop remains active, before rebasing into
terminal position hold. The reference joint mode retains its prior contract.
A moving-chassis regression verifies that the hold transition waits for quiet;
separate tests preserve wheel, force and retention failures during verification.

After this change, **both analytic controller configurations complete the first
pick-and-place transfer**, including destination support, regional placement,
open/clear gripper, quiet robot and two seconds of independent object stability.
Neither detects wheel unloading; maximum wheel gaps are 0.031–0.038 mm and peak
reported contact force is about 3.73 N. Both then exhaust the existing return
planner's budget. This is one frozen physical case repeated with two analytical
configurations, not two independent qualification cases or learned success.

![Hold dynamics and continuous quiet](../../paper/figures/mars_hold_dynamics_20261025.png)

![Selected Cartesian teacher delivery and wheel-gap comparison](../../paper/figures/mars_cartesian_oracle_20261025.png)

The final-source learned replay still fails all three pregrasp stages; the
terminal-braking repair does not imply the learned policy reaches transport.
The complete public suite, loaded generalization, normal-observation agent and
Herman gates remain open. There are no hardware commands in this work.

## Evidence and bounded next run

`paper/data/whole_body_control/` contains checksummed summaries and raw archives:

- `hold_precision_training_v4`: original training source/model/weight bindings,
  numbered checkpoint manifests, commands, scalar logs and all validation/teacher
  comparisons. It is also usable as training evidence for the local candidate.
- `hold_cartesian_validation_v4`: matched pose and public joint/Cartesian results
  before the final transport-result repairs.
- `hold_cartesian_final_policy_v5`: final-source learned public replay, preserving
  all three confirmed pregrasp stops.
- `cartesian_pose_oracle_v5`: all three analytical implementation snapshots,
  including the overwritten timeout, corrected safety failure, and verified
  delivery after quiet braking.

The candidate weights remain local at
`logs/manipulation/hold-fixed-contract-feedback-v8/seed-64/step-8192`.
Reproduction of the public replay uses its archived declarative program and
`--arm-execution cartesian`; no controller IK or per-waypoint joint matching is
introduced. Geometric planning/replanning still uses IK and is reported separately.

The bounded longer refinement specification is
`configs/benchmarks/mars_hold_training_feedback_v5.json`: 65,536 additional steps,
one CPU environment, checkpoints every 8,192 steps, exploration standard deviation
0.05, learning rate 5e-5 decaying to 1e-5, and the actual TAMP endpoint precision of
3 mm / 0.02 rad. It retains the same low randomization and tracking/manipulation
mixture. Launch requires a fresh verified source-bound regression receipt.
This research run cannot promote a checkpoint or bypass carry/task qualification.
Its launch status is recorded separately; planned steps are not completed steps.

The final frozen implementation passes **737 source-bound tests**,
with a verified JUnit/receipt pair named
`hold-cartesian-final-v5-regressions-20261025`. Runtime package parity, relevant
lint and the paper build also pass. Intermediate checks invalidated by subsequent
UI/reporting repairs were not accepted as source-bound evidence.

The bounded refinement **started at 2026-10-26T00:31:24.741961+00:00** from source
`d57e25f1`, after verifying the 737-test receipt.
It runs in `logs/manipulation/tamp-pose-precision-v9/` on CPU 14;
`pose_precision_refinement_v5_launch.json` records the exact command, source
weights, configuration checksum and launch code. This immutable launch record
is not a completion report. Current progress and exits stay in the local run
directory. The canonical carry-qualification supervisor remains stopped.

## Cartesian input phase correction

Raw actor inputs exposed a further integration error: opening the gripper set
the servo phase to `release`, and installing a Cartesian schedule did not restore
the execution stage. In the preserved replay, 370 pregrasp actor calls therefore
received the release one-hot, while the public trace correctly labeled pregrasp.
The joint backend already updated the phase through its normal reference setter.

The Cartesian backend now uses that public setter before each schedule, without
clearing measured history, grip force or faults. It restores retention intent,
keeps the unloaded measured aperture when resuming after release, requires an
open gripper for pregrasp/grasp, and cannot grant release authorization. Release
preview uses the authorized open target. Tests inspect the actual first actor
observation for pregrasp, carry and release, and reject ungranted release.

A paired replay at the same 24,576-step checkpoint and identical initial physics
confirms the corrected actor phases. Both runs still time out at pregrasp; this
fix does not by itself establish useful learned execution. The raw paired data,
phase counts and a fresh source-bound regression receipt are preserved in
`cartesian_phase_handoff_v6` and its adjacent archive/receipt. Diagnostic quiet
metrics now accept explicit pose bounds and default to the bundle's declared
hand contract, so 5 mm / 0.05 rad development checks are not confused with TAMP's
3 mm / 0.02 rad endpoint.

The 65,536-step refinement has completed, bringing newly completed PPO work in
this iteration to **91,136 steps**. Its final nearby positions are approximately
0.40/0.42 mm, but angles remain 0.0232/0.0245 rad. Wrist-only position errors are
about 2.5 mm, with 0.054/0.059 rad angular residuals. These fail the strict angular
contract despite substantially better position tracking. Corrected Cartesian
public replay still times out at pregrasp. At the stalled intermediate reference,
hand position lag is about 3.1 mm and angle lag about 0.016 rad: the binary
reference-clock gate is using final endpoint precision at every intermediate
sample. This exposes a separate trajectory-following bottleneck; the next
controlled execution comparison keeps final and safety bounds unchanged.

## Completed precision refinement and continuous reference clock

The completed 65,536-step run is archived in `pose_precision_training_v6`
with all nine numbered checkpoint manifests, actual original source/configuration
bindings, offline W&B scalar records, validation and teacher comparisons. Together
with the earlier runs in this iteration, **91,136 PPO steps completed**. Weights
remain local. The final diagnostic comparison remains ineligible: the policy
loses the teacher's ordinary-reach success and retains a disturbance failure.
One episode per category is diagnostic, not a qualification denominator.

`pose_precision_validation_v6` preserves all 15 final pose probes and six
joint/Cartesian program comparisons, including explicit strict and loose quiet
metrics. Nearby position residuals improve to **0.420/0.402 mm**, but orientation
residuals **0.0245/0.0232 rad** exceed the unchanged **0.02 rad** limit. Wrist-only
residuals improve to **2.52/2.57 mm and 0.0585/0.0539 rad**, without the earlier
base-envelope abort in these two probes. Selected stationary holds remain quiet;
these results do not establish loaded or moving-goal safety.

A second execution repair replaces the binary reference-clock gate with
continuous lag scaling. The old clock required final 3 mm / 0.02 rad precision
at every intermediate sample and froze at approximately 3.1 mm lag. The new clock
slows proportionally within the existing 10 mm / 0.10 rad tracking tube. It never
runs faster than the original schedule. The path tube, 1 cm / 0.02 rad base drift
boundary, force/retention/wheel checks, final 3 mm / 0.02 rad accuracy, settling
and total time cap are unchanged. Invalid retiming tolerances stop before actor
inference. Regression tests cover continuity around the old threshold and
rejection before motion.

`cartesian_clock_v7` contains all three final public replays. They still complete
**0/3 programs and 0/6 transfers** with confirmed stops. Two stop on pregrasp
timeout after 371 actor calls each; the 50 g case stops on the tracking envelope
after 198 calls. In the first case, the new clock presents the full 4.827 s
reference instead of freezing at 2.25 s. Its final measured error is **4.21 mm /
0.0324 rad**. The clock repair removes a demonstrated execution deadlock; it does
not turn this policy into a successful manipulation controller.

The same archive includes a fresh analytic baseline on one frozen 20 g case.
The 0.15 m/rad metric again completes the first delivery, passes all independent
physical predicates and two extra seconds of hold, then exhausts the return
planner budget. The 1.0 m/rad metric stops on `policy_cycle_deadline` during place.
Both outcomes are retained. The previously successful two-metric replay remains
historical evidence, not a claim that both currently pass. No learned success or
hardware readiness follows from the analytic comparison.

The next bounded reward ablation is specified in
`configs/benchmarks/mars_hold_training_feedback_v6.json`: two sequential 16,384-step pilots from the same
final checkpoint and seed, validation every 4,096, same mixture, exploration,
decay and physical limits. The control retains the old objective; the treatment
enables the existing `goal_gated_settling` option. The
previous run left this option false, permitting quiet bonuses and endpoint motion
penalties before meeting the pose contract. The ablation awards that bonus and
applies that extra endpoint penalty only once in goal; ordinary effort, body
motion, jerk, force and safety costs remain. This is a hypothesis to measure,
not a claim that more training will resolve the remaining angular error. Launch
requires the fresh source-bound regression receipt and records its actual status
separately from this specification. Automatic promotion remains disabled.

The final phase/clock implementation passes **744 fresh source-bound tests**,
with verified XML and receipt `hold-cartesian-clock-v7-regressions-20261025`.
Runtime package parity, relevant Ruff/whitespace checks and the updated paper
build pass. This receipt supersedes earlier snapshots for current-code launch;
all older reports remain bound to their original sources.

The matched control/treatment pilots **started at 2026-10-26T01:28:00.229388+00:00**
from committed source `da2cad3a`, after verifying all
744 source-bound tests. They run sequentially on CPU 14 in
`logs/manipulation/tamp-goal-gated-precision-v10/`. The immutable
`pose_goal_gated_refinement_v6_launch.json` contains both exact commands,
configurations, original weights, source state and the evaluation driver.
After both complete, the driver evaluates six strict pose probes and the three
frozen public programs per variant, asserting identical initial physics across
paired programs. Results go to `logs/eval/goal-gated-pair-v10-20261025/`.
Launch is not a completion or improvement claim; local status and exits are
recorded separately. No automatic promotion is enabled.
