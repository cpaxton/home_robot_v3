# Why supported-lift success remains low

The current learned controller has two independent deficits: its arm response is
too slow/inaccurate for the timed lift, and its base response fails to hold the
commanded pose. Reward and coverage contribute; simply increasing gain causes
retention failures. This is not explained by the pose target being universally
unreachable, normalization clipping, or the arm hitting its actuator force cap
on the tested cases.

This follows the [supported-lift experiment](mars_grasp_lift_20261027.md).
The selected coupled-reward diagnostic is `v18/precision/step-15360`. It was
selected using one training case before independent replay; the existing eight
assessment cases are now repeatedly used for diagnosis and must not be called a
new qualification population. All interventions are simulator-only. No PPO,
production policy change, threshold relaxation or hardware motion occurs here.

## Matched arm/base channel isolation

Each intervention starts from identical admitted physical state on seeds
3102000–3102007. The targets, 12-second horizon, force/retention/wheel guards,
3 mm / 0.02 rad hand endpoint and base limits are unchanged. Reference actions
use the teacher's privileged joint witness and serve only to isolate channels;
hybrids are not learned-policy successes or proposed deployment backends.

| Arm / wheel control | Success | Mean final hand position / rotation | Outcomes |
| --- | ---: | --- | --- |
| Learned / learned | 0/8 | 13.48 mm / 0.0295 rad | 4 full-base stops, 4 timeouts |
| Learned / zero wheel request | 0/8 | 14.02 mm / 0.0129 rad | 8 timeouts |
| Learned / reference wheels | 0/8 | 13.88 mm / 0.0125 rad | 8 timeouts |
| Reference arm / learned wheels | 0/8 | 2.51 mm / 0.0027 rad | 7 full-base stops, 1 timeout |
| Reference / reference | 7/8 | 2.06 mm / 0.0010 rad | 7 successes, 1 terminal-settling timeout |

Replacing wheels removes the hard base stops but leaves arm position error.
Replacing the arm makes its pose much more accurate but does not fix learned
base motion. Thus neither a base-only repair nor a hand-only repair is enough.
The full reference result reproduces the earlier teacher supplement, including
its failure; the small assessment set is not proof of universal feasibility.

![Matched channel isolation](../../paper/data/whole_body_control/lift_failure_diagnosis_v1/channel_isolation.png)

The learned controller has substantial endpoint debt. During moving references,
the 95th percentile of its maximum requested arm-joint rate is **0.068 rad/s**,
versus **0.207 rad/s** for the full reference; the native arm limit is 0.6 rad/s.
The learned controller's 95th-percentile commanded-position lag is only
0.00385 rad, versus 0.01067 rad for the successful reference. None of the arm
actuators reaches its configured ±2 actuator-force limit in these interventions.
Measured motion follows the small requests with some lag. These measurements
support insufficient/poorly coordinated policy commands rather than a saturated
arm as the immediate limitation. They do not establish real-world torque or
thermal margins or rule out all physics-model errors.

Base correction is also faulty. Among the learned controller's 727 running
samples with heading error above 0.01 rad and a nonzero yaw request, **79.9% of
requests turn away from the base goal**. The reference has zero such away-turning
requests in its 361 eligible samples. These are pooled, correlated diagnostic
samples, not independent statistical trials. Arm/base coupling may motivate
transient motion, but it is not correcting the active fixed-base constraint.

## What the success gate is rejecting

All eight learned rollouts finish outside the active 0.01 rad heading success
limit. Four hit the distinct 0.02 rad full-body rotation safety boundary and stop
early. Of the four that consume the full horizon, three also miss hand position
or orientation. The remaining case ends at **2.89 mm / 0.0071 rad**, but heading
error is **0.0176 rad**. Its hand/lift/arm-quiet predicates overlap for only
0.15 seconds in the recorded rollout, short of the required continuous second.
Consequently these full-length failures are not explained solely by base grading.
Dropping a predicate from a stopped trace is not a valid counterfactual physics
rollout, and no such result is reported as a success.

The goals move for approximately 4.5–7.3 seconds and then hold. Several learned
rollouts are still converging late; there is no demonstrated hard 4 mm kinematic
floor. The strict gate requires the original pose **and** quiet for one second,
while retaining sufficient object lift and safety. A single accurate endpoint
sample or a low average error is insufficient.

## Increasing gain trades error for loss of retention

Two additional matched diagnostics use reference wheels and multiply only learned
arm requests, retaining the same slew and native limits:

| Arm multiplier | Success | Retention losses | Peak monitored force |
| --- | ---: | ---: | ---: |
| 1× | 0/8 | 0/8 | 4.09 N |
| 2× | 0/8 | 1/8 | 7.53 N |
| 3× | 0/8 | 6/8 | 13.71 N |

Two surviving 2× cases finish inside hand-pose tolerances but have only 0.9 and
0.5 seconds of contiguous settling when the horizon ends. Other cases still
miss position or lose the object. The force ceiling is not crossed, but increasing
load and loss of retention make this an unsuitable shortcut. No wheel unloading
occurs in these component interventions; public-delivery unloading from the
previous mixed-training treatment remains a separate, unresolved failure.

## Objective and input problems

The old reward gives independent fixed-base holding credit even while the hand
misses the target. Parking on the fixed development case earns 43.02 total
reward, versus the successful teacher's 62.85. At the actual discount 0.99 the
returns are 18.84 and 23.43. The teacher is still preferred; this is a strong
competing parking incentive, not proof that parking is globally optimal.

A timing mismatch was found in the earlier external diagnostic wrapper: it
subtracts base credit using the **previous tick's** hand goal, whereas the native
environment scores the advanced current goal. Historical results remain valid
for that wrapper, but it is not an exact native same-tick reward implementation.
A new matched accounting replay retains both calculations: the old wrapper gives
parking/teacher totals **2.67 / 49.81**; same-tick coupling gives **2.48 / 49.34**.
The difference does not reverse the finding, but the production correction must
use the goal actually scored, with a regression test for moving references.
There is no claim that the old single-case wrapper is already production-ready.

The actor also consumes absolute world-coordinate context. A counterfactual
input audit translates all historical measured robot poses and active targets
together, leaving joints, velocities, phase and the relative preview unchanged.
A 5 cm translation can change an arm request by **0.0613 normalized units**, or
**0.0368 rad/s**, and a base request by 0.0316 normalized units. The gripper head
also changes. No input clipping occurs in the 27 sampled initial/moving/terminal
observations; gross normalization clipping is not the explanation in these
samples. The context can choose different gains for the same relative task at a
different world origin. This exposes a generalization burden, not proof that
canonicalization alone will solve execution. These are input counterfactuals,
not physically translated scene trials or task-success measurements.

Finally, the mixed treatment contains only **2,038 supported-lift transitions**
out of 16,384 PPO steps: about **102 seconds of simulated lift data at 20 Hz**.
It does not contain 16,384 loaded steps, nor do the 90,624 aggregate experiment
steps belong to one policy. The better diagnostic is trained on one posture;
assessment changes the initial elbow and lift duration. Broad coverage and
loaded settling are under-trained. Actual reset/unique-trajectory counts were
not logged and cannot be inferred from step share. Low randomization also keeps
mass near 60 g; this does not cover the entire configured mass range or qualify
20 g public deliveries.

## Implementation priorities and intermediate gates

1. Add a native, opt-in hand-conditioned base-credit correction using the current
   scored goal. Preserve base-only reward and inactive-mask semantics. Test
   moving-goal timing, reward accounting and unchanged safety before more PPO.
2. Reduce dependence on arbitrary world origin through a versioned relative
   representation or consistent coordinate augmentation. Keep public world-pose
   commands unchanged. Test equivalent translated inputs; rotations must preserve
   component-mask semantics rather than treating partial XY constraints as a
   full pose. Compare against the unchanged input representation.
3. Cover supported lift start, intermediate lift, late lift and loaded hold as
   bounded parts of the tracking-majority mixture. Physically verified training
   prefixes can expose later phases without making the actor solve a long failed
   prefix first. Include small base-heading recovery states and mid-motion goal
   changes. Reset assistance ends before scored motion; there is no runtime
   teacher fallback, teleport, attachment or contact permission relaxation.
4. Before scaling up, require learning on several newly generated feasible
   development trajectories, correct-sign base recovery and contiguous terminal
   settling. Then compare matched mixed-training variants on **new** assessment
   seeds and the unchanged public delivery/multi-transfer contracts, retaining
   all force, jerk, retention and unloading failures. Advance randomization only
   after validation; do not spend another long run on a one-case overfit alone.

The result is an actionable diagnosis, not a controller qualification. All 56
channel/gain intervention episodes, nine input-audit rollouts and five accounting
replays are retained. Raw exploratory gate counters use per-axis speed checks;
published endpoint blockers use native norm scales, and success/continuous
settling come from the environment itself. The production source is unchanged;
the existing source-bound **801-test receipt still verifies**. No new full suite
run is claimed for these data/documentation additions.

Evidence: `paper/data/whole_body_control/lift_failure_diagnosis_v1/summary.json`
binds complete raw traces, input/accounting audits, all failures, scripts and
source/config/weight identities with SHA-256. Weights remain local. No hardware
commands were sent.
