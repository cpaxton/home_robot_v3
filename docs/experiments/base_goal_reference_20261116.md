# Base endpoint references and velocity feedforward

Simulator-only follow-up to the [signed-feedback readout](base_direction_bootstrap_20261116.md).
The earlier `823b3d1d` candidate remains the retained baseline: 24/24 timed base
paths, 10/10 small general, 6/10 larger general, 6/6 unloaded TAMP grasp-pose,
4/4 moving base-to-hand handoffs, but only 2/8 raw static base goals. Those
results and failures are not replaced or pooled with this experiment.

## Diagnosis and fixed development cases

The same eight static endpoint goals use seeds 35850000–35850007, a 25-second
horizon, 5 cm XY acceptance, 5-degree heading acceptance and a two-second
measured quiet hold. The hand is inactive; precise active-hand contracts stay
unchanged. Goals include lateral and diagonal translations, forward/reverse
motion, and final heading changes. No hardware or new PPO steps are involved.

The actual Mars drive-axle offset is about 14 micrometres, not a hypothesized
large base-link offset. Scratch geometry checks admit all eight reference
paths. Geometric admission does not establish dynamic tracking feasibility.

A continuous turn–drive–turn reference alone stays at 2/8. The controller lags
the first turn while the fixed reference clock advances into translation.
Slowing reference progress by measured lag alone also gives 2/8, with some
paths incomplete at the unchanged deadline. Errors to an unfinished active
reference must not be described as final endpoint errors.

Explicit announced-twist feedforward gives 4/8, as does its combination with a
conservative smooth progress governor. Keeping the inner XY region active
throughout motion improves the fixed-clock result to 6/8: millimetre XY errors
no longer compete with the turn. Calibrating only base-only heading correction
to twice its previous magnitude gives 8/8. Four times also gives 8/8 but has a
higher worst measured yaw jerk (63.19 versus 44.68 rad/s^3), so the two-times
setting is frozen before the fresh audit. These are development-selected
diagnostics, not eight independent held-out successes per attempted variant.

## Implemented interface and scope

Source `0a682deb` adds `BaseGoalReference`, a reusable continuous, axle-aware,
forward/reverse turn–drive–turn compiler for Mars and Stretch. It wraps both
turns to avoid a full spin across ±pi. Each rest-to-rest segment uses an S-curve
with trapezoidal velocity under the native profile speed/acceleration and the
explicit experimental jerk limit 4 × acceleration. Optional phase bounds
reserve derivative capacity for a progress governor. The derivative certificate
is kinematic, not a force, wheel-support or dynamic-feasibility certificate.

The compiler only creates references on scratch FK state. It does not reset
physics, history, faults, grip authority or the controller. Its record has a
distinct format; display samples are not a wire-compatible linearly
interpolated `task_pose_trajectory_v1` schedule. The native runner separately
checks the measured rest boundary and geometry before installation.

`BaseNavigationFeedback(reference_tracking=True, heading_gain_scale=2)` opts into
the continuous XY region and direct physical-unit feedforward of the announced
reference twist. Learned gains still supply pose corrections. The heading
scale is an explicit development calibration, not a PPO update. No teacher
action or temporal output filter is introduced. The active-hand/partial-base
mapping stays bitwise unchanged and native actuator/stop guards remain active.

The nested contract is `signed_base_navigation_v2` inside the existing
`emet-whole-body-v3` actor bundle. Both weights and manifest pin it. An older
v3 runtime must reject the new nested contract; legacy defaults continue to
serialize with their original nested v1 contract. Targeted tests cover both
robots, exact feedforward units, saturation, scope, metadata and base path
geometry/derivative bounds. All 83 targeted tests pass.

This does **not** wire rest-to-rest paths into continuous UI dragging. Moving
retargets need a boundary-matched braking/replan reference; restarting a
zero-velocity path while moving would violate the advertised reference bounds.
No service default or actor is automatically promoted.

## Source-bound native and regression verification

The source archive contains 3,470 checksummed inputs and has SHA-256
`f9ca3fb2b69986cd055d3b618970b98fef18f20fa79a52e5a6f47f22c87b983c`.
It is unpacked and verified at `/tmp/emet-base-goal-reference-0a682deb`.
Native job `20261116_131735_782166` uses one CPU14 worker; regression job
`20261116_131731_f6ab79` uses one CPU15 worker. Numerical threads are one,
CUDA is hidden and no packages or drivers are changed.

The native run exports and publicly reloads the actual configured bundle,
checks all parameter tensors against its initializer, and repeats the existing
trajectory/general/TAMP/handoff cohorts. It scores raw static targets separately
from compiled endpoint references. The fresh compiled audit is fixed before
launch: 16 targets at radii 0.10/0.18 m over eight bearings, alternating final
headings ±0.2 rad, with seeds 35860000–35860015. No failed admission is removed
from the denominator. All stops, quietness, force, wheel unloading, measured
jerk and endpoint error remain recorded.

The completed trajectory comparisons preserve 24/24 base, 10/10 small general,
6/10 larger general and 6/6 original unloaded TAMP grasp-pose. The two larger
general tracking timeouts are measured-quiet hand misses: 4.61 mm / 0.0109 rad
on the training coupled case and 10.09 mm / 0.0283 rad on the audit coupled
case. Their base positions are already within the coarse region. The other
two larger-general failures are contact stops at 34.65 N and 31.55 N, not
wheel-lift events. These remain next-step arm/coupled tracking and contact
failures; making base-only navigation better does not solve or waive them.

The actual exported candidate completes **8/8 compiled development endpoints**
and **16/16 fresh compiled audit endpoints**, while raw static goals remain
**2/8**. All 24 compiled stops confirm with no unsafe-contact or wheel-lift flag.
Audit worst final errors are **25.01 mm / 0.05051 rad** (2.89 degrees). Audit
first-settle time is median 14.58 s, worst 20.35 s; maximum recorded hold drift
is 0.209 mm. Development worst first-settle is 21.45 s and hold drift 0.300 mm.
All acceptance bands, the deadline and measured-hold requirement are unchanged.

Smoothness remains a limitation. Audit peaks reach **10.75 m/s^3 linear jerk**
and **70.51 rad/s^3 yaw jerk**, measured by native finite differences at 50 ms.
The worst yaw event is around 6.9 s, early in the final turn after translation,
not at activation of the stationary terminal region. Its sampled yaw command
changes progressively while measured yaw velocity fluctuates. That suggests
examining actuator/servo and contact response with measured-versus-commanded
traces; it does not establish a friction, geometry or solver defect. Reference
C2 bounds do not certify measured acceleration/jerk. Do not hide this event
behind endpoint successes or treat these magnitudes as hardware damage ratings.

The native job is complete with **4/4 moving base-to-hand handoffs**, preserving
physics and history. The full readout has 86 scored cases across the separately
reported cohorts, including all raw-goal and larger-general failures.

Fresh source-bound regressions pass **1,480/1,480**, zero failures/errors/skips,
in **1,064.40 s** of JUnit runtime (1,066.13 s for the receipt-producing run).
The receipt verifies against the canonical source. Runtime-package parity
passes and the paper builds successfully (59 pages). The sequential finalizer
job is `20261116_132524_63b32b`; offline W&B evaluation run is `ybaxsdnt`.
No PPO is active and no new policy-gradient updates are claimed.

[Compact evidence and plot](../../paper/data/whole_body_control/base_goal_reference_v1/)
retain all cases, exact native/probe runners, raw/source/receipt hashes and
offline logging identity. Raw traces, bundles and source archives stay ignored;
no compressed generated evidence is committed. These endpoint results do not
qualify loaded manipulation, disturbance robustness, UI moving retargeting or
Herman hardware motion.
