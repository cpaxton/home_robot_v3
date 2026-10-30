# Jerk-limited TAMP references and reward-curriculum retraining

The new simulator reference path preserves the TAMP planner's geometric route,
removes redundant collinear vertex stops, and supplies continuous FK poses with
linear/angular velocity and acceleration. Native reward coupling and a later
smoothness curriculum are implemented. Both bounded retraining runs and final matched replays are complete.
The learned controller remains **unqualified**; see the completed results below.
Earlier launch/progress snapshots are preserved as historical evidence.

## Reference and physical preflight

Eight paths generated through `PhysicalPickPlaceExecutor.plan_joint_target`
cover Mars and Stretch. All eight plans admit. New nominal durations are:

| Robot | Legacy duration range | Continuous FK duration range |
| --- | --- | --- |
| Mars | 2.72–2.87 s | 1.08–1.14 s |
| Stretch | 9.36–12.27 s | 1.54–2.94 s |

This is a component reference comparison, not measured execution speed or
full pick/place qualification. The older conversion stopped at each densely
sampled planner vertex. Exact collinear compression preserves the route; real
corners still stop. Native joint velocity/acceleration bounds and an explicit
experimental jerk ceiling are checked at polynomial extrema. This is not a
manufacturer jerk rating. A bounded smooth reference-clock governor accounts for
retiming derivatives, and FK Jacobian/Jacobian-derivative checks agree with finite
differences on both robots without modifying live physics.

![Mars nominal reference](../../paper/data/whole_body_control/jerk_tracking_preflight_v1/innate_mars-reference.png)

![Stretch nominal reference](../../paper/data/whole_body_control/jerk_tracking_preflight_v1/stretch-reference.png)

Legacy Cartesian acceleration is a finite difference on a roughly 2 ms grid;
its interpolation discontinuities are not presented as measured robot loads.
The new plotted Cartesian motion uses analytic FK twist/acceleration. The
internal continuous record retains its polynomial source rather than pretending
its exported pose samples define execution. Existing v1 wire commands reject
this internal v2 reference; simulator use is explicitly selected.

Fresh physical teacher probes admit **20/20** box resets: eight full lift starts
and twelve start/mid/late/hold phase-mixture cases. The teacher settles at least
once on **20/20**, but holds final success on only **10/20** (4/8 full-start,
6/12 phase-mixture). All other outcomes are settling timeouts, with no force,
wheel, retention or invalid-physics stops. The baseline remains imperfect.
Cylinders and canonical carry qualification remain unresolved and unqualified.

Complete schedules, witnesses, admitted/rejected states, measured teacher traces,
all timeouts, source/model identities and derivative traces are archived at
`paper/data/whole_body_control/jerk_tracking_preflight_v1/`. Prefixes physically
execute the teacher during reset only and reject any unsafe contact, wheel,
retention, collision or base-boundary failure before actor handoff. No object
teleport/attachment or retention-reference rebasing is used.

## Native training changes

The native tracking reward now offers same-tick hand-conditioned base credit,
with tests preserving base-only reward and exact physical replay across the
reward change. The previous external wrapper's goal-timing mismatch is removed
from this implementation. The candidate starts at precision level 2 (25 mm /
0.15 rad reward scales); acceptance remains 3 mm / 0.02 rad throughout.

Motion/action/jerk regularization starts at 0.05 of its configured weight.
Curriculum promotion requires at least 20 evaluation episodes per scenario plus
the forced full-start loaded comparison, measured accuracy, zero safety/admission
failures and active base-heading gates. Precision and regularization then tighten;
phase-specific counts are logged but are not individually qualified by this gate.
Force and retention costs and physical stop limits retain full strength. Strict
full-distribution checkpoint validation uses final reward precision and full
motion regularization. No elapsed-time-only curriculum promotion occurs.

The 70% general / 30% manipulation task mix is retained. Supported lift remains
9% of expected resets, now divided into 50% starts, 25% mid-lift, 15% late-lift,
and 10% loaded holds. Scored time begins at handoff while explicit reference
phase offset preserves the remaining trajectory. Raw reset/step counts expose
actual coverage. Retargets, disturbances, other carry, pregrasp and placement
remain in the distribution. Low randomization still concentrates mass near
60 g; this does not qualify the entire configured mass range or hardware.

Velocity and acceleration matching rewards are **disabled in this first matched
comparison**. The reference derivatives are available for audit; training first
isolates the corrected pose/curriculum bundle. The original masked pose API,
seven-sample preview and actor layout remain compatible. There is no online IK
velocity controller, DAgger, teacher residual or runtime fallback.

## Matched run and remaining gates

`configs/benchmarks/mars_jerk_tracking_curriculum_v1.json` fixes two sequential
65,536-step pilots with seed 86: old objective/final precision/full motion cost
versus the new native reward/curriculum bundle. They start from identical actor
and critic tensors, frozen normalization, fresh optimizers and a recorded
zero-PPO current-source transfer of the earlier learned diagnostic. Changes are
a bundle comparison, not independent causal ablations of every reward term.

Before launch, fresh source-bound full regressions and current-source preflight
must verify. Training runs one CPU environment on CPU 14 with CUDA disabled and
one numerical thread. W&B is offline scalar-only; weights remain local. The
persistent supervisor records exact commands, hashes, initialization equality,
recovery attempts and every checkpoint. Resume restores optimizer and learned
exploration; environment/RNG resets are not bitwise continuation. No checkpoint
promotes itself.

After both runs, the same supervisor replays post-update `last` bundles against
the source on new full-start and phase-mixture seeds, general base/hand/combined,
goal-change and disturbance cases. All three public deliveries and the original
multi-transfer programs run under both legacy and new arm reference profiles;
the reference baseline is repeated separately. Goals, thresholds and failure
cases are unchanged. Numbered pre-final-update checkpoints are retained and are
not silently substituted for the final updated policy.

Full-base/retention/force/wheel checks and measured stop confirmation remain
independent of reference smoothness. The small terminal exact-goal SE(3)
correction has explicit task-space derivative bounds but no joint witness or
dynamic certificate. Continuous reference math alone cannot qualify its execution.
Transport/base reference improvements, explicit derivative actor features,
velocity/acceleration reward ablations, wider mass coverage, cylinders and Herman
remain separate later gates. No hardware commands were sent.


Fresh source-bound regressions pass **830 tests** with no failures/errors/skips;
verified XML/receipt are `jerk-tracking-regressions.*` under
`paper/data/whole_body_control/`. The focused preview/runtime/reference/curriculum
checks pass 128 tests. Core runtime parity, relevant Ruff and diff whitespace
checks pass. These tests validate implementation contracts; the 20-case physical
preflight remains 10/20 final settling and is not upgraded to carry qualification.


A final trace-lifecycle check found and fixed a stale reference-governor binding
when a jerk-limited arm stage hands off to a plain transport reference. A new
physical regression exercises that handoff with observation recording enabled.
Fresh source-bound validation now passes **831 tests**; current XML/receipt are
`jerk-tracking-v2-regressions.*`. The repeated current-source preflight is archived
under `jerk_tracking_preflight_v2/` and has the same 20/20 admission and ever-
settling, 10/20 final success. Earlier evidence remains historical and unchanged.

The first supervisor attempt (`v19`) was interrupted during startup validation,
with **zero additional PPO steps**. Its initial reports and launch are retained.
The replacement is `logs/manipulation/mars-jerk-tracking-v20/`, with the curriculum
candidate first and control second. A two-case-per-scenario startup sanity probe
cannot advance precision/regularization; all periodic validation retains 20 cases
per scenario and the forced full-start loaded gate. This avoids spending a full
qualification-sized evaluation before the first update and does not weaken the
training curriculum or final replay gates.


The replacement supervisor launched from committed source `9e183ade`.
The curriculum variant has passed startup validation and is performing PPO
updates; its archived snapshot records 16,384
collected steps and 307 optimizer epochs/updates
reported by SB3. This is progress evidence, not new held-out success. The control
variant follows sequentially. Exact launch, initial actor/value/normalization
identity, zero-step transfer evidence, frozen specification and persistent
recovery/evaluation drivers are archived under `jerk_tracking_launch_v20/`.
Current local progress is `logs/manipulation/mars-jerk-tracking-v20/status.json`;
the scalar stream is `curriculum/seed-86/metrics.jsonl` within that run. Sources
remain frozen during training; all gates must still pass before any promotion.


The first 16,384-step forced full-start lift check is **0/20 success**, with
45% retention failures and no wheel unloading; teacher execution on the same
20 cases is 14/20 final success and 20/20 ever settled. The candidate mean final
hand error is 48.4 mm / 0.169 rad. This remains an early negative result, not a
claim that the reward changes solved manipulation. Training and the full paired
comparison continue under the same frozen configuration and safety gates. The
raw first-check report and scalar stream snapshot are retained in the launch
archive; subsequent checkpoints and final replay determine the outcome.


## Completed matched results (v20)

Both variants completed 65,536 additional PPO steps without recovery attempts.
The final evaluation exited successfully at 16:50 UTC on 2026-10-27, using the
post-update `last` policies. Frozen source remains `9e183ade`; its source-bound
831-test receipt was checked before replay. No policy was promoted.

The new independent full-start lift seeds show lower terminal hand position
error, but **no successful learned full-start lifts**:

| Policy | Final success | Mean terminal position error | Mean terminal orientation error |
| --- | --- | --- | --- |
| Starting learned checkpoint | 0/8 | 39.23 mm | 0.134 rad |
| New reward/curriculum | 0/8 | 21.54 mm | 0.148 rad |
| Old-objective control | 0/8 | 16.56 mm | 0.074 rad |
| Teacher, identical initial physics | 2/8 | — | — |

These are errors at each episode's termination, including early safety stops;
they are not equal-horizon trajectory errors. The teacher settled at least once
in 7/8, then frequently lost final settling. It also had one retention failure.
This eight-case comparison is diagnostic, not a population success estimate.

The curriculum candidate stopped for five fixed-base violations, one wheel-lift
and two tracking timeouts; the control had one fixed-base violation and seven
timeouts. Neither learned variant lost retention in these eight full-start
cases. For the eight phase-mixture resets, success was 1/8 for the curriculum,
0/8 for the control and source, and 3/8 for the teacher. One reset was rejected
identically for all four controllers; it remains in the end-to-end denominator.

Periodic validation also remained 0/20 at every 16,384-step checkpoint for both
variants. Candidate mean position error improved to 14.8 mm at 49,152 steps but
was 19.3 mm at 65,536; orientation worsened from 0.053 to 0.147 rad. This was not
monotonic convergence. The validation gate correctly kept curriculum precision
at level 2 and motion regularization at 0.05 throughout; randomization stayed
0.05. Lower training cost did not establish strict endpoint tracking or stability.

All three public single-delivery cases failed for both learned variants under
both reference profiles. Legacy execution stopped for base drift during the
arm stage in all six retrained deliveries. With the jerk-limited profile, all
three curriculum deliveries timed out; the control stopped on two arm-configuration
collision checks and one Cartesian path tracking envelope. Original multi-transfer
programs failed at their first delivery with the same respective reasons. Every
learned program replay recorded confirmed stop. The public reference baseline
completed 3/3 single deliveries, but 0/3 original multi-transfer programs: those
completed the first transfer and failed subsequent planning with
`no_plan_within_budget`. That planning failure is separate from learned execution.

Generalization probes (two cases per mode) also failed acceptance: each retrained
policy scored 0/10 across base, hand, combined, retarget and push. Hand-only
terminal position errors improved, but combined/retarget/push still triggered
unsafe-contact stops. The small sample cannot qualify safety, and reference
smoothness does not imply smooth or safe executed motion.

The new curriculum bundle did not outperform the old-objective control on these
loaded errors. The next experiment should isolate learned base-hold behavior and
arm tracking/settling, audit training/runtime observation and actuator semantics,
and test relative-frame inputs and small heading-recovery resets before a larger
training budget. Velocity/acceleration rewards remain disabled in this comparison;
their effect has not been tested. Original task thresholds stay unchanged.

Portable complete summary, raw scalar streams, checkpoint identities and all
periodic loaded validations are in
`paper/data/whole_body_control/jerk_tracking_results_v20/`. The 204 MB compressed
full local trace bundle remains at
`logs/eval/jerk-tracking-v20-20261027/report.json.gz`; portable provenance records
both its compressed and uncompressed SHA-256. It is deliberately not duplicated
in Git. Launch evidence and all earlier failures remain preserved. No hardware
commands were sent.
