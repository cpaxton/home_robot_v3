# Current-COM feedback through reset: strong general tracking, open TAMP gap

The corrected inner loop passes the actual-reset, reverse/yaw, finite-push and
measured-stop checks. The retained legacy actor now passes **12/12** on the
common small benchmark and **151/164** general cases. It still passes **0/6
actual TAMP pregrasps**, versus teacher **5/6**. These results support the loop
correction and a stronger general baseline, not a qualified manipulation policy.
No weights were changed, no PPO ran, and no hardware moved. Defaults remain off.

## Actual reset, motion and stopping

The [predeclared protocol](../plans/mars_current_feedback_validation.md) crosses
cached/current feedback enabled before reset with twelve existing conditions:
five nominal base primitives, those five with a finite +X end-effector push,
and two mid-motion cancellations. The push is 0.5 N over six control ticks,
using the existing rise/fall envelope. There are no new seeds, friction changes,
geometry edits or reference/acceptance relaxations.

Both variants admit and complete all ten non-cancellation conditions and
confirm all twelve measured stops. Maximum stop time is **0.15 simulated
seconds**, below the unchanged 0.75-s bound. There are no wheel-lift,
unsafe-contact or invalid-physics results in these twenty-four episodes.
This exercises the analytic controller and low-level loop; it is not a learned
policy disturbance result. The finite external wrench is removed before stop
verification, so it is not a sustained-force stop test or hardware latency claim.

| Native measurement | Cached COM | Current qvel, same COM |
| --- | ---: | ---: |
| Nominal hold linear jerk, m/s³ | 0.09594 | 0.000353 |
| Nominal hold yaw jerk, rad/s³ | 0.08065 | 0.001547 |
| Nominal left-yaw linear jerk, m/s³ | 0.45105 | 0.53302 |
| Nominal left-yaw angular jerk, rad/s³ | 5.62030 | 1.95698 |
| Nominal right-yaw angular jerk, rad/s³ | 5.69942 | 2.35941 |
| Confirmed stops | 12/12 | 12/12 |
| Maximum measured stop, simulated seconds | 0.15 | 0.15 |

Improvement is not uniform across every metric: left-yaw linear jerk increases.
Original metrics are preserved. Stop checks use measured arm velocity and all
six floating-base velocity components, with no reset, rearming or grip release.

## Frozen controllers on the corrected plant

Current feedback is enabled during initialization, not switched on afterward.
The analytic teacher completes **164/164 general cases**. Each frozen controller
uses those fresh teacher budgets and matching current-source reset/reference
hashes. The original `3ac722e1` budget matrix is retained as a separate comparison;
it is never relabeled as current-source admission.

| Controller / population | Fresh-budget full | Original-budget comparison |
| --- | ---: | ---: |
| Retained legacy, general | **151/164** | 149/164 |
| GRU control, common C1 cases | 2/12 | 2/12 |
| GRU candidate, common C1 cases | 8/12 | 8/12 |

The fair shared twelve-case comparison is teacher **12/12**, retained legacy
**12/12**, GRU control **2/12**, GRU candidate **8/12**. Retained legacy passes
all ten arm cases in this subset. Across the complete general population its
stage results are **49/50 base, 100/106 arm and 2/8 coupled**. The thirteen
remaining failures include seven tracking timeouts and six tracking-envelope
exits, with overlapping terminal-hold/smoothness reasons retained per case.
This is an existing development population, not an untouched statistical audit.

The GRU candidate retains seven full arm passes, but only **9/10 arm smoothness**
under actual reset, versus the prior matched-reset intervention's 10/10. Full
supported-up still fails tracking and now also linear jerk. The previous
[8/12 matched-reset result](mars_aligned_jerk_measurements_20261115.md) remains
valid for its original producer and initialization; it is not substituted for
these actual-reset measurements.

## Actual public-tool trajectories

The primitive environment executes real oracle TAMP discovery/planning and
approach/pregrasp schedules on two distinct supports. It stops before task-owned
grasp closure; none of these results claims pick/place, loaded carry or retention.

The teacher succeeds **5/6** and retained legacy **0/6**. All twelve baseline
execution stop receipts confirm. Legacy failures are one transport timeout,
three Cartesian tracking timeouts, one Cartesian tracking-envelope exit, and
the same case-ID failure also seen by the teacher. General pose success does
not establish task-trajectory tracking in these fixtures.

The teacher's `development:left:50g:2` now reports `operation_failed` during the
approach's lift stage, with a confirmed stop and no wheel/force/physics/retention
safety flag. Its object moves about 0.589 mm and rotates 0.01799 rad. Only the
initial clearance command is recorded; the tool reports eight execution IK
calls and no completed task operations. This is a changed measured outcome,
not the earlier `object_changed_replan` code. The existing experimental
admission gate correctly rejects it. No exception list or threshold was weakened.

A second predeclared **24-episode TAMP-only study** tests physical settling and
other frozen trajectory controllers. The zero-settle teacher/legacy baseline
above supplies the anchors; no adaptive checkpoints or extra seeds are selected.

| Follow-up | Extra pre-planning settling | Success | Recorded confirmed stops |
| --- | ---: | ---: | ---: |
| Teacher | 2 s | 5/6 | 5/6 |
| Retained legacy | 2 s | 0/6 | 5/6 |
| Frozen velocity-feature legacy final | 0 s | 0/6 | 6/6 |
| Frozen GRU candidate | 0 s | 0/6 | 6/6 |

After settling, the difficult fixture is rejected during planning with
`no_plan_within_budget`, before actor-controlled execution. Those two early
planning errors do not return a stop receipt, so their stop status is **unknown**,
not silently counted as confirmed or diagnosed as a physical stop failure.
Other learned failures remain tracking-related. Extra waiting and the existing
velocity-feature/frozen-GRU alternatives do not close the gap. The worker's
initial-state hash precedes optional settling; it is not a post-settling identity.

The task-level failure is initialization-sensitive, and learned execution of
the five demonstrated trajectories also needs attention. These observations
do not establish a single root cause or justify an automatic PPO extension.

## Compatibility and reproducibility

TAMP adds a fixed support to the training scene. The adapter reconstructs and
verifies each bundle's original full-scene fingerprint, then compares the
existing conservative compiled-robot identity before inference. Robot geometry,
dynamics, indices, actuators, meshes, couplings and solver settings must match;
the additional support is permitted. Tests cover scene transfer and rejection
of a changed robot before bundle inference. Actor inputs contain only the
established sensor/target fields; privileged critic fields are filtered out.
Weight and environment-option provenance remain explicit; bundles are not retagged.

Immutable runtime source **`35bb4fa6`** passes **1,299/1,299** fresh full
regressions in 1,165.6 s; its receipt binds that baseline's runtime/test
sources. Later optional tracing changes in `a8c9e9d7` require their own full
receipt; see the [four-episode input diagnostic](mars_tamp_input_diagnostic_20261115.md).
The focused compatibility/reset tests pass, and Ruff passes. General
case progress is saved atomically every ten completed episodes. All native
jobs use one numerical thread on CPU14; full checks use CPU15, CUDA hidden.

| Evidence | Producer | Managed job |
| --- | --- | --- |
| Actual reset/motion/stop, 24 episodes | `35bb4fa6` | `20261115_130428_c58ed8` |
| Full regressions | `35bb4fa6` | `20261115_130431_da6cb7` |
| General/TAMP frozen baselines, 364 episodes | `35bb4fa6` | `20261115_130647_75d834` |
| Additional TAMP comparison, 24 episodes | `71c9e0f4` | `20261115_134425_601332` |

The follow-up retains the same verified runtime/test sources; only diagnostic
script/documentation files were added. Source hashes, actor hashes, per-case
tables, both budget matrices and raw-archive digests are in the
[compact data](../../paper/data/whole_body_control/mars_current_feedback_validation_v2/README.md).
Raw commands, gzip traces and weights remain in ignored artifact storage.

The initial `24c14b27` attempt is retained separately: its twenty-four loop
cases pass, but broad evaluation and full checks were interrupted to replace
an overly strict whole-scene adapter check. Its last saved teacher prefix is
120/120, not a complete admission result. The corrected loop archives reproduce
both original gzip files byte-for-byte. No partial receipt is reused.

## Next decision

Keep retained legacy as the stronger general-control baseline; these results
do not justify preferring the GRU architecture or spending more time on the
old three-case C1 objective. Keep the direction/audit manifests from PR #240
as coverage tools, not admitted or trained results.

Next instrument the failed TAMP stages to compare announced targets/velocities,
actor inputs, requested motion and measured motion against the successful
teacher traces. Check pose/posture/phase coverage and executor timing before
selecting a training change. Separately diagnose the initialization-sensitive
planning case while keeping its goals and thresholds unchanged. Then admit
the resulting trajectory distribution and a fresh audit before optimization.
Actual grasp/lift/retention and hardware testing remain subsequent stages.
