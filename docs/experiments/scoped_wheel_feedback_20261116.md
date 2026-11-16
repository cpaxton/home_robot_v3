# Versioned scoped wheel feedback: matched native comparison

This follows the [15-case diagnostic](base_goal_replan_20261116.md). Its
supported implementation adds `wheel_feedback_base_only_gains`, an optional
linear/yaw proportional gain pair selected at the physics boundary when all
three base constraints are active and the hand target is inactive. Partial
base masks and active-hand commands retain `wheel_feedback_gain`. The selected
pair is `(0.0, 0.5)` and the fallback remains `0.5` for this experiment.

The setting requires Mars, current-COM feedback and physical wheel slew. It
changes no learned tensors, reward, target tolerance, native velocity/force
limit or stop behavior. Selection does not reset history or the body-velocity
integral; existing target-update invariants remain. It is opt-in; defaults and the retained selectable checkpoint remain unchanged.

An `emet-whole-body-v4` bundle binds the scope, gains, feedback measurement and
physical slew/jerk settings in both weights and manifest. The public loader
requires agreement with the validated environment configuration, rejects a
silent downgrade and preserves v2/v3 legacy behavior. Viewer selection, TAMP
attachment and spawned primitive execution propagate the setting. The primitive
benchmark retains its established reset procedure, applying execution settings
after fixture admission; it is not a new reset-dynamics validation.

## Frozen comparison

One sequential native worker compares baseline scalar 0.5 with the scoped
candidate on identical cases and seeds: **113 attempts per contract**, including
moving targets, viewer execution, base paths, both coupled distributions,
unloaded TAMP, raw/compiled endpoints, original base-to-hand handoffs, eight new
reverse audits and four new repeated mask-handoff audits. Audit seeds and all
updates are frozen before launch. No case mining, PPO or automatic promotion.
Paired repeats are not independent reliability trials.

Repeated mask handoffs preserve the same underlying admitted reference and its
velocity, changing base/hand masks at ticks 20, 40, 60 and 80. Tests assert that
physics, history and arm/grip targets survive installation. Physical reference
samples remain continuous; activating a hand intentionally enables its motion hint.
All original support, force, precision, settling and stopping failures remain
in their denominators. Post-episode stop confirmation uses unscored stepping
with a 0.75 s physical budget; no legacy 32 ms retries are reused.

Targeted implementation checks pass 179 tests; the final execution unit file
passes 15/15. Source `4b3c405960876825c57f5925300f001ee1d1ae78` is frozen with
3,504 checksummed inputs. Compressed archive SHA-256:
`786337e1d3df9bdc28c8a3f4a86e8fd4e819f065254b2d1ac0b3943fa1d3acc8`;
uncompressed tar SHA-256:
`c9e7fe6545f51518976c3c61c1737a2c203693f5d6dbdf45702549d61c82b292`.

## Completed readout

Source-bound full gate `20261116_164218_a4b9d1` passes **1,538 tests**, with zero
failures/errors/skips, in 1,209.58 s (receipt interval). Verification checks the
canonical source digest and exact test identities. This replaces, rather than
inherits, the earlier 1,520-test source receipt.

| Cohort | Scalar baseline | Scoped candidate |
| --- | --- | --- |
| Moving goals | 10/12 | **12/12** |
| Viewer | 3/3 | 3/3 |
| Base paths | 24/24 | 24/24 |
| Small coupled | 10/10 | 10/10 |
| Larger coupled | 6/10 | 6/10 |
| Unloaded TAMP grasp pose | 6/6 | 6/6 |
| Raw, uncompiled endpoints | 2/8 | 2/8 |
| Compiled endpoints (development + audit) | 24/24 | 24/24 |
| Original base-to-hand handoffs | 4/4 | 4/4 |
| New reverse audit | 8/8 | 8/8 |
| New repeated mask handoffs | 2/4 | 2/4 |

All **113/113 stops confirm per contract** using unscored physics and the
unchanged all-six-base-DOF/arm quiet thresholds. The scoped candidate resolves
the two original reverse wheel events. It adds no wheel/unsafe-contact failure
in these cases. Both contracts retain the two larger-coupled unsafe-contact
stops at 34.65 N and 31.55 N; those failures are not waived. The other two
larger-coupled misses finish at 4.61 mm/0.01089 rad and 10.09 mm/0.02833 rad.

The candidate's repeated-mask misses finish at **6.26 mm/0.01708 rad** and
**11.78 mm/0.03189 rad**, versus baseline 6.13 mm/0.01676 rad and
11.90 mm/0.03221 rad. Base errors remain within the coarse region, so these are
hand-tracking failures. All four mask cases stop cleanly without wheel or
unsafe-contact flags. Do not loosen the 3 mm/0.02 rad hand contract.

TAMP initial-state hashes, compiled robot identities, model fingerprints,
terminal poses and object motion are **identical** between contracts. These are
unloaded, open-jaw prefixes, not grasp/retention or pick/place qualification.
General/TAMP solver configurations still differ; the comparison does not replace
per-plant checkpoint admission for actual agent execution. Native general cases
use randomization scale zero; viewer cases retain scale one with paired seeds.
No new external-wrench disturbance population is claimed.

Smoothness is mixed: moving-audit peak yaw jerk falls **150.09 → 109.63 rad/s³**,
but peak linear jerk rises **20.68 → 28.10 m/s³**. Mask-audit yaw jerk falls
54.03 → 31.06 rad/s³ while linear jerk rises 9.73 → 13.73 m/s³. These are scored
motion measurements, not force/damage ratings or post-fault stop envelopes.
Retain the option for simulator experiments; do not change defaults or promote
it as a globally safer/hardware-ready controller.

Native baseline job `20261116_162848_dc5934` finishes 109 cases before its new
audit assertion aborts; continuation `20261116_164653_17158b` completes the four
mask cases and all 113 candidate cases. Producer hashes and the original
partial report remain intact. Finalizer `20261116_164739_e113e5` verifies the
fresh receipt, passes cold v4 viewer startup/public selection/resume/batched
nudges/pause/render/clean shutdown, and logs evaluation to offline W&B
**`w2ld2dmi`**. No PPO steps, upload, hardware motion or automatic promotion.

Complete cases, exact producer scripts and the comparison figure are in
[scoped_wheel_feedback_v1](../../paper/data/whole_body_control/scoped_wheel_feedback_v1/README.md).
The experimental v4 actor is `logs/eval/scoped-wheel-feedback-v2/candidate/actor`;
the retained v3 actor remains `logs/eval/base-goal-reference-native-v1/actor`.
Viewer job `20261116_170847_aa8444` exposes both roots on localhost8766 for
explicit selection, starting with the default feedback controller. Over SSH,
forward port8766 and open `http://localhost:8766`; reset a finished/faulted scene
before selecting an actor, then press `1` for base targets. No hardware.

## Next decision

Keep the v4 option experimental. Diagnose measured linear jerk with the existing
physics-rate observer before hardware or a smoothness claim. Generate analogous
training cases for hand activation, partial-base masks and return-to-hand holds,
without training on these audit seeds. Preserve regular tracking and the outer
50/50 general/TAMP physical-tick mix. Use hand error, settling, support/contact
loads and actual retention as distinct outcomes. Per-plant agent admission,
loaded grasp/lift/carry and Herman remain separate gates.

## Disk cleanup

Disposable uv package-cache contents (970.1 MiB) were removed without touching
installed environments. Four completed duplicate runtimes were removed only
after verifying every source file against its retained archive, checking local
Git references and absence of active jobs/processes. Small generated URDF/cache
extras were archived and checksummed separately. All checkpoints, experiment
traces, original source archives, models, datasets and unrelated work remain.
The cleanup receipt stays under ignored `logs/recovery/disk-cleanup-20261116/`.

The first full-gate attempt is cancelled after four report-unit failures caused
by missing Git metadata in the extracted source tree. A local Git view now
pins HEAD to the exact frozen revision using existing objects; runtime source
bytes are unchanged. All four checks pass in a focused rerun. The fresh full
run uses a new output path and will not reuse partial XML or receipts.

## New mask-audit producer correction

The first repeated-mask audit incorrectly asserted that the entire masked
nine-value motion hint must remain identical when enabling a hand target.
Previously inactive hand velocity entries correctly change from zero to the
reference velocity. The corrected audit compares the physical reference's
base/hand/gripper samples at -50 ms, -1 ms, 0, +1 ms and +50 ms, retaining the
same path, goal masks, target tolerances and event times. It records both motion
hints separately and continues to assert unchanged physics, history and grip.

The original attempt and 109 completed baseline cases remain unmodified. A
separate completion producer reruns only the four interrupted mask-audit cases,
checks the original source/producer hashes, and writes `run.completed.json`
without replacing `run.json`. Baseline mask cases pass 2/4; the two cleanly
stopped tracking misses finish at 6.13 mm/0.01676 rad and 11.90 mm/0.03221 rad,
with no wheel/unsafe-contact flag. Native continuation job
`20261116_164653_17158b` then runs the complete scoped candidate under the same
frozen protocol. No audit episode is converted into controller success by the
producer correction. The separate partial audit attempt is retained as an
infrastructure failure, outside the completed-case table.
