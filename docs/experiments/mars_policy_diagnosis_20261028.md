# Mars tracking diagnosis and failure-family curriculum

The completed v25 refinement regressed base holding and remains unqualified.
A separate 1.1x-reference-duration teacher ablation succeeds on all 19 admitted
development lifts (19/20 including the original reset rejection); the archived
17/19 baseline is unchanged. Full results appear below.

The evidence points to a learned tracking deficit, with a real servo-distribution
mismatch. It does not justify making the controller infer tasks or repair
infeasible TAMP plans. On new development seeds, the teacher completes 17/19
admitted smooth lifts while the learned policy completes 0/19; one additional
reset rejects for both. Mean terminal hand position error is 0.58 mm versus
14.64 mm. These are development probes, not qualification.

## Responsibilities

TAMP chooses grasps, task stages, release and geometrically valid routes. The
trajectory-generation/retiming layer must account for robot velocity,
acceleration/jerk limits, payload and clearance, and reject paths it cannot
support. It may accept paths without absolute timestamps and generate timing.
The controller tracks the resulting masked base/hand/gripper references,
reports tracking problems, and enforces measured stop limits. Neither a task
stage selector nor grasp/lift/place decisions are added to the actor here.

An XY base region with optional yaw is a useful planning contract. Existing
base masks represent inactive yaw, but the native fixed-base arm witness and
its full-body motion bounds cannot simply be dropped without replanning and
physical revalidation. A position-only hand command is also not implemented by
the current all-or-nothing hand mask. These interface extensions remain explicit
follow-ups, separate from improving the learned trajectory tracker.

## Matched actuator intervention

On four previously evaluated diagnostic seeds, switch final wheel-motor slew
only **after** identical physical initialization. This retains the original
hand goals, horizon, safety boundaries and object state:

| Controller | Legacy final wheel command | Physical motor slew |
| --- | --- | --- |
| New-curriculum policy | 0/4; 21.71 mm terminal hand error | 0/4; 21.19 mm |
| Old-objective policy | 0/4; 19.88 mm | 0/4; 19.68 mm |
| Teacher | 1/4; 1.41 mm | 3/4; 0.65 mm |

The actuator mismatch materially affects teacher settling in this small probe,
but fixing it does not solve the learned controller. The candidate has one
wheel-lift stop in legacy mode; in physical-slew mode it has two fixed-base
stops and two timeouts. Control has one fixed-base stop and three timeouts in
both modes. Hand-current/future error terms dominate the learned yaw head's
mean absolute preactivation contributions over direct base-error terms. This
is an audit of correlated closed-loop signals, not proof that deleting those
terms would improve physics or permit hand-only whole-body motion.

The first attempt toggled slew before reset, changing initialization, and failed
its pairing assertion. Its incomplete `v21_unmatched` reports are retained and
are not interpreted causally. `v22_matched` corrects this. All intervention
traces, action-head decomposition and source identities are in
`paper/data/whole_body_control/policy_diagnosis_20261028/`.

## Removing timing is not an endpoint step

A second probe issues the exact final hand pose immediately, with and without a
base-yaw objective. Both learned policies score 0/4 under both treatments. Even
the teacher scores 0/4: three wheel-lift stops and one fixed-base stop. For the
old-objective policy, dropping the yaw objective reduces mean terminal position
error from 22.16 to 16.48 mm, but still yields two fixed-base stops and two
timeouts. Existing full-body bounds remain active in both treatments.

This probe bypasses smooth reference generation. It establishes a boundary
requirement, not evidence that the existing public TAMP plans are intrinsically
infeasible: the public reference baseline already completes 3/3 deliveries.
The correct untimed interface is a geometric path plus constraints passed to a
smooth retimer; the controller then tracks that reference. Do not use a jump
to the final pose as the proposed planner/controller interface.

## Coverage and implemented training support

Each completed 65,536-step v20 run contains only 13 full-start supported lifts,
plus 18 mid/late/hold starts (31 supported-lift starts total). The candidate has
6,442 supported-lift steps and the control 6,269. The main 70% tracking mixture
is intentional, but the failing transition received little repeated practice.
PPO's low exploration, decayed learning rate, reference timing and absolute-
context generalization remain hypotheses; this experiment does not isolate their
individual causal effects.

The new opt-in `FailureReplay` wrapper samples failed trajectories and nearby
goals **in training only**. It preserves the declared outer task mix and replaces
a bounded fraction of carry resets. Supported lift uses the existing physical
grasp reset, with small reset-only joint-endpoint perturbations passed through
the TAMP planner/FK admission checks. The actor sees its existing masked pose
preview and measured history, with no joint witness, object ground truth or
new task-decision input. Rejected children remain failures; reset does not
keep sampling until success. All physical stop limits and grip authority remain.

A dedicated 33-million seed namespace excludes the older frozen assessments.
Twenty new development parents yield 14 train families and three validation
families with matched teacher final success; two teacher retention failures and
one reset rejection are retained outside that debugging subset. Whole parent
families, including all their nearby goals, stay on one side of the split.
Teacher feasibility of a parent does not certify its perturbed children. The
initial child audit retains teacher failures rather than relabeling them as
feasible or excluding them from its published denominator.

Training now exposes `physical_wheel_slew`, preserves the legacy default for
historical experiments, rejects incompatible exact resume, and records a change
as non-preserving dynamics during explicit weight transfer. Raw replay steps,
episode starts and parent IDs are logged. The source report, configuration,
model, corpus and source manifests bind replay data to a concrete experiment.

The next bounded PPO pilot should use the better v20 control as its initializer,
physical wheel slew, and a 70/4/20/4/2 reset mix. With half of carry resets assigned
to replay, 10% of expected resets revisit failure families; 70% remain general
tracking. Independent family children, general tracking and unchanged public
TAMP programs remain acceptance gates. No policy promotion or hardware command
is authorized by these results.


Fresh source-bound regressions pass **841 tests**, with no failures/errors/skips.
Focused checks pass 41 tests; relevant Ruff, runtime package parity and diff checks
pass. Corpus, complete development reports, 12 matched nearby goals under both
controllers, generation scripts, and verified regression XML/receipt are archived
under `paper/data/whole_body_control/failure_replay_development_v1/`. Each child
partition has teacher 5/6 and policy 0/6, with one teacher retention failure retained.
Training has not yet improved these results.


## Active bounded refinement

Source `49a11690` is committed and pushed. A 16,384-step pure-PPO pilot is now
running as `logs/manipulation/mars-failure-replay-v25/`, seed 87, on one CPU core
with CUDA off and offline scalar-only W&B. Its archived snapshot has
12,800 collected PPO steps. The actor/critic tensors and
normalization arrays match the source before updates (excluding the explicitly
changed exploration standard deviation, 0.03 to 0.05); deterministic mean actions
are identical on audited inputs. The optimizer is fresh. Physical wheel slew is
explicitly a dynamics change, not an exact continuation.

The candidate retains full precision/motion regularization, uses the declared
70/4/20/4/2 task mix with half the carry resets assigned to training families,
and keeps ordinary start/mid/late/hold samples elsewhere. The learning rate is
constant 3e-5 for this short pilot; discount is 0.9975 to give later trajectory
tracking/settling more weight than the previous 0.99. This is a bundled debugging
treatment, not independent causal evidence for each change. No DAgger, teacher
residual or runtime fallback is enabled. Validation families never enter training.

The persistent supervisor verifies the fresh regression receipt and independently
rebuilds the corpus from measured parent reports before launching. It then tests
the post-update `last` policy on the exact child-audit goals and all three public
single deliveries under legacy and jerk-limited profiles. It retains every stop
and failure; multi-transfer and hardware remain later gates. No performance gain
from this refinement is claimed yet. Launch/progress/initialization evidence is
archived in `paper/data/whole_body_control/failure_replay_launch_v25/`.

The first launcher attempt (`v24`) serialized optional/tuple arguments incorrectly
and exited during CLI parsing, with zero PPO steps. Its command/error metadata
remain preserved; `v25` fixes serialization without changing the tested production
sources or overwriting that attempt.


A measured reward audit on the ten children where the teacher finishes successfully
prefers the teacher over the starting policy in **10/10** cases. The minimum
teacher-minus-policy discounted return is 9.48 at gamma=0.99 and 27.69 at 0.9975.
This rules out the successful teacher being globally disfavored on those recorded
rollouts; it does not rule out local optimization traps or poor reward conditioning.
The learned yaw head's hand-error contributions are much larger than its direct
base-error contributions in the matched audit. Conditioning small base-recovery
signals and separating arm/base exploration merit controlled follow-up tests;
neither is silently enabled in this pilot.


The pilot has now collected all 16,384 steps and is running checkpoint validation.
Its numbered checkpoint's forced-lift check remains **0/20**, mean terminal hand
error 24.46 mm / 0.0757 rad, with no retention, wheel or unsafe-contact stops in
that check. This numbered checkpoint precedes the last optimizer update; final
`last`-bundle child/public replays are still pending and must not be inferred from
it. Actual training coverage was 14 supported-lift episode starts, including five
failure-replay starts across four distinct training parents (831 replay steps).
The short run has not exercised all 14 training families. These are coverage and
negative progress observations, not controller qualification.

## Completed refinement and matched follow-up

The v25 pilot and post-update `last`-bundle replays completed. On the six held-out
family children, final success remains **0/6**. Mean terminal hand error changes
from 13.91 mm / 0.0631 rad under the source to 23.58 mm / 0.0642 rad under the
refinement. All six refined episodes stop on the fixed-base boundary, whereas
all six source episodes run to tracking timeout. These terminal errors include
different stopping times and are not an equal-horizon arm-error comparison.
The six training-family children also remain 0/6, all fixed-base stops.

Public single deliveries remain **0/3 under each reference profile**. Every
refined program stops on the Cartesian path tracking envelope during lift, with
confirmed stop. The strict periodic broad evaluation still has wheel unloading
and retention failures, especially in disturbance/pregrasp cases. No safety,
carry, multi-transfer or hardware gate passes. The run is stopped and is not
extended unchanged.

Final scalar streams, checkpoint identity, full child traces and all public
outcomes are in `paper/data/whole_body_control/failure_replay_results_v25/`.
The 39 MB compressed public observation/physics trace remains local at
`logs/eval/failure-replay-v25-20261028/report.json.gz`; portable provenance pins
its compressed and uncompressed hashes. Earlier launch/progress snapshots remain
historical, not claims of ongoing training.

### Matched channel isolation (v26)

All interventions replay the same six validation children from identical initial
physics, with unchanged goals, references, deadline and safety limits. Hybrids
are diagnostics, not deployable policies or policy successes:

| Arm / wheel requests | Final success | Mean terminal hand position / rotation | Outcomes |
| --- | --- | --- | --- |
| Refined / refined | 0/6 | 23.58 mm / 0.0642 rad | 6 fixed-base stops |
| Refined / zero | 0/6 | 13.15 mm / 0.0332 rad | 6 timeouts |
| Refined / source | 0/6 | 23.82 mm / 0.1055 rad | 3 fixed-base stops, 3 timeouts |
| Refined / teacher | 0/6 | 13.15 mm / 0.0332 rad | 6 timeouts |
| Teacher / refined | 0/6 | 1.96 mm / 0.0021 rad | 3 fixed-base stops, 3 timeouts |

Replacing wheel requests removes hard base stops but leaves arm tracking debt.
Replacing arm requests makes the hand accurate but leaves base/settling failures.
This is evidence for separate control deficits, not a reason to move task-stage
logic into the actor. Small base-error conditioning, asymmetric arm/base
exploration and targeted physical recovery coverage remain controlled next
experiments. A larger replay budget alone is not demonstrated to solve them.

### The teacher is a comparator, not a performance ceiling (v27–v29)

The original teacher's two failures occur during lift with hand tracking around
1–2 mm; relative object rotation crosses the unchanged 0.1 rad retention limit.
Testing the permitted gripper action offsets does not fix either case: the
measured gripper actuator remains at its existing -0.08 Nm torque cap throughout.
No torque/force cap is increased. Larger 1.25–2x time dilations cannot fit the
original horizon plus settling allowance and are recorded as not executed.

A **1.1x reference duration** fits the original 12-second horizon and fixes both
retention failures. It preserves the complete geometric path, endpoint and
physical initial state. Uniform quintic time scaling lowers nominal speed,
acceleration and jerk; no controller task decision is introduced.

A subsequent uniform 1.1x-duration replay across **all 19 originally admitted
parents succeeds 19/19**, with zero retention loss, wheel unloading or unsafe
contact. Mean terminal hand position error is 0.47 mm. The original support-margin
reset rejection remains unchanged, so this is **19/20 end to end** over the original
20 candidates. This is a retiming ablation on a repeatedly used development
population, not a new held-out qualification or replacement of the archived
17/19 baseline. Reference timing changed; the goals, 12-second horizon, geometry
and acceptance thresholds did not. It does not establish public multi-stage
pick/place or broad-mass performance.

For the two repaired cases, peak monitored force falls from 5.35/5.28 N to
4.94/4.76 N and maximum relative rotation stays below 0.075/0.060 rad. The probe
supports fixing reference dynamics in the trajectory-generation/retiming layer.
It does not demonstrate that the learned controller has improved or that its
final performance should be bounded by the old teacher. The policy emits full
velocities without a teacher-action residual; teacher-success filtering was only
a temporary debugging subset, not a permanent training ceiling. All teacher
failures remain in the full benchmark and archives.

Complete channel/retiming/grip traces and drivers are archived in
`paper/data/whole_body_control/tracking_and_teacher_diagnosis_v26_v29/`. Production
sources and thresholds are unchanged by these diagnostics; the existing 841-test
receipt still verifies. No additional PPO or hardware motion occurs after v25.
