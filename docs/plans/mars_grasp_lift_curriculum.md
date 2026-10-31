# Supported grasp-to-lift curriculum

Ordinary carry resets currently perform a teacher lift before actor handoff.
Pregrasp episodes include lift only after a successful learned approach/closure,
which can starve the loaded transition of data. Add an opt-in carry subtype that
starts at a physically closed, table-supported grasp, before lift-off.

The matched experiment retains the existing 70% tracking / 30% manipulation mix:

| Reset category | Expected share |
| --- | ---: |
| Ordinary base/hand/combined tracking | 35% |
| Nearby orientation/combined reaching | 21% |
| Stationary reach holds | 14% |
| Existing pregrasp | 10% |
| Supported box grasp-to-lift | 9% |
| Remaining ordinary/hold carry | 3% |
| Placement | 5% |
| Disturbance | 3% |

The new subtype occupies 75% of the existing 12% carry share. Actual episode and
step shares are logged separately. Both arms retain mid-motion goal changes and
bounded disturbances. History/preview, architecture and pose/retention rewards
are unchanged between arms.

Reset requires one second of bilateral contact and a quiet half-second grasp
window: less than 1 mm / 0.005 rad relative drift and 0.01 rad/s gripper speed.
Rejected attempts remain in evidence. Continuous nearby elbow configurations
include straighter public-execution postures. Private FK/IK and the existing
TAMP planner admit a 4.8–6 cm lift with fixed hand attitude, native derivative
bounds, collision checks and at least 5 mm sampled static support margin. A
separately checked direct shortcut avoids braking at every planner sample.
Neither generation nor admission moves live physics.

The actor receives only measured history and announced SE(2)/SE(3) preview.
During rollout, a fresh measured-state collision guard retains frozen shallow
support-departure permissions, clearing them on failure/reset/close. Preserve
force, retention, wheel, release and full body drift guards, the 3 mm / 0.02 rad
hand requirement, and existing jerk/energy terms. The teacher's heading deadband
is below this subtype's 0.01 rad success condition; other tasks retain the old
default. Sampling is not continuous dynamic certification.

Intermediate gates are focused reset/derivative/stop tests, a teacher gallery,
a small PPO smoke test, and fresh source-bound regressions before the matched
pilots. Mixed-shape probes expose cylindrical payloads spinning/sliding out;
quiet closure and grasp-depth sweeps do not resolve it. The first lift subtype
therefore selects boxes only; other tasks retain their original object mix.
All cylinder failures stay in the reports and retain the strict limits.

The box teacher preflight admits 8/8, settles at least once on 7/8, and finishes
successfully on 5/8. One loses retention and two lose terminal settling; there
are no force or wheel failures. This is research preflight, **not canonical carry
qualification**. The canonical supervisor and promotion gates remain unchanged.

`configs/benchmarks/mars_grasp_lift_curriculum_v1.json` specifies two matched
16,384-step PPO pilots from the same explicit 3x learned hand-gain diagnostic
bundle. That initialization adds no PPO steps; the pilots use no DAgger or
teacher residual. Freeze copied normalization, decay the small LR, use one CPU
environment and offline scalar W&B. Pin source/config/weights, preflight,
regression receipt and persistent recovery/evaluation drivers at launch.
Preserve completed variants across crashes; resumed environment/RNG state is
not claimed bitwise identical. No candidate promotes itself.

Compare source/control/treatment on identical independent loaded cases, ordinary
base/hand/combined goals, retargets, disturbances, all three public deliveries
and the unchanged original multi-transfer programs. Keep all failures and report
first failing phase, final/ever settling, drift, jerk, force, retention and wheel
unloading. Teacher failure is not an infeasibility certificate. Herman remains
a separate later gate.

Generate full targets and measured teacher traces from the persistent checkout:

```bash
taskset -c 14 env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m emet.rl.manipulation.grasp_lift \
  --teacher --samples 8 --seed 3101000 --output FRESH_DIRECTORY
```

Omit `--teacher` for geometric export or supply `--policy CHECKPOINT` for measured
policy rollouts. Reports include rejected cases, initial physics, reset attempts,
joint witnesses, complete task-pose schedules, tracking/load traces and an
equal-scale target plot. Keep CPU 14 for one train/evaluation worker and CPU 15
for one test worker; disable CUDA and numerical multithreading.

The implementation passes **801 fresh source-bound tests**, runtime parity,
relevant lint and the paper build. The 512-step forced-lift PPO smoke completes.
All preflight outcomes, including cylinder/depth failures, are archived in
`paper/data/whole_body_control/grasp_lift_preflight_v1.*`; the target gallery is
`grasp_lift_targets_v1.png`.

The matched pilots launched from committed source `4f8456e8` on 2026-10-27.
The old gain bundle's runtime source check correctly rejected the changed
controller source. A named, zero-PPO training transfer creates a separate current
bundle without changing the original; source provenance, normalization and
recorded action equality are retained. The frozen launch records that transfer,
the full regression receipt, teacher preflight and both persistent drivers.
Both matched pilots and paired replay have completed. The identical-physics
teacher supplement finishes 7/8 independent lifts; source/control/treatment each
finish 0/8 and 0/3 public deliveries. Isolated coupled-reward diagnostics improve
independent mean tracking error but still finish 0/8. No candidate is promoted.
The [complete experiment report](../experiments/mars_grasp_lift_20261027.md)
records all training, objective audits, failed integrated cases and next gates.
Persistent progress is `logs/manipulation/mars-grasp-lift-v15/status.json`; portable
paired replay and all failures are archived under
`paper/data/whole_body_control/grasp_lift_v15/`.


## Reward and timed-preview follow-up

The actor already receives seven SE(2)/SE(3) reference samples from -0.05 to
+1.0 seconds. The MLP context reads the full snippet; the structured feedback
tail explicitly uses the current and +0.2 second pose errors. Future motion is
available, but reference twist, remaining duration and a terminal-hold flag are
not explicit inputs. Derive those quantities from the announced task-pose
schedule, with coherent timestamps and the executor's declared progress rate;
no robot-joint IK witness is needed for these features. Keep world-frame API
commands and move policy context to an appropriate relative representation.
A transformer is an architecture ablation, not a prerequisite for this interface.

[UMI-on-Legs](https://umi-on-legs.github.io/static/umi-on-legs.pdf), section 3.2,
uses future task-space pose samples and a coupled position/orientation reward
with a precision-scale curriculum. It reports that tightening the scale over
training is important. Our new loaded pilots started at the finest level
(5 mm / 0.05 rad reward scales), so they received no coarse-to-fine curriculum
for the newly introduced task. Keep the 3 mm / 0.02 rad acceptance gate fixed
while broadening training reward support and advancing precision only after
stratified general and loaded validation. An alternative ablation retains a
small fine-scale reward alongside a broad tracking term.

There is already dense pose-progress shaping in `env.py`; do not add a duplicate
forward-motion bonus. Native hand-conditioned base credit must use the goal
actually scored at the same tick, preserving base-only/masked task semantics.
Keep motion/action/jerk regularization subordinate to accurate execution, and
reward sustained quiet after the active trajectory really finishes. Force,
retention and wheel-stop guards remain independent of reward tuning.

[DeepMimic](https://arxiv.org/abs/1804.02717), section 5.1, includes reference
velocity matching in motion imitation. For this controller, a small Cartesian
linear/angular velocity tracking term is an explicit optional ablation, not a
claim that UMI uses one or that it is required. Pure speed toward a target can
reward overshoot. Matching only reference velocity also gives a stationary
reference zero speed even when position error remains. Compare a bounded
error-corrected desired twist (reference twist plus pose-error feedback) against
the simpler corrected pose/curriculum baseline, and retain smoothness, contact
and retention outcomes. At the reached final pose its desired twist must vanish.

A confirmed clock inconsistency has been repaired separately: retimed preview
and task goals followed announced progress, while environment completion used
elapsed physics time. Retimed completion now waits for the actual endpoint;
a separate timestamp makes the terminal hold ramp grow after progress is capped
at the end. Repeated endpoint packets preserve that hold age, new paths reset it,
and preview reconfiguration preserves the same progress/rate. Episode horizons
and executor deadlines are unchanged. This does not explain the earlier fixed-
clock lift 0/8 and does not requalify any old checkpoint. Fresh source-bound
regressions and matched learning tests are required before another training run.

## Jerk-limited references and the next matched run

`mars_jerk_tracking_curriculum_v1.json` specifies two sequential 65,536-step
research pilots from identical transferred actor/value weights and frozen
normalization. Both retain 70% general tracking and the same 9% expected
supported-lift reset share. Within that subtype, physically verified reset
prefixes sample 50% lift start, 25% mid, 15% late and 10% loaded hold. Prefixes
retain the original grasp reference, stop/reject on force, retention, wheel,
collision or full-base failures, warm measured history, and finish assistance
before scored motion. Reference phase offset is explicit; scored time starts at
handoff and no object is teleported or attached. Raw reset/step counts by phase
are logged so cumulative shares need not be inferred from averaged fractions.

The control uses the previous reward precision and full motion penalties. The
candidate uses same-tick hand-conditioned base credit, precision level 2 and
motion-penalty scale 0.05. Validation can advance precision and motion penalties
only after at least 20 cases per scenario, plus the forced full-start
loaded comparison, pass measured accuracy/safety and base-heading gates. Force
and retention costs and all physical stop limits stay active at every stage.
Strict acceptance remains 3 mm / 0.02 rad; full-distribution validation uses the
final precision and full motion-penalty scale. Velocity/acceleration rewards are
**disabled in this first comparison**, pending complete reference/rollout
validation; the nominal reference derivatives are now available to audit.

The new path compiler removes only duplicate and redundant collinear vertices;
real corners retain rest-to-rest stops, preserving the checked geometric route.
It uses C2 quintics and polynomial extrema to bound velocity, acceleration and
jerk. A two-pole reference-rate governor makes lag-retiming acceleration
continuous; conservative chain-rule bounds account for its acceleration/jerk.
Runtime pose sampling uses private continuous FK, rather than linear interpolation
between exported pose samples. Cartesian twists/accelerations use FK Jacobians
and their time derivatives. No online IK action loop or teacher fallback is used.

The jerk ceiling is an explicit experimental reference limit (4× native joint
acceleration in the TAMP adapter), **not a manufacturer jerk rating**. Limits
apply to the nominal/governed kinematic reference; measured controller jerk,
loads, retention and unloading remain separate rollout gates. A short exact-goal
SE(3) correction has explicit task-space derivative bounds, but does not claim a
joint witness or a dynamic certificate. Frozen TAMP goals and thresholds remain
unchanged. The internal continuous reference records as `task_pose_fk_reference_v2`
and is rejected by the existing v1 wire sender/decoder; this adds no hardware
route. Simulator use is explicit via `arm_reference="jerk_limited"` or the public
sequence runner's `--arm-reference jerk_limited` with Cartesian checkpoint mode.
Legacy references remain available for paired comparisons.

Reference preflight exports full schedules and derivatives from eight real
TAMP-planner component paths (Mars and Stretch). Physical loaded teacher probes
cover eight full-start and twelve phase-mixed resets. Admission/ever-settling and
terminal settling are reported separately; neither component preflight nor a
training launch is canonical carry qualification. Public single deliveries and
original multi-transfer programs run after training under both legacy and new
reference profiles, alongside paired new-seed loaded/general/retarget/disturbance
checks. Final replay uses the post-optimizer-update `last` bundle, preserving all
numbered checkpoints and failures. Policy selection/admission is independent.

One CPU environment runs on CPU 14, CUDA is disabled, numerical threads are
limited to one, and W&B stays offline scalar-only. The source-bound full test
receipt, preflight/model/source/config/weight identities and persistent recovery
and evaluation drivers must verify before training. Completed runs survive
restarts; optimizer and learned exploration are restored, while environment/RNG
resets are explicitly not bitwise continuation. No automatic promotion or Herman
motion is enabled. See Modern Robotics on
[trapezoidal versus S-curve/polynomial time scaling](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-2-of-2/)
and [Ruckig](https://docs.ruckig.com/) for jerk-constrained trajectory-generation
principles; no new dependency was installed here.


The implementation and follow-up handoff regression pass 831 fresh source-bound
tests. Current launch/preflight details and the preserved zero-PPO interrupted
startup are in the [experiment report](../experiments/mars_jerk_tracking_20261027.md).
The curriculum pilot runs first; its startup probe uses two cases per scenario
and cannot advance the curriculum. Later validation uses 20 per scenario.

## Responsibility boundary and failure neighborhoods (2026-10-28)

TAMP owns task decisions: grasp selection, stage ordering, release, clearance,
and load-aware geometric planning. Its trajectory-generation/retiming layer must
turn a geometric path into a smooth reference obeying robot limits, or reject it.
A path may omit absolute timing, but this does not authorize an abrupt endpoint
jump. The controller tracks the resulting masked poses, exposes tracking/stop
status, and enforces measured limits. It must not infer the next task stage or
repair an infeasible task plan. Training task labels are sampling/scoring metadata;
this work adds no new task-decision input to the actor.

The desired planning interface supports a base XY region with optional yaw and
ordered task-space paths without mandatory timestamps. Hand orientation remains
constrained where the grasp/placement needs it. Existing generic base masks can
represent inactive yaw. The current native Mars arm planner/executor still uses a
fixed-base witness and independent full-body motion bounds; dropping its yaw
objective is not permission to silently drop those bounds or reuse a witness
outside its valid base region. Extending that TAMP planning/execution contract
needs new geometric and physical revalidation. No new wire format or fully
integrated untimed-stage execution is claimed by this training change.

Use separate development failure families to improve tracking coverage. The new
`failure_replay` module mines complete supported-lift policy/teacher reports in a
dedicated development seed namespace. Initial states, planned trajectories,
model fingerprints, source manifests and configuration must match. For this
initial debugging subset, retain only failed policy parents with teacher final
success. Preserve rejected parents, teacher failures and successes separately;
the teacher is not a universal feasibility bound or the eventual policy ceiling.

Draw bounded joint-endpoint perturbations (up to 0.03 rad, initially 0.005 rad)
and regenerate the corresponding full hand-pose trajectory through the existing
TAMP planner/FK checks. This is reset-only target generation, not a joint hint or
IK action loop in the actor. Reset the grasp through physics; never inject saved
payload state. Recheck joint margins, lift height, collision, static support,
reference derivatives and any physical prefix. Invalid children reject; they
are not silently replaced until a successful sample appears. Their dynamics
remain unverified until rollout. Grasp reference and all measured failure limits
remain intact.

Split by parent family before sampling neighbors. Neither frozen assessment seeds
nor public regression seeds can enter the corpus. Keep the outer task mixture
explicit (70% general tracking); replay replaces a bounded fraction of carry
resets. With no explicit task weights, replay selects the 70% tracking-retention
mix; experiments should supply explicit weights. Log raw steps, episode starts
and parent identities, not just sampling
probabilities. Compare against source and teacher on identical child cases and
on the unchanged public TAMP tests. A gain in the replay family does not admit
multi-stage manipulation or hardware.

Training now has an explicit `physical_wheel_slew` flag, defaulting to legacy
behavior for historical fixtures. New matched training should enable it to match
the final wheel-motor slew already used in native TAMP execution. Exact resume
rejects a change in this setting. Training-only weight transfer records it as a
dynamics change; it is not an exact physical continuation.
