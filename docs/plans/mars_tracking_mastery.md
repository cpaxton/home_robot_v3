# Mars tracking mastery before broad training

The first goal is to fit a small deterministic control problem. A single policy
receives the established masked SE(2) base target, SE(3) hand target, gripper
reference and measured history/pose preview. TAMP/trajectory generation supplies
the paths; the actor chooses complete velocity commands. Case IDs and benchmark
stage names are logging/reset metadata and are never added to actor inputs.

The frozen suite is `configs/benchmarks/mars_tracking_mastery_v1.json`.
It contains 13 training cases and 13 nearby validation cases:

| Stage | Training cases | Validation cases | Motion |
| --- | ---: | ---: | --- |
| base | 5 | 5 | Hold, forward, reverse, left yaw, right yaw |
| arm | 4 | 4 | Two pitch reaches and two wrist rolls, with active base hold |
| coupled | 2 | 2 | Axle-correct turn/drive with simultaneous arm motion |
| loaded | 2 | 2 | Physically grasped lift and hold using a 1.1x-duration reference |

Unloaded validation uses separate reset seeds/postures and 15% larger target
motions. Loaded cases deliberately reuse prior development families, including
both original teacher retention failures in training. They are held out from
this learner's training split, not claimed as a new qualification population.
There is no domain randomization, task mixture, goal switching or push in this
first problem. Those return after this finite capability test passes. This
isolated diagnostic does not replace the broad 70% tracking training distribution.

## Reference generation and physical checks

Arm targets come from FK of finite joint witnesses. Base paths turn about the
actual driven axle, then translate along its heading; they never request sideways
wheel motion. Private continuous FK generates world hand targets at every sample.
Rest-to-rest quintic references obey native velocity/acceleration limits and
explicit experimental jerk bounds. Unloaded references last at least three
seconds; loaded paths uniformly slow the existing admitted TAMP-generated path.
The original 12-second episode budget remains.

Every reset repeats the physical fixture preparation, with no replayed object
state or attachment injected into execution. Joint margins, TAMP collision
semantics and sampled support margins are checked before empty-hand motion.
Loaded fixtures retain their original grasp reference, support-departure guard,
retention/force/wheel checks and physically generated grasp. Execution adds a
fresh measured collision check and a tracking tube; the independent low-level
force/wheel/invalid-physics stop logic remains active. Admission is not a dynamic
certificate: the teacher must execute all selected cases before training starts.

All eight action channels remain available. Arm-only cases constrain the base
through pose targets and reward/safety checks, rather than silently setting wheel
actions to zero. Base-only cases mask the hand goal through the existing command
contract. No teacher residual, DAgger, online IK action loop or task-stage decision
is introduced into the actor. The explicit comparator uses the existing measured
Jacobian task-pose teacher and the same pose schedule.

## Mastery gate

Every case must pass individually; an average cannot hide one failed direction.
Both the training and validation partitions must pass before adding complexity.

- Active hand endpoint: 3 mm position and 0.02 rad orientation.
- Active base endpoint: 5 mm XY and 0.01 rad yaw.
- At least two continuous seconds of accurate terminal holding, with measured
  whole-body speed below the existing stop thresholds and arm speed below
  0.03 rad/s. Motion continues through the full episode to expose wandering.
- Hold drift at most 3 mm; retain the existing loaded/fixed-base boundaries.
- No force, wheel-unloading, retention, invalid-physics, forbidden-collision or
  path-envelope stop. The hand path tube is 10 mm / 0.1 rad; base tube is
  15 mm / 0.08 rad.
- Peak monitored force and separate joint/base-linear/base-yaw jerk must be
  within 1.25x the paired teacher, with explicit small floors in the frozen suite
  for nearly stationary references. These are experimental comparator bounds,
  not manufacturer damage or jerk ratings.

Reports separate tracking/settling, drift and each smoothness reason. A native
`ok` is not automatically a mastery pass. Every rejection stays in the denominator.
Initial physical states and exact continuous reference records are hashed and
must match between teacher and learned-policy evaluations.

## Run and inspect

Run in the controller-integration checkout, using the existing environment:

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl

taskset -c 14 .venv/bin/python -m emet.rl.manipulation.mastery preflight \
  --output logs/eval/mastery-teacher
```

`report.json.gz` contains every case, FK witness, dense pose samples, admission,
measured action/pose trace and failure reason. `summary.json` separates stages and
splits. `targets.svg` shows base and hand paths, endpoint yaw arrows and measured
hand errors. Inspect an individual trajectory without running physics:

```bash
.venv/bin/python -m emet.rl.manipulation.mastery plot \
  --preflight logs/eval/mastery-teacher/report.json.gz \
  --case-id train:coupled:forward_turn --output logs/eval/mastery-one-case
```

The single-case view includes a hand orientation triad. Internal FK records are
visualization/execution records for this simulator; they do not extend the v1
hardware wire protocol. Plot sub-selection cannot weaken a training gate.

Produce fresh source-bound regression evidence before training:

```bash
taskset -c 15 .venv/bin/python - <<'PY'
from pathlib import Path
from emet.rl.regression_gate import run_regressions
run_regressions(Path('logs/eval/mastery-regressions.xml'))
PY
```

Start with the five base cases. Defaults use scratch pure PPO, one CPU environment,
fixed learning rate, separate base/arm exploration scales and balanced shuffled
cycles through training cases. Validation cases never enter the sampler.

```bash
taskset -c 14 .venv/bin/python -m emet.rl.manipulation.mastery train \
  --stages base --steps 16384 \
  --preflight logs/eval/mastery-teacher/report.json.gz \
  --regression-receipt logs/eval/mastery-regressions.receipt.json \
  --wandb-mode offline --output logs/manipulation/mastery-base
```

Training refuses stale sources, mismatched suite identity or an incomplete/failed
teacher preflight for either split. Output paths must be new. W&B is optional,
offline and scalar-only. Each checkpoint reports every selected train/validation
case; the final `last` bundle is saved and evaluated after the optimizer update.
Evaluation preserves Python/NumPy/Torch RNG state so loading a checkpoint cannot
reset exploration. Raw per-case steps and episode starts expose actual coverage. Those counters are
cumulative: read the last metrics record rather than summing records.
No best checkpoint is silently selected using validation results.

After both partitions pass, retain base rehearsal while adding arm cases:

```bash
taskset -c 14 .venv/bin/python -m emet.rl.manipulation.mastery train \
  --stages base arm --initialize-from logs/manipulation/mastery-base/last \
  --preflight logs/eval/mastery-teacher/report.json.gz \
  --regression-receipt logs/eval/mastery-regressions.receipt.json \
  --output logs/manipulation/mastery-base-arm
```

Then add `coupled`, and finally `loaded`. Stage selection is explicit and supports
isolated diagnostic training; there is no automatic stage advancement. Warm-start
uses a fresh optimizer and is recorded as weight transfer, not exact continuation.
Re-evaluate a bundle independently with the same preflight:

```bash
taskset -c 14 .venv/bin/python -m emet.rl.manipulation.mastery evaluate \
  --stages base --policy logs/manipulation/mastery-base/last \
  --preflight logs/eval/mastery-teacher/report.json.gz \
  --output logs/eval/mastery-base-final
```

Passing this small set is a prerequisite to broader randomization, disturbances,
mid-motion goal changes and public multi-stage TAMP regression testing. It does
not promote a policy or authorize real-robot control.

## Longer matched base-only experiment

The first 16,384-step pilot is a smoke-scale training budget: 64 PPO rollout/update
cycles and about 20 episodes per training case. Full mastery did not improve,
but forward terminal XY error fell to 2.5 mm with remaining heading/settling error;
left yaw error fell from 0.040 to 0.026 rad. Reverse remained outside the path tube.
Do not equate an unchanged mastery count with no tracking improvement.

The bounded debugging experiment uses 262,144 steps for seed 90 on **one CPU
worker**. Repeated seeds 91/92 are deferred until this configuration shows useful
learning. The active run starts from scratch on the same five base cases, with
unchanged model, action scales, reward, limits and train/validation targets.
Seed 90 provides the longer-budget comparison. Checkpoints evaluate both complete
splits every 16,384 steps and after the last update. No automatic extension beyond
the budget or advancement to arm/coupled/loaded stages is enabled.

A simulator-only sign probe compares zero action, small signed direct pulses and
the task-pose teacher on identical initial states. Positive/negative forward and
yaw requests must produce correspondingly signed measured motion. The existing
`base_linear_velocity_m_s` metric is an unsigned displacement speed; the new audit
uses body-local MuJoCo spatial velocity for a signed command-response comparison.
This adds diagnostics without changing the observation, reward or actuator path.

The mastery trainer now emits:

- `metrics.jsonl`: cumulative case coverage and per-rollout, per-case reward sums,
  reward components, action means/standard deviations/clipping, command/measured
  base velocity, errors and terminal codes. Unequal episode lengths remain visible.
- `updates.jsonl`: post-optimizer entropy loss, approximate KL, clipping fraction,
  value loss/explained variance and per-channel Gaussian standard deviation.
  The last optimizer update is included. Undefined statistics serialize as null.
- Checkpoint evaluations: full case-level tracking/holding/smoothness reports,
  actions, signed command/response and reward components.

W&B receives offline scalar coverage, optimizer, rollout and evaluation metrics.
Weights and full traces remain local. Check the series `status.json` and each
seed's metrics rather than assuming a launched run has finished. If translation
remains flat, audit reward/observation/action sensitivity and use smaller-motion
training curriculum while preserving the frozen evaluation suite. Add parallel
simulation only after learning is demonstrated and the host is stable.

## Opt-in turning/jerk correction

The v2 suite retains every v1 case/reference/gate while opting into final wheel
command smoothing and dense fine-yaw/full-body-quiet/normalized-jerk rewards.
The frozen old actor improves from 0/10 to 6/10 full mastery under the declared
execution change, with 10/10 smoothness passes. Turning remains unmastered.
A single bounded warm-start is running; it is separate from the intervention.
See [implementation, checks and launch](../experiments/mars_turning_jerk_20261029.md).
Select `--suite configs/benchmarks/mars_tracking_mastery_v2.json` explicitly;
the legacy default and existing comparison evidence remain reproducible.
