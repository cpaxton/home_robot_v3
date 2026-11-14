# Reference velocity and orientation feedback prototype

Branch: `feat/mars-reference-velocity`, based on `0fba6121` from
`feat/mars-base-isolation`. Simulator research only.

## Current status (2026-11-14)

[PR #232](https://github.com/cpaxton/home_robot_v3/pull/232) contains opt-in
legacy-MLP features, training tools and experiment evidence. The completed
six-run comparison does **not** establish a policy improvement: arm tracking
remains 63–64/64, while base-linear jerk causes most combined-contract losses.
Retain the original source checkpoint. The subsequent frozen-base/context
pilot has passed 1,178 full regressions and 27 focused checks, but has recorded
**zero PPO transitions** and is waiting for an explicit CPU14 handoff.

The separate GRU/multi-critic C1 comparison on the parent branch stopped in its
candidate settings check before the candidate's first optimizer update. That
incomplete comparison provides no shared-versus-per-head result; its owner
has retained CPU14 pending recovery. See the
[C1 attempt record](https://github.com/cpaxton/home_robot_v3/blob/feat/mars-base-isolation/docs/plans/mars_whole_body_advantage_pair.md#recorded-attempt-2026-11-14).
This PR is not a dependency of the GRU experiment and does not modify its actor.

Correctness, behavioral improvement and hardware readiness are separate
decisions. Correct, tested experimental code may merge with documented
negative results; passing every behavioral gate is not a merge prerequisite.
Keep the options off by default. A checkpoint needs measured improvement
before promotion, and these results do not authorize motion on Herman.

The actor can optionally consume a local reference twist derived from its
ordinary pose preview. This makes anticipation explicit while preserving the
stationary equilibrium of the learned gain head. It does not invoke the teacher
or add an unconstrained context-to-motion bias.

## Options

- `feedback_reference_velocity=true` (`--feedback-reference-velocity` in the
  ordinary manipulation trainer) adds nine feedback columns: base XY/yaw
  velocity, hand linear velocity, and hand angular velocity.
- `feedback_orientation_error="rotvec"`
  (`--feedback-orientation-error rotvec`) replaces each rotation-column residual
  with the principal axis-angle error, padded with three zero columns. The
  ordinary context still includes the original rotation-column preview.

Both default to the existing behavior. The ordinary preview format stays v1;
the policy constructor and environment metadata record the feature options.
Loading/resuming a bundle with inconsistent options fails.

## Geometry and equilibrium

Reference velocity uses a forward secant over announced offsets 0 and +0.05 s.
It is expressed in the **current** base/hand frames, in m/s and rad/s, using the
fixed physical normalization of the feedback observation contract. Angular
velocity uses `log(R_future @ R_now.T) / dt`; it is not a difference of rotation
vectors. Yaw differences wrap through the shortest angle. Target masks apply.

The preview sampler already accounts for actual executor progress rate.
Paused and completed references repeat their targets and produce zero
feedforward. This remains a finite-horizon velocity estimate: it can anticipate
a corner within the next 50 ms, and observation clipping can distort estimates
far outside the normal tracking region. No new corner gate is claimed.

The mean remains a learned matrix times pose errors, reference velocity and
measured velocity, followed by the existing tanh. A stationary reference at zero
error and measured velocity has exactly zero mean motion for any learned gains.
The original gain divisor is retained when adding reference columns, so the
existing restoring-yaw branch is unaffected at initialization.

Training transfer can add reference columns with zero influence on the actor
and critic. Existing angular-scale migration still works. Removing trained
reference columns or changing orientation representation during transfer is
rejected; use scratch training for the four-way orientation comparison.

## Matched pilots

`scripts/run_mars_feedback_ablation.py` uses the existing source-bound regression
receipt and mastery teacher preflight gates. It preserves the canonical cases,
physics, reference tubes, smoothness limits, and terminal hold requirements.
It logs each command, initial and final results, and the existing detailed
checkpoint/evaluation artifacts. Defaults are four scratch variants, one seed,
4096 transitions per variant, and the arm stage. This is a development pilot,
not a training-budget-matched reproduction of the existing long runs.

```bash
export PYTHONPATH=src MUJOCO_GL=egl
python scripts/run_mars_feedback_ablation.py \
  --regression-receipt logs/manipulation/reference-velocity-prototype/regressions.receipt.json \
  --output logs/manipulation/reference-velocity-prototype/scratch \
  --steps 4096 --seeds 90
```

For a function-preserving checkpoint continuation, select
`--variants baseline velocity --source /absolute/path/to/actor`. Orientation
variants require scratch initialization. Continuation keeps the copied
observation statistics frozen to isolate the feature change. Additional paired seeds and longer
budgets are needed before interpreting a tracking difference as a robust gain.

## Fault-stop evidence

The archived `mars_tamp_translation_v1` report records 52/72/52 learned calls
and `cartesian_path_tracking_envelope` stops; `mars_tamp_postures_v1` records
57/73/57 with the same reason. Stops are confirmed in each case. These are the
closest inspected reports to the quoted 53/73/53 counts, not an exact match.
They support investigating Cartesian tracking lag. They do not support adding
a premature-stop penalty or weakening executor stops. These observations are
from archived reports, not a new physical replay.

## Validation

Focused tests cover small rotations and half turns against SciPy, finite
geometry gradients, non-identity coordinate frames, yaw wrapping, masks,
retiming/pause/completion, PPO/export/load for all four variants, stationary
equilibrium, and actor/critic fidelity across velocity-feature transfer.
The full source-bound regression gate passed **1165 tests**, with zero failures,
errors, or skips, in 663.8 seconds. The receipt and pilot artifacts live under
`logs/manipulation/reference-velocity-prototype/`.

## Completed continuation pilot

Both variants started from the same `mars-base-curriculum-legacy-v1-seed90`
checkpoint, weights SHA-256
`e7b425699510cc635485bdd8f59a5b25692fb165faf248aaecbc2707c0dbb883`.
The teacher mastered all 16 arm admission cases. Each actor initially mastered
all eight canonical arm cases. On 160 observations across these eight cases,
the maximum initial action difference after adding zero-influence velocity
features was `5.59e-9`. Sampled p95 prediction time was 0.586 ms for baseline
and 0.673 ms for velocity, under concurrent regression work; this is not an
isolated latency benchmark.

The continuation used seed 90, 4096 transitions per variant, learning rate
`1e-5`, arm action standard deviation `0.03`, frozen copied normalization, and
the source's executor action masks. Both used the existing native rewards and
identical physical cases and reference paths. The runner records exact commands
in `continuation/results.json`.

| Final measure | Baseline | Reference velocity |
| --- | ---: | ---: |
| Tracking and settling, all canonical arm cases | 8/8 | 8/8 |
| All gates, training cases | 4/4 | 3/4 |
| All gates, validation cases | 4/4 | 4/4 |
| Validation motion position RMS | 0.750 mm | 0.787 mm |
| Validation motion orientation RMS | 0.01035 rad | 0.01059 rad |

Motion RMS pools samples during each announced reference's active interval,
excluding the subsequent endpoint hold. The velocity actor's sole combined-gate
failure was `train:arm:roll_left`: peak base-linear jerk was 0.20851 m/s³ against
its teacher-relative budget of 0.20068 m/s³. It still passed tracking and settling.
The compact readout is `continuation/pilot-summary.json`; full traces and
initial/final checkpoints remain alongside it.

This small pilot provides **no evidence of an improvement**. It verifies the
new feature/transfer/training path and exposes a smoothness regression. One seed,
a short continuation, and already well-tracked canonical cases cannot establish
whether explicit velocity helps the harder expanded-arm or TAMP distribution.
The rotvec option passed geometry/PPO/export tests but was not trained in this
continuation pair; comparing orientation representations requires the scratch
four-way experiment. Neither checkpoint is promoted.

## Expanded paired experiment protocol

The expanded experiment compared baseline and reference velocity on
`mars_tracking_mastery_tamp_postures_v1.json`, using the same retained source
checkpoint as the pilot. It trains the arm stage for **16,384 transitions per
variant and seed**, with paired seeds **90, 91, 92**, LR `1e-5`, arm exploration
standard deviation `0.03`, frozen normalization, and unchanged executor masks.
The budget is six serial runs / 98,304 total training transitions. It uses only
the final checkpoint for the declared comparison, with no automatic extension,
restart, best-checkpoint selection, or promotion.

Teacher preflight covers the full admission population for base, arm and
coupled stages. Initial and final actor snapshots are evaluated on that same
population; training still uses only its declared training split. Added
near-straight arm postures are training/development cases, not new held-out
validation cases. Report canonical validation separately. Base and coupled
readouts check regressions outside the arm training stage. A useful treatment
must improve tracking across paired seeds without trading away smoothness or
the existing validation/base behavior. Keep failed and stopped cases in the
denominators and compare tracking errors over matching time coverage.

On Olympia, training/evaluation is one worker pinned to CPU14, with CUDA hidden,
rendering disabled, and one Torch/BLAS/OMP/MKL thread. Focused and full regression
tests use CPU15 before training starts. The managed supervisor waits for the
existing CPU14/15 workloads to exit, uses a separate immutable runtime checkout,
copies the source bundle, and stops if tests, teacher admission, source checks,
or its fixed timeout fail. It does not stop or modify other jobs. Caliban is not
used for this comparison because the recorded ARM frozen-fixture qualification
remains unresolved; those gates must pass before moving policy training there.

The runner now supports `--evaluation-stages` and `--evaluation-population`:

```bash
taskset -c 14 env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_mars_feedback_ablation.py \
  --regression-receipt /path/to/current/regressions.receipt.json \
  --suite configs/benchmarks/mars_tracking_mastery_tamp_postures_v1.json \
  --source /path/to/frozen/source/actor --variants baseline velocity \
  --stages arm --evaluation-stages base arm coupled \
  --evaluation-population admission --steps 16384 --seeds 90 91 92 \
  --learning-rate 1e-5 --arm-std 0.03 --output /path/to/fresh/expanded-experiment
```

This protocol was declared before the run. The completed results follow.

## Expanded experiment result: continuation regresses smoothness

All six fixed-final runs completed, for 98,304 training transitions. The fresh
source-bound gate passed 1165 tests and the orchestration checks passed 2/2.
Teacher admission passed all 122 selected base/arm/coupled cases for each
variant. No candidate is promoted.

| Variant | Seed | Arm tracking/settling | Arm all gates | Base all gates | Coupled all gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, initial | all | 64/64 | 59/64 | 42/50 | 2/8 |
| Velocity, initial | all | 64/64 | 60/64 | 43/50 | 2/8 |
| Baseline, final | 90 | 64/64 | 38/64 | 42/50 | 2/8 |
| Baseline, final | 91 | 64/64 | 19/64 | 43/50 | 2/8 |
| Baseline, final | 92 | 64/64 | 28/64 | 41/50 | 3/8 |
| Velocity, final | 90 | 63/64 | 25/64 | 41/50 | 2/8 |
| Velocity, final | 91 | 64/64 | 38/64 | 42/50 | 2/8 |
| Velocity, final | 92 | 64/64 | 22/64 | 44/50 | 2/8 |

Initial and final reset/reference hashes match for all 732 case pairs. The
initial base/arm full-pass counts differ by one case across representations;
retain that measured difference and compare each final with its own anchor.
This is not an initial performance gain attributable to feedforward.

Base-linear jerk explains the arm full-pass losses: baseline final arm jerk
misses are 26/45/36 across seeds, versus 5 initially; velocity has 39/26/42,
versus 4 initially. Tracking mostly survives, but continuation degrades the
combined contract in both representations. The evidence does not support more
training with this unchanged setup or adoption of the velocity treatment.

The [compact result and provenance](../../paper/data/whole_body_control/mars_reference_velocity_expanded_v1/results.json)
include per-seed train/validation groups and failures. Full traces/checkpoints
remain in `logs/manipulation/reference-velocity-expanded-v1/comparison/`.

## Channel-restoration diagnostic: preserve the learned base function

A separate 32-rollout, zero-training intervention uses the first declared seed
(90) and four **training-only** cases both initial actors passed and both finals
failed: pitch down, wrist down, and near-straight A/B upward translation. It
retains each variant's own initial anchor to avoid mixing representations.

For fully constrained base goals with an active hand, evaluate the final actor
with only its two base commands replaced by its initial actor's commands on the
same live observation. The reciprocal intervention replaces only the five arm
commands. Gripper commands, inputs, targets, native execution limits, timing,
teacher budgets and checkpoint files are unchanged.

| Commands used | Baseline all gates | Velocity all gates |
| --- | ---: | ---: |
| Initial actor | 4/4 | 4/4 |
| Final actor | 0/4 | 0/4 |
| Final arm, initial base | 4/4 | 4/4 |
| Final base, initial arm | 0/4 | 0/4 |

All 16 unmodified initial/final replays reproduce their prior full-pass flags
and peak base-linear jerk exactly (maximum jerk difference zero), with matching
reset/reference hashes. Restoring the initial base commands also recovers the
velocity actor's selected tracking-timeout case. Restoring the initial arm
commands recovers tracking there, but leaves all four jerk failures.

This implicates the changed learned base function in these selected regressions.
It does not isolate base readout weights from shared actor-context changes, nor
establish generalization: these cases were selected after observing training
failures. The two-policy in-memory intervention is **not** an exportable single
actor or canonical qualification. No policy is changed, exported or promoted.
The exact intervention script hash, case metrics and replay checks are retained
in the compact result. Full diagnostic artifacts are in
`logs/manipulation/reference-velocity-expanded-v1/channel-diagnostic-v1/`.

The next justified training hypothesis is to freeze the retained base function
**including the shared actor context**, train only arm feedback/feedforward
coefficients, and suppress wheel exploration during arm adaptation while
preserving the existing base mean commands. Freezing base readout weights alone
would still allow shared-context changes to alter the base function. This next
training mode is now implemented; its bounded pilot is queued as described
below, with the full base/arm/coupled and smoothness measurements retained.
Axis-angle remains a separate untrained option.

## Frozen-base/context arm-adaptation pilot

The opt-in `--arm-adaptation-only` mode now implements that hypothesis. It keeps
one eight-output feedback actor. Actor-context parameters, base gain rows,
restoring-yaw parameters and gripper parameters are frozen. Only the five arm
gain rows, arm exploration scales, and critic are trainable. The original
normalization is frozen too. Exact protected-tensor checks run at rollout,
checkpoint and training-end boundaries; a violation stops the run.

Base/gripper sampling uses their learned deterministic means. Those commands
are not zeroed or filtered. PPO likelihood and entropy use the active arm
marginal; deterministic base/gripper channels contribute no policy gradient.
The underlying Gaussian scale tensor still stores frozen base/gripper entries,
so those entries are not effective sampling standard deviations. Runtime
deterministic means use the ordinary actor path. No second actor or teacher is
invoked, and no new task/physics inputs or thresholds are introduced.

The mode requires an initialized source bundle, arm-only training stages,
executor masks and frozen normalization. It defaults off. Weight decay is
rejected because it would move gradient-masked base rows. The focused
adaptation/executor/orchestration checks pass **27/27**, including actual PPO
updates, bitwise preservation of base/grip means and context tensors, continued
arm/critic learning, deterministic wheel requests, checkpoint transfer, and
optimizer/export/load round-trips with both feature widths.

The first declared pilot uses **two runs**, baseline and reference velocity,
**seed 90**, **16,384 transitions each / 32,768 total**. It retains the earlier
source, TAMP-postures training population, LR `1e-5`, arm standard deviation
`0.03`, rewards, limits, and full 122-case initial/final evaluation. This compares
with the completed unfrozen seed-90 controls; it is a deliberately bounded test,
not a new three-seed claim. No automatic extension, retry, best-checkpoint
selection, or promotion is permitted. More seeds are a separate decision.

A fresh full source-bound regression gate and current-source teacher admission
must pass before PPO. Use CPU15 for checks and one CPU14 worker for training,
CUDA hidden, rendering disabled and one numerical thread, with an isolated
runtime and coordinated reservations. Add `--arm-adaptation-only` to the
expanded-runner command, with `--seeds 90 --steps 16384`. This section records
the implementation and launch protocol; a completed training result is pending.

Managed job `20261114_091806_52e91d` now runs this protocol from immutable
implementation `7400cef1fd12fbd7079f2d00b86caa634a6c3c8c`. Its fresh full
regression gate passed **1,178/1,178**, with zero skips, errors or failures;
the focused gate passed **27/27**. The full receipt binds the tested source
and XML. CPU15 was released after host-visible gate/pytest child exit.

The supervisor is waiting for the existing C1 reservation to explicitly
release CPU14. C1 currently stopped at a settings-comparison error, so that
release has not occurred. Once released, the supervisor runs
current-source teacher admission, both fixed training runs, all declared
evaluations and a serialized protected-function check without another launch
request. No new PPO transitions are recorded yet. No automatic retry,
extension or promotion is enabled. A failed gate stops the job.

Launch protocol, source copy, receipt, progress and eventual results remain in
`logs/manipulation/mars-frozen-base-arm-v1/`; `supervisor.json` is the live phase
and transition-count record. This launch status is not a behavioral result.
