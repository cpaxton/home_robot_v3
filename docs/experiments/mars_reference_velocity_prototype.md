# Reference velocity and orientation feedback prototype

Branch: `feat/mars-reference-velocity`, based on `0fba6121` from
`feat/mars-base-isolation`. Simulator research only.

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
