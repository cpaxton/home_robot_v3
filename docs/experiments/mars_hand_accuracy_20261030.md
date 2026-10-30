# Mars hand accuracy setup and functional check, 2026-10-30

The latest base actor is not yet a usable manipulation policy. Fresh original
expanded teacher execution passes **66/66**, while that actor passes **0/8 arm**
and **0/4 loaded** tracking/holding cases. It retains **50/50 base tracking/holding
and 34/50 full base mastery**. The comparisons use matching physical starts and
references; rejected/stopped cases remain failures.

In the isolated arm cases, peak requested joint rates are only
**0.00041–0.00061 rad/s**. Motor-target lag is around 0.0014 rad and peak torque
around 0.18 Nm. These traces support under-commanding as the immediate deficit;
they do not establish real-world actuator limits or prove sag is solved. Pitch
cases stop at the 10 mm tube after 1.55–1.75 s. Roll cases finish with position
within 0.72 mm but nearly the entire 0.08–0.092 rad rotation still uncorrected.

## Implemented treatment

The [hand plan](../plans/mars_hand_accuracy.md) retains canonical cases and all
accuracy, orientation, quiet, path, force, retention and wheel/jerk gates.
Stage-aware sampling assigns 75% arm / 25% base reset probability despite having
45 base cases and only four original arm cases. Add quarter/half arm amplitudes
from the existing training reset postures, producing 12 arm training cases. No
validation seed or target enters those additions. Every generated target retains
its continuous FK witness, collision/support admission and bounded reference.

Enable the existing native hand-gated base credit. Dense pose/progress rewards
remain at 20 Hz; history/preview, architecture and complete velocity actions
stay the same. No new force cap, torque gain, teacher action loss, IK action loop,
residual or output filter is introduced. Evaluation-only measured-Jacobian
diagnostics expose slewed/measured joint rates, motor-target error, torque and
requested/measured hand velocity toward the active goal.

The new 66-case base/arm teacher preflight passes on both the local host and
Caliban. Local verification passes **929 source-bound controller tests**, including
36 focused mastery tests, plus two separate headless LLM import tests. A small
import repair defers microphone/Whisper dependencies until voice is requested.
Runtime-package parity, relevant Ruff and generated-artifact checks pass.

## Completed 8,192-step functional check

One CPU14 worker, CUDA hidden, one Torch/BLAS thread, seed90, fresh optimizer,
LR5e-5, base/arm std0.003/0.05, gamma0.9975, 256-step rollouts and 16,384-step
command-penalty warmup. This deliberately short check stops before completing
that ramp; independent evaluations use full cost. It is separate from the planned
eight-worker experiment, which starts from the preserved base actor. W&B is
offline and scalar-only. An initial logging-service sandbox rejection performed
zero optimizer steps; the successful retry preserves that failed setup.

The check completes 8,192 transitions and 160 optimizer epochs with 42 episode
starts, 28 arm starts. Actual sampling counts are retained rather than claiming
an exact 75% frequency in this small draw. Every original arm training primitive
and smaller amplitude is visited. Step-zero and final physical/reference hashes
match on all 18 canonical evaluation cases.

- **Arm remains 0/8** tracking/holding and full mastery. Pitch trajectories still
  stop at the unchanged 10 mm tube; final errors from different stopping times
  are not equal-horizon accuracy comparisons.
- Peak arm requests increase to **0.0077–0.0170 rad/s**. Wrist endpoint rotation
  decreases from **0.080–0.092 to 0.043–0.070 rad**, but wrist position error
  worsens from **0.57–0.71 to 3.62–7.88 mm**. This is a pose tradeoff, not success.
- Position RMS over matching time prefixes improves on all four pitch cases:
  **4.79→4.33, 4.96→3.98, 4.86→4.56, 4.69→4.22 mm**. Wrist position RMS worsens
  to 5.0–7.22 mm. Base endpoint error stays below 0.30 mm on these arm cases, so
  base displacement does not explain those larger hand errors.
- Original base tracking/holding remains **10/10** and full mastery **6/10**;
  paired smoothness failures change cases. No wheel-lift, unsafe-contact,
  invalid-physics or retention flags occur in the 18 final finite cases. This
  unloaded check is not retention/compliance or hardware qualification.

The actor is learning nonzero arm responses, but meaningful arm coordination and
position accuracy remain unresolved. This result does not justify raising gains
or relaxing thresholds.

## Parallel pilot state and artifacts

Caliban's bounded 65,536-step eight-worker supervisor is launched. At the time of
this setup report, **its fresh source-bound regression gate is still running;
production PPO has not started**. Teacher admission is complete. The supervisor
refuses PPO if either gate fails, evaluates original base/arm cases on schedule,
then all expanded base/arm cases and separate coupled/loaded probes. It performs
no automatic extension, checkpoint selection, stage advancement or promotion.

Local persistent state/driver/logs: `logs/recovery/hand-accuracy-v1/`.
The completed local check is
`logs/manipulation/mars-hand-accuracy-smoke-v1-retry-seed90/`.
Caliban's RAM-backed workspace is
`/dev/shm/cpaxton-mars-parallel-20261030/native`. A bounded read-only mirror copies
state, weights and reports into ignored local storage once per minute; it exits
on completion/failure or after three hours. Check `pilot-state.json` and the
receipt, not just process existence, before claiming training has started.

Compact [baseline](../../paper/data/whole_body_control/hand_accuracy_v1_setup/baseline.json),
[initial command audit](../../paper/data/whole_body_control/hand_accuracy_v1_setup/initial-command-audit.json)
and [matched functional-check metrics](../../paper/data/whole_body_control/hand_accuracy_v1_setup/smoke-comparison.json)
are tracked. Full compressed traces and weights stay in ignored storage. All
failures are preserved; no Herman motion occurred.
