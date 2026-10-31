# Preserve the arm result while restoring whole-body learning

The five-output component passes all 16 unloaded arm admission cases. Transfer
it into the existing eight-output, history-and-preview whole-body actor. This
is a training warm-start, not a second runtime policy, teacher residual or new
task manager. Runtime still consumes masked base SE(2), hand SE(3) and gripper
targets and applies the existing executor and physical safety checks.

1. Verify the producer's completed run, exact environment, stable model/control/
   observation sources, all eight canonical component passes and checksums of
   its final actor, statistics and evaluation **before deserialization**. Copy
   the shared actor context, critic, five arm gain rows and normalization. New
   general base rows and grip offset start at zero. The existing restoring-yaw
   branch uses its normal constructor initialization; the earlier base policy
   is not preserved. Do not mutate the producer checkpoint.
2. Test actual PPO-produced weights, arm means, critic values, normalization and
   full runtime export/reload. Reject modified artifacts and incompatible
   configuration. Audit the real checkpoint's predictions, then repeat the
   unchanged physical arm cases before training. Run the complete source-bound
   regression gate and fresh matched base/arm teacher admission.
3. Run one bounded **32,768-step** local PPO pilot, seed90, one environment on CPU 14,
   CPU inference and one Torch/BLAS thread. LR 3e-5, gamma 0.9975, std 0.01 for each
   of eight outputs, rollout/batch256, five epochs, clip0.1 and target KL0.01.
   Freeze the copied normalization to preserve the arm feature basis. Rehearse
   75% arm /25% base episodes, with the existing smaller training amplitudes;
   validation resets remain excluded from sampling. Use full command costs
   from step0; no new action filter or teacher action loss. Record offline W&B
   scalars and evaluate all 18 canonical cases at step0, every 8192 and final.
4. Keep each scheduled checkpoint, all failures, teacher-paired jerk/force and
   quiet-hold metrics. Evaluate the final checkpoint on all 66 expanded cases.
   Report arm and base separately; an aggregate score cannot hide forgetting.
   If arm regresses or base does not learn, inspect measured/commanded motion,
   clipping, reward terms and optimizer diagnostics before extending training.
   Frozen arm-only statistics may clip unfamiliar base inputs: measure it rather
   than silently adapting the normalization during this first experiment.
5. Only after reliable base/arm tracking, add simultaneous targets, loaded hold/
   lift transitions and the unchanged public TAMP pick/place programs. Those
   remain separate acceptance cases; empty-hand success does not qualify carry,
   agent execution or Herman. Do not weaken pose, path, jerk, force, wheel-load,
   stop-confirmation or frozen-fixture gates to start another run.

Use `configs/benchmarks/mars_tracking_mastery_arm_to_whole_body_v1.json` and
`mastery train --initialize-arm-from .../last --arm-evidence .../results.json
--freeze-normalization`. The component's producer files stay immutable and
unmodified components remain non-deployable. Generated weights and traces stay
in ignored `logs/`; only compact metrics, hashes and reports enter Git.

## Control-step exposure follow-up

The completed first pilot retains all 16 expanded arm tracking passes but only
8/16 full arm passes; base tracking remains 38/50 with 33/50 full passes. Straight
base motion receives just389/32,768 transitions (1.19%). Episode probabilities,
yaw augmentation and early failed-path stops explain this sampling gap. A
single analytic translation-gain probe worsens jerk/quietness and is rejected.

Before another architecture or optimizer change, test one distribution change:

- `mars_tracking_mastery_step_budget_v1.json` retains the exact previous config,
  canonical cases, augmentation, gates, initialization and PPO hyperparameters.
  Target 75% arm /25% base **control steps**, rather than episode starts. Within
  base, target 40% forward, 40% reverse, 5% hold and 7.5% for each turn direction.
  This assigns 20% of total transitions to translation instead of1.19%.
- Select the next stage/family by accumulated control-step deficit, then cycle
  its cases without replacement. Count actual policy transitions, including
  failed episodes, without synthesizing data or changing physical state. All
  validation seeds remain excluded. Budgets are per native environment; the
  current pilot still uses only one worker.
- Test unequal 39/35-step failed translations versus 240-step episodes, step and
  family proportions, independent family coverage, deterministic reseeding,
  invalid recipes and native step/reset accounting. Preserve the old episode
  sampler by default and keep evaluation independent of training budgets.
- Obtain fresh current-source teacher admission and full regression receipts.
  Start one 32,768-step seed90 pilot from the same original arm component with
  frozen normalization, LR 3e-5/std 0.01 and evaluations every 8192. No reward,
  exploration schedule, mean-action prior, teacher loss or actor architecture
  changes are combined with this comparison. Report all 18 canonical and66
  expanded cases plus realized transition exposure; do not automatically extend.

The separate arm/base finite suite still does not qualify simultaneous goals,
loaded hold/lift, public agent pick/place or hardware. Full smoothness remains
required even when endpoint errors are small.

## Executor-visible requests and command history

The sampling-only follow-up passes 957 source-bound tests and logs 8,192 PPO
steps before being intentionally stopped. Its partial checkpoint is retained.
Inspection finds another concrete distribution defect: frozen arm-component
statistics assign wheel/grip request history a standard deviation of about
0.000114. The full actor samples an unused reaching gripper channel at std0.01,
and the executor records that request even though it ignores its correction.
Reconstructed final arm traces saturate this history channel about67–100% of
the time. Zero-mean Gaussian noise alone would saturate it about82% of the time.

Fix this without changing the native controller or adding task networks:

- Use one existing eight-output feedback actor/trunk. Its Gaussian distribution
  emits zero for channels ignored according to visible target masks and grip
  phase/retention intent. Apply the same mask to sampled actions, deterministic
  means, log probabilities and entropy. Preserve base authority when an active
  hand has no base target. Executor holds, faults and release authority remain
  independent and unconditional.
- Keep normalized wheel/grip requests in actor units, preserving the learned
  arm-command statistics. The old component's corresponding inputs and means
  must be exactly zero before this declared normalization migration. Preserve
  component arm/critic behavior on that domain; do not claim identical statistics
  or equality on arbitrary new base-command inputs. Check real reset predictions
  and all 16 physical arm cases again.
- Test likelihood/entropy and gradient masking, shared/unshared feature paths,
  actual PPO head updates, native ignored-command history, runtime export/reload,
  fixed command scales and invalid migration/configuration rejection. A later
  training transfer must not silently discard the executor mask.
- Use `mars_tracking_mastery_command_history_v1.json`: retain the control-step
  budgets, unchanged physical targets/reward/guards, original arm initialization
  and one 32,768-step seed90 pilot. Obtain fresh teacher/regression receipts.
  This combines the measured sampling and history repairs; it is no longer a
  sampling-only ablation. Preserve every checkpoint and all 66 final outcomes.

Only the executor-ignored output dimensions are zeroed. Active policy means
receive no temporal filter or teacher residual. The original component and
interrupted run stay immutable; the corrected pilot uses a fresh directory.

The [completed command-history pilot](../experiments/mars_command_history_20261031.md)
retains16/16 arm tracking but only4/16 full passes; base is38/50 tracking and32/50
full passes. Translation receives19.43% of actual transitions. Keep it
unpromoted. A separate inertia/body-frame telemetry correction reproduces every
physical metric and reward on66 cases, and final current-source regressions
pass976 tests. Isolate base optimization with the successful arm function fixed
before another mixed extension or loaded-stage advancement.
