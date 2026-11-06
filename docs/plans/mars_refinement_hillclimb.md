# Mars whole-body refinement hill climb

Objective: a smooth, accurate, reactive controller that can execute the agent's
task-pose contracts on real Innate Mars. Keep one runtime actor. Retain every
attempt; promote a development candidate only using explicit physical gates.

## Current baseline

`mars-base-curriculum-legacy-v1-seed90/last/actor` passes 49/50 expanded base
tracking, 42/50 full base and 15/16 full arm cases, with 16/16 arm tracking.
Remaining failures: one larger forward reference, seven other base smoothness
misses and one small-amplitude arm jerk miss. No unsafe-contact, wheel-lift or
invalid-physics flag trips. This is a development baseline, not a hardware-ready
policy. Source robot/model, normalization and all acceptance thresholds remain
fixed for the next comparison.

The previous cold-start aligned reward removes 50 native completion points but
earns no full-contract bonuses during exploratory training. Test a gradual
transition and quieter exploration from the stronger baseline instead.

## Continuation fidelity and reward schedule

The default initial arm admission remains strict. The explicit
`--initial-admission source_fidelity` mode supports simulator repair of a known
partly qualified source. First replay the source on all 66 expanded cases under
current code and require its safety flags to remain clear. Then require the
merged actor's canonical and expanded-arm metrics, actions, rewards, physical
poses, stop codes and reset/reference identities to match that fresh source
exactly. Record existing failures without granting qualification. Missing,
duplicate, changed or stale coverage fails before optimization. Final evaluation
and hardware-promotion gates are unchanged.

For fraction `f`, use `native_reward - f * native_bonus - f * excess_costs +
f * terminal_contract_bonus`. At zero this is exactly native reward; at one it
is the previous fully aligned objective. Latch `f` per episode so changing a
global schedule cannot change reward accounting halfway through a trajectory.
Native and terminal completion credit together remain bounded by their declared
mixture. Full qualification always uses the fixed original gates.

## Next matched refinement pair

Use the same 8k baseline above, seed 90, fresh optimizer, frozen arm/context/grip
and normalization. Run sequentially on CPU14, CUDA hidden, one Torch/BLAS thread,
offline scalar W&B. Require fresh source-bound full regressions and teacher
66/66 before either run.

- Each run: 16,384 transitions, 80% base / 20% moving-arm control-step budgets,
  existing base primitive/posture mixture, LR 3e-5 and initial gSDE coefficient
  std 0.1 rather than 0.5. Keep 8-tick resampling, gamma 0.9975, rollout/batch
  256, five epochs, clip 0.1 and KL target 0.01.
- Control: native reward throughout.
- Scheduled: jerk/peak weights 0.1/0.2, full-contract bonus 10, native warmup for
  4,096 transitions, linear transition over the next 4,096, then full alignment.
  Episode boundaries can delay the applied fraction; log the actual fraction.
- Evaluate all 18 canonical cases at step zero, every 4,096 and final. Evaluate
  all 66 expanded cases at final. Score every result; do not select an earlier
  checkpoint using validation. This comparison is exploratory hill climbing on
  development cases, not an untouched generalization estimate.

## Selection and further work

Retain the source unless a fixed final candidate has no safety events, preserves
16/16 arm tracking and at least 49/50 base tracking, preserves at least 15/16
full arm and 42/50 full base passes, and strictly improves at least one of those
full-pass counts. Report all failures and numeric jerk/error changes even when
a count stays unchanged. If neither improves, diagnose the specific failed
trajectories and training signal before declaring another bounded recipe.

After these finite cases are mastered, expand nearby reachable poses, base
headings, goal changes and disturbances, then coupled and continuous loaded
TAMP trajectories. Generate new audit cases before claiming generalization.
Verify real velocity-command framing, limits, watchdog, measured stop and
telemetry-only integration before a reviewable, bounded Herman motion test.
No simulator score alone authorizes uncontrolled hardware motion.

## Refinement results and critic calibration experiment (November 6)

The fixed native and scheduled finals both improve expanded base tracking to
50/50. Native achieves 43/50 full base and 11/16 full arm; scheduled achieves
40/50 and 2/16. Both retain 16/16 arm tracking and clear the recorded safety
flags, but neither preserves the baseline's 15/16 full arm passes. Retain the
source. All checkpoint results and failures are in the
[refinement report](../experiments/mars_refinement_20261106.md).

The source also fails all four coupled cases (teacher 4/4), so separate base and
arm success cannot qualify it as a useful whole-body controller. Keep these
failures visible while repairing the simpler distribution first.

An on-policy audit of the native refined actor, using its saved exploration and
8-tick resampling, finds reset-value residuals of about 25 on a base hold and
67--69 on two arm trajectories. These include SB3's time-limit bootstrap. Two
exploration seeds per case are diagnostic evidence, not a calibrated estimate
of broad performance. Critic underfit is a testable explanation for poor policy
updates, not an established cause of the smoothness regression.

Next bounded experiment, before any further reward change:

- Restart from the retained source, same 80/20 base/arm control-step mixture,
  coefficient std 0.1, seed 90, actor LR 3e-5 and native reward throughout.
- Train for 24,576 transitions: 8,192 critic-only transitions, then 16,384 PPO
  actor transitions. Freeze both actor means and noise during critic warmup;
  verify exact tensor identity at its boundary and evaluate physical behavior.
- Use a separate critic LR multiplier of 30 (effective 9e-4). During critic
  warmup use 20 epochs and GAE lambda 1; restore five epochs and lambda 0.95
  for actor learning. Leave gamma, rollout/batch size, clipping and KL unchanged.
  The optimizer must preserve its role ratio when SB3 applies LR schedules.
- Require fresh full regression receipt and teacher 66/66 under the new source.
  Evaluate canonical 18 at zero, every 4,096 and final; expanded 66 at final.
  Repeat the noisy value diagnostic at the warmup boundary and final. Do not
  select a checkpoint by repeated validation scores.
- Use the same physical nonregression/strict-improvement selection gate above.
  This changes training only; the exported inference actor, target interface,
  limits, measured feedback and executor safety remain the same.

If this does not improve the fixed final, examine task reward/value scale and
arm-error-to-base coupling before adding more training time. If it does, add
coupled goals, mask combinations and dynamic target changes, then loaded TAMP
plans and passive Herman interface validation. Hardware readiness additionally
requires verified SI feedback, freshness, odometry/twist, watchdog and measured
whole-body stop on the actual robot.

## Coupled-goal curriculum and critic units

The critic-warmup final preserves 49/50 base tracking and 42/50 full base but
regresses to 5/16 full arm. Reject it; see the
[complete report](../experiments/mars_critic_calibration_20261106.md). Its critic
improves base-hold calibration while saturating on arm states. A fixed-target,
fixed-batch value regression improves native-unit audit MSE from 791.57 to
108.50 when rewards/critic units are scaled by 0.01. This is a training
conditioning result, not improved control.

The coupled teacher-base/learned-arm diagnostic passes 4/4, while removing the
heading selector's hand exclusion still gives 0/4. Direct hand-to-base coefficient
restoration recovers 14/16 arm cases, below the retained source's 15/16. Do not
deploy either ad hoc ablation. Add the missing coordinated data while preserving
rehearsal, rather than requiring perfect primitive jerk before exposing the
controller to simultaneous goals.

Next fixed final, starting from the same retained source:

- 36,864 transitions: 4,096 critic-only warmup (20 epochs, lambda 1), then
  32,768 ordinary PPO transitions (five epochs, lambda 0.95).
- Base/arm/coupled control-step budgets 40/20/40. All base coefficients remain
  trainable; no new hand-to-base mask, teacher action loss or runtime filter.
  Learned arm/context/grip and normalization stay fixed for this diagnostic:
  the teacher-base intervention shows that the preserved arm can execute these
  four coupled references. Unfreeze arm/context later if evidence requires it.
- Scale every training reward by constant positive 0.01, without clipping or
  changing reward shape. Rescale the initial value head into these units.
  Actor outputs, observations, physical limits and native evaluation are
  unchanged. Save critic units in bundles and restore native units when a
  generic trainer transfers them. This critic is not an infeasibility certificate.
- Actor LR 3e-5, critic multiplier 10 (3e-4), coefficient std 0.1, seed 90.
  Keep gamma 0.9975, batch/rollout 256, eight-tick resampling, clip 0.1 and KL 0.01.
- Require fresh source-bound full regressions and teacher 70/70 (50 base,
  16 arm, four coupled). Initial source fidelity now covers all 70. Record its
  coupled failures; initial fidelity does not grant qualification.
- Evaluate canonical 22 at zero, warmup boundary, every 8,192 and fixed final;
  expanded 70 at final. Require exact warmup physical replay and repeat noisy
  value audits in native units at warmup and final. Keep all results.

A development candidate must preserve source base/arm tracking and full-pass
counts, have zero recorded safety events, and strictly improve at least one
stage's full-pass count. Passing any subset of coupled cases is progress, not
completion. All 70 physical gates, dynamic masks/goal changes, load/disturbance
coverage and integrated TAMP/agent execution remain required before simulator
qualification, followed by actual Herman interface and bounded-motion gates.

## Gain-bias optimizer units: matched follow-up

The normalized coupled final is rejected: base 49/50 tracking, 38/50 full;
arm 16/16 tracking, 5/16 full; coupled 0/4 tracking/full. Fresh regressions pass
1,041 and teacher 70/70; initial full transfer and warmup are physically exact.
See the [complete report](../experiments/mars_coupled_units_20261106.md).

General yaw gain biases move from approximately 0.00457 to 0.00651/0.00699,
despite 32,768 actor transitions. The restoring-turn head's biases remain near
2.0, while context readout changes can affect the arm's base reactions. Test
gain-parameter learning units before adding policy structure or another long
unchanged run.

Repeat the preceding 36,864-transition recipe from the same retained source,
changing only `--gain-bias-lr-multiplier 1000`: effective gain-bias LR 0.03,
other actor readout/noise LR 3e-5 and critic LR 3e-4. The actor context stays
frozen. Keep the same seed, 4k warmup,
40/20/40 step budgets, reward scale 0.01, exploration, evaluation cadence and
fixed final. The optimizer preserves all LR ratios under SB3 schedules and
reload, and frozen arm rows remain unchanged. Log the full base gain-bias
vectors per rollout. Initial mean action, inference architecture and safety
limits are unchanged. Fresh full regressions and teacher 70/70 are required.

Apply the same physical nonregression selection gate and retain every failure.
Do not select an intermediate validation checkpoint. Quarter-amplitude training
parents pass 2/2 under the source; half and three-quarter parents fail tracking.
This establishes an easier curriculum starting point for a later experiment,
without changing any canonical goal or threshold.
