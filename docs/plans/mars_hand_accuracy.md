# Mars hand tracking accuracy

First teach the existing pose-conditioned actor meaningful arm corrections on a
small reachable set, retaining base rehearsal. Then add loaded hold/lift, followed
by the public agent pick/place cases. This finite diagnostic does not replace
the broad tracking/manipulation distribution or qualify hardware.

## Evidence and hypothesis

The latest base-trained actor has 50/50 base tracking/holding but 0/8 arm cases.
Its isolated arm requests peak around 0.001 normalized units, below roughly
0.001 rad/s physically. Pitch reaches hit the 10 mm path tube; wrist rolls leave
almost their complete 0.08–0.092 rad target rotation. The teacher executes all
66 original expanded cases, including loaded lifts, under identical geometry
and limits. This is not evidence of universal unreachable targets or a torque
ceiling. Earlier loaded-arm diagnostics likewise isolated under-commanding from
physical tracking lag.

The current expanded sampler has 45 base training cases and only four arm cases.
Uniform case cycling would devote about 8% of starts to the arm. The base-only
reward also gives independent base-hold credit while an active hand is wrong.
Use explicit stage weights and the existing native hand-gated base reward before
adding a new architecture, IK/residual action path or arbitrary action filter.

## First bounded comparison

Preset: `configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json`.

- Preserve every canonical reset/target, continuous FK path rule, 12-second
  horizon and gate. Active hand endpoint remains 3 mm / 0.02 rad, with two
  continuous quiet seconds. Preserve the 10 mm / 0.1 rad path tube, force,
  retention, wheel, speed/acceleration and paired jerk/drift limits.
- Reset selection is 75% arm / 25% base, independent of stage case counts.
  Cycle cases without replacement within each selected stage. Retain all 45
  base training pairs and all four original arm training cases. Add quarter-
  and half-amplitude arm motions, using the same training reset postures and FK
  witnesses. Validation seeds/targets are never included in these additions.
- Enable existing `hand_gated_base_reward`: with an active hand, base credit
  requires pose tracking. Retain dense 20 Hz hand pose/progress rewards,
  measured-jerk and physical costs. Base-only reward semantics are unchanged.
- Warm-start the completed restoring-yaw actor with a fresh optimizer. Preserve
  its observation/history/preview contract and all action channels. No teacher
  action loss, DAgger, residual, new task IDs or privileged joint target enters
  the actor.
- One 65,536-step seed-90 pilot: LR 5e-5, gamma 0.9975, base std 0.003,
  arm/grip std 0.05, 16,384-step training-only command-smoothness warmup and
  scheduled evaluations every 16,384 steps. Independent evaluation always uses
  full command cost. No hard safeguard is scheduled down.
- Use eight native CPU worlds on Caliban only after fresh host-specific source
  regressions and teacher preflight. Aggregate PPO rollout is 2,048 transitions
  (256/world), not the earlier single-worker batch. Record this optimization
  change explicitly; it is not a matched scaling-only comparison. If local
  fallback is necessary, use one CPU14 worker and record the different batch.
- No repeated seeds, automatic extension, validation-selected checkpoint,
  loaded-stage advancement or hardware promotion.

## Intermediate tests and measurements

1. Preserve current-source baseline teacher/actor reports on base, arm, coupled
   and loaded stages, with physical-start and reference hashes.
2. Test stage proportions, deterministic reseeding, within-stage coverage,
   invalid recipe rejection, unchanged canonical/validation contracts, and
   smaller target admission. Run the complete source-bound regression gate.
3. Teacher preflight covers every expanded base/arm training case and original
   validation case. A same-shaped but incomplete preflight must still fail.
4. At step zero, verify warm-start physical metrics against the preserved actor;
   changed returns from hand-gated reward are not learning gains.
5. Each checkpoint scores the original 18 base/arm cases. The final report also
   scores all 66 selected admission cases, preserving per-case failures and the
   separate 50-case base score. Do not hide rehearsal regressions in an average.
6. Report endpoint/trajectory hand position and orientation separately, path
   stops, quiet duration, drift, jerk, force and wheel/retention events. New
   evaluation-only diagnostics compare slewed joint-rate requests, measured
   joint rates, motor-target lag and actuator torque; a measured Jacobian
   projects requested and actual hand velocities toward the current goal.
   These read state only and never modify observations/actions or solve IK.

## Following gates

If small arm tracking fails, inspect whether requests point toward the target,
their amplitude, physical lag, reward components and independent orientation
control before spending a larger budget. Keep every negative result.

After arm/base tracking improves without new safety failures, introduce a
bounded loaded hold/lift share using physically admitted existing resets. Keep
ordinary arm/base rehearsal. Score supported-to-free load transfer, hand pose,
relative object motion and force/unloading separately. Cylinder rotation remains
a separate physics/grasp issue and is not fixed by an empty-hand success.

Then replay the unchanged public agent pick/place programs through TAMP and the
learned-control plugin. A component endpoint, teacher retiming success or base
benchmark cannot substitute for the complete learned-controller delivery.
