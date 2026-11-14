# Contract-aligned Mars control curriculum

The November 1 mixed pilot improves base tracking to 48/50 but passes only
15/50 full base cases and 0/16 full arm cases. Its arm weights remain fixed,
yet new base outputs respond to hand errors and alter physical hand behavior.
Use the successful pre-PPO arm transfer as the source, not that regressed actor.

## Reward alignment

`mastery_objective.py` shares the evaluator's force and jerk budget calculation:
each budget is the maximum of its physical floor and 1.25 times the paired
teacher's peak measurement. Teacher case identities, reset states and reference
hashes must match before a budget can be used. These measurements are training
labels, never policy observations or online controller inputs.

Keep native dense progress, pose tracking, quiet-hold, command smoothness and
physical costs. The optional contract reward adds:

- A control-tick jerk excess cost integrated over physical time. For each jerk
  group use `h(r) = sqrt(1 + max(r - 1, 0)^2) - 1`, where `r` is measured jerk
  divided by its acceptance budget. Template weight 1.0; first-pilot weight 0.1.
- An incremental peak excess cost across force and jerk groups, template weight
  2.0 and first-pilot weight 0.2.
  Charge only newly accumulated excess; do not integrate a stale maximum.
- A terminal full-contract bonus of 10. Remove the native first-settle bonus.
  Award the replacement only when the complete episode passes tracking,
  reference completion, quiet duration, drift, force/jerk and safety gates.
  A later failure therefore cannot retain an earlier completion bonus.

Weights are explicit CLI options and are saved in the run and exported bundle's
training recipe. Base-only and general mastery trainers both support them.
The wrapper preserves observations, actions, references, physics and stopping
behavior. Evaluation retains the original acceptance thresholds and native
reward trace, so cross-checkpoint return comparisons stay on a common objective.
Training returns and their component costs are recorded separately.

## Bounded first comparison

Before optimization, require a fresh full regression receipt, teacher 66/66 and
the source actor's initial 16/16 physical arm passes. Verify the teacher's native
metrics and rewards reproduce the historical teacher; audit the shaped reward
on admitted teacher and zero-base-request trajectories before using it for PPO.

Run two sequential seed-90 pilots from the same source, each with 8,192 base-only
transitions and evaluations every 2,048 steps:

1. Existing reward, base-only sampling.
2. Contract-aligned reward, identical base-only sampling and PPO settings.

Before either pilot, four low-amplitude teacher-plus-oscillation training-case
probes expose excessive initial soft costs: one 0.5 mm/s perturbation retains
about +20 native return but receives -59 with template weights. Calibrate both
new soft weights to 0.1 of template strength, giving that fixed trace about +5
against +20 for smooth completion. The full-contract bonus remains strict.
Record sensitivities at 0.05, 0.1, 0.25, 0.5 and 1.0 multipliers; this selection
uses training-case diagnostics, not trained validation checkpoints.

Keep actor context, arm/grip rows and normalization fixed. Retain existing held
arm postures, hold/forward/reverse/left/right cases and 80% translation transition
share. Use the existing coherent exploration, LR 1e-4, gamma 0.9975, rollout and
batch 256, five epochs, clip 0.1 and KL target 0.01. One CPU14 world, one
Torch/BLAS thread, hidden CUDA, offline scalar W&B. Each final evaluation covers
all 66 expanded cases, including moving-arm regressions. Keep every checkpoint
and failure; do not select one using validation or extend a run automatically.
This isolates the reward package within the base-only distribution; it does not
identify the independent contribution of each new reward term.

## Gated progression

| Phase | Reset and goal distribution | Training and advancement |
| --- | --- | --- |
| Base fundamentals | Held arm across existing postures; base holds, forward/reverse and turns, followed by quiet holds | Train base rows/critic with the arm function fixed. Require canonical and expanded full passes, not only tracking. |
| Moving arm, constrained base | Start with 80% base and 20% moving-arm transitions, then expand the hand share after validation | `--training-stages base arm --arm-step-fraction 0.2`; preserve the arm function initially and retain base regressions. Teach quiet base behavior while the hand moves. |
| Coordinated motion | Retain previous cases; add coupled base/hand paths, unconstrained-base hand goals and mid-motion goal changes | Use the general mastery trainer and a qualified source actor with fresh optimizer, low randomization and reachable FK/IK witnesses. Validate the actual mask semantics; an inactive base target must permit hand-driven base motion. |
| Loads and disturbances | Retain ordinary tracking; add windowed pushes, payload variation, grasp-to-lift, carry, placement and public TAMP trajectories | Require separately admitted teacher references, retention and wheel support throughout continuous execution. Expand randomization only after full gates pass. |

Later phases do not start merely because a training budget expires. If a phase
fails, report position/orientation, settling, drift, per-axis jerk, force,
wheel unloading, requested/measured motion and reward terms. Inspect feedback
signs and reward tradeoffs before increasing scale. Runtime remains one actor.
Herman testing remains downstream of simulator qualification.

## Reproduction

Generate fresh current-source preflight and regression receipts first, then:

```bash
python -m emet.rl.manipulation.base_component \
  --source logs/eval/mars-command-history-v1-transfer/actor \
  --preflight logs/eval/mars-base-curriculum-v1-teacher/report.json.gz \
  --regression-receipt logs/recovery/hand-accuracy-v1/base-curriculum-full.receipt.json \
  --training-stages base --steps 8192 --evaluation-every 2048 \
  --contract-reward --jerk-excess-weight 0.1 --peak-excess-weight 0.2 \
  --completion-bonus 10 --wandb-mode offline \
  --output logs/manipulation/mars-base-curriculum-contract-v1-seed90
```

Omit `--contract-reward` and weight overrides for the legacy-reward comparison,
using a distinct output directory. Run with the bounded CPU/thread settings
above. Generated models, full traces and W&B files remain in ignored `logs/`.

## Completed first comparison

The [complete report](../experiments/mars_contract_curriculum_20261103.md) records
1,005 passing regressions and a reproduced 66/66 teacher. Native-reward base-only
training reaches 49/50 base tracking, 42/50 full base and 15/16 full arm passes;
the aligned reward reaches 48/50, 41/50 and 12/16 respectively. Both retain
16/16 arm tracking. Initial metrics and final teacher/reset identities match.
No later phase starts automatically and neither actor is promoted.

The aligned exploratory rollout pays no full-contract bonuses after removing
50 points of native completion credit. Keep the native curriculum as baseline
and test a gradual reward transition plus quieter refinement exploration next,
with unchanged acceptance gates. Explicitly verify source-transfer fidelity
when continuing a partly qualified checkpoint, rather than bypassing admission
or weakening the final arm/smoothness gates.
