# Frozen-base/context arm adaptation: bounded follow-through

The original job `20261114_091806_52e91d` timed out waiting for CPU14 and
recorded zero PPO transitions. Preserve that attempt and its software receipts.
The user's 2026-11-15 instruction to proceed authorizes this new bounded launch;
it does not turn the expired job into a completed experiment.

## Fixed comparison

| Setting | Declared value |
| --- | --- |
| Variants | Existing baseline and preview-reference-velocity legacy MLP |
| Seed and budget | Seed90; 16,384 transitions per variant; 32,768 total |
| Source weights | Retained source SHA-256 `e7b425699510cc635485bdd8f59a5b25692fb165faf248aaecbc2707c0dbb883` |
| Source manifest | SHA-256 `fb0df2c1f737017e208d2849a7cc2523aaa3a0d60a73f71871bbc623603345a1` |
| Suite | `configs/benchmarks/mars_tracking_mastery_tamp_postures_v1.json` |
| Training | Arm stage; `--arm-adaptation-only`; LR `1e-5`; arm std `0.03` |
| Frozen function | Actor context, base/grip function, restoring-yaw parameters and normalization |
| Trainable | Arm feedback/feedforward gain rows, arm exploration scales and critic |
| Wheel/grip sampling | Retained deterministic learned means |
| Evaluation | All 122 admission cases: 50 base, 64 arm, eight coupled; own initial and fixed final |
| Runtime | One ordinary eight-output actor; no second policy, teacher residual or new filter |

This is the already implemented legacy-MLP intervention. It does not change
the GRU/multi-critic C1 experiment. Orientation representation, rewards,
references, native speed/force/support/retention limits and acceptance
thresholds remain unchanged. The completed unfrozen seed90 baseline/velocity
controls are historical references with their original source identities;
this does not create a contemporaneous four-arm ablation.

## Admission and resources

Finish the corrected C1 candidate and its fixed-final evaluation before taking
CPU14. Check the owning thread's explicit release and host-visible native child
exit. Do not infer vacancy from sandbox-isolated `/proc` or from a failed
supervisor alone.

Use the same immutable fresh-gated source snapshot as C1 when available, since
it contains the legacy actor and ablation runner too. Verify its exact revision,
source-bound full regression receipt and source-copy hashes. Reusing a receipt
requires identical bound source; an older receipt is not evidence for changed
code. Obtain the runner's own matching 122-case teacher admission for each
feature variant before its PPO run. Keep the original teacher budgets.

Use CPU14, one environment, one numerical thread, CUDA hidden and rendering
disabled. CPU15 is for any required checks. No local GPU, Caliban qualification
bypass or hardware motion is part of this trial. Start the active runtime budget
after the actual training-lane handoff, rather than consuming it during another
experiment's reservation. Record source/protocol/script hashes, managed job ID,
commands, actor source copies and actual transition progress in ignored logs.

## Readout and completion

Preserve each initialized actor and evaluate every declared initial/final case.
Check reset/reference identity pairing and compare each final with its own
anchor. Report training and validation partitions, tracking/settling, full
passes, position/orientation error and separate base-linear/base-yaw/joint jerk.
Do not conceal a full-contract failure behind accurate hand tracking.

Assert protected tensors at the existing rollout/checkpoint/final boundaries.
Reload the serialized initial and final bundles and verify the protected
function remains bitwise identical, while recording whether arm and critic
parameters changed. Parameter preservation alone does not establish physical
nonregression on changed closed-loop observations.

The experiment completes when both fixed runs and their declared readouts are
recorded, including failed outcomes. Keep the source unless a candidate passes
its declared behavioral gates. No automatic retry, extension, extra seed,
intermediate checkpoint selection or policy promotion follows this pair.

Record results in the [experiment report](../experiments/mars_reference_velocity_prototype.md)
and update the [ordered TODO](../../TODO.md#current-wbc-follow-through-2026-11-15).
Actual run identity and results are added after handoff; this protocol alone
does not claim a running job or completed PPO.
