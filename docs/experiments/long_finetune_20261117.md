# Longer ordinary-reward fine-tuning — complete

The noise and hand-priority probes do not improve the combined accuracy/contact
result. Both short runs expose each hand trajectory twice and each base case
once. This experiment tests longer exposure with smaller updates before changing
more reward terms or actor structure.

Start from the completed ordinary-noise actor/critic/Adam checkpoint at 16,384
decisions, actor SHA `e87a8e747136f2ea511ae7884f43bf0eb1966b146118fd0682c1148a91e96bfa`.
Verify the training checkpoint checksum, exact loaded actor/state tensors and
same source-bound 1,585-test receipt. Preserve critic fitting, original reward,
ordinary learned Gaussian exploration, task distribution, references, acceptance
and physical guards. No new teacher loss or hardware use.

Train **65,536 additional decisions**, actor LR **1e-5 → 1e-6**, critic LR 0.001,
three independent critic epochs, rollout 256, batch 64, clip 0.05 and target KL
0.01. There is no fresh-critic warmup: all critic/Adam state is restored. Physics,
history, scheduler/case cursors and RNG restart explicitly with seed 108. This is
checkpoint fine-tuning after reset, not an exact environment continuation or a
fresh paired-initialization comparison. The initial state-restoration checks
pass before updates. Track new decisions separately from the source's 16,384.

The outer general/TAMP control-tick ratio remains 50/50, with 80/20 hand/base
inside general. Cases are unchanged; more repeats do not establish posture or
reset diversity. Export every 8,192 decisions and retain crash checkpoints under
ignored logs. Evaluate only the fixed final actor against the source ordinary
policy on identical 24 development, ten existing-coupled and six unloaded TAMP
cases. No intermediate checkpoint selection, automatic extension or promotion.
Baseline executions are reused, not independent fresh trials. Previously inspected
audits are development evidence; fresh confirmation remains necessary.

Source `9d0620f9` is restored from its verified archive under ignored
`logs/runtimes/hand-priority-9d0620f9/`. Job `20261117_083406_6c3285` uses one
CPU14 native worker, one numerical thread and hidden CUDA. Offline W&B ID
`6vuul7j4`; outputs are `logs/manipulation/long-finetune-v1/`, runners/protocol
`logs/recovery/long-finetune-v1/`. The retained viewer is paused and automatically
restored after completion/failure. The run is bounded to two hours plus final
readout, with its supervisor bounded longer to permit cleanup. All failures and
stop outcomes remain recorded. No learned controller is newly qualified.

A saved-state audit at 7,936 additional decisions verifies all 31 optimizer
parameter clocks equal the source clocks plus actual new actor/critic updates
(364/372). The source clocks are 624 actor / 768 critic steps. This catches
optimizer reinitialization without claiming exact physical-state continuation.
Compact producer/result are in `long_finetune_v1/`; live progress remains in
ignored run/updates logs.

A separate [fresh confirmation set](fresh_tracking_confirmation_20261117.md)
freezes 24 new cases while training runs. Its CPU15 worker evaluates teacher
and source, then only the fixed final continued actor. These cases never enter
PPO; no training settings change in response to their readouts.

## Completed results

All 65,536 additional decisions finish: cumulative lineage 81,920 decisions,
77,452 new physical ticks. The fixed final policy is evaluated without selecting
an intermediate checkpoint. Source-bound regressions remain 1,585 and the paper
build succeeds. The retained viewer is restored; no PPO job remains active.

On reused development cases, ordinary → continued is **20/24 → 19/24**;
existing coupled stays **7/10**, unloaded TAMP stays **6/6**. The same three
contact failures remain; worst endpoint error improves 13.10 → 12.37 mm but
worst yaw jerk rises 53.36 → 70.64 rad/s³ and peak contact load 34.96 → 42.56 N.
All 40 new candidate stops confirm. Reused baseline rows are not fresh trials.

The independent [fresh confirmation](fresh_tracking_confirmation_20261117.md)
is more encouraging: **12/24 → 16/24**, safety flags **4 → 1**, wheel lifts
**2 → 0**. All initial states/configuration/reference hashes match across teacher,
source and final policy. Teacher is 17/24 with two flagged episodes. Every one
of the 24 generated cases executes; none is filtered or fed into PPO.
Peak linear/yaw jerk still rises **14.88/65.57 → 28.68/78.54** in SI units.
This establishes a scoped generalization gain, not a qualified hardware policy.

The gain is concentrated in free-base hand and base-only cases. Two-stage goal
changes remain **0/4** (source 1/4, teacher 2/4), motivating analogous new training
cases. The maintained [mixture ledger](../rl/TRAINING_MIXTURE.md) distinguishes
current 40/10/50 exposure from the planned 30/10/10/50 allocation. Reward, guard
and reference contracts are not relaxed to improve success counts.
