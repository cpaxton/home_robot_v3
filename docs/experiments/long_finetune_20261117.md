# Longer ordinary-reward fine-tuning

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
