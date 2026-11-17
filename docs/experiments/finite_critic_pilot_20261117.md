# Corrected finite-contract critic pilot

This comparison follows the [actor/reward/critic review](../rl/WHOLE_BODY_ARCHITECTURE_REVIEW.md).
It holds the original reward, actor, exploration and task distribution fixed,
and tests joint versus independently scheduled critic fitting after correcting
finite tracking deadline semantics. The completed previous reward pilot is
historical; its negative physical results are not relabeled as a corrected run.

Frozen source: `a179dceb7335902befdb5108a62024662396c137`, 3,564 packed inputs.
Source hashes/archive record are in
`logs/recovery/finite-critic-a179dceb-source-record.json`.
The corrected wrappers physically stop and score a finite deadline as terminal;
generic collection truncations still bootstrap. Actor-only inference and guards
are unchanged. The new critic fitting option is training-only.

Both arms use retained scoped-feedback v4 actor weights
`46291c71ea4dccd93feaf94d4b0e88f3deb1ee5945667964300f3c9f1c165813`, fresh critics,
seed 107, 16,384 decisions, 1,024 critic-only warmup decisions, three PPO actor
epochs, 64-sample minibatches, 256-decision rollouts, initial std 0.02, actor LR
5e-5 → 1e-5, clip 0.05 and target KL 0.01. Both use the unchanged legacy wrapper
reward and corrected finite-contract targets. `finite_joint` keeps historical
joint value updates; `finite_independent` schedules three full value epochs
after actor optimization, including when actor KL stops early. This changes
critic update scheduling/minibatch ordering, not just a scalar budget. Critic
permutations preserve the actor RNG stream.

The exact twelve training and twelve disjoint audit cases from the preceding
pilot are unchanged. Outer general/TAMP exposure remains 50/50 by physical ticks;
within general it is 80/20 hand/base. TAMP is still unloaded, open-jaw grasp-pose
prefix execution. Fixed final checkpoints are evaluated on the same 24 cases,
existing ten-case coupled suite and six TAMP prefixes. No audit labels enter
training, no intermediate checkpoint selection, and no automatic budget extension.
This does not establish loaded carry, full agent deployment or hardware readiness.

The frozen native admission replays the existing results: teacher 12/12 training,
11/12 audit and 6/6 TAMP. Retained actor is 20/24 new cases, 6/10 existing coupled
and 6/6 TAMP, with all 40 stops confirmed. Known contact failures remain counted.
176 targeted tests pass on the correction and critic isolation. Full regression
job `20261117_002837_d29cc2` and native admission job
`20261117_002841_fc6dd2` are the explicit prerequisites for training job
`20261117_003315_6197ed`. The earlier queued supervisor was replaced before
learning to preserve virtualenv-based viewer restoration. At preparation time,
PPO remains waiting for the complete fresh source receipt.

Exact protocol, cases and runners are under ignored
`logs/recovery/finite-critic-pilot-v1/`; all checkpoints, losses, per-head GAE fit
statistics, native metrics and offline W&B records go under ignored
`logs/manipulation/finite-critic-pilot-v1/`. A successful paired run automatically
checks matched initial states/first actions, produces physical-outcome tables,
and restores the retained-policy localhost viewer on port 8766. Failures abort
learning/comparison and preserve their records; no candidate is promoted.

The first full gate at source `a179dceb` completes with 1,571 passing tests and
two compact test-policy interface failures in the new value-fit diagnostics.
No corrected PPO starts; the prerequisite supervisor aborts and restores the
viewer. The learner now uses the existing `evaluate()` interface as a fallback
when a policy has no separate `values()` query. Both fitting schedules pass
the analytic one-step learning and actor-update isolation checks. This is a
compatibility repair, not a change to production actor inference or rewards.
The failed source/gate/native reports remain preserved; corrected-source jobs
use fresh paths and receipts.
