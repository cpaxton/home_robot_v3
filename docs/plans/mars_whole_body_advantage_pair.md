# Controlled C1 advantage-normalization comparison

The cold 8k pilot does not establish useful tracking. Before widening the
distribution or budget, compare two normalization schemes on basic tracking
and hold. This is a diagnostic experiment, with independent implementation,
performance and hardware-readiness decisions. Keep every original goal and
acceptance threshold; correct experimental code need not wait for perfect
behavioral mastery to be useful.

[Frozen protocol](../../configs/benchmarks/mars_whole_body_advantage_pair_v1.json)
specifies **two runs × 4,096 decisions**, seed90, identical cold actor/critic/
log_std initialization, optimizer, rewards, initial Gaussian std0.1, and zero
randomization. No critic warmup, teacher actions, runtime filter, new actor
channels, model tuning or intermediate selection is introduced.

The only between-run setting is `--advantage-normalization per_head|shared`:

- Historical `per_head` standardizes each head before summing. Positive reward
  coefficients within a head largely cancel, and the sum has variable scale.
- `shared` sums the raw weighted GAE estimates, then standardizes once. GAE
  linearity preserves the specified total-reward priorities. The combined
  actor signal has unit standard deviation unless constant.

This compares both relative weighting and aggregate advantage scale. Same
hyperparameters do not isolate those effects. Log combined advantage std,
per-head raw/contribution scales, measured first-minibatch unclipped actor
gradients, total pre-clip gradient norm, clipping and KL. Per-head gradient
diagnostics exclude entropy and value losses; they are not a decomposition of
all later clipped PPO updates.

## Evidence motivating the comparison

The predeclared initial-state check evaluates twelve unchanged existing cases
without training. Initial zero-head actor full/tracking2/12; the cold8k final
full0/12, tracking1/12; retained source full11/12, tracking12/12. Initial and
reference hashes pair exactly, with no hard safety flag. The actor's gain/grip
heads are exactly zero at initialization; this is a deterministic zero-command
baseline, not a teacher or a function-preserving source transfer.

Across the original32 PPO updates, median tracking/stability advantage
correlation is−0.794. Separate standardization amplifies stability relative to
tracking by a median29.18 based on their raw advantage standard deviations.
Twenty rollouts have zero contact reward, but their median contact advantage
std is0.0359 before normalization. Five update signals have negative covariance
between the summed standardized advantage and tracking, including three with
zero contact reward. These statistics identify a plausible credit/variance
problem, not a measured actor-gradient or isolated causal finding. Cold critic
error and legitimate future-value estimates must not be conflated.

## Data and evaluation

[Basic manifest](../../configs/benchmarks/mars_whole_body_basic_learning_v1.json)
selects only three original training cases: base hold, pitch-up@0.25 and
pitch-down@0.25. Balance base/arm control ticks equally. Preserve each case's
reset seed, FK reference, limits, 3mm/0.02rad hand tolerances and two-second
settling requirement. The original164-case and public TAMP suites remain
unchanged and separate.

Each arm records initial and fixed-final results for all twelve predeclared
evaluation cases. They include the three trained cases, other existing
free-space goals and two supported-contact probes. This is a development
diagnostic, not a new held-out/general qualification suite. Every decision must
advance exactly one control tick; no TAMP/executor-hold time enters this pair.
Record actual case sequence, episode starts/outcomes and ticks. Later on-policy
episode lengths and case exposure may diverge between schemes.

Before PPO, persist actor, all critics, log_std, optimizer and NumPy/Torch RNG
state and hashes. Export the initial actor separately. Require exact agreement
of initial numeric-state hashes and of the first pre-update rollout's
observations, actions, rewards and raw GAE. Preserve the first rollout locally
for independent inspection. Do not require later rollout equality after the
first optimizer update.

## Launch and reporting

Pass math/optimizer learning tests, a fresh source-bound full regression receipt
and the original teacher preflight before either run. Teacher admission remains
general164/164 plus the already declared safe-stop TAMP challenge; no new
performance exception is introduced. Record committed source/protocol hashes,
commands and worker IDs. Use the existing single-thread CPU lanes, CUDA hidden,
offline scalar W&B and one serial training process. Check actual occupancy with
the legacy-controller owner before launch.

Report every fixed-final case with tracking, error, settling, drift, jerk,
force and wheel support separately. A low aggregate full score does not replace
that breakdown. Initial/final comparisons measure this small learning setup;
source comparisons are a reference, not an equal-training-budget ablation.
Both outcomes are retained even if neither works. No automatic retry, extra
seed, budget extension, policy promotion or hardware motion follows this pair.

## Recorded attempt (2026-11-14)

Immutable implementation `e0d8df02`, managed job `20261114_092650_d2cc46`:
the fresh source-bound regression run passes **1,187/1,187**, with zero skips,
errors or failures. Teacher admission again passes 164/164 general cases and
5/6 TAMP pregrasp, retaining the already declared object-motion challenge.

The historical `per_head` control completes **4,096 decisions / 4,096 physical
ticks**, with 16 PPO updates. On the twelve development diagnostic cases, its
initial full/tracking counts are 2/12 and 2/12; final counts are **0/12 and
1/12**. These results concern the small cold-start diagnostic, not the retained
source or a full general-controller qualification.

The `shared` candidate records its initial evaluation and first **256
unoptimized rollout ticks**, then stops **before its first optimizer update**:
`paired PPO settings differ beyond normalization scheme`. Every checked
initial-state and first-rollout hash matches the control, including observations,
actions, rewards and raw GAE. The serialized PPO settings also match apart
from the intended normalization option. The check compares serialized
`advantage_weights` as a JSON list with the current in-memory Python tuple,
which produces a false mismatch.

This is a real orchestration correctness bug, not a behavioral qualification
failure. Preserve the incomplete attempt; **no paired normalization conclusion
is available**. Repair and test the serialized/in-memory comparison before a
declared recovery. No retry, extension or promotion has occurred. The separate
PR #232 adaptation job remains queued because CPU14 has not been released.

The comparison repair is implemented in `29a9fcec`: both PPO setting dictionaries
are compared in the JSON domain, preserving all value checks while accepting
the list/tuple serialization boundary. Tests now exercise an actual candidate
training call against the control's saved `run.json`, verify the pre-update
guard succeeds, and reject changes to every other PPO setting and each paired
state/data field. This repairs orchestration; it does not complete the pair.
The original frozen attempt and its 1,187-test receipt remain historical.

The separate PR #227 reward-clock review also found that executor hold ticks
were summed without intra-transition discounting, despite continuation using
`gamma ** ticks`. That correction must be tested before further general/TAMP
training. It does not change this basic experiment's one-tick, zero-TAMP
distribution. Do not silently overwrite old evidence or reuse a source-bound
receipt for changed training code.

[Compact attempt evidence](../../paper/data/whole_body_control/mars_whole_body_advantage_pair_v1/attempt.json)
records outcome groups, matching hashes and artifact checksums. Raw logs,
checkpoints and first-rollout arrays remain under the ignored
`logs/manipulation/mars-whole-body-advantage-pair-v1/` directory. The original
retained checkpoint remains unchanged; no hardware was used.
