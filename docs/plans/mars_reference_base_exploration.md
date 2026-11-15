# Reference-conditioned base exploration: bounded coordination test

The 2026-11-15 continuation authorizes this new experiment after the completed
C1 and frozen-base trials. Their outcomes remain unchanged. This is a fresh
pair, not an extension or recovery of either prior budget.

## Hypothesis and intervention

C1 improves hand following while learning base motion that violates quiet-hold
and jerk contracts. Replacing final base commands with initial zero commands
helps some arm cases but loses pitch orientation. The frozen legacy actor's
velocity result is a separate, small positive single-seed signal. Neither
observation proves that base sampling noise causes the C1 failures.

Test that specific hypothesis in the GRU actor. The opt-in
`--stationary-base-std-scale 0.1` multiplies only forward/yaw sampling standard
deviations when all of the following measured/reference conditions hold:

- The current hand target is active.
- All three base pose coordinates are explicitly constrained, including in
  every current/future preview sample.
- All announced current/future base errors are identical, and announced base
  reference velocity is exactly zero. The past preview sample is excluded.

Free-base goals and upcoming moving references retain normal exploration.
A distant static base target or a moving physical base can still meet this
condition: it describes the announced reference, not a stationary robot.
Deterministic means, action masks and correction authority remain unchanged.
This is not the teacher's arm-reachability criterion or a guarantee that base
motion is unnecessary. Arm/grip standard deviations remain unchanged.

PPO must use the same conditional distribution for sampling, old/new action
likelihoods and entropy. Default scale 1.0 preserves seeded samples and
probabilities. The setting belongs to training state/provenance; exported
single-actor inference has no sampling gate or second controller.

## Fixed comparison

| Setting | Declared value |
| --- | --- |
| Architecture | Existing GRU, fixed scaling and three privileged critics |
| Control/candidate | Base-noise scale 1.0 / 0.1 under the same reference condition |
| Seed | 90 in both runs |
| Budget | Exactly 4,096 decisions each, 8,192 total; no continuation |
| Initialization | Fresh identical zero-head actors, optimizer state and RNG |
| Learning | Shared advantage normalization; LR 3e-4, five PPO epochs, batch128, rollout256 |
| Training cases | Existing basic general-only C1 three-case set |
| Evaluation | All twelve existing C1 cases, initial and fixed final for both runs |
| Unchanged | Observations, actor mean/masks, references, reward, limits and acceptance |

Before the candidate samples its first rollout, bind the current-source control
record and verify identical source, preflight, cases, seed, complete PPO settings,
actor/policy/optimizer hashes, initial observation and RNG. Only the declared
exploration scale differs. First rollouts intentionally differ after sampling;
do not reuse the normalization pair's exact-rollout claim or source bridge.

Record the condition/reduced-noise decision counts per rollout, each learned
log standard deviation, actual case/tick exposure, episodes and optimizer
diagnostics. Compare all initial/final reset/reference identities and every
failure reason, including base-contract stops, hand orientation, actor-active
quiet hold, base-linear/base-yaw/joint jerk and independent safety flags.
Inspect physical traces of failed quiet holds instead of inferring rest from
small hand errors. Native initial/final results use the exported actor means.

## Admission and resources

Freeze the implementation revision before checking it. Obtain a fresh full
source-bound regression receipt and current-source teacher preflight. Preserve
the teacher's diagnosed 5/6 pregrasp outcome and all 164 general cases; do not
claim success on the known object-motion challenge. Both gates must pass before
either PPO run. The two runs share that same immutable implementation/preflight.

Olympia: one CPU14 teacher/training worker; at most one CPU15 check worker;
one numerical thread, CUDA hidden, ordinary rendering disabled. Verify vacancy
using host-visible processes. No GPU, Caliban or hardware motion. Use managed
jobs with a 90-minute preflight/check limit and a separate 60-minute paired
training/readout limit. Fail without automatic retry if either gate or run
fails. A timeout is an outcome, not permission to extend the budget.

## Decision

Complete all fixed readouts and preserve failures, even if the candidate loses.
This is one finite diagnostic seed, not a generalization study or checkpoint
promotion. Keep existing production defaults/source. More seeds, stronger
gates, loaded stages or a different base-authority mechanism are separately
declared work; no automatic sweep, best-checkpoint selection or extra training.

Follow-through and PR status belong in [TODO.md](../../TODO.md#current-wbc-follow-through-2026-11-15).
Add measured outcomes and compact provenance after the actual run finishes.

## Recorded launch (2026-11-15)

[Draft PR #238](https://github.com/cpaxton/home_robot_v3/pull/238) contains the
implementation and 48 passing focused checks. Immutable implementation
`3ac722e1096dc901d68192b803dc9596471c46c7` is checked by managed full-regression
job `20261115_101800_b63738` on CPU15 and teacher job
`20261115_101800_2b8ed4` on CPU14. Pair supervisor
`20261115_102028_c99a87` waits for both successful gates and actual host-visible
CPU14 child exit. It has a separate active budget; waiting does not consume
PPO decisions. This registration records zero trained decisions until actual
rollout/update records appear.

Protocol, exact supervision script, gate receipts, progress and eventual raw
readouts belong in ignored
`logs/manipulation/mars-reference-base-exploration-v1/`. Do not turn a pending
or failed managed run into a successful experimental outcome.
