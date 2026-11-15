# Recover the interrupted C1 comparison

**Completed:** [fixed-final results and limits](../experiments/mars_c1_normalization_recovery_20261115.md).
Shared normalization improves hand endpoints but preserves only the same two
full successes as its initial actor. Fresh checks pass 1,249/1,249; the arm
follow-through has received the CPU lane. The protocol below is the executed
plan, not another requested launch or an automatic budget extension.

Continue only the shared-normalization half of the frozen comparison. The
completed historical control remains 4,096 decisions; the previous candidate
collected 256 ticks but made no optimizer update. Preserve both attempts and
all source identities. The remaining candidate budget is 4,096 decisions,
seed 90, the same three training cases and twelve diagnostic evaluation cases.
There is no additional variant, seed, smoothing layer or promotion decision.

## Source and numerical equivalence

The [fixed recovery recipe](../../configs/benchmarks/mars_whole_body_advantage_recovery_v1.json)
pins the original control file and records every changed teacher-source module
with exact before/after hashes and a reason. Actor, observation, physical
reward, control, profile and reference code are protected. The physics/runtime
and assets must remain unchanged. The existing generic pairing guard remains
strict unless the explicit C1-only recovery proof is supplied.

[The offline proof](../../scripts/prove_mars_c1_recovery.py) checks the declared
source delta, writes the full source diffs, and verifies unchanged ASTs for
ManipulationEnv, MasteryEnv, the factory, GAE and advantage combination. It
replays all 256 saved actions to reconstruct the original optimizer RNG, then
runs the saved first PPO update with original and current implementations in
disposable copies. Every policy and optimizer tensor must match exactly. This
adds no simulator ticks, candidate interaction budget or actor export.

Fresh current-source regressions and teacher preflight still run. The proof
also reads the actual old/new teacher artifacts and requires all 164 unique
cases to retain their reset/reference identities and pass mastery. It binds
both preflight hashes without rewriting either. The current candidate must
still reproduce every original initial numeric-state and full first-rollout
hash before its first update; all PPO settings except normalization must match.
Single-tick rewards remain bitwise unchanged by the bundled-reward repair.

This proof is limited to the declared basic, general-only experiment. It does
not claim equivalence for multi-tick TAMP training, a different distribution or
new robot/runtime versions. It is not a safety or hardware certificate.

## Execution and reporting

Use an immutable committed source snapshot with its own fresh regression and
teacher receipts. Validate the proof independently before any candidate PPO
update. Run one single-thread CPU14 process, CUDA hidden, with offline W&B.
CPU15 is the regression lane; coordinate any handoff with its actual owner and
host-visible child exit. No automatic restart follows a failed check.

Record source/proof hashes, commands, job IDs, attempted/optimized decision
counts and end-to-end time. Keep the old control and failed candidate outputs
unchanged and write the recovered run to a new directory. Compare fixed-final
results on all twelve cases: tracking, settling, error, drift, jerk, force and
wheel support. Both outcomes belong in the report even if neither improves.

After C1 completes and the worker exits, explicitly hand CPU14 to the separately
bounded frozen-base arm experiment. Its previous handoff timeout made zero PPO
steps and remains a failed launch record. Reusing the identical gated snapshot
is allowed; new arm teacher admission and declared initial/final evaluations
remain required. No hardware motion or automatic policy promotion follows.
