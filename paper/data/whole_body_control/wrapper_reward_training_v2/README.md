# Wrapper reward pilot v2 producers

Frozen runtime: `5f0eb3c2`. These small text files preserve the exact protocol,
cases and runner. See [the pilot report](../../../../docs/experiments/wrapper_reward_training_20261116.md).
The executable runner uses the declared ignored `logs/recovery/` and
`logs/manipulation/` paths; copy protocol/cases/runner to the declared recovery
path in the frozen runtime to reproduce, then run `run.py admission` followed
by `run.py train` with a fresh complete regression receipt. The native runs use
CPU14 and single numerical threads with CUDA hidden and offline W&B.

The legacy-reward control and contract-reward arm share initial actor, critics,
seed, distribution and budget; the runner checks initial state/first actions.
No raw traces, compressed archives, credentials or weights belong here.
Preparation and passing teacher cases are not policy-training outcomes.

Both fixed runs and all final readouts are complete. `physical-comparison.json`
is a compact physical-metric projection of the ignored 2.3 MB native report,
with its full-report SHA-256. `physical-comparison.svg` excludes inactive hand
errors and marks safety stops. Run metadata, optimizer diagnostics and all
producer/evidence hashes are included. No checkpoint is promoted: retained,
new reward and legacy PPO score 20/24, 18/24 and 19/24 on new cases. All retain
6/6 unloaded TAMP prefixes; legacy improves existing coupled 6/10 to 7/10.
