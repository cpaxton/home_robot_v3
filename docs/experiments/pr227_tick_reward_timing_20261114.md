# Consistent reward timing across bundled executor ticks

The TAMP primitive environment can run several executor-owned hold/stop ticks
between actor decisions. Previously it returned their undiscounted reward sum,
while grouped GAE discounted the next value by `gamma ** elapsed_control_ticks`.
A delayed reward, including completion credit, therefore had a different weight
than the same reward collected one physical tick per decision. This affected
mixed general/TAMP training; one-tick general-only training did not have this
timing discrepancy.

## Correction

The environment now preserves the three reward groups for every physical tick
in `info["reward_group_ticks"]`. It adds terminal success/failure credit to the
last tick, matching the timing of a one-tick general environment. A failure
before any actor-applied tick retains the existing one-tick transition convention.
The environment's scalar reward, `reward_groups` totals and evaluation outcomes
remain undiscounted.

The collector uses the learner's gamma to construct the training reward for a
decision spanning `k` ticks:

```text
decision_reward = sum(gamma**j * tick_reward[j] for j in range(k))
value_bootstrap = gamma**k * next_value
```

Thus terminal credit at the final tick has weight `gamma ** (k - 1)` within
that decision. GAE retains its existing termination, timeout and trace handling.
The collector rejects multi-tick transitions without individual rewards or with
inconsistent totals, and PPO rejects a rollout prepared with a different gamma.

The single-tick path preserves existing float32 reward bytes, including signed
zeros. `reward_discount` is rollout metadata checked by PPO; gamma is also
recorded in `run.json`'s PPO configuration. The numeric first-rollout archive
and hash schema exclude that metadata, preserving their one-tick array contract.
Run metadata explicitly names the reward timing. PPO's `raw_reward_mean/std`
describe discounted training groups before advantage normalization, whereas
environment/evaluation totals remain undiscounted.

## Validation and limits

The targeted suite passes **61/61** in **31.27 seconds**, using CPU15 with CUDA
hidden, rendering disabled and one numerical thread. It covers the new timing
tests plus whole-body v2, multi-critic objective and basic-learning tests:

- Bundled versus per-tick returns with a 49-tick terminal transition, three
  discounts, success/failure credit, and terminal or timeout bootstrapping.
  Return equivalence is tested with lambda one; it is not a claim that skipping
  intermediate critic evaluations preserves every lambda-return estimate.
- Bitwise single-tick rewards, invalid timing rejection, explicit collector
  gamma, learner/collector agreement and a real PPO update.
- Native successful pregrasp through executor-owned terminal hold/stop, checking
  every tick reward, raw totals and the final-tick success bonus.

Ruff, formatting and whitespace checks pass. This is a correctness repair,
not a new training result. Historical runs and their source-bound evidence
remain unchanged; old full regression or teacher receipts do not qualify the
modified training source. Fresh admission is required before further training.
