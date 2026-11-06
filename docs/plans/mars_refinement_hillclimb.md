# Mars whole-body refinement hill climb

Objective: a smooth, accurate, reactive controller that can execute the agent's
task-pose contracts on real Innate Mars. Keep one runtime actor. Retain every
attempt; promote a development candidate only using explicit physical gates.

## Current baseline

`mars-base-curriculum-legacy-v1-seed90/last/actor` passes 49/50 expanded base
tracking, 42/50 full base and 15/16 full arm cases, with 16/16 arm tracking.
Remaining failures: one larger forward reference, seven other base smoothness
misses and one small-amplitude arm jerk miss. No unsafe-contact, wheel-lift or
invalid-physics flag trips. This is a development baseline, not a hardware-ready
policy. Source robot/model, normalization and all acceptance thresholds remain
fixed for the next comparison.

The previous cold-start aligned reward removes 50 native completion points but
earns no full-contract bonuses during exploratory training. Test a gradual
transition and quieter exploration from the stronger baseline instead.

## Continuation fidelity and reward schedule

The default initial arm admission remains strict. The explicit
`--initial-admission source_fidelity` mode supports simulator repair of a known
partly qualified source. First replay the source on all 66 expanded cases under
current code and require its safety flags to remain clear. Then require the
merged actor's canonical and expanded-arm metrics, actions, rewards, physical
poses, stop codes and reset/reference identities to match that fresh source
exactly. Record existing failures without granting qualification. Missing,
duplicate, changed or stale coverage fails before optimization. Final evaluation
and hardware-promotion gates are unchanged.

For fraction `f`, use `native_reward - f * native_bonus - f * excess_costs +
f * terminal_contract_bonus`. At zero this is exactly native reward; at one it
is the previous fully aligned objective. Latch `f` per episode so changing a
global schedule cannot change reward accounting halfway through a trajectory.
Native and terminal completion credit together remain bounded by their declared
mixture. Full qualification always uses the fixed original gates.

## Next matched refinement pair

Use the same 8k baseline above, seed 90, fresh optimizer, frozen arm/context/grip
and normalization. Run sequentially on CPU14, CUDA hidden, one Torch/BLAS thread,
offline scalar W&B. Require fresh source-bound full regressions and teacher
66/66 before either run.

- Each run: 16,384 transitions, 80% base / 20% moving-arm control-step budgets,
  existing base primitive/posture mixture, LR 3e-5 and initial gSDE coefficient
  std 0.1 rather than 0.5. Keep 8-tick resampling, gamma 0.9975, rollout/batch
  256, five epochs, clip 0.1 and KL target 0.01.
- Control: native reward throughout.
- Scheduled: jerk/peak weights 0.1/0.2, full-contract bonus 10, native warmup for
  4,096 transitions, linear transition over the next 4,096, then full alignment.
  Episode boundaries can delay the applied fraction; log the actual fraction.
- Evaluate all 18 canonical cases at step zero, every 4,096 and final. Evaluate
  all 66 expanded cases at final. Score every result; do not select an earlier
  checkpoint using validation. This comparison is exploratory hill climbing on
  development cases, not an untouched generalization estimate.

## Selection and further work

Retain the source unless a fixed final candidate has no safety events, preserves
16/16 arm tracking and at least 49/50 base tracking, preserves at least 15/16
full arm and 42/50 full base passes, and strictly improves at least one of those
full-pass counts. Report all failures and numeric jerk/error changes even when
a count stays unchanged. If neither improves, diagnose the specific failed
trajectories and training signal before declaring another bounded recipe.

After these finite cases are mastered, expand nearby reachable poses, base
headings, goal changes and disturbances, then coupled and continuous loaded
TAMP trajectories. Generate new audit cases before claiming generalization.
Verify real velocity-command framing, limits, watchdog, measured stop and
telemetry-only integration before a reviewable, bounded Herman motion test.
No simulator score alone authorizes uncontrolled hardware motion.
