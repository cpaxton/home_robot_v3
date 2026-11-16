# Prospective native TAMP layouts and bounded recovery

PR #242 is frozen at `855348a2`. This evaluation lives on
`feat/tamp-prospective-evaluation`; it changes the evaluator, not the native
motion policy, robot physics or acceptance thresholds.

Before any live run, commit [the complete six-case protocol](../../configs/eval/tamp_prospective_20261116.json),
compile every fixture and validate its task/model signature. Freeze that clean
source for all native and agent launches. No failed case may be replaced,
excluded or silently retried. If evidence informs a fix, retain the entire
original result and call subsequent replays development, not held-out evidence.

Four new layouts combine XY shifts and cube yaw that were not executed during
PR242 development. They retain the two-goal and buffered-swap families, canonical
robot/contact settings and original destination regions. This is a small
prospective within-family test, not a new environment/robot reliability claim.

| Case | Family | Common X/Y offset (m) | Cube yaw (rad) | Transfers |
| --- | --- | --- | --- | --- |
| prospective-two-goal-1 | Two goals | +0.022 / +0.007 | +0.08 | 2 |
| prospective-two-goal-2 | Two goals | -0.024 / -0.009 | -0.08 | 2 |
| prospective-swap-3 | Buffered swap | +0.0125 / +0.006 | +0.07 | 3 |
| prospective-swap-4 | Buffered swap | -0.0175 / -0.007 | -0.07 | 3 |

Run all four through both native public tools and the actual `run agent` CLI,
using the same frozen motion implementation and remote Caliban model. Keep the
4-task/10-transfer native and agent denominators separate. Require successful
final receipts plus independent whole-footprint/support/stillness goals. Record
initial states, source/fixture hashes, model calls, audits and all failures.
Server-default model sampling and independently evolving startup states remain
limitations; no exact reset-paired certificate is claimed.

Two separate native scripted recovery trials use familiar layouts:

- Move the head through its normal controller after planning. Require
  `stale_task_plan` without actuation, then `unknown_plan` on reuse. A separately
  preauthorized scripted request rediscovers, replans and executes fresh goals.
- After one real, measured completed transfer, inject one no-command admission
  refusal at the provider boundary. Require `partial`, one completed step and
  `failed_phase=preflight`, then a consumed handle. Rediscovery must admit the
  measured supported scene before a fresh request completes the remaining work.

The second fault represents planning admission/budget refusal. It does not
simulate grasp loss, change sensor feedback or qualify held-object recovery.
Neither scripted scenario qualifies the LLM's handling of partial execution;
actual-agent stop handling remains scored separately with the existing explicit
invalid-handle/new-user-turn check. Unexpected failures stop the recovery script.

Use managed CPU-only jobs, at most two simulator bridges: serialized native
CPU0–7 and serialized agent CPU8–13. Keep local GPU/VLM use disabled. Respect
other controller work; do not modify its stage-prefix, actor or training code.
Raw outputs stay in `.tamp-runs/prospective-20261116/`; commit compact results
and update the paper only after every declared run terminates.

## Completed frozen gate

All ten declared episodes completed on clean source `63b85dfa`, with unchanged
PR242 motion-module hashes. Prospective native and actual-agent cohorts each
pass 4/4 tasks and 10/10 transfers. Both native scripted recovery trials pass,
including all five expected boundary checks. The affected suite passes 416 tests
without failures/errors/skips. [Results and limitations](../experiments/tamp_prospective_20261116.md).
No native policy was tuned or failed case replaced during this cohort.
