# Contextual TAMP retry through the actual agent

The corrected stop-only control passes and the contextual retry completes its
task. This is a two-object, privileged simulated Stretch development diagnostic
using remote Caliban, not observed or hardware acceptance. No controller,
collision, contact, retention or goal threshold changes accompany this work.

The actual `emet.app.run_agent` CLI receives the original two-goal instruction.
An evaluator-only wrapper refuses admission for step two after a real measured
cylinder transfer. After verifying partial feedback, supported/settled objects,
no additional commands and exactly one no-command evaluator fault, the recovery
trial receives a new stdin turn containing only:

> Please retry the remaining work.

The harness does not restate the goals or insert tool calls. The stop-only
control receives no second request and must remain incomplete. The task source,
protocol and model are frozen before each pair. Both pairs use the same declared
[protocol](../../configs/eval/tamp_contextual_recovery_20261116.json).

| Frozen source | Stop-only control | Contextual retry task | Audited commands | Remote model calls | Affected tests |
| --- | --- | --- | --- | --- | --- |
| `0e4e3959` | Pass; task incomplete | Fail; cube unmoved | 546 | 10 | 482 pass |
| `ebbdb1d9` | Pass; task incomplete | Pass; both original goals hold | 799 | 12 | 485 pass |

All four CLI processes exit cleanly, with zero forbidden commands and zero
shutdown warnings. There are no test failures, errors or skips in either frozen
source's broader regression run. Pre-freeze focused tests caught two fixture/
assertion mistakes; those logs and a status-helper invocation error before job
creation are retained separately. They are not extra robot trials.

## Failure and correction

The first source supplies bounded historical task context, excluding tool plan
handles and sensor payloads. The model remembers the original goals but bundles
discovery, planning and execution, inventing `plan_123` before seeing the real
planner response. Discovery and a real one-step plan succeed; the invented
handle is rejected with `unknown_plan`. No cube command is sent. This failed
attempt remains in the report and is not rescored after the correction.

The corrected CHAT loop dispatches only the first proposal from an all-TAMP
batch and returns its actual result to the model. Later proposals are logged as
deferred, never executed or queued automatically. The prompt also explicitly
requires one call per round and copying actual returned handles. Failed,
partial and discovery-only results still end that user turn. Mixed capability
batches retain their previous stop behavior.

On the corrected contextual trial, the new turn calls exactly `scene_tasks`,
`plan_task`, then `execute_task_plan`. The model requests only the remaining cube
goal and chooses beam search; the cylinder's current location is protected by
the task compiler. Planning returns one step and a fresh handle. Execution
verifies that one cube transfer, then independent final checks confirm both
objects remain in their original goal regions, supported and settled. The model
reports that both tasks are complete, consistent with those checks. The corrected
live pair proposes one call per round; repeated bundling and first-call failure
are exercised separately in real-loop mocked regressions.

Partial feedback correctly names the completed cylinder, unfinished cube,
preflight refusal and stop. The receipt is more explicit than the prose:
`execution_started=true` for prior task progress,
`failed_step_execution_started=false`, `completed_steps=1`,
`failed_phase=preflight`, `plan_consumed=true`, `goal_verified=false`.
The model does not claim overall success at this boundary. This is manual
text review against receipts and independent physical predicates, not an
automated semantic score. Verbatim reports and both source histories are in the
[compact JSON record](tamp_contextual_recovery_20261116.json).

## Artifacts and limits

Raw runs are under `.tamp-runs/contextual-recovery-20261116/` and
`.tamp-runs/contextual-recovery-serial-20261116/`. The JSON record includes
relative artifact paths, hashes, receipts, predicates, model text and test
counts. Corrected jobs are `20261116_100322_b36902` (stop-only),
`20261116_100323_ef6a13` (contextual retry) and `20261116_100325_0eb8b6`
(regression). Both live sources were clean at launch.

These are familiar layouts and development repeats, not new held-out instances,
an exact reset-paired causal comparison or a broad reliability estimate. The
refusal is injected planning admission, not physical solver exhaustion, contact
loss or sensor failure. Unknown dispatch, held-object/grasp-loss recovery,
observed grounding, navigation, exploration and hardware remain unqualified.
Stop-only correctness is never counted as complete-task success. PR #243's
source and historical evidence remain unchanged; this is the separate
`feat/tamp-recovery-dialogue` follow-up.
