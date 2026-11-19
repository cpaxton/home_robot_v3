# Actual-agent partial execution feedback — November 16

The actual agent reports partial progress correctly and stops. After a separately
authorized request, it rediscovers the scene, obtains a fresh one-transfer plan
and completes only the remaining object. These are privileged simulated Stretch
trials with remote Caliban, using an explicit evaluator-only admission refusal.
They do not simulate grasp loss or qualify held-object recovery.

| Frozen source | Stop-only feedback/control test | Authorized-recovery test | Affected tests |
| --- | --- | --- | --- |
| `d9fb0273` | Pass; task intentionally incomplete | Fail after correct partial feedback: discovery rejected configured robot selector | 458 pass |
| `d2b24610` | Pass; task intentionally incomplete | Pass; one remaining transfer, both final goals satisfied | 468 pass |

Both complete declared pairs are preserved. The stop-only test passing is not a
full-task success: only the cylinder goal is satisfied. The recovery test requires
both final physical goals and a successful fresh execution receipt. There are no
forbidden commands, CLI infrastructure failures or shutdown warnings in these
four episodes. One earlier launch registration failed before any job/episode
started because the wrapper environment was omitted; it is recorded separately.

## What the agent receives

Previously, the public receipt supplied counts and a step index without naming
which action completed. Planning now returns public-role `step_details`;
execution returns the verified `completed_actions`, `failed_action`, remaining
`unverified_actions`, `plan_consumed` and explicit goal-verification status.

Relevant fields from the actual partial receipt are:

```json
{
  "status": "partial",
  "code": "task_motion_budget_exhausted",
  "data": {
    "execution_started": true,
    "completed_steps": 1,
    "planned_steps": 2,
    "failed_step_1based": 2,
    "failed_phase": "preflight",
    "failed_step_execution_started": false,
    "plan_consumed": true,
    "goal_verified": false,
    "completed_actions": [{"step_1based": 1, "action": "transfer", "object_name": "red cylinder", "from_region": "cube goal", "to_region": "cylinder goal"}],
    "failed_action": {"step_1based": 2, "action": "transfer", "object_name": "blue cube", "from_region": "outside", "to_region": "cube goal"}
  }
}
```

Overall execution has started because the cylinder moved. The failed cube step
has not dispatched. The complete receipt also lists that cube action as
unverified. Effects enter `completed_actions` only after measured verification;
these describe history, not a guarantee against later disturbance. Indices are
local to each plan, so the subsequent one-action recovery plan starts at step 1.
Only public role/region names are described; private operator refs/body IDs are
never serialized. Unknown exceptions retain possible dispatch, rather than
claiming the failed step did not start.

The fixed-source model's verbatim partial report is:

> The red cylinder has been successfully moved to the cylinder goal. However, the blue cube could not be moved to the cube goal due to a motion budget exhaustion during preflight. I stopped the task to avoid further issues. Let me know if you'd like me to retry or adjust the plan.

Manual comparison against the receipt and independent physical scoring confirms
that it names the completed object correctly, leaves the cube unfinished and
reports a stop. It does not claim full success. The reason is an explicitly
injected admission refusal, not a naturally exhausted solver budget. The machine
receipt precisely identifies no dispatch at the failed step; the prose identifies
its preflight stage.

## Verified stop and recovery boundary

The evaluator invokes the actual `emet.app.run_agent` CLI via a bootstrap that
replaces only the native provider constructor with `RefuseAfterOneTransfer`.
All motion, planning, prompts, dispatch and feedback run through production code.
The first request does not reveal which action will be refused.

After the forced final reply, the evaluator independently verifies one satisfied
goal and two supported, settled objects, the one-shot no-command refusal record,
and a quiet actuation audit for one simulation second. No subsequent tool is
dispatched before a new user turn. Only then does the recovery condition receive
a new stdin request restating the original goals. The stop-only condition ends
without reauthorization and remains incomplete.

The first recovery exposes two real integration defects: an unavailable inspection
tool proposal followed by rejection of `scene_tasks(robot="stretch")` by the bound
native session. The fix accepts its own configured robot, advertises `scene_tasks`
as task-state inspection and stops mixed TAMP batches at unavailable tools. It
does not alias invented tool names or relax object-filter admission. Both trials
are repeated on the new source with unchanged user requests.

In the fixed replay, the recovery request dispatches `scene_tasks`, `plan_task`,
and `execute_task_plan` with a fresh handle. That receipt verifies exactly one
completed action: moving the blue cube. Both original goals remain independently
satisfied. The agent subsequently performs additional read-only rediscovery and
a zero-step plan for the already satisfied cylinder goal. No duplicate physical
transfer occurs, but this is not perfect tool-call efficiency. The model's final
statement that both goals are satisfied is supported by the measured state.

## Validation and limits

All **468 affected tests pass**, zero failures/errors/skips, on `d2b24610`
(job `20261116_083517_25e5e2`). Tests cover named verified prefixes, failed-step
possible dispatch, no leaked private refs, immutable stored descriptions, matching
robot selectors, rejection of unknown robots/filters, and stopping TAMP batches
after unavailable calls. New evaluator checks reject incorrect partial receipts
or unsafe measured state before authorizing recovery. The earlier 458-test run
and failed recovery remain recorded.

The corrected pair uses 12 remote model calls and 799 audited commands. Sampling
is server-default and unpinned; the layouts/instructions are familiar development
cases. Natural-language feedback is reviewed manually, separately from machine
receipts and physical scoring. A bare "retry" without restating goals, unknown
execution outcome, physical contact/grasp failure, observed grounding and
hardware are not qualified here.

[Compact receipts, verbatim feedback and artifact hashes](tamp_partial_agent_20261116.json),
[predeclared protocol](../../configs/eval/tamp_partial_agent_20261116.json),
and [public feedback contract](../api/tamp_task_plans.md).
Verbose raw records remain in ignored `.tamp-runs/partial-agent-20261116/` and
`.tamp-runs/partial-agent-recovery-fixed-20261116/`; they are not a public raw-data archive.
