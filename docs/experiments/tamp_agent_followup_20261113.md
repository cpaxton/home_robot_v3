# Actual agent repeat with shared native policy — November 13

The actual production `run agent` CLI, using Caliban's remote
`Qwen/Qwen3-VL-8B-Instruct`, passes **3/4 complete development tasks and
8/10 declared transfers**. An independent scorer verifies
**6/8 final object goals**. All four original November 1 pilot cases
remain in the population: two-goal and buffered-swap controls, then the same
families with +15 mm initial object X positions. This is a separate repeat;
the historical 2/4 result remains unchanged.

| Case | Complete task | Transfers | Execution result |
| --- | --- | --- | --- |
| two-goal-control | pass | 2/2 | verified completion |
| swap-control | pass | 3/3 | verified completion |
| two-goal-shift | pass | 2/2 | verified completion |
| swap-shift | fail | 1/3 | cube lift verification after cylinder relocation |

The agent chooses and calls `scene_tasks`, `plan_task` and
`execute_task_plan` through the normal CHAT tool registry. Task JSON defines
roles, regions and final goals; the experiment does not supply a transfer order
or directly call the native executor. The shared opt-in
`agent.native_tamp_policy=bounded_release` profile is configured through the
normal CLI. A completed result requires both the public `goal_verified` receipt
and independently measured supported, settled whole-footprint final goals.
Supported objects after a partial receipt cannot make the task pass.

The control additionally requests one invalid plan-handle execution, checks the
rejection, and then issues a separate authorized user turn for the real task.
All 4/4 trials satisfy the JSON result contract. The 2
observed stop boundaries have zero subsequent dispatches before a new user turn.
These interface checks are distinct from physical task success or recovery
with a held object.

There are 18 inference calls, 2,216 audited commands, zero
forbidden commands and zero CLI infrastructure failures. Sampling uses server
defaults and is not independently pinned. Source is `c4fa1668`; source/artifact
hashes and job IDs are retained in the [compact manifest](tamp_agent_followup_20261113.json).
Raw logs, fixture XML, launch configuration, protocols, chat results and audits
remain in ignored `.tamp-runs/agent-followup-20261113/`.

Scope is **privileged stationary simulated Stretch**, native asynchronous
MuJoCo physics, no cameras or local GPU inference. Caliban hosts the language
model. There are no attachments, object-pose teleports or hardware trials.
These are one live trial per development case, not held-out reliability or
exact reset-paired evidence. Agent startup and native-reference startup differ;
no cross-harness paired-state certificate is claimed.

The separate [broader native replay](tamp_native_followup_20261113.md) is only
4/8 tasks and retains cylinder release, cube rotation and post-release joint
limit failures. The three successful agent cases do not resolve
those failures, observed full-arm/self-occupancy/contact admission, household
adapters or exploration. Keep those acceptance gates explicit before expanding
paper claims or making this the default policy.

The failed swap returns `task_execution_failed` in its preserved public receipt.
Its native log attributes failure to `grasp_verification_failed` during cube
lift, after one completed cylinder relocation. This log does not distinguish
the one-second hold verifier from the additional loaded-handoff check, so no
more specific contact/rotation diagnosis is claimed. A subsequent API-only fix
maps native `*_verification_failed` reasons to the existing safe public
`verification_failed` code; it changes no robot motion or historical scores.
