# Multi-step TAMP task plans

`plan_task` and `execute_task_plan` are CHAT tools using the existing
[schema-version-1 JSON envelope](tamp.md). They require a qualified provider;
registration alone does not enable multi-step execution on a connected robot.
The production loop omits these tools when no session is bound, preserving
existing agent conditions. Direct registry calls without a provider return
`task_provider_unavailable`, with no physical command.

`plan_task(instruction, strategy="beam")` accepts the user's complete final goal
and returns an opaque `plan_ref`, number of steps, strategy and
`execution_started: false`. Currently implemented strategies are `greedy`,
`beam` and `mcts` (the existing UCT implementation). Other methods explicitly
return `strategy_unavailable`; they are not aliases. Planning calls a shared
operator/feasibility provider and must not mutate live simulator state.

`execute_task_plan(plan_ref)` consumes the handle once, including on rejection.
It rejects changed model, robot profile, boot, reset, revision or initial
predicates. After each native skill it reads measured predicates, checks the
operator's effects and protected goals, and revalidates before the next skill.
A command returning success is insufficient. Any failed/unknown result stops the
plan and the production CHAT dispatch batch. There is no automatic retry.

Successful execution reports `completed_steps`, `execution_started` and
`goal_verified`. Failures retain completed-step counts and return `partial` if
execution may have begun. Public errors never expose provider exception strings
or private body names. Inspect the scene before deciding how to recover.

Python integration: construct `TaskSession(provider, budget=BudgetSpec(...))`
and pass it as `tamp_task_session` to `run_agent_with_robot`. The provider owns
fresh grounded `TaskState`, instruction-to-`GoalSpec` compilation, qualified
operators, private-state feasibility and measured native execution. It must
bind the connected robot and correct input mode; this seam does not waive any
native/observed qualification requirement. Session locking serializes its own
calls, not unrelated external controllers; provider execution must atomically
validate its planning snapshot before commanding hardware.

Benchmark support and remaining qualification gates:
[MuJoCo implementation plan](../plans/2026-10-22_mujoco_tamp_benchmarks.md).

## Privileged native Stretch integration

`run agent --native-task-config FILE` opts into stationary region-transfer
planning. It requires online LLM CHAT, `--state-only-sim`, physical manipulation
mode and privileged inputs. [Example configuration](../../configs/eval/native_stretch_transfer_debug.json)
declares public roles, private body bindings, one support and disjoint XY regions.
Its diagnostic `reset_id` is not an upstream reset certificate. The configuration
must match the connected scene; it is not portable table calibration.

Discovery reports public object and region names, measured locations, free
regions and supported instruction forms. Body bindings stay private. The bounded
compiler accepts `swap OBJECT and OBJECT` or semicolon-separated clauses such as
`move OBJECT to REGION`. Unknown roles, conflicting goals, overlapping regions,
uncertain occupancy and unsettled/held objects are refused. It does not claim
unrestricted natural-language grounding or observed-input support.

Plans report `planning_scope: task_skeleton`, `motion_refinement: per_step`.
Search checks occupancy/footprint fit and probes native arm/payload planning for
candidate first moves from the measured state. Infeasible first moves are pruned
before selecting ordering. Hypothetical later states remain symbolic: a returned
skeleton is not a full-episode collision-free trajectory certificate. Every
transfer is refined again from fresh measured geometry before native actuation.
Motion probes use private data, discard unused motor-plan handles, share a
bounded planning-IK allowance and record semantic rejection counts. Execution-time
replanning and outer LLM calls are not yet included in that allowance; no complete
matched-budget baseline claim follows from it.

`TaskExecutionResult(success, execution_started, failure_code)` distinguishes a
known precommand failure from possible physical progress. A failed command or a
failed later step preserves partial progress and stops execution. Legacy boolean
receipts cannot establish no actuation. Measured effects, support, containment,
stillness and final goals remain mandatory regardless of command receipts.


Failures also identify `planned_steps`, `failed_step_1based` (or null before a
step/at final verification), and `failed_phase`. For example, one completed step
followed by step-2 preflight rejection is partial progress, not a failed first
transfer. Debug mode can include private simulator state in JSON logs for offline
replay; public tool responses and public motion evidence never include it.
See [measured integration cases and captured-state controls](../experiments/tamp_native_task_provider_20261022.md).

## Provider failures and budgets

Discovery and planning use the same public failure translation as execution.
A changed scene during planning returns `stale_task_plan` with `recovery: replan`
and stores no plan. Typed unavailable observations return `stale_observation`
for known stale measurements, otherwise `observation_unavailable`. These errors
remain failures and do not authorize an automatic execution retry.

Execution preserves allowlisted provider reasons, including
`task_object_not_settled_on_support`, `ambiguous_region_occupancy`,
`native_execution_unavailable` and `object_not_lifted`, through both exceptions
and typed receipts. Unknown provider errors return `task_provider_failed` without
exception text. Partial-progress counts and one-shot consumption still apply.

`TaskBudget` is currently tested infrastructure without a production caller.
`BudgetSpec` bounds task search, and native planning probes separately consume
the provider's IK allowance. Neither constitutes a shared whole-agent ledger;
CHAT calls and execution replanning must be connected before matched-budget
whole-agent comparisons are claimed.

## Preserved objects and execution preflight

Explicit move goals protect the current `at` predicates of every unmentioned
object. For example, `move alpha to right` cannot silently relocate beta from
right into a buffer; that request is infeasible under the preserve-beta
constraint. Explicitly include beta's desired destination, or request a swap,
when moving it is authorized. Swap goals likewise protect other objects.

The region provider converts errors from its read-only snapshot, feasibility
and operator-selection preflight into typed no-actuation receipts. The transfer
callback is outside that exception boundary: unknown exceptions there still
mean execution may have started. Earlier measured progress is preserved in either
case. This distinction does not infer no actuation from arbitrary provider errors.

`--native-task-config` is incompatible with `--eqa-eval`; it cannot silently
configure a different orchestrator. Both remote OpenAI text and vision clients
keep robot telemetry live during inference. Local inference retains its existing
stream-pause guard.

Native execution refinement also has a planning-only exception boundary: timeout,
malformed-result and invalid-handle errors before `adapter.execute` return
no-actuation receipts. Typed failed receipts with `execution_started: false`
report `failed_phase: preflight`, even when an earlier step made physical
progress. Exceptions from the actual adapter executor remain possible physical
execution and are never reclassified by the planning-only catch.
