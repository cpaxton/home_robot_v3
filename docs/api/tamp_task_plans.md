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
