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
