# Final TAMP dialogue integration — November 16

Clean frozen source `a7ee2ab4` includes the updated native foundations and named
partial feedback. Integrating landed parent main `6d0750bd` at `40b0b6f0`
preserves the complete tested `src`, `scripts`, `configs` and `paper` trees.
Subsequent changes only update the ledger, TODO and this validation record.
The bibliography resolution preserves Agentic GenTAMP and all main controller
citations. The stale draft/unmerged ledger finding from independent review is
resolved. No other production fix was requested by that review.

**230 integration tests pass**, zero failures/errors/skips. An independent
review agent separately passed 55 focused checks and found no production
blocker; those checks overlap this suite. Ruff/whitespace and the full paper
build (70 pages) pass.

The [predeclared protocol](../../configs/eval/tamp_pr244_integration_20261116.json)
uses two actual `run agent` sessions with Caliban, camera-free privileged
simulated Stretch, local GPU disabled and independent familiar-layout startups.
After one real verified cylinder transfer, an evaluator-only seam refuses the
second transfer before commands. This simulates admission failure, not natural
solver exhaustion, grasp loss or an unknown command outcome.

- **Stop-only control passes**, deliberately leaving the full task incomplete.
  The receipt names the completed cylinder action and unfinished cube action,
  records one of two steps complete, step 2 failed at `preflight`, overall
  `execution_started=true`, `failed_step_execution_started=false`, consumed
  plan and unverified final goal. No subsequent dispatch occurs in that turn.
  285 audited commands; four model calls; clean exit, no shutdown warning.
- **Contextual retry passes** after the separate request “Please retry the
  remaining work.” The trace is `scene_tasks` → `plan_task` →
  `execute_task_plan`, with a fresh one-step plan, its actual returned handle
  and exactly the remaining cube transfer. Both original independent final
  goals are in-region, supported and settled. 509 audited commands; eight
  model calls; clean exit, no shutdown warning.

The pair has **794 audited commands, zero forbidden commands and 12 remote
model calls**. Stop compliance and full-task completion are different metrics;
only the retry case finishes the complete task. Server sampling is default,
not independently pinned. The [compact record](pr244_main_integration_20261116.json)
contains public receipts, exact source/job identities and nine artifact hashes.
Verbose local logs remain under `.tamp-runs/native-stack-20261116/pr244/`.

Historical successful and failed cohorts remain unchanged and are not pooled
with this pair. This is a familiar two-object integration diagnostic, not a
held-out reliability estimate or contact-loss, held-object, unknown-dispatch,
observed-grounding/exploration or hardware recovery qualification. Those gates
remain separate; observed self-occupancy/contact admission is tracked in #245.
