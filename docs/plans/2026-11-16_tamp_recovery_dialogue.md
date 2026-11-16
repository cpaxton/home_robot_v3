# Bounded task dialogue and contextual recovery

Follow-up branch: `feat/tamp-recovery-dialogue`, based on frozen PR #243
(`117ed129`). Do not change #242/#243's reported source or rescore their trials.

The CHAT model resets on each user turn. Retain at most two consecutive TAMP
turns as explicit historical text: original requests and public outcome summaries.
Strip executable handles, private identifiers and sensor payloads. Preserve
completed actions, incomplete goals, and unknown/per-step dispatch flags. Drop
oversized records whole rather than truncating goal constraints. Clear history
after a non-TAMP turn. New user instructions alone authorize work; past requests
do not. Keep existing hard failure and consumed-plan boundaries unchanged.

Test stop, explain and plan-only follow-ups with the real loop; final summaries
cannot dispatch tools. Then freeze source before a managed actual-agent trial
on Caliban, using the existing one-shot admission refusal after a measured first
transfer. Only after checking partial feedback, supported/settled objects and a
quiet audit interval, send exactly “Please retry the remaining work.” Require
fresh discovery and planning, exactly one remaining transfer, both original
physical goals, and no forbidden commands. Repeat the stop-only control on the
same source. These are development diagnostics, not a new held-out cohort or
contact/unknown-dispatch recovery qualification.

The first frozen pair at `0e4e3959` passes stop-only and returns correct partial
feedback, but contextual recovery fails: the model bundles discovery, planning
and execution with an invented `plan_123`. Planning returns a real one-step
handle; the invented handle is rejected with `unknown_plan`, with no second
transfer. Preserve this attempt and its 482 passing affected tests.

Serialize all-TAMP model batches: dispatch only the first proposal, log later
ones as deferred, and require the model to reconsider after the actual result.
Deferred calls are never queued automatically. Preserve hard failure boundaries
and mixed capability batch semantics. Add real-loop regressions for repeated
bundling, actual returned handles and first-call failure, then freeze a new
source and repeat both declared trials without replacing the first pair.

The repeat at `ebbdb1d9` passes stop-only and contextual recovery. The new retry
turn uses fresh discovery, a one-step beam plan and the returned handle, then
completes exactly the remaining cube transfer. Both original physical goals
hold. There are 799 audited commands, zero forbidden commands or shutdown
warnings, and 485 affected tests pass with no errors/skips. See
[both frozen source histories and limitations](../experiments/tamp_contextual_recovery_20261116.md).

## Relation to Agentic GenTAMP

[Coding Agents for Generalized Task and Motion Planning Problems](https://arxiv.org/abs/2609.30233)
and its [official experiment code](https://github.com/tomsilver/robocode) motivate
a separate frozen-program baseline. Such a baseline must use the same permitted
robot tools, observations, goal checks and deployment limits as our online agent
if compared within our environment. Record offline synthesis cost separately
from online model, search and execution cost. Match model and access level for
method comparisons; keep fully observed and observed-input cohorts separate.
Freeze programs before testing new layouts and object counts. Porting an
environment to MuJoCo requires a new dynamics/goal parity check; it is not an
official benchmark reproduction by name alone. No synthesized-policy baseline
or paper-to-paper numerical comparison is claimed in this change.
