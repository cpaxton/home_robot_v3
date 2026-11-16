# Bounded task dialogue and contextual recovery

Follow-up branch: `feat/tamp-recovery-dialogue`, based on frozen PR #243
(`117ed129`). Do not change #242/#243's reported source or rescore their trials.
Review: [PR #244](https://github.com/cpaxton/home_robot_v3/pull/244), ready for
review and stacked on #243.

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

## Review and merge status (November 16)

Dependency order is #226 → #231 → #234 → #242 → #243 → #244. None is merged.

| PR | Scope | Work before landing |
| --- | --- | --- |
| [#226](https://github.com/cpaxton/home_robot_v3/pull/226) | Observed evidence/adapter foundation | Draft; GitHub reports conflicts with main. Resolve integration and obtain scope/review agreement on the open self-occupancy admission concern. Default observed discovery stays read-only; no observed task success is claimed. |
| [#231](https://github.com/cpaxton/home_robot_v3/pull/231) | Shared bounded native release policy | Foundation review and integration checks after #226; keep its historical failed tasks and config-test results. |
| [#234](https://github.com/cpaxton/home_robot_v3/pull/234) | Native IK reserve, grasp ordering and retention diagnostics | Current-head review must cover additions beyond the original margin-only review. |
| [#242](https://github.com/cpaxton/home_robot_v3/pull/242) | Checked unloaded approach and loaded placement refresh | Posted review finds no incremental code blocker. Stabilize the reported two-second snapshot fixture flake before claiming a clean expanded integration suite; preserve the production freshness guard. |
| [#243](https://github.com/cpaxton/home_robot_v3/pull/243) | Named partial receipts, prospective trials and restated-goal recovery | Author self-review at `117ed129`: no production blocker; 199 exact-head targeted tests pass, no errors/skips; Ruff and whitespace clean. 26 artifact hashes and six parent motion/command hashes verified. Independent approval and parent integration remain. |
| [#244](https://github.com/cpaxton/home_robot_v3/pull/244) | Contextual retry, serial TAMP proposals and prior work | Ready for review; corrected live pair and 485 affected tests pass on `ebbdb1d9`. |

The [#243 self-review](https://github.com/cpaxton/home_robot_v3/pull/243#pullrequestreview-5922413427)
is author review, not independent approval. It notes that #243's original recovery
scorer does not automatically enforce the fresh-admission sequence; the recorded
trace supplies that evidence and #244 adds explicit checks. No GitHub commit
statuses or Actions check runs exist on #243's head; local validation is not
described as hosted CI. The new 199-test review run does not replace or rerun its
previous 468-test/live cohorts.

Resolve #226 against current main first, preserving independently landed state
and controller fixes. Keep its foundation-only scope explicit while observed
execution remains gated, and settle the review concern before promotion. Then
merge in dependency order and retest each integrated child. Preserve stack
ancestry with merge commits, or explicitly rebase children if squashing parents;
blind retargeting after a squash can reintroduce their commits. Repeat affected
API/provider/command/CHAT tests and a native plus actual-agent smoke after main
integration. Qualification still stops at privileged simulated tabletop tasks;
observed admission/exploration and household/more-object generalization are next.
