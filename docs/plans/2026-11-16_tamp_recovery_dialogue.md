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

Dependency order is #226 → #231 → #234 → #242 → #243 → #244. The first
three foundations are merged. Remaining children receive integration checks
against the resulting main before landing.

| PR | Scope | Work before landing |
| --- | --- | --- |
| [#226](https://github.com/cpaxton/home_robot_v3/pull/226) | Observed evidence/adapter foundation | Merged as `fb9f9bf1`; foundation scope only. Default observed discovery stays read-only. Open self-occupancy/tracking limitations are tracked in #245; no observed task success is claimed. |
| [#231](https://github.com/cpaxton/home_robot_v3/pull/231) | Shared bounded native release policy | Merged as `e9478d3d`. Current-main integration: 1,103 tests and native/actual-agent two-goal smokes pass. Independent review agent: no blocker, 30 focused tests pass. Historical failed cohorts remain unchanged. |
| [#234](https://github.com/cpaxton/home_robot_v3/pull/234) | Native IK reserve, grasp ordering and retention diagnostics | Merged as `db35f08b`. 467 integration tests and native 2/2 control pass; independent review agent passed 19 focused tests. Review includes closing, ordering and verification additions. Integrating landed #231 preserved the complete tested tree. |
| [#242](https://github.com/cpaxton/home_robot_v3/pull/242) | Checked unloaded approach and loaded placement refresh | Original and updated-parent integration reviews find no blocker. Integration tests and shifted-swap native/actual-agent smokes are the gate. Parent fixture fix reuses the loaded model; production freshness guards remain unchanged. |
| [#243](https://github.com/cpaxton/home_robot_v3/pull/243) | Named partial receipts, prospective trials and restated-goal recovery | Author and independent-agent reviews find no production blocker. 199 exact-head targeted tests pass; original recovery sequence was manually verified. Parent integration remains required. |
| [#244](https://github.com/cpaxton/home_robot_v3/pull/244) | Contextual retry, serial TAMP proposals and prior work | Independent review agent: no production blocker; 55 focused tests pass. This ledger update resolves its documentation nit. Parent integration remains required; historical corrected live pair and 485 tests remain separate evidence. |

The [#243 self-review](https://github.com/cpaxton/home_robot_v3/pull/243#pullrequestreview-5922413427)
is author review, not independent approval. It notes that #243's original recovery
scorer does not automatically enforce the fresh-admission sequence; the recorded
trace supplies that evidence and #244 adds explicit checks. No GitHub commit
statuses or Actions check runs exist on #243's head; local validation is not
described as hosted CI. The new 199-test review run does not replace or rerun its
previous 468-test/live cohorts.

Preserve independently landed state and controller fixes, and keep #226's
foundation-only scope explicit while observed execution remains gated. Merge
remaining children in dependency order and retest integrated changes. Main uses
squash merges; integrate resulting main into each child and inspect its diff
before landing. When resolving duplicate ancestry, verify the complete tested
tree remains identical rather than assuming a clean retarget is sufficient. Repeat affected
API/provider/command/CHAT tests and a native plus actual-agent smoke after main
integration. Qualification still stops at privileged simulated tabletop tasks;
observed admission/exploration and household/more-object generalization are next.
