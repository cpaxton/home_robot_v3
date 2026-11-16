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
