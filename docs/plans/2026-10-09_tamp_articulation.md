# Assisted articulation follow-up

Stack base: #186 (`c3570773`); sibling of metrics #187, not dependent on it.
The closed dishwasher diagnosis motivates an explicit generic open/close tool,
not obstacle removal or object-specific planner exceptions.

Implemented: simulator joint-state descriptors, verified assisted JSON action,
geometry refresh, pre-pick goal access checks, and plan invalidation. See
[contract and limits](../apis/tamp_articulation.md). Unsupported fixtures remain
visible with a rejection reason instead of disappearing from the denominator.

Validation: 145 focused tests pass across TAMP APIs, articulation, scene geometry,
CHAT routing, scripted harness and physical-motion contracts. Ruff passes. Tests
exercise the actual server handler on a small MuJoCo fixture, both endpoints,
geometry changes, unchanged dynamics during metadata refresh, stale receipts,
stale sessions, physical-mode rejection and unsupported joint configurations.
Live cycle `20261009_155138_272c3a` at `8df25141` passed both verified states
through CHAT/ZMQ in the actual kitchen. Artifacts:
`~/runs/emet/tamp-articulation-r2-20261009/job/job.log`. Initial attempt
`20261009_154943_dd5e13` failed before startup because the isolated worktree lacked
its MolmoSpaces environment link; both attempts are retained. This open/close
success does not imply complete pick/place or multi-action task success.

Previous integrated gate `20261009_123154_9bfdbc` is now terminal: CHAT failed,
0/3 admissions, no repeats. Preserve that baseline. Do not replace it with an
assisted opening result or claim that the closed door explains all clutter cases.
