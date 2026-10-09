# Assisted articulation follow-up

Stack base: #186 (`c3570773`); sibling of metrics #187, not dependent on it.
The closed dishwasher diagnosis motivates an explicit generic open/close tool,
not obstacle removal or object-specific planner exceptions.

Implemented: simulator joint-state descriptors, verified assisted JSON action,
geometry refresh, pre-pick goal access checks, and plan invalidation. See
[contract and limits](../apis/tamp_articulation.md). Unsupported fixtures remain
visible with a rejection reason instead of disappearing from the denominator.

Validation: 144 focused tests pass across TAMP APIs, articulation, scene geometry,
CHAT routing, scripted harness and physical-motion contracts. Ruff passes. Tests
exercise the actual server handler on a small MuJoCo fixture, both endpoints,
geometry changes, unchanged dynamics during metadata refresh, stale receipts,
stale sessions, physical-mode rejection and unsupported joint configurations.
Live simulator validation is pending; passing this feature's open/close test does
not imply complete pick/place or multi-action task success.

Previous integrated gate `20261009_123154_9bfdbc` is now terminal: CHAT failed,
0/3 admissions, no repeats. Preserve that baseline. Do not replace it with an
assisted opening result or claim that the closed door explains all clutter cases.
