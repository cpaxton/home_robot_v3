# Measured articulation follow-ups — October 12

PR #179 merged into main as `ccd69863`. Its broader seven-case placement run
remains **3/7**; these follow-ups do not rescore it or claim full placement success.

PR #201 (`22b7c2ec`) adds measured Stretch articulation and explicit task admission.
The physics process publishes privileged scene qpos over IPC. The ZMQ process
reconstructs fixture FK from that snapshot, updates geometry, and publishes
request-correlated command results. XML defaults and stale parent MjData are not
accepted as live state. Missing/malformed snapshots disable the capability.
This simulator metadata is not a new real-robot sensor.

The script resolves `$task_ref` against `access.available_actions`, logging total
and eligible tasks plus its selection as JSON. Explicit task handles are never
substituted. Unsupported multi-joint fixtures remain unsupported; no eligible
task means a failed case, not a denominator exclusion. This is a different
selection policy from the historical first-task benchmark.

Managed GPU job `20261012_153010_4c32aa` completed successfully:

| Case | Source | Result |
| --- | --- | --- |
| Sourccey kitchen 0 startup and assisted open/close | #200 `2808c547` | Pass, exit 0; 54.3 s |
| Stretch kitchen 0 assisted open/close | #201 `22b7c2ec` | Pass, exit 0; 14/24 tasks eligible, selected task 1 |
| RBY1 kitchen 2 assisted open/close | #201 `22b7c2ec` | Pass, exit 0; 4/24 tasks eligible, selected task 21 |

Both state changes returned `verified: true` in each case. Artifacts, exact
commands, source revisions and per-case results are under
`~/runs/emet/articulation-startup-fixes-20261012/`. All use joint teleport with
`collision_scope: none`; no door-sweep collision, native contact, arm placement,
or sustained payload support was tested. Stretch still logged a lift-init timeout
and multiprocessing connection errors during teardown; successful articulation
does not establish that those separate startup/shutdown issues are resolved.

All 98 tests passed (zero skips) across TAMP, scripted selection, Stretch telemetry
and camera-frame contracts. New regressions require measured state after command
receipt, reject missing snapshots, verify geometry refresh, and ensure no
unsupported task is silently actuated. Ruff passes. See
[the articulation API](../apis/tamp_articulation.md).

#195 now targets main but needs a restack/conflict review after its parent was
squash-merged; its native-execution work is not validated by these assisted tests.
