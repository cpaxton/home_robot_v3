# Main refactor and observed-TAMP integration — November 18

Branch start: `8a627be1` on `fix/tamp-observed-admission`.
Fetched main: `f1b7bb5b`, including #247 context/grounding changes, #248 tool
binding consolidation and #249 runtime simplification. The separate dirty
`feat/tamp-agent-execution` checkout was not modified.

## Merge and review

Resolved three conflicts without discarding the observed-admission changes:

- `TODO.md`: retain both work inventories and the completed #246 integration.
- `physical_pick_place.py`: retain the observed target-contact preclose guard,
  adopt main's simulation-rate-aware settling window and formatting.
- `zmq_client.py`: retain original RGBD registration provenance together with
  main's observation/history updates.

Read-only review found that optional `describe_scene(pan_rad/tilt_rad)` gaze
could move the camera without the existing TAMP world-change barrier. CHAT now
passes tool arguments to that barrier and invalidates before/after non-null gaze,
including zero-valued angles, failures and in-flight inference. Caption-only
calls and explicit null angles remain read-only. Tests use real observed frame
epochs, task sessions and adapter handles; old and during-motion plans cannot
survive the action. Provider/handle bindings, typed public TAMP JSON and stop-on-
failure continuation otherwise retain their prior behavior.

## Validation

All native work uses detached CPU jobs; no local VLM or GPU experiment is required.
Artifacts are under `.tamp-runs/main-integration-20261118/` in the owning worktree.
The reviewed source files and hashes are retained alongside the gaze patch.
The [compact JSON record](main_refactor_tamp_integration_20261118.json) pins test
reports, runtime inputs, the simulator receipt and the prior-command comparison.

| Job | Scope | Result |
|---|---|---|
| `20261118_011459_c7af21` | Initial agent test launcher | Two incorrect test paths; no product result |
| `20261118_011535_6d0313` | Broad agent selection | Cancelled after unintentionally including opt-in simulator cases |
| `20261118_011814_b7fa88` | First gaze contract batch | 163 passed; optional Habitat client package absent from Python path |
| `20261118_012006_065593` | Agent, image history, tool bindings, native CHAT contracts | 728 passed, four simulator opt-ins skipped |
| `20261118_012757_f0301f` | Final agent pack with ordinary project paths | 728 passed, four simulator opt-ins skipped |
| `20261118_012006_10eb4b` | Final gaze, real evidence epochs/handles and dispatch | 164 passed, zero failures/errors/skips |
| `20261118_011500_15edd0` | Native TAMP/contact/geometry, original-ray provenance and simulation-clock checks | 717 passed, zero failures/errors/skips |
| `20261118_011814_8ebb1f` | Merged paper | Builds to 75 pages |
| `20261118_011535_be8c56` | Full assisted clutter control on CPUs 12–13 | Passed through departure |

The final agent checks use `emet test --no-sim`, retaining the four skipped
simulator cases explicitly. The optional Habitat acquisition-time client test
now imports its local source with pytest's scoped path fixture, matching existing
repository tests. The final pack needs no extra package path and creates no
Habitat simulator or GPU context. The separate full clutter control uses
CPU OSMesa and the existing provided-map fixture, task, source lifetime and
bounded-release policy. Failed test invocations are not counted as task failures
or silently promoted into passing tests.

The paper and CLI import/help checks pass. Independent review confirms the gaze
barrier correction; it does not qualify unrelated main features or model quality.

## Task scope

The complete clutter trial passes pick/carry/place/release/retreat/remapping and
departure using the existing vertical plus 1 cm lateral correction. All **115
command records match the previous successful trial exactly**, including numeric
commands and simulation timestamps; both canonical command hashes are
`a8b09ca96cbda02349b5c836cb5406d85a98cda563c50ef7c1a92d98cb7cd85d`.
This checks one deterministic development control, not every refactor behavior.

The released object drifts 0 m during departure. All five maps unload, with
24,114,174 peak counted bytes and zero resident bytes after cleanup. Planning and
physical transfer take 478.53 s and 413.64 s wall time respectively, including
nested map work; CPU timing variation is not a refactor speedup claim.

This is native-state/provided-map assistance, not an LLM trial, hardware test or
observed-only qualification. Historical cohorts and failures remain unchanged;
tests alone do not rescore earlier experiments.
