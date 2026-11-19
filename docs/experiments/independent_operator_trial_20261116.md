# Independent operator trial: production CHAT harness

The requested fresh-context subagent actually operated the harness in live
simulation. Neither the first attempt nor its informed follow-up localized the
microwave. The useful result is a concrete control/feedback failure breakdown,
not an accuracy comparison or evidence that a stronger model solves the task.
No production source changed during these trials.

## Protocol and provenance

- Source: `7014124fb972b67bdee8970a65d59acf46841ef1`, feature branch
  `fix/closer-view-feedback`.
- Environment: `configs/sim/molmospaces_ithor_train_0.yaml`; registry robot
  `rby1`, which currently uses the Galaxea R1 MJCF proxy. This does not validate
  native RB-Y1 hardware. Physical execution enabled with an audit path; base
  teleport explicitly disabled. No real robot or manipulation.
- Exact task: “Find and independently localize the microwave using camera
  evidence and existing tools. Do not touch or manipulate objects. Report
  limitations honestly if no admitted position can be obtained.”
- The fresh-context operator received actual production prompts, tool schemas,
  receipts and PNGs. It received no target location, simulator ground truth,
  scene XML, earlier solution traces or private process logs.
- A temporary loopback OpenAI-compatible model endpoint let the subagent supply
  coordinator replies to the unchanged `run_agent_with_robot` CHAT loop. The
  endpoint saved requests/images and transported replies; it executed no tools.
  The production dispatcher executed them. This is a live operator trial, not
  a mocked executor or an offline review of traces.
- Perception remained Qwen via Caliban, with shared coordinator/perception
  dialogue disabled. Caliban dtype/revision/decoding are unverified. The actor
  change does not isolate every model dependency.
- Limit: 12 coordinator rounds, eight base-action requests, no manipulation.
  Existing safety admission remained active. Jobs ran serially under the GPU
  lease on CPU cores 12,14,20,21, with offline weights and one-thread math.
- Cold-start CHAT entry did not use the benchmark's four-view prescan. These
  outcomes are not paired against earlier Qwen benchmark scores.

Durable artifacts are under `/home/cpaxton/runs/emet/`:

| Attempt | Directory | Managed job | Outcome |
| --- | --- | --- | --- |
| Setup r1 | `external-coordinator-trial-20261116-r1` | `20261116_123534_fc0d03` | Missing physical audit path; rejected before any operator request. |
| Independent r2 | `external-coordinator-trial-20261116-r2` | `20261116_123958_ef2d7f` | Six rounds; no identified microwave or admitted position. |
| Informed follow-up r3 | `external-coordinator-trial-20261116-r3` | `20261116_125311_e6557e` | Eight rounds; three head-pose failures; final FIND feedback lost to coordinator. |

Each live directory contains `public/request_NNNN/` with the full model request,
readable prompt, image index, PNGs and operator response. `private/` contains the
temporary runner, actuation audit and perception output. `operator/` preserves
the actor's original notes/results. `manual_evaluation.json` records task
failure separately from the raw runner's `complete: true, success: null`.
Managed process logs are in `jobs_runs/external-operator-trial-20261116-r{2,3}/`.
The setup failure remains in the record; it is not a model or navigation miss.

## Observed behavior

The initial image is almost entirely a brown surface, with no recognizable
microwave. The first operator requested description, scan, diagnostics, an
explicit 90-degree turn, and targeted floor inspection before stopping.

The scan failed with “Mapping scan turn failed; refusing to count the unchanged
view as a new heading.” The explicit turn supplied more useful detail:
`rejected_swept_footprint:unobserved_footprint`, `execution_started: false`,
36 footprint cells, nine unknown, none occupied. Diagnostics reported 10 m
obstacle clearance, which does not imply floor support is observed. Showing
both quantities together would make the admission boundary clearer.

Targeted floor recovery chose an unknown footprint cell at `[1.2, 1.8, 0]`
(a floor hypothesis, not an object location). Its camera projection was behind
the camera. The implementation stops on this candidate before head motion,
rather than considering another unknown cell. This particular attempt added
no floor evidence.

The operator stopped at six of twelve rounds without trying the already-exposed
manual `observe_floor(target_blocker=false, pan_rad, tilt_rad)` interface.
The debrief corrected that overbroad conclusion: one failed targeted inspection
does not establish that all recovery is impossible or that a head tool is absent.

The informed follow-up exercised three distinct manual head configurations:

| Requested pan, tilt (rad) | Measured pan, tilt | Receipt |
| --- | --- | --- |
| -1, -1 | 0, 0 | `head_pose_unconfirmed` |
| 1, -0.7 | 0, 0 | `head_pose_unconfirmed` |
| 0, -1 | 0, 0 | `head_pose_unconfirmed` |

A -30-degree turn was rejected before execution, and scan again failed. A
scene-graph query offered only a weak related-view reference without a precise
world position; the operator correctly declined to treat it as localized geometry.
The final `find_objects("microwave")` call exposed an independent feedback bug.

Unchanged task-image content does not prove stale acquisition. Public metadata
reported `frame_sequence: null`, `image_timing: {}`, and explicitly unverified
freshness. No successful viewpoint change was reported. Floor observation also
restores the task gaze by design. We have not diagnosed camera freshness or the
exact head-controller root cause from these observations.

## FIND feedback failure: source and live log agree

The final r3 log reports navigation failure and no plan, followed by:

```text
Navigation Failure: Could not find the object microwave
Executor ran: find -> ok
Virgil: Done (find_objects).
```

There was no ninth coordinator request. This was normal loop termination,
not a crash or timeout. Three contracts contribute:

1. `DynamemTaskExecutor.__call__`'s `find` branch receives `_find` returning
   `None` without updating `_last_exec_ok`; the dispatcher prints `ok`.
2. `find_objects` advertises `returns_info=True`, but its executor mapping takes
   precedence over its detailed `func` result. The executor dispatch branch does
   not propagate that information flag.
3. The CHAT loop consequently treats the call as action-only and ends the turn.
   It appends the generic result to history but makes no subsequent model call.

Thus the actor did not receive either the navigation failure or an opportunity
to recover. The raw exit-zero completion is not localization success. Relevant
source: `src/emet/agent/loop.py::_dispatch_tool_calls`,
`src/emet/agent/tools.py::find_objects`, and
`src/emet/controller/task/dynamem/dynamem_task.py` (`find` / `_find`).

## Small next repair sequence

1. **Measured head contract:** in this same adapter, probe command → actual
   applied controls → measured joints → optical pose/image. Determine whether
   command application, physics ownership, hold behavior or telemetry is wrong.
   Add a settled-pose integration check; mock actuator-assignment tests alone
   cannot validate this. Keep failed pose confirmation a failure.
2. **Recoverable FIND feedback:** preserve task success separately from executor
   liveness; deliver the detailed find result to the coordinator, including a
   failed result, without executing the command twice. Check analogous scan
   paths. Test both successful and failed mapped information tools through the
   production dispatch/follow-up boundary.
3. **Coverage diagnostics:** distinguish obstacle clearance, measured floor
   support and unknown footprint cells. For targeted floor inspection, consider
   visible/reachable unknown cells using existing head capabilities. Do not mark
   unseen floor free or relax negative-obstacle rejection.
4. **Requalify:** repeat the bounded operator task after these contracts pass,
   then paired small EQA and OVMM gates. Resume the frozen current-decision
   image-packet comparison from the history audit separately; do not attribute
   control failures to image-history or model precision.

This supplies specific next repairs with no new recovery architecture. The
trial does not exhaust all poses/tools, establish global unreachability, prove
a camera failure, or measure a model-performance advantage.

## Follow-up repair: FIND feedback

After the trials, FIND now uses its registered function once, returning a
structured task outcome and the existing bounded navigation feedback. A missing
find point sets `_last_exec_ok=False` while executor liveness remains true.
Executor-mapped information tools also propagate `returns_info`, allowing the
coordinator to continue within its existing round budget. No navigation safety,
head behavior, model-answer policy or recovery strategy changed.

The new cases first reproduced four failures on the trial source. After repair,
95 focused tests pass across `test_dispatch_tool_calls.py`,
`test_nav_tool_returns.py`, `test_chat_camera_history.py`,
`test_agent_manip_tool_sequence.py`, `test_dynamem_sim_manip_gating.py`,
`test_call_llm.py`, `test_tool_outcome.py`, `test_agent_prompt_and_tools.py` and
`test_look_around_skip_sweep.py`. Ruff check/format and diff whitespace checks pass.
The CHAT/HTTP cases exercise the real loop, tool registration and client request
format with controlled executor outcomes and model transport; they are not a
new live simulator success claim. Three manipulation assertions were updated
to the already-existing rendered tool-result format; their old JSON expectation
also fails with the previous dispatcher, independently of this repair.

Next remains the measured-head simulation contract probe, followed by the
bounded operator retry. The failed live trials above remain the recorded
localization outcomes until that retry is actually run.

## Physics-log audit correction

The subsequent head investigation found a startup exception in r3's
`private/server.log`: `_sim_loop` called the legacy idle base-pose hold, and
the physical-execution guard raised
`forbidden_physical_actuation:stationary_base_pose_hold`. The observation
threads remained alive after the physics thread died. Thus the head failures
are not evidence of a functioning dynamics loop rejecting these configurations:
this run has an infrastructure failure. Public `complete: true` never checked
physics liveness. This also explains why frame-delivery sequence alone is not
an adequate simulator progress gate. The independently traced FIND feedback bug
still stands, but no model-performance conclusions should use these trials.

The physical-mode repair skips legacy idle pose pinning and deactivates its
named world weld, leaving support to dynamics; the guard still rejects direct
pose-hold actions. Headless physics failure clears server running state and is
re-raised on the main server thread. A small MuJoCo control verifies time,
measured joint and optical-pose change without pinning; separate tests cover
legacy-mode compatibility, fatal physics failure and step-limit shutdown.
Furnished-scene head qualification is recorded below.

## Repaired head contract: live Molmo probe

Source `01b9a39a812c27ef2e5601f6106c5e4f6e303706`; job
`20261116_162519_d93809`; artifacts:
`/home/cpaxton/runs/emet/sim-head-contract-20261116-r1/`.
The archived `runner.py` sends head commands only through the normal benchmark
server/client, on the same scene and proxy robot. It uses no perception model.
Before actuation, image simulation time advances from 2.392 to 2.590 seconds.

| Requested pan, tilt (rad) | Measured pan, tilt | Within existing 0.12 rad tolerance |
| --- | --- | --- |
| -0.3, -0.4 | -0.241, -0.310 | Yes |
| -1, -1 | -0.915, -0.922 | Yes |
| 1, -0.7 | 0.906, -0.716 | Yes |
| 0, -1 | 0.098, -0.961 | Yes |

Applied torso targets match each request; optical poses and image hashes change.
Image simulation time reaches 6.700 seconds. Base-up dot world-Z remains above
0.999999 during these observations. Six audited commands are accepted, with no
forbidden actuation. The process finishes successfully. Parent visual inspection
of `pose_1.png` confirms revealed floor/kitchen furniture; `pose_2.png` still
faces a nearby surface. Confirmed gaze movement does not guarantee task
usefulness for every requested angle.

The focused physical-motion/head pack passes 76 CPU tests, including the new
real MuJoCo dynamics control and preservation of the direct pose-hold guard.
No head tolerance was widened. This qualifies this simulator/proxy combination,
not all robot adapters or real hardware.

## Informed operator retry r4

Source remains `01b9a39a`; job `20261116_162850_28730f`; artifacts:
`/home/cpaxton/runs/emet/external-coordinator-trial-20261116-r4/`.
Source stayed unchanged throughout the trial. This is the same informed actor,
not a fresh blind comparison. The runner checks advancing image simulation time
and base uprightness before handing control to it. Exact prompts, pixels,
responses, physical audit and the runner remain archived as before.

The operator used 12 responses, four base-tool requests and two head/floor
requests, with no manipulation. Base-tool requests are macro calls, not a count
of every underlying primitive; the actuation audit has 103 accepted commands
and zero forbidden entries. Public session end reports no error. Task outcomes:

| Gate | Observed result |
| --- | --- |
| Initial stationary floor inspection | Measured head arrival and fresh RGB-D; 77 additional observed cells. |
| Scan | Completes and returns a new kitchen view. |
| Failed FIND feedback | Footprint rejection and geometry details reach the next coordinator round. |
| Targeted floor recovery | Six unknown footprint cells become zero; checked footprint becomes valid, with route replanning still required. |
| Replanned FIND | Base advances about 0.6 m; later returns an occupied-footprint rejection (full coverage; reported clearance 0.28 m versus 0.33 m required). No safety bypass. |
| Visual identification | Actor and parent identify the silver microwave above the stove in `public/request_0005/message_009_image_01.png`. |
| Object geometry | **No admitted microwave 3D position.** Scene/memory queries give unrelated low-confidence frontier/viewpoint coordinates, which the actor rejects. |

The control/recovery contracts improved concretely; end-to-end localization
still fails. Reported occupied cells were not independently validated against
simulator ground truth, so this does not establish an optimal clearance threshold
or that no collision-free route exists.

`face_toward` also exposes a legacy path: it uses raw local base pose with a
world voxel target and reports a turn from executor liveness. The trial's voxel
search reference is unresolved, and the resulting image faces the fridge area.
The frame mismatch is a source-confirmed bug, but the weak target reference also
prevents attributing the wrong view solely to that mismatch.

After r4, facing now converts the base into the target's planning frame and
reuses the existing validated `rotate_base` handler. It forwards measured
success/rejection instead of executing the legacy command or treating an
unresolved location as object identity. Four regression combinations cover
translated/rotated origins, cached controller transforms, and rejected versus
successful turns; the 72-test focused CHAT/motion suite passes. No live trial
has yet been rerun with this additional facing repair.

The next task-level repair should make current-view object grounding reachable
from the coordinating CHAT path using existing RGB-D admission machinery.
`prepare_query_target` already supports current-view grounding, while query-mode
navigation first relies on a grounded target or a retrieved search candidate;
the public CHAT toolset has no standalone current-view grounding call. A visible
object should not require completing a frontier search to request its geometry.
Keep this separate from clearance validation and fresh manipulation authorization.
Then repeat this bounded case and small paired EQA/OVMM gates, without a broad
sweep or interpreting the infrastructure-invalid trials as model accuracy.

## Disk cleanup during qualification

Lossless compression of 6,970 inactive October 29 baseline debug arrays recovered
2.811 GiB. Each gzip archive was decompressed and SHA256-checked before removing
its raw copy; an interrupted temporary archive was removed with its original
retained. Models, environments, source changes and evidence content were preserved.
Manifest: `/home/cpaxton/src/home_robot_v3/logs/.compressed-debug-20261116.jsonl`;
summary alongside it. Restore a raw array with `gzip -dk path/to/array.npy.gz`.
Package-cache pruning also ran. Other cleanup happened on the shared host; the
whole change in free disk space is not attributed to this session.
