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
