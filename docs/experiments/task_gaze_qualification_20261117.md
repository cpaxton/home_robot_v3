# Measured task gaze and bounded approach qualification

The existing view tools now support explicit absolute head pan/tilt: CHAT
`describe_scene`, EQA `look_around`. Both use one measured stationary capture
primitive shared with floor inspection. No new tool is added, and no-argument
caption/scan behavior is retained. Floor observation keeps downward bounds,
calibrated-depth requirements, footprint checks and task-gaze restoration.

Live source `c9c648f0c8bc5311d2b9183e018b2365e3bc152c`; managed job
`20261117_092902_0e6a21`, completed. Artifacts:
`/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/`.
The parent operates the production CHAT mailbox, using actual schemas,
images and receipts. This is an informed interface diagnostic, **not independent
usability or an autonomous Qwen-coordinator benchmark**. Separate Caliban Qwen
performs grounding; its deployment precision/revision remains unverified.

## What the live run establishes

| Check | Observed result |
| --- | --- |
| Out-of-range request | `describe_scene(tilt_rad=9)` rejects as `invalid_head_tilt`, returning active tilt `[-1.8326, 1.5708]` and pan `[-3.0543, 3.0543]`; no clipping/success claim. |
| Task gaze | After one in-place scan, tilt `-0.3` is requested, measured `-0.418933` within the existing 0.12-rad tolerance. Base translation during that capture is approximately 4 micrometers. Subsequent frame reception is confirmed. |
| Useful pixels | Parent inspects the next camera image: the previously clipped microwave is fully visible. |
| Grounding | Qwen recognizes the microwave on both attempts; neither admits XYZ because all proposed depth pixels exceed the unchanged 2.5 m gate. |
| Exact evidence | Both calls pass source RGB, displayed RGB and saved sensor-depth hash checks against their decision IDs. Live map scope is also retained. |
| First approach | A 0.3 m request rejects before execution for unknown swept footprint: 14 unknown cells in the rejection. |
| Floor recovery | Two bounded calls reduce the fixed checked pose's unknown cells `12→4→0`. The first recheck starts at 12 after intervening map updates; it is not the original 14-cell rejection. Both calls restore a useful task view. |
| Replanned approach | The new 0.3 m request executes, reporting 0.230212 m measured forward progress and negligible lateral/yaw change. Requested distance is not treated as travel. |
| Longer approach | A 0.8 m request rejects before execution for an occupied footprint cell, with full observed coverage. Floor sensing is not retried to erase occupancy. |
| Tool surface / audit | All twelve requests advertise the same 32 tools; registry remains 34. Twenty-five actuation-audit rows, zero forbidden entries. No manipulation. |

The microwave task is **incomplete**. A useful view and semantic recognition are
not object localization. The occupied-route rejection requires an alternative
approach or further map diagnosis, not weaker clearance thresholds.

The first rejected surface proposal has 329 measured pixels at median optical
depth 3.375 m; the second has 3,047 at 3.554 m. Different masks/head poses mean
these medians cannot be subtracted to infer travel or distance improvement.
Neither mask's localization accuracy is ground-truth-scored; the first proposal
covers only a narrow part of the visible appliance. Correct semantic recognition
does not establish a complete mask. Both remain rejected proposals.

Startup image simulation time advances 1.672→2.000 s; base-up dot world-Z is
0.9999994. Scene/robot/providers and physics remain as in the preceding
[decision-view diagnostic](decision_view_handoff_20261117.md): Molmo iTHOR train
index 0, `rby1` with Galaxea R1 MJCF proxy, physical guards, teleport disabled,
renderer sensor depth, `depth_candidates` / `yoloe_sam2`, no GT object masks/poses.
Acquisition timestamps are empty; transport reception is measured, not promoted
to an independent sensor-acquisition freshness guarantee. The 12-response budget
and eight-base-request limit are unchanged; four base macros are requested here.

## Inspect the context

[Comparison figure: scan, task gaze, restored task view](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/figures/task_gaze_views.png)
uses actual coordinator images. `figures/manifest.json` retains input file
hashes. The parent inspected the individual views and the assembled figure.

- [Post-gaze coordinator image](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/public/request_0004/message_007_image_01.png).
- [Post-motion reacquisition](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/public/request_0009/message_017_image_01.png).
- [Request/receipt summary](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/request_summary.json).
- [Hash verification](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/handoff_verification.json) and [separate evaluation](/home/cpaxton/runs/emet/task-gaze-qualification-20261117-r1/manual_evaluation.json).

`check_decision_handoff.py` is preserved beside the run so the read-only checks
can be repeated. Full public requests, replies, actual grounding RGB-D,
calibration, raw model output, navigation capture evidence and actuation audit
remain in the artifact directory. Raw completion is not rewritten as success.

## Diagnostic repair and remaining acceptance

The final navigation diagnostic also says base clearance is 0.00 m. Source audit
found `format_base_clearance_hint` querying the world-frame A* field with a local
robot pose. Follow-up `df8db87d` uses the controller's planning-world conversion;
shifted/rotated-origin tests reproduce the contract and verify the repair.
This warning cannot establish actual zero clearance. The occupied swept-route
rejection uses a separate world-frame validation path and is not cleared by
fixing the hint. The hint repair is CPU-qualified, not live-rerun in this trial.

The combined focused pack has **659 passing tests**, including floor recovery,
head capability, measured arrival, stale/base-drift rejection, RGB-only task
views, actual CHAT/HTTP images and history, explicit EQA argument forwarding,
default routing, admission provenance and diagnostic frames. Ruff/commit hooks
pass. No new router, automatic retry policy, answer override, tool or default
perception switch is introduced.

Next run the small paired EQA and simple OVMM gates. Separately inspect the
occupied cell and qualify an alternative close-view route. This diagnostic
does not establish full navigation reliability, collision-proof execution,
microwave XYZ accuracy, manipulation or benchmark superiority; there is no
reason to launch a broad sweep from it alone.
