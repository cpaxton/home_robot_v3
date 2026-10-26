# Observed native geometry integration, October 26

This is internal geometry/executor evidence, not observed agent task success.
Observed task discovery remains read-only. The privileged stationary Stretch
control remains a separate evaluation tier.

## Implemented behavior

The robot-only native collision adapter consumes a supplied conservative payload
envelope and measured reference pose without a simulator object body. It uses
full robot/payload volumes, requires known-free workspace coverage, and rejects
stale or changed scene revisions. Speculative lift/release replanning restores all
attachment state, including the released-volume hypothesis. Expected missing
payload evidence fails preflight before any arm or gripper command.

The depth workspace builder marks cells free only when the whole cell projection
is in front of valid sensor depth with an error margin. Missing/out-of-range
returns, image clipping and space behind measured surfaces remain unknown. The
pinhole pixel-footprint coverage model and supplied calibration/error bounds are
assumptions; this is not qualification of hardware sensing or capture timing.

Tests use synthetic coherent observations and measured-joint feedback on small
MuJoCo robot-only rigs. They exercise actual native pose planning, dense segment
revalidation and the executor command seam. They do not simulate grasp dynamics or
provide evidence that a real tracker can produce these payload envelopes.

## Offline live-capture replay

Input: the previously saved registered ZMQ capture
`.tamp-runs/capture-binding-20261026/postmerge/capture-0.npz`, containing sensor
depth, K and camera pose. No object body positions or scene collision geometry
were used. This replay makes no new VLM calls and is not a new live agent episode.

A 0.6 m cube centered 0.75 m along the camera optical axis, at 0.025 m resolution,
contains 13,824 cells: 8,343 free, 35 occupied, and 5,446 unknown. Construction took
0.139 s on this host. Replacing depth with missing returns produces zero free
cells. Raw output: `.tamp-runs/observed-native-20261026/workspace-diagnostic.json`.
The counts describe one sensor snapshot, not task success or collision recall.

## Remaining acceptance gates

No observed multi-step execution is claimed. Still required: a coherent observed
adapter, independently bounded payload geometry, pose hypotheses/observability,
qualified fresh capture timing, target/robot occupancy provenance, multi-view
refresh, visual grasp/retention/release effects, and held-out same-agent Caliban
trials. Current simulator self-occlusion limitations still apply. See the
[execution plan](../plans/2026-10-26_observed_native_execution.md).


## Regression procedure

The managed CPU-only suite covers all of `src/test/motion`,
`src/test/controller/task/tamp`, native input provenance, live-capture tool binding
and observed motion-epoch invalidation. It uses `MUJOCO_GL=disable` and no local
GPU model. The first run (`20261026_184331_3ec92d`) had 515 passes and three failures
in the pre-existing A* start-escape fixtures: unrestricted `MagicMock` invented an
unconfigured `get_navigation_map` method. Restricting the mock's declared interface
restored the intended voxel-map fallback; the three isolated tests then passed.
No production A* behavior changed. Additional final geometry tests were added
before the final managed rerun; the runs are not independent task attempts.


Final managed run `20261026_184551_55ff3a`: **521 passed, zero failures, errors or
skips**, 123.454 s test time. Ruff on all changed Python files and `git diff --check`
passed. Raw JUnit: `.tamp-runs/observed-native-20261026/final-regression.xml`;
log: `.tamp-runs/observed-native-20261026/final-job/job.log`. No new live contact or
VLM agent episode was run for this internal geometry change.
