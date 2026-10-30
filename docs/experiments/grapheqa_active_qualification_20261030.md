# GraphEQA active connection qualification

This follows the [model API smoke](grapheqa_model_api_connection_20261030.md).
The external controller now has an active sensor/graph/executor connection.
This is an explicitly adapted GraphEQA baseline, not an unchanged upstream
reproduction or a completed paired active comparison with emet.

## Input and execution contract

- Scene/spawn selection stays in the renderer. The policy, mapper and planner
  receive no simulator global pose, semantic sensor, object annotations,
  enrichment labels, full-scene bounds, global maps or navmesh routes.
- The renderer retains Habitat's internal navmesh **motor collision filter**.
  It refuses base motion if that filter is unavailable. No pathfinder route,
  snapped target or simulator pose is returned to the planning adapter.
- RGB-D odometry starts at a local origin with calibrated camera height/tilt.
  Bounded commands initialize registration; the accepted pose comes from RGB-D.
  Failed depth/registration stops motion, without command-odometry fallback.
  This is local odometry without loop closure or relocalization.
- Fresh YOLOE masks use the pinned upstream global vocabulary. Background stays
  unknown. Learned labels are described as unverified in model context.
- Geometry uses the shared voxel floor map, grid search, footprint sweep and
  measured route executor. Only the occupied circular Habitat footprint gives
  proprioceptive floor support; a surrounding free disk is not invented.
- A persistent native Hydra process sees only learned sensor packets, static
  vocabulary and reference code. It has no GPU, network or simulator assets.
  Native agent nodes update less frequently than captures; context explicitly
  includes the latest estimated camera pose and timestamp.
- One reference controller persists across decisions. Its history includes
  intended target geometry and actual motion receipts. A fresh capture follows
  each attempt, including a rejected motion. The final model answer is retained.
  Scoring happens after all episodes, and failed rows remain in the denominator.

The odometry backend is the existing Open3D RGB-D registration API with sampled
depth/appearance agreement checks. The camera-to-world pose composes with the
inverse source-to-target transform. See the
[official Open3D description](https://open3d.org/docs/latest/tutorial/pipelines/rgbd_odometry.html).
Registration success and self-estimated arrival are not ground-truth accuracy.

## Completed motion gate

Job `20261030_180943_037ded`, source `eaeb5276`, passed in development scene q2:

| Case | Receipt |
| --- | --- |
| Observed 0.3 m translation | Success; estimated endpoint residual 2.1 mm, yaw residual 1.61 degrees |
| Subsequent 90-degree turn | Success; estimated XY residual 2.5 mm, yaw residual 0.12 degrees |
| Unknown/out-of-grid goal | Rejected before issuing motion |

Artifacts: `/home/cpaxton/runs/emet/grapheqa-navigation-20261030-r3/`.
They contain registered RGB-D, poses, images, timestamps and shared executor
receipts. There were 38 executor primitives, no VLM calls and no detector calls.
This is one short control check, not navigation robustness across environments.

Two earlier failures are preserved:

1. `20261030_180403_b4e6d2`, source `40e1140a`: 4 cm geometry aggregation left
   isolated unknown floor cells; full-footprint admission disconnected the
   robot's reachable region. The default 1 cm geometry voxels restored floor
   connectivity without filling unknown cells or weakening hazard rejection.
2. `20261030_180646_6a6bab`, source `4b51a9be`: the new primitive adapter switched
   an in-place turn to steering toward a tiny XY correction as odometry drift
   crossed a separate 2 cm threshold. It oscillated in heading and failed. The
   repair preserves turn intent, uses the shared 2.5 cm position tolerance and
   rejects excessive drift. A regression test covers this failure.

A replay of the earlier ten learned RGB-D packets through the persistent native
worker also passed: graph updates at frames 5/10 and a complete context with an
observed frontier at frame 10. This is cached-stream integration validation,
not a new online-segmentation episode. Replay output:
`/tmp/grapheqa-native-stream-yfto0tvp/`.

Validation: 63 targeted emet tests and 21 fork API/controller/protocol tests pass.
The upstream native simulation tests require its isolated Hydra runtime and are
not included in the host-only count.

## First model-driven diagnostic

At emet `983e409c` / fork `c94c480`, jobs `20261030_181600_a659c7` and
`20261030_181913_6d9bd7` completed q2/q6/q12 without infrastructure errors. All
three answers were wrong. q2/q6 returned no action and low confidence; q12
stopped confidently. No model-selected motion was executed. This does not pass
active-agent acceptance despite successful HTTP/graph plumbing.

The saved context exposes distinct causes:

- q2 has eight reachable frontiers, but Qwen chooses no action while admitting
  that the rug is not visible. This is a planning failure, not a missing tool.
- q6 has no reachable frontier because the fixed zero-height assumption labels
  measured floor near -0.125 m as a drop. Replaying **identical RGB-D** with a
  floor reference measured under the stationary occupied footprint restores
  1,894 reachable cells and eight frontiers. Calibration is startup-only; mixed
  height/insufficient returns are rejected, and subsequent drops remain hazards.
  It does not query scene geometry or move the prescribed spawn. Real robots
  require independent support/mount calibration; this convention repair is
  confined to the external Habitat adapter.
- q12 treats the lack of a bedside-table label as proof of absence. The current
  view shows a sofa/doorway, not a complete bedroom survey. The model's confident
  answer is retained as wrong; no harness rule replaces it.

The first adapter also left the head down after floor inspection. The repaired
acquisition now restores the task-facing tilt before assessment, retaining
floor measurements in the map. The next matched development check repeats all
three IDs with startup floor calibration and this view restoration. It is not
a larger benchmark sweep or a claim of improved accuracy.

## Next live gate and expansion

From clean committed emet and fork checkouts, run the existing managed API check
with `--active-question-ids 2 --max-decisions 4`, using the root `.venv` Python.
It owns the local Qwen8B int4 server and starts the Habitat child in
`.venv-habitat`, preserving the GPU lock. Keep CPU affinity 12, all BLAS/OpenMP
thread counts at one, caches offline and the 20 GiB free-space reserve.

The initial acquisition sequence is fixed: current frame, nine five-degree
left turns, three five-degree downward head steps, stationary measured floor
calibration, and three upward steps restoring the task-facing tilt. Every RGB-D
frame is segmented online and integrated. Model decisions then select observed objects
or reachable frontiers. Motion and registration failures are returned as
feedback, not interpreted as visual absence or used to replace an answer.

After this one-case wiring gate, use the already selected q2/q6/q12 development
population. Do not call an uncertain no-action response a navigation pass. Match
pose/perception/motion budgets when connecting the emet side. A matched active
emet comparison and matched-history image-only controls remain acceptance work;
do not compare this new sensor-only row directly with historical supplied-pose,
navmesh-assisted scores. Expand to the fixed twelve-question pilot only after
both sides meet those contracts. No full sweep is needed at this stage.
