# Task-independent RGB-D history pilot

The [November 3 frozen answering pilot](frozen_eqa_comparison_20261103.md)
qualified three answer paths but exposed limited target coverage. The next
development step is better acquisition, without tuning GraphEQA's model or
repeating until a correct answer appears.

## Declared collection

Use the existing q2, q6, q12 scene/spawn bindings, in that order, one managed GPU
job at a time. These are inspected development scenes, not a holdout. The
collector receives no question, choices, labels, detector output or scene
geometry. It reads only scene/floor from the dataset to initialize the renderer.
No answer model runs during collection.

Each history uses the existing `SensorOnlyHabitat`, RGB-D odometry,
`ObservedSpaceNavigation`, and `HabitatMotionExecutor`:

- Same 0.1 m circular body and 0.05 m local observed-floor grid as the GraphEQA
  adaptation. No global pose, semantic sensor, route oracle or full-scene bounds.
  Habitat still applies its internal navmesh motor collision filter.
- Initial 360-degree yaw panorama in calibrated 5-degree primitives. Preserve
  the existing stationary 45-degree scan / downward floor calibration / head
  restoration before completing the panorama. Every frame is saved.
- At most two frontier legs, each an admitted route prefix of at most 0.75 m
  cumulative planned translation, followed by a fixed 90-degree yaw scan.
  Choose the shortest admitted frontier route, stable candidate order for ties;
  exclude previously attempted targets within 0.35 m. No semantic ranking.
- At most 200 attempted primitives and 240 seconds after sensor initialization.
  All yaw/head/translation primitives count. Every operation checks the budget;
  an in-flight synchronous primitive may finish after its time deadline. Budget
  expiration is retained as partial collection, not successful exploration.
- Stop on a failed measured motion or rejected registration. Do not silently
  replace the case, add retries, reduce clearance, or supply a commanded pose
  as a measured pose. No admitted frontier is also a retained outcome.

Record lossless RGB, metric depth/calibration, timestamps, full odometry
receipts, attempted commands, map masks, candidate/rejection events, full motion
receipts, source/configuration pins and stage outcomes. Path length is estimated
camera motion, including drift during turns, not a global distance score. There
is no target-visibility, EQA or task-success metric in this collection.

## Resource and replay gates

Check registered/unmanaged GPU work before each short job; RL retains priority.
Use `emet jobs --gpu-exclusive --cpu-safe`, CPU 12, one BLAS/OpenMP thread,
cached environments and offline assets. Never queue all three behind RL or run
them concurrently. Keep the shared 20 GiB disk reserve and preserve earlier
experiments. No real robot commands are involved.

```bash
PYTHONPATH=src /path/to/existing/habitat/python \
  scripts/collect_exploration_history.py --question-id 2 \
  --output /path/to/new-collection/q2
```

Before replay, audit registration receipts, actual movement, final map, and
uniformly sampled RGB for every history including partial failures. Freeze
histories before asking any model. Then compare question-only/current/uniform
and selected history under predeclared image/token budgets. Construct native
GraphEQA and emet memories from the same learned packets when testing memory;
do not substitute an empty-graph answering row for the full harness.

Full active agents still need their physical/input contracts matched. This
collector establishes reusable acquisition evidence; it does not close that
comparison, qualify production exploration, or change agent policy.
