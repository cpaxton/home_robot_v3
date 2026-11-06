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

## First three attempts

All three ran serially at `6a2a0aef`, without model calls or scorer access.
All 486 recorded frames have accepted registration receipts and zero reported
collisions. Accepted odometry is not ground-truth localization accuracy.

| Scene binding | Frames | Completed legs / 2 | Estimated camera path m | Wall s | Outcome |
| --- | --- | --- | --- | --- | --- |
| q2 | 126 | 0 | 0.433 | 41.15 | Fresh sweep rejects unobserved floor after about 0.35 m displacement |
| q6 | 176 | 1 | 0.815 | 58.80 | Second leg stops on 25.17 mm estimated position drift during a turn |
| q12 | 184 | 2 | 1.545 | 59.54 | Both legs and arrival scans finish |

Jobs are `20261106_065232_7b313c`, `20261106_065501_4e8a98`, and
`20261106_065735_48fa39`. Raw histories and uniform RGB/map panels are under
`/home/cpaxton/runs/emet/exploration-history-20261106/`. The standalone
`scripts/analyze_exploration_history.py` verifies ordered frames, saved PNG/depth
hashes and aligned depth timestamps before exporting figures. Manual review of
all three panels confirms full-room yaw coverage and small translations; it
does not establish visibility of the queried targets. q12 still inspects the
living room and doorways rather than entering the bedroom.

The first q6 leg arrives with 2.50 mm position and 0.000188 rad yaw residuals
relative to odometry. Its next turn stops with 25.17 mm position residual and
0.353 rad remaining yaw. These are separate outcomes; do not label the whole
episode successful or loosen the 25 mm bound after observing the miss.

## Geometry sampling and replay qualification

The initial collector did not fix/save Torch's CPU RNG. `SparseVoxelMap.add`
uses `point_update_threshold=0.9`, randomly retaining 10% of incoming geometry.
Thus frozen RGB-D and source pins alone do not exactly reproduce these maps.
An independent q2 replay admits the original route but does **not** reproduce
the final map exactly. Preserve this limitation; do not present its point counts
as an exact reconstruction of the original failure.

That replay shows a relevant mechanism at local cell (-0.45, 0.10) m: measured
floor support remains, but points 30.5–32.7 mm below the fixed reference change
the cell from observed to unknown. `floor_evidence` disqualifies a cell when
*any* point lies outside its 30 mm floor band. Neither drop nor raised-hazard
flags are set in this replay. This explains how additional observations can
reduce certified floor coverage; it does not prove the world is safely flat.
No tolerance, hazard or motion-admission setting is changed.

The collector now seeds NumPy/Torch acquisition with declared seed 0 and saves
the exact CPU mapping RNG state before its first update. One separate q2 seed-0
repeat is predeclared solely to verify exact CPU map replay from saved packets
and RNG. Original failures remain original outcomes; this is not an additional
independent scene or an accuracy comparison. Preserve both replay drivers and
their hashes with the artifacts. Full active agents will also need their map
sampling stream isolated from model inference before making strict pairing claims.

### Seeded check result

Job `20261106_070407_e1d7c1`, source `3666a9b9`, completes inference-free
collection with the same q2 rejection after 126 frames (40.91 seconds). It is
retained separately at `q2-seed0/`, not pooled into the three-scene outcome.
Replaying its saved packets with the saved CPU RNG state produces **identical
final obstacle and explored masks**. The original route also passes admission
at frame 79 in this exact replay.

At frame 99 the diagnostic cell changes from observed to unknown despite
retaining 24 points within the floor band: two points at -30.52 and -31.48 mm
fall just beyond its -30 mm lower boundary. No drop/raised flag is set for that
cell. This confirms the classification mechanism on a reproducible realization,
without establishing that the physical floor is safe or that odometry is exact.
The exact replay took 9.10 seconds on CPU 12, without rendering or inference.
Its first launch hit the Habitat environment's existing libstdc++ import-order
conflict; using the collection driver's import order resolves that setup error.
Both setup logs and the final driver are retained, not benchmark task rows.

The output retains `floor-diagnostic/result.json` for the original non-exact
replay and `q2-seed0/floor-replay/result.json` for the exact seeded replay, with
driver hashes. The repo's `paper/data/exploration_history_20261106/summary.json`
records the source-linked aggregate and figure hashes. Seven new contracts and
34 existing packet/navigation contracts pass (41 tests); pre-commit checks pass.

## Next physical gates

Before an active paired evaluation, replay this floor-conflict case alongside
clear floor, true drops, raised obstacles and missing depth. Assess measured
support, pose/calibration uncertainty, and what a bounded fresh inspection can
recover; do not mark conflicting returns as missing sensor data or erase them
with a visited disk. Preserve hard terrain/occupied-space rejection.

For the q6 turn, separate odometry drift from physical base displacement through
an independent scorer-side audit. Keep simulator state out of agent inputs.
Use the same measured executor, pose basis, footprint, map sampling and action
budget for the existing emet agent and GraphEQA adaptation. These are physical
contract gates, not reasons to tune Qwen answers or force its movement.

The histories, including failed attempts, are available for predeclared frozen
image-selection comparisons. They do not yet demonstrate useful target views,
full EQA improvement, small OVMM acceptance or a matched active baseline result.
