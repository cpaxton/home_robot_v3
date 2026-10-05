# Executed alternative approaches (2026-10-01)

**All three fresh-process repeats passed: 3/3 full cleanups, 9/9 object
relocations.** This validates the preselected approaches on the original scene00
cleanup fixture. It is assisted GT/MCTS kinematic-latch execution, not autonomous
approach selection, physical grasp acceptance, or an expanded benchmark score.

> **Superseded.** The preselected-approach adapter
> (`scripts/validate_tamp_approaches.py`) is now replaced by online approach
> selection with read-only collision queries (`check_base_poses`). See
> [automatic approach selection](tamp_auto_approaches_20261001.md).

## Fixed intervention

The preceding [contact diagnosis](tamp_contact_tracking_20260930.md) identified
wall-intersecting potato and apple approaches. A static audit sampled sixteen
55 cm standoff directions per object, finding seven and nine clear,
position-IK-reachable alternatives respectively. Before execution, this run
selected the first qualifying candidate in audit order for each exact object.
The kettle retains its original clear approach.

| Object | Base X (m) | Base Y (m) | Yaw (rad) |
| --- | ---: | ---: | ---: |
| Potato | 1.530677 | 1.956429 | -0.785398 |
| Apple | 0.680915 | 1.960673 | 0 |
| Kettle | 1.241190 | 1.735251 | -1.570796 |

`scripts/validate_tamp_approaches.py` supplies the existing `approach_pose`
candidate argument to the shared MCTS grounder through an explicit diagnostic
adapter. Its two regression tests check collision/IK filtering, exact object
identity, immutable candidate inputs, and rejection of missing or ambiguous
selection evidence. No scorer or fixture file is modified. The output manifest
records the assisted scope, chosen poses, selection rule, source and input hashes.

## Protocol

Three fresh-process repeats were declared before seeing results, using the same
poses, scene, seed and source. They run serially under managed GPU exclusivity
and CPU affinity. Each repeat has a 900-second wall limit. The internal IK budget
is 8.75 mm; measured arm arrival remains 35 mm and lifted-object target-distance
acceptance remains 80 mm. Base endpoints still undergo the new collision check.

Managed job: `20261001_005055_4719fa`.
Frozen source: `a36b27f0` (original r3 harness plus tested controller repairs and
validation adapter). Main follow-up adapter commit: `e4845913`.
Artifacts: `~/runs/emet/tamp-validated-fixtures-20260928/approach-validation-r10/`.
Each repeat contains the unchanged task-score JSON, initial scene evidence,
CSV and an `approach_validation.json` manifest. The root contains process logs,
runner script and archived server traces.

## Results

| Repeat | Full task | Objects relocated | Accepted arm motions | Max arrival error | Max lifted-object target error |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | pass | 3/3 | 18/18 | 34.902 mm | 6.062 mm |
| 2 | pass | 3/3 | 18/18 | 34.891 mm | 6.079 mm |
| 3 | pass | 3/3 | 18/18 | 34.892 mm | 6.004 mm |

All 54 measured arm motions and all nine lifted-object checks passed. There
were zero reported motion failures. Manipulation wall time was 87.2–89.4 s per
repeat. The managed job is terminal (`done`). All three objects completed
approach, grasp and placement in every repeat; no pose selection was changed
between trials. `summary.json` preserves these counts and measured extrema.
Arrival errors are the first accepted measurements, not settled-state residuals.

## Interpretation

The intervention isolates approach choice from controller repair: the previous
repaired-controller run relocated only the kettle (1/3 objects), rejecting the
two wall poses before movement. The selected alternatives are evaluated with
that same controller and acceptance limits.

These are repeats of **one fixture**, not three distinct tasks or evidence of
broad environmental robustness. Simulator base teleportation and object latch/
placement assistance remain enabled. RBY1's visual-only arm links also prevent
full physical collision certification. Measured arrival checks establish the
required EE position at acceptance, not settled tracking or grasp orientation.
The next gate is shared-planner approach generation and collision-aware selection,
followed by fresh trials without the preselected-pose adapter.
