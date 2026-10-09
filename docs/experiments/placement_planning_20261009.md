# Placement planning branch (2026-10-09)

The October 8 diagnostic smoke (`20261008_174720_1cbb35`, source `4f16f27a`)
failed at placement base approach. Command 47 was rejected with
`teleport_endpoint_collision`: the proposed base intersected the dishwasher and
nearby cabinetry, with maximum reported penetration 0.116262 m. Pickup/lift
passed; placement arm motion never began. API output correctly reported partial
completion. This establishes a geometric approach failure, not a tracking failure.

The isolated `feat/placement-collision-planning` branch starts at `046a4eae`.
It adds generic 3D scene/payload geometry, bounded multi-candidate placement
search, and kinematic-executor integration. See [the API](../apis/placement.md)
for geometry assumptions and unsupported execution guarantees. There are no
object-category exceptions or relaxed measured-arrival thresholds.

Tests exercise real MuJoCo FK/IK with synthetic geometry: bulky and offset
payloads, rotation, arm-volume collisions away from link origins, obstacles along
edges with clear endpoints, alternate base candidates, multiple complete paths,
source-state preservation, occupied voxel height, unknown-space predicates,
missing geometry and execution failure propagation.

The previous failed smoke remains the baseline. A new full TAMP success claim
requires rerunning that same actual CHAT task and its final-scene scorer after
integration, followed by the admitted-task repeat matrix. Unit tests establish
geometry/search contracts, not real-robot readiness.

Validation before simulator submission: 136 targeted tests pass across placement,
TAMP API/search, kinematic base frames, physical motion contracts, arm RRT and
MCTS planning. Ruff passes for all changed Python files.

The payload-detour regression routes around an obstacle blocking the direct path
and rechecks the returned path at 0.005 joint-coordinate spacing. RRT construction
and final validation now both use at most 0.025 spacing. A superseded smoke
(`20261009_073710_72783d`, source `91f29ef6`) was cancelled while waiting for the GPU;
it produced no simulator result.
