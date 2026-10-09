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

Final managed CHAT smoke: `20261009_073929_9b876c`, frozen source `dbb4cac2`
in `/tmp/emet-placement-validation-final-20261009`. Artifacts:
`~/runs/emet/placement-collision-final-20261009/`. At PR submission it is waiting
for the shared GPU lock; no new full-task success result is available yet.

The initial full CHAT smoke finished **failed**: pickup/lift passed, but placement
submitted 49 base candidates to a command accepting at most 32. No placement path
search or base movement occurred. The adapter now queries batches of 32 and 17,
validating every response before search. Regressions cover order preservation and
an invalid second response. Placement plus TAMP tests: **65 passed**; Ruff passes.

## Stacked smoke and geometry repair

`20261009_074642_72de9c` at `0b1a34fe` reached placement search, but failed with
43 simulator endpoint rejections and 6 payload/scene rejections. No placement
motion was executed. Reconstruction from the saved scene and measured arm state
(inferred base height 0.300538 m) identified a backsplash's whole visual AABB
spanning the room interior. Its actual declared collision components are three
thin boxes. Using individual collision-component bounds leaves clear candidates
in that reconstruction. This is supporting diagnostic evidence, not a successful
live execution result.

The simulator now publishes disjoint collision-component bounds alongside semantic
bounds; placement prefers the components. Regression coverage includes a hollow
fixture with a room-sized visual box, serialization, and updated component poses.
No body/category is excluded to obtain clearance.

Review hardening adds runtime/timeout handling, occupancy-change detection with
fresh-path revalidation, private collision buffers, transport attachment-offset
verification, and removal of the unused single-approach helper and float-key lookup.

The new collision component field is preserved through the actual session reader
as well as server serialization; the hollow-fixture test exercises that round trip.
The queued `20261009_075535_03f90f` run was cancelled before acquiring the GPU to
include this reader repair.

Additional reconstruction with a 5×5 support grid rejected all 25 sampled targets:
12 preplace volumes intersected a mug, 8 intersected lettuce, 3 intersected a paper
towel object, and 2 final placement volumes intersected countertop geometry.
These counts describe the saved-scene reconstruction and current top-surface
interpretation, not a proof of task infeasibility or a live success/failure result.
The planner now checks target occupancy before spending its IK budget. No obstacle
is removed and no target is silently substituted to obtain a passing score.

## Full stacked readiness submission

- Source: `f743caf2`; checkout `/tmp/emet-placement-full-stack-20261009`.
- Managed job: `20261009_080207_3f0e72`.
- Artifacts: `~/runs/emet/placement-full-readiness-20261009/`.
- Protocol: actual CHAT smoke, fresh admission of the original scene00/02/12
  fixtures, then three fresh-process repeats for admitted cases. Failed admissions
  remain failures; missing cases do not become successful trials.
- Validation before submission: **176 tests passed**, Ruff and whitespace clean.
- At submission the full gate is queued for the shared GPU; readiness is unproven.
