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

## Free-region placement before rearrangement

Placement now subtracts clutter footprints enlarged by the payload footprint from
explicit support patches. An off-grid-slot regression demonstrates a feasible
region missed by the former 5×5 lattice. The full vertical approach volume must
be clear, not just its endpoints. Separate patches cannot support an object by
bridging a gap. Region budgets and blocker volumes remain available for future
rearrangement reasoning; no blocker is automatically removed.

The simulator publishes support patches separately from semantic object bounds.
The initial supported GT extraction is horizontal top faces of axis-aligned box
collision geoms. Mesh/rotated supports require an explicit geometry provider;
missing surface geometry fails closed. This prevents treating an appliance's
whole visual bounding-box top as a measured support surface. Observed providers
can supply one or multiple explicitly grounded support patches through the same API.

The grounded-surface reconstruction found 35 free payload candidates on interior
support patches, while its original search spent all 96 IK calls at the first
base pose. Per-base quotas now prevent that starvation. A regression with an
unreachable first viewpoint and a reachable later one requires the later solution.
This distinction matters: free object space, arm reachability and approach-path
clearance are separate tests; failure of one does not by itself justify clearing
objects from a countertop.

## Current validation queue

- Final code source: `1335d5d5` in `/tmp/emet-placement-fair-search-20261009`.
- Full stacked readiness job: `20261009_081750_e342cc`.
- Artifacts: `~/runs/emet/placement-fair-search-20261009/`.
- **182 tests pass** across the placement and existing TAMP/motion contract suites;
  Ruff and whitespace checks pass.
- Earlier queued jobs `20261009_080207_3f0e72` and
  `20261009_081232_2f7efa` were cancelled before acquiring the GPU so the full gate
  runs the current code once. They have no simulator result.
- The full gate still waits for the shared GPU. No full-task success is claimed.

## Completed readiness result (supersedes queued status above)

Job `20261009_081750_e342cc`, code `1335d5d5`, finished with exit 1.
`results/readiness.json` reports CHAT exit 1, **0/3 admitted fixtures**, and no
repeat results; `ready=false`. The nine planned repeats were not executed.

- CHAT: pickup/lift passed; placement returned zero paths. Of 49 base candidates,
  43 failed endpoint checks, three failed robot/scene checks, and one failed
  payload/scene checks. The remaining search recorded 20 RRT iteration-limit
  failures, four pose-IK failures, and two per-base budget events. Final object
  error was 2.2031 m. This is search failure, not proof that rearrangement is needed.
- Scene00: one placement exhausted 96 pose-IK calls across eight approaches; a
  later placement had no accepted surface candidate. A grasp also failed joint
  bounds.
- Scene02: all three object attempts failed measured grasp/pregrasp tracking.
- Scene12: placement failed with `Explicit support patches required`; subsequent
  attempts failed tracking. These later attempts are not independent diagnostics:
  the clutter runner continues after failure without establishing held state.

Separately, two approximate offline reconstructions exited with SIGSEGV; Python
faulthandler located one crash at the pose-IK named-joint lookup. An isolated
96-call IK loop completed. A 90-second gdb attempt ended without a native crash
trace. These are unresolved diagnostic failures, not the cause established for
the live gate. See the [merge review](../plans/2026-10-09_tamp_merge_and_metrics.md).

## Follow-up repairs

The scene12 missing-support failure was reproduced from its saved MJCF. Its bin
has five collision boxes, but measured rotation entries around 1.6e-5 exceeded
the old exact-axis 1e-6 test. Support extraction now inscribes a conservative
rectangle in a box face's XY projection, with arbitrary yaw and at most 1 mrad
tilt. Sloped/mesh faces remain unsupported. Tests check every rectangle corner
against the oriented face and reject larger tilt; 18 metadata tests pass. This
repairs support availability, not a claim of successful live placement.

The exact live CHAT snapshot from job `20261009_121359_46c309` exposed stale
held-object geometry: EE Z was 1.291729 m, while cached payload bounds still
spanned Z 1.101594–1.190639 m at the original tabletop position. The simulator
attachment callback patched `pos` only. It now recomputes pose, bounds, collision
components and support patches from live MuJoCo state for all cached bodies
sharing the moved freejoint ancestor. A regression exercises the actual server
callback, parent/child motion, rotation and an intentionally wrong command target.
Metadata/contact suites: 34 passed. This identifies a stale-volume bug; a fresh
full gate is still required to establish its effect on placement success.

The exact pre-fix snapshot replay (`20261009_122347_8b00fe`) completed normally
with zero solutions, 22 preplace RRT budget failures and two IK failures. It did
not reproduce the earlier approximate native crash; this does not prove the
crash resolved. Diagnostic gate `20261009_121359_46c309` finished with CHAT failure,
0/3 admissions and no repeats. Support extraction reached IK in scene12 rather
than failing for missing patches; uncertain-state continuation stopped as intended.

The next gate (`20261009_122159_920b79`, `7e20a9b3`) failed CHAT at lift tracking
(51.38 mm against the unchanged 35 mm bound), before testing placement. Scene00
also failed pregrasp tracking. These are not a placement-success measurement.
Metadata refresh is now deferred from every attachment physics tick to telemetry
publication and uses kinematics only; tests require contact and acceleration
buffers to remain unchanged. This removes unnecessary physics-loop work without
relaxing tolerances; it is not proof of the cause of the tracking failures.

Arrival-clock review found that trajectory sleeps used the simulator rate but
`_wait_measured_ee` used a fixed three-wall-second deadline. The wait now uses the
same existing capped ratio scaling and logs its actual wall budget. Clear delayed
tracking can finish within its simulated-time allowance; a blocked joint still
fails at the deadline. Contact/lift evidence suites: 23 passed. This is a timing
contract change, not a relaxed distance bound or an equal-wall-budget comparison.


## Integrated gate and closed-appliance diagnosis

Integrated code `2fcbfbd1` passed 259 focused CPU tests. Managed job
`20261009_123154_9bfdbc` runs that immutable source. Its CHAT phase passed
pickup/lift, then returned zero placement paths (18 pose-IK failures, 18 RRT
budget failures). Detach was not attempted. Admission phases are still running
at this checkpoint; no full-task success is claimed.

The command line requests bowl → microwave, but its explicit CHAT calls filter
`scene_tasks` only by bowl and select the first result. The actual selected goal
is dishwasher. The verification correctly checks that selected goal; this is a
fixture-intent mismatch, not a false positive in the final success check.

Exact snapshot:
`~/runs/emet/tamp-integrated-20261009/snapshots/daa7e5efe0e74d27a6c680c45360056a`.
Its 15 door-component AABBs exactly match the asset's default closed dishwasher
door (box indices 57–71). A bounded, six-connected voxel diagnostic inflated
scene boxes by the fixed measured-orientation payload half extents plus the
5 mm planner margin and half a cell. Results:

| Diagnostic | Resolution | Targets connected to each of six starts |
| --- | --- | --- |
| Full payload, original scene | 20 mm | 0/35 |
| Point payload, original scene | 10 mm | 0/35 |
| Full payload, only door boxes removed | 20 mm | 31/35 |

The four remaining targets intersect conservative occupied cells. Bounds extend
0.6 m beyond starts/targets. Voxel discretization and bounded search can reject
real paths; no arm collision checks are included. Door removal is a diagnostic
ablation, not a valid opening action or new benchmark outcome. Nevertheless,
matching the closed-door geometry and changing only those components provides
strong evidence that access, rather than just sampling budget, blocks this CHAT
placement. Automatic articulation is not part of the current three-operation
approach/grasp/place plan. No planner obstacle was removed in production.

The same ongoing gate's scene00 admission has found three placement paths both
before and after base transport. That is search evidence only; release, retreat
and complete cleanup must finish before reporting task success. The gate remains
not ready because CHAT failed regardless of later admission results.

## October 11 acknowledgement-timeout diagnosis

Command receipt publication held the dispatch lock while deep-copying up to 256
receipts, including retained base-query contact results. The concurrency regression
blocks that copy deliberately and verifies that a new command still dispatches,
that concurrent updates do not mix receipt revisions, and that published results
cannot mutate tracker state. It fails before the fix. Publication now captures
receipt dictionaries under the tracker lock and copies their owned, replace-only
results outside both locks. The command timeout, receipt capacity, identity,
collision checks, and measured-arrival thresholds are unchanged. The 26 focused
command protocol tests and Ruff pass.

The initial October 11 crash diagnostics were contaminated by the diagnostic
`faulthandler.dump_traceback_later` watchdog on this host's Python 3.10.12. A native
core shows its `dump_frame` freeing code/frame memory concurrently with a servo
thread. A Python-only receipt-copy reproducer passes roughly 172,000 iterations
without the watchdog and segfaults immediately with frequent timed dumps. These
crashes must not be attributed to MuJoCo, GPU, CPU affinity, or robot contact.
Periodic dumps were removed; the retained native analysis is under
`~/runs/emet/tamp-timeout-both-20261011/core-thread3.txt`.

The allocator-instrumented diagnostic reached a measured pre-place pose (3.32 cm
position error, 0.065 rad orientation error), then failed closed with
`placement_attachment_changed`, before release. It is not a placement success.
Clean live validation without the periodic watchdog is required before claiming
that the original acknowledgement timeout is resolved.

Clean baseline `20261011_111410_11a9ae` at `7a190fc0` completed without an
acknowledgement timeout or crash, but rejected changed attachment geometry after
base transport. Its GDB wrapper exit status is not the task score: the public
execution envelope is partial, with release not attempted.

Fixed-source validation `20261011_111933_a934cf` at `65fd0cef` completed 407
acknowledged commands with **zero acknowledgement timeouts**, median 0.0511 s,
maximum 1.3587 s, against the unchanged 5 s acknowledgement deadline. The full
CHAT sequence was `scene_tasks → set_receptacle_state(open) → plan_pick_place →
execute_pick_place_plan` in `molmospaces_ithor_train_0` with RBY1, a bowl and the
dishwasher. This is explicitly assisted kinematic execution with GT geometry and
teleport base navigation, not native-contact or learned-agent acceptance.

The 360-waypoint pre-place segment reached measured position error 0.033735 m
and orientation error 0.070452 rad, within the existing 0.035 m / 0.1 rad limits.
The next refresh rejected `placement_attachment_changed`: corresponding EE-local
payload corners differed by up to 0.028188 m; the center differed by about 0.026 m.
The public result is partial / `placement_invalidated`, with release not attempted.
The source currently derives this volume from asynchronous world AABBs, while the
assisted latch preserves position but not relative orientation. Establish coherent
rigid attachment/geometry evidence before changing that guard; do not increase its
tolerance to make this trial pass.

Artifacts: `~/runs/emet/tamp-timeout-fixed-20261011/` (job log, command timings,
payload geometry and replay snapshots). The 54 command/runtime, navigation retry,
ZMQ protocol/parity and Stretch snapshot tests pass; Ruff is clean. These results
validate the dispatch-lock regression and this live timeout check, not a causal
claim that every earlier timeout had the same source, nor full placement success.

## October 11 rigid attachment and coherent payload geometry

The acknowledgement fix exposed `placement_attachment_changed` after pre-place.
Two defects were present: assisted attachment maintained only EE-relative position
(the free object could rotate), and refresh transformed asynchronous world AABB
corners into a newer EE frame. AABB corners are not corresponding material points
when the object rotates.

The server now preserves the full measured relative pose, including when the
semantic object is a welded child of a free root. Privileged payload metadata
measures a stable root-local volume and EE together under the simulator lock.
It covers the entire free-root assembly using actual mesh vertices and
conservative primitive bounds. Observed providers are unchanged; no privileged
fallback or synthetic hardware sensor was added. The 1 cm attachment guard,
35 mm arrival threshold, 0.1 rad orientation limit and command deadlines remain.

The first run, `20261011_132632_ece95e` at `a169d8d7`, rejected all support
candidates: using each mesh's rotated local AABB inflated the bowl footprint to
about 27 cm. The corrected implementation bounds actual mesh vertices in the
stable root frame; a tetrahedral-mesh regression detects this inflation.

Full public CHAT rerun **passed**: `20261011_133133_a84009`, source `10076be1`.
Artifacts: `~/runs/emet/tamp-attachment-tight-20261011/` (`job.log`, two private
search snapshots). Configuration remains `molmospaces_ithor_train_0`, RBY1,
bowl → dishwasher, `scene_tasks` → assisted open → plan → execute, with GT
geometry, object latch/placement assistance and base teleport. EGL rendered on
the GPU; `--cpu-only` disables neural models, not rendering.

- Initial search returned two paths; measured-pose search returned three.
- Pre-place executed 505 waypoints; place executed 11; retract also completed.
- EE position errors: pre-place 0.033263 m, place 0.032666 m, retract 0.032779 m.
  Orientation errors: pre-place 0.098691 rad, place 0.011669 rad. These pass the
  existing limits; the pre-place orientation is close to its limit.
- Corresponding payload vertices changed by less than 9e-16 m across transport
  and pre-place refresh. A deliberate 3 cm slip still fails the unit regression.
- Public execution returned `status: ok`, all operations completed, detach
  returned, and `placement_verified: true`.
- The existing final-scene smoke check passed: measured object displacement
  2.3536 m and distance to selected receptacle center 0.1545 m (<0.25 m).
  This is a proximity check, not a sustained support/contact audit.
- 575 successful acknowledgements; maximum latency 1.210183 s; no execution
  timeouts. Cleanup then rejected `cancel_navigation` with “cancellation target
  has no confirmed stop”. That separate shutdown/control-handoff issue remains;
  it occurred after the successful task and final-scene check.

Final targeted suite: 117 tests across payload frames, contact tracking,
placement, simulator geometry publication and the TAMP API; Ruff passes. A final
error-path regression ensures missing payload volume is omitted (placement then
fails closed) rather than interrupting simulator telemetry/command receipts.
That error-path hardening follows the GPU-tested source above.

This is one successful assisted control, not a repeatability rate, learned-agent
result, native-contact acceptance, or real-robot readiness. Repeat the admitted
task/robot matrix and strengthen settled support evidence before broader claims.

## October 12 review fixes and broader acceptance triage

Review fixes at `35b40446` reject singular support faces without losing obstacle
geometry, suppress articulation-jitter rescans while retaining cumulative drift,
convert payload-center release targets to the actual body origin/orientation, and
remove the unused grid sampler. The observed-geometry constructors remain provider
APIs; no live observed provider is supplied by this PR.

The seven-case run `20261012_082533_8dc0aa` at `48d5307b` completed **3/7**.
Artifacts: `~/runs/emet/placement-broad-20261012/` (protocol, summary and per-case
logs/results). This used the same public four-call assisted flow, selecting the
first returned bowl task. All cases, including unsupported tasks, remain counted.

| Case | Outcome | Failure stage |
| --- | --- | --- |
| RBY1 kitchen 0, repeat 1 | Pass | Full assisted task completed |
| Stretch kitchen 0 | Fail | Open rejected: articulation unsupported |
| Sourccey kitchen 0 | Fail | Simulator initialization, before any TAMP command |
| RBY1 kitchen 1 | Fail | No accepted support candidate; no placement IK/path execution |
| RBY1 kitchen 2 | Fail | First task targeted oven; open rejected as unsupported |
| RBY1 kitchen 0, repeat 2 | Pass | Full assisted task completed |
| RBY1 kitchen 0, repeat 3 | Pass | Full assisted task completed |

The three passing repeats had maximum acknowledgement latencies 1.664225,
1.511503 and 1.461880 s. Each subsequently reported the separate shutdown
cancellation rejection. Kitchen 1 rejected 40 of 49 base endpoints and produced
zero support targets; this is not proof that the task is infeasible or requires
rearrangement. Unsupported articulation must be handled at task admission or by
an appropriate alternative plan, not relabeled as successful placement.

Sourccey's startup failure was reproduced on current main `7f65dbef` and PR
`32303c8c`, with the same regenerated kitchen-0 scene and MuJoCo 3.5.0. The actual
server load/auto-placement/home sequence produced base x=2.691665650967536e22 and
y=-5.688219143012023e22, followed by `FactorizeHessian: rank-deficient sparse
Hessian`. Plain model load/home without auto-placement succeeded on both. This
localizes a pre-existing initialization problem, not a placement regression or a
completed root-cause fix. Scripts, logs, scene hash and provenance are preserved
in `~/runs/emet/placement-merge-triage-20261012/`.

Review-head smoke `20261012_083715_a1db36` (`35b40446`) failed before release:
pre-place position error 0.034923 m met the 0.035 m limit, but orientation error
0.118330 rad exceeded 0.1 rad. The wait returned as soon as position converged,
then tested orientation once. Payload geometry remained coherent (<1.4e-15 m
corner changes), with 521 acknowledgements, maximum 1.108636 s, and no execution
timeout. It did not exercise the corrected release behavior.

Commit `32303c8c` waits for position and orientation together under the same
existing deadline. It does not loosen either tolerance or increase the budget.
Four regressions cover delayed orientation convergence, sustained orientation
error, sustained position error, and stale observations. All **243 changed-file
tests pass with zero skips**; the focused affected suite passed 101 tests, and
Ruff passed. Pytest's periodic traceback watchdog was disabled because of the
separately reproduced host-level CPython watchdog crash.

The revised full public CHAT smoke **passed**: `20261012_101729_f324e0`, frozen
source `32303c8c`, terminal job status `done`. Artifacts:
`~/runs/emet/placement-pose-wait-20261012/`. Initial search found three paths;
post-transport search found two. Pre-place executed 675 waypoints and place 12.
Measured position/orientation errors were 0.034867 m / 0.068379 rad at pre-place
and 0.034591 m / 0.009639 rad at place. Retract position error was 0.034029 m.
Public execution returned `ok`, detach returned and `placement_verified` was
true. Payload corner changes stayed below 2.04e-15 m. There were 751 successful
acknowledgements, maximum latency 0.707331 s, and no execution timeouts.

The final proximity smoke passed (object displacement 2.3539 m, distance to the
receptacle center 0.1947 m, threshold 0.25 m). Cleanup then reported the existing
cancellation rejection. The revised assisted live smoke gate is satisfied;
sustained support/containment, cross-robot coverage and native execution remain
unproven. All these runs use privileged geometry and assisted execution.
