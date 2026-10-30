# Observed native TAMP execution

Status: attachment, reference tracking, visual-effect checks and the explicitly
injected native/task adapter are implemented and pass synthetic integration tests.
The default RGBD provider remains read-only pending live source assembly and
sensor/geometry qualification.
October 29: synchronized simulator images/motors and motor-only rigid EE FK pass
live development checks. Full observed transfers remain unqualified: symmetry,
unmeasured gripper geometry and contact/occupancy integration are the next gates.
Built after live-capture binding `ae924531`.
First target: stationary Stretch, rigid opaque objects, observed tabletop supports,
and the existing `scene_tasks` / primitive / multi-step JSON tools. Preserve the
privileged control as an independent regression and evaluator. Add Sourccey only
after passing the same gates. Do not start by adding more task-search baselines.

## Starting state on October 26, before the body-independent adapter

- Live registered RGBD capture, bounded semantic-reference reassociation, capture
  ages and motion invalidation exist. Discovery produces visible positions and
  heuristic support patches; it does not provide executable object geometry.
- `PhysicalPickPlaceExecutor` accepts an observed pose provider, but payload
  attachment/replanning still calls `collision.set_payload(model, data, body,
  ee_body)`. That assumes an object body in the MuJoCo scene. Its pose contract
  does not express uncertainty or symmetry.
- `NativeTableAdapter` synchronizes full simulator qpos/qvel, discovers model
  bodies and verifies gripper/support contacts, body velocities and final bounds.
  It is explicitly privileged; changing a source string cannot make it observed.
- `PlacementScene`, `HeldObject` and `PlacementCollisionChecker` already support
  scene boxes and an EE-relative payload volume. However `known_free` is optional,
  the checker is not a drop-in native attachment adapter, and the old voxel arm
  checker is a 2D map projection. Neither is a complete observed native provider.

Keep MuJoCo for the robot's calibrated kinematics/collision geometry, using a
robot-only model. Environment/object observations must not be simulator bodies.

## Milestone 1: separate evidence from simulator bodies

Introduce an `ObservedNativeAdapter` alongside the privileged table adapter,
implementing the same tool/session boundary. Extract shared robot-only IK, arm
command conversion, private planning data and measured-joint settling; retain the
existing privileged adapter and receipts as regression controls.

Define immutable, revision-bound evidence records for:

1. An object identity, reference-frame ID, measured surface points, pose hypotheses
   or bounded SE(3) uncertainty, observable axes, capture identities and ages.
2. Payload geometry in that reference frame, its construction assumptions and
   uncertainty/coverage. Visible-point bounds alone are not a complete volume.
3. Workspace occupancy and known-free cells, support geometry, unknown-space
   policy and the revisions/captures establishing them.
4. Effect checks returning `satisfied`, `violated` or `unknown`, with evidence IDs,
   observation intervals and measured bounds. A controller acknowledgement is
   separate from an observed effect.

Add a geometric attachment interface accepting payload vertices/voxels and an
EE-from-reference transform. Clearing/attaching it must not read object bodies,
write object qpos, or invoke a scene attachment service. Keep public handles
opaque; diagnostics stay structured JSON, with existing execution-started and
preflight/partial receipt semantics.

Gate: the shared executor can plan and command a robot-only rig with supplied
observed evidence; tests deliberately make scene snapshots, body poses, simulator
contacts and attach/set-body RPCs unavailable. Existing privileged tests pass.

## Milestone 2: construct observed occupancy and payload bounds

Build a bounded manipulation workspace from calibrated RGBD. Integrate rays as
known-free only up to measured surfaces, with a noise margin. Surface endpoints
are occupied; space behind surfaces, missing returns and unobserved cells remains
unknown. Do not derive free space from the absence of accumulated map points.

Use object-specific masks across measured views to build a reference-frame
surface/occupancy model. Construct a conservative possible-object volume from
multi-view visibility and depth, retaining occluded ambiguity. If that volume
cannot be bounded inside the scanned workspace, request another view or reject.
Do not label the visible-point AABB as complete geometry. A known-model prior can
be an explicitly separate input tier, supplied independently of simulator truth;
it must not become an object-name-specific implementation or an open-world claim.

Track robot and target contributions separately from scene occupancy. Removing
the robot/held target must not erase nearby obstacles or support voxels. Keep
provenance for cells instead of cropping a large exclusion box around a target.

Adapt `PlacementCollisionChecker` to the native attachment interface, checking
full robot-link and payload volumes, dense path edges and self collision. Require
a known-free predicate for observed mode. Existing observed occupancy constructor
APIs currently allow that predicate to be absent; observed execution must not.
Handle intentional finger/object contact and terminal support approach explicitly,
by phase and geometric contact region, without exempting an entire receptacle.

Gate: observed planning rejects a narrow payload-only collision, occluded swept
space, a hidden/unbounded object extension, stale geometry, and an obstacle moved
between segments. It also finds feasible paths after additional views reveal free
space. Independently compare accepted paths against evaluator-only scene truth;
never feed that truth back into runtime geometry.

## Milestone 3: track a reference frame with observability

Extend local surface association into measured rigid registration over the object
reference model and fresh masks. Evaluate overlap, residuals, spatial support and
pose observability; calibrate thresholds against independent held-out scenes.
Reject background leakage, insufficient overlap and competing identities.

A reference coordinate frame may be defined on the initial measured model; it
need not equal a simulator/manufacturer body origin. Store that definition and
express every payload vertex and subsequent pose in the same frame. An arbitrary
initial coordinate convention is not evidence of physical orientation. Subsequent
motion must come from measurements, not propagation of commanded EE motion.

Represent symmetry or unobservable rotation explicitly. Do not force an arbitrary
unique yaw or report zero uncertainty from a low registration residual. Check all
relevant pose/geometry hypotheses; request another view or reject when the motion
or placement depends on an unresolved axis. Shape-invariant rotations need not
cause false rejection when every admissible volume satisfies the same constraints.

Gripper closure establishes only measured aperture/stillness. A provisional
attachment hypothesis is needed to collision-check the first lift, but it is not
a verified grasp. Keep the possible still-unheld target in the collision state
until lift/co-motion is observed; validate the initial bounded lift probe against
both outcomes. Promote attachment to verified only after that visual evidence,
then compare fresh observations with expected EE-relative geometry. A static or disappearing target
must not pass because the arm moved or object height was guessed.

Gate: known relative translations/rotations, symmetric objects, small visible
patches, texture/shape ambiguity, distractors, occlusion and camera-pose error all
produce appropriate bounded estimates or explicit unknown outcomes. No reference
geometry or timestamps silently renew when tracking is lost.

## Milestone 4: observed grasp and placement effects

Add a visual effect verifier; do not reuse `NativeTableAdapter._verify` as an
observed verifier. It may reuse robot measured-joint settling, not scene contacts
or free-joint velocities.

- Grasp/lift: the identified object moves with the measured gripper and separates
  from its previous support, with consistent relative geometry across captures.
  Height gain alone, finger proximity or successful closure is insufficient.
- Transport: keep measured retention evidence fresh. When occlusion or uncertainty
  makes it unavailable, stop with an unknown/partial receipt; do not silently
  substitute predicted attachment pose or simulator contact.
- Release/place: the object remains visible in the intended region after opening
  and retreat; its conservative footprint fits, its estimated bottom is consistent
  with observed support, and repeated captures establish settling. This verifies
  visual placement evidence, not a hardware contact sensor we do not possess.
- Distinguish unknown from measured failure. Unknown physical progress stops;
  automatic regrasp or held-object recovery is outside this first milestone.

Start with stop-and-observe transitions and stationary base/head during
manipulation. Pose/capture alignment still needs explicit timing bounds: local
receive freshness alone is insufficient. Use measured settling and fresh captures
with validated pipeline latency bounds, or proper sensor-clock alignment. Reject
when the timing relationship cannot be bounded; do not claim that a new image ID
alone proves capture after motion. Moving-time verification requires a separately
qualified joint/camera interpolation or conservative motion-error bound.

Gate: dropped object, empty closure, object pushed upward without grasp, target
occluded at lift, stale camera stream, release outside the region, unsupported
placement and an unsettled object cannot pass. Include successful release/retreat
and symmetry-tolerant retention cases to avoid a verifier that only rejects.

## Milestone 5: same agent and tools, held-out task trials

Bind observed predicates/effects into the existing task provider/session. Predicates
come from one coherent evidence revision; task planning and every execution segment
revalidate them. Unknown is not false: absence from a mask or map cannot become
`not_at`, a free destination, or an unprotected object. Initially admit a complete
symbolic snapshot only when the required goal, protected-object and action
predicates are observed; otherwise request inspection/reject before task search.
Keep the same greedy/beam/MCTS implementations, tool names,
prompts and execution policy. Capability reporting states exactly which observed
operations are qualified. No task discovery result alone enables execution.

Run the actual CLI with Caliban inference and no local model/GPU requirement:

1. Single transfer on held-out initial placements and object geometries.
2. Two-goal transfer and a buffered swap, including occupied destinations.
3. Explicit failures: missing/stale evidence, ambiguous identity, tracking loss,
   moved obstacles and infeasible placement. Count valid task failures; separate
   fixture-construction/runtime crashes and preserve all attempts.
4. Navigate to the workspace, invalidate old plans/references, acquire a fresh
   stationary scan, then run the same multi-step tools. Exploration and viewpoint
   recovery become scored capabilities only when this whole chain is exercised.

Independent evaluator truth scores identity, collision violations, retained/lifted
objects, support, settling, final constraints and false-positive verifier decisions.
Runtime code receives only sensor/joint observations and declared robot calibration.
Include sensor misregistration, delayed frames and self-occlusion. The current head
simulator hides robot geom groups for mapping; correct sensor rendering plus
robot-point filtering must be qualified before claiming realistic self-occlusion.

Report task success, false-positive/unknown verification rates, collision/unknown-
space rejection, recovery-boundary compliance, VLM calls, capture ages and planning/
execution latency. Freeze source, scene population, budgets and scoring rules before
paper comparisons. Sensor smoke counts are not a task-success denominator.

## Implementation order and review units

Use the existing feature branch for the plan and small contract work. Keep code
changes reviewable in dependency order: (1) simulator-independent evidence and
attachment seam, (2) 3D free/occupied/unknown geometry, (3) measured pose hypotheses,
(4) visual effects, (5) observed adapter/session and actual agent trials. Contracts
and their first real consumer should land together; avoid adding unused scaffolding
or one PR for each small correction. Keep privileged controls green throughout.

The first implementation should target milestone 1 plus the first real geometry
consumer from milestone 2. This exposes the deepest coupling before investing in
a pose estimator that the current body-based executor cannot actually use.


## October 26 implementation checkpoint

Implemented the first concrete consumer of observed geometry:

- `PayloadGeometryEvidence` records an opaque object handle, reference frame,
  immutable envelope vertices, construction tier, geometry revision and measured
  pose. The provider supplies the complete bound; this change does not construct
  one from a single visible mask or qualify pose observability.
- `ObservedNativeCollisionChecker` reuses full robot/payload volume checking on a
  robot-only MuJoCo model. Its native `set_payload` seam transforms supplied
  reference geometry into the EE frame, never resolving the object handle as a
  simulator body. It requires a bounded observed workspace and whole-volume
  known-free checks, rejects stale/changed scene evidence, and binds retention to
  the original geometry reference/revision. Release retains a conservative static
  object volume for retreat; it does not verify physical release.
- `ObservedWorkspace.from_depth` constructs an immutable 3D snapshot from registered
  depth, intrinsics and the optical camera pose. Complete projected cell volumes
  must precede all valid returns by an explicit margin before becoming free;
  clipped, missing, out-of-range and occluded volumes remain unknown. Occupied
  endpoint cells cannot erase the surrounding unknown region. It does not fuse
  views, remove target/robot contributions, establish capture timing, or infer
  complete payload geometry.
- Native collision checkers now capture/restore their complete attachment state
  around speculative executor replanning. Expected missing observed geometry
  produces a failed preflight result before arm/gripper commands.

Synthetic robot-only regressions exercise the real executor and measured-joint
command seam without object bodies or scene/contact/attachment RPCs. They are not
physical grasp or sensor qualification. An offline replay of the saved live ZMQ
capture also checks that missing depth creates no free cells; see
[the implementation evidence](../experiments/observed_native_geometry_20261026.md).

Next: implement the actual `ObservedNativeAdapter`, coherent bounded pose/geometry
production and visual effects before binding agent execution. In particular, the
current strict two-second scene snapshot will reject long manipulation without a
freshly rebuilt snapshot; do not extend its age or renew timestamps to bypass the
missing refresh integration. Keep observed task discovery read-only until those
gates pass, then run the frozen Caliban agent trials in milestone 5.


## October 26 execution integration checkpoint

`RigidRgbdTracker` now produces fixed-reference measured rigid poses from distinct
RGB material features and paired depth points; RANSAC rejects outliers and poor
spread, and lost/ambiguous tracking clears cached poses. It requires a current
instance mask and a separately supplied complete prior. Textureless/symmetric
orientation remains unknown; no canonical yaw is invented.

`ObservedVisualVerifier` is consumed by the physical executor's observed grasp
and place paths. It requires bounded capture intervals, explicit alignment/error
bounds, measured robot brackets and fresh repeated co-motion or post-retreat
settling evidence. Full calibrated gripper volumes must clear the object. Missing
evidence is unknown; a command acknowledgement, closure or height gain cannot
substitute. Observed placement requires a verified lift reference. Hardware-clock
gripper settling no longer depends on private MuJoCo simulation time advancing.

`ObservedNativeAdapter` now binds the existing primitive JSON tools to robot-only
motion refinement and visual effects, refreshes actual scene evidence, preserves
single-use plans and records the actual command boundary. `bind_regions` connects
it to the same task search/session and protected-goal machinery. A three-transfer
swap passes through the real tools in a synthetic sensor/controller fixture.
CHAT has an explicit Python factory hook; no CLI/default provider is silently
enabled. External motion invalidates both primitive and task handles.

Validation: 610 checks passed with zero skips/failures on committed implementation
`c532a758` in managed CPU job `20261026_205621_349c68`; no new live sensor, contact or LLM agent task denominator.
See [the integration evidence](../experiments/observed_execution_integration_20261026.md).

Next is concrete live source assembly/qualification, not another symbolic search
baseline: mask/identity production, complete payload geometry and uncertainty,
clock/inter-sample motion bounds and robot/target occupancy provenance still need
real sensor qualification. Current depth carving alone does not remove those
couplings. Preserve unknown space and all failed attempts, then exercise held-out
stationary Stretch tasks with Caliban before claiming observed agent success.

## October 27 live audit

Managed CPU/Caliban job `20261027_001043_cb29cc` passed 5/5 fresh surface refreshes,
but the actual rigid tracker's feature gate rejected the 217-pixel surface (15
interior pixels, zero landmarks). The source still lacks qualified timing,
complete payload geometry, occupancy provenance and support grounding. Default
planning/execution correctly refused and sent no extra commands. This is a live
prerequisite audit, not observed task acceptance. Prioritize a suitable measured
view and source assembly before a full native agent trial; see
[the evidence and next steps](../experiments/observed_live_readiness_20261027.md).

## October 27 source assembly progress

Actual 2× head renders with self-occlusion exposed proposal overflow; measured
local-plane normals fixed that condition without changing depths or thresholds.
Both repeats passed 5/5 sensor refreshes. The 800-pixel object still yields only
five material landmarks and remains pose-unknown. Static scan fusion and target
endpoint provenance now preserve unknown/obstacle cells. Simulator acquisition
intervals and same-host clock mapping are implemented and exercised live; the
RGB/depth union spans 203.5 ms and does not establish encoder alignment. See
[all three attempts, validation and remaining source gates](../experiments/observed_source_assembly_20261027.md).
Observed execution remains disabled; do not infer a transfer/swap score.

## October 28 support/enclosure milestone

A measured partial-support producer now projects full grid footprints into
observed mask/depth evidence, fits a near-horizontal plane, and remeasures fresh
geometry without filling unknown holes. The explicit semantic-seeded support
path produced an 83x10 cm fresh patch with 0.64 mm residual in live CPU/Caliban
trial `20261028_085139_b4f42d`; the earlier broad-proposal overflow is retained.
Measurement margins are not yet independently calibrated.

A free-space-enclosure constructor now rejects visible-only object bounds when
the possible connected volume reaches a scan boundary. All five recorded frames
remain `object_geometry_unbounded` in offline spatial analysis. It preserves
unknown/background contacts rather than carving them away; support-contact
separation or explicit shape priors are still needed. No object yaw is inferred.
See [validation and remaining gates](../experiments/observed_support_enclosure_20261028.md).

## October 28 geometry tracking checkpoint

Merged `main` at `cd169673` via `1844e4be`, preserving both multi-turn VLM history
and request timing. Geometry fitting now shares an interface across explicitly
declared primitive priors and measured multi-view surface models. Distinct poses
and unobservable symmetry remain explicit; stale/lost fits are discarded.
Geometry co-motion checks account for shape symmetry but do not qualify temporal
alignment or contact.

Live CPU/Caliban repeat `20261028_182754_73e62d` passed 5/5 geometry fits and 5/5
observed discovery refreshes with a declared cylinder prior. The first attempt
retains its 1/5 discovery failure: independent support grounding consumed the
semantic reference lifetime. Scheduling support after object refreshes resolved
that diagnostic without changing age gates. Surface residual is approximately
0.30 mm, not pose accuracy. Main/source regression: 748 checks; final focused
geometry suite: 9 checks, overlapping. Paper builds at 52 pages.

See [evidence, figure and remaining gates](../experiments/geometry_tracking_20261028.md).
Symmetric raw fits still cannot substitute for the native interface's unique
pose. Independent calibration, encoder alignment, moving-instance segmentation,
observed enclosure/contact policy and concrete source consumption remain before
live observed task execution; no new task-success denominator is claimed.

## October 29 sensor-input checkpoint

Co-sampled simulator images now retain their actual camera pose and a separate
normal eight-channel motor packet. Motor-only rigid EE FK matches the separate
full-robot debug reference in 5/5 live development captures. Full-qpos debug
contains passive finger/telescope coordinates and is explicitly rejected by
observed effect admission; it cannot substitute for real robot sensors.

One repeat rejected a frame after fitting exceeded the original two-second age
gate. Deduplicating only exact declared shape-symmetry search starts resolved the
latency issue without relaxing age/geometry checks. The successful fast repeat
fit in 0.40–0.47 s. A JSON fixture/manifest generator makes the development scene
and robot-only model reproducible from packaged assets.

Final affected regression: 787 checks with zero failures/errors/skips. Each of two
privileged calibration runs passed both native contact controls. The phased audit
places the large 0.358 rad finger discrepancy at fixture initialization, while
execution contact reaches 0.00827 rad coupling error and 0.11877 rad rubber angle.
These are correlated debug measurements from development controls, not certified
geometry bounds or new observed task successes. The paper still builds at 52 pages.
See [all attempts, input tiers and remaining source work](../experiments/observed_encoder_sampling_20261029.md).

Next implement the shape-symmetry payload contract, bound unmeasured gripper/link
geometry from actual motor signals, associate moving masks independently of
commanded motion, and assemble phase-specific contact/unknown
space policy with persistent map/support evidence. Keep original evidence times
and explicit static-scene assumptions. A fresh image cannot renew hidden map
cells, and partial support observations must not silently become a complete
fixture. Then run the first complete stationary observed transfer through the
same public tools before held-out multi-step or exploration trials.

## October 29 moving-mask / gripper checkpoint

Implemented a separate shape-equivalent payload contract through public planning,
retention and visual effects, preserving unknown material yaw. Whole-tool gripper
volumes now use analytic intervals over unmeasured joints, anchored at motor-only
rigid EE FK. Two native controls pass; all 175 sampled envelopes contain actual
collision geometry, including 75 contact samples. These are conditional model
checks, not certified global or hardware bounds; full-arm uncertainty remains.

Moving RGBD association now follows whole current components within a declared
motion envelope and latches ambiguous/lost identities. The four rendered-stage
cases pass, including 24 cm translation/8 cm lift and explicit distractor/occlusion
stops. Actual head-camera contact testing is still incomplete: the final
registered RGBD repeat tracks 11 captures, then stops on insufficient geometry
near the gripper as target visibility drops from 1349 to 513 pixels. Privileged
control success does not turn that sensor failure into observed task acceptance.

810 affected regression checks and 65 overlapping final focused checks pass.
[Evidence, all attempts, figure and videos](../experiments/observed_moving_geometry_20261029.md)
identify the remaining view/occlusion problem. Next qualify wrist/head view
continuity, then assemble support/map/contact/full-arm policy in a concrete source.
The first full observed transfer remains pending; do not skip to multi-step scores.

## PR #226 review: independent self-occupancy blocker

The review identifies a planning failure before the observed transfer milestone,
independent of the head-mask visibility problem. Robot surfaces occupy their own
depth map; their interiors and shadows remain unknown. The current checker
requires every whole robot-geom AABB to be free and therefore rejects the initial
configuration. A MuJoCo ray/depth regression now reproduces `invalid_start`, both
before and after perfect self-endpoint exclusion. Synthetic adapter/swap tests
use an externally supplied all-free predicate and validate the interface only.

Keep the PR scoped to an explicit integration foundation. The working live arm
executor remains blocked until a separate source/query policy can establish:

1. Calibrated robot ownership tied to the same capture/motor pose, retaining mixed
   obstacle cells and original evidence times. An outer geometry envelope is not
   proof that all enclosed space belongs exclusively to robot material.
2. Distinct robot-interior and environment-free evidence. Unobserved space behind
   a robot and hollow/empty regions inside its AABB must not become free. Use
   AABBs as broad-phase bounds, with a conservative material-volume query where
   self ownership matters; exempting an entire outer box is not sufficient.
3. Actor/phase-specific robot, payload and support queries, with any mechanical,
   clearance or static-scene assumptions explicit. Changing the contact phase
   must not authorize arbitrary unknown swept volumes or erase nearby obstacles.
4. Tests with self-occlusion, a nearby obstacle, mixed ownership and movement into
   an unseen region, followed by the actual stationary observed transfer.

More informative wrist/head views remain needed for tracked payload evidence,
but do not substitute for this self-occupancy policy. The checker deliberately
continues to reject raw self-occluding maps; no unsafe carve or bypass is added.

## October 30: head/wrist visibility and historical background

The native contact development repeat now completes with one continuous RGBD
track over 81/81 captures, including release and retreat. Head-to-wrist handoff,
whole-fragment union under a declared complete-shape prior and opt-in neutral
highlight handling resolve the fixture's visibility failures. Wrist intrinsics
now match the actual rendered dimensions/FOV; physical calibration is unchanged.
Negative tests retain ambiguity/loss, incompatible geometry and replay stops.

`ObservedBackground` supplies original per-cell age and explicit static-scene
validity to native free-space queries. A prior depth scan can support a short path
through later self-occlusion; expiry and unseen-volume motion still reject. No
robot AABB is declared free. This is query plumbing, not a qualified live scan,
robot-interior policy, support certificate or phase-specific contact exception.

Next assemble those policies in the actual observed source, with full-arm and
camera/encoder uncertainty, then run the stationary public-tool transfer. The
default source remains read-only and full observed task acceptance remains open.
Keep this separate from the privileged control, diagnostic candidates and any RL
training work. [Attempts, protocols and video](../experiments/observed_handoff_20261030.md).
