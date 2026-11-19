# Observed native TAMP integration

This is the contract/diagnostic foundation in
[PR #226](https://github.com/cpaxton/home_robot_v3/pull/226). Live self-occupancy
and phase-specific contact admission remain tracked in
[issue #245](https://github.com/cpaxton/home_robot_v3/issues/245). Merging the
foundation does not qualify or automatically enable a live observed executor.

The production CHAT tools can now use an explicitly injected
`ObservedNativeAdapter`. It shares `scene_tasks`, `plan_pick_place`,
`execute_pick_place_plan`, `pick_place`, and optionally `plan_task` /
`execute_task_plan` with the privileged control. No simulator body names, object
pose writes, scene snapshots or contacts are used by this adapter.

**Status:** implementation and synthetic integration regressions pass. A complete
live RGBD source is not assembled or qualified by default. The usual RGBD
`scene_tasks` provider remains read-only. This is not an observed-task benchmark,
a contact-execution result, or a hardware sensor-availability claim.

**Raw-depth planning blocker:** a self-occluding head-camera map marks robot
surfaces occupied and their interiors/occlusions unknown. The default checker
requires complete known-free AABBs and rejects that start. The opt-in
`ObservedMaterialAdmission` path now separates captured, calibrated robot
material from newly swept space. It queries actual solids, erodes material by
its supplied uncertainty, preserves environment occupancy, and requires measured
free evidence outside that material. Candidate poses never construct ownership.
Meshes require explicit solid declarations; hollow CAD/AABB volumes cannot be
promoted to owned material. Source-backed raycast regressions cover the start,
unknown shadow, measured prescan, false labels and original evidence ages.
Phase-specific target/support policies now have opt-in implementations. Full
Stretch uncertainty qualification and live source assembly remain required
before claiming a complete observed transfer.
Its synthetic transfer/swap fixture supplies `known_free=lambda _: True` to test
tool orchestration. This is not evidence that a real self-occluding map admits
those paths. This planning blocker is separate from moving-mask visibility loss.

## Components

| Component | Responsibility | Rejects / limits |
| --- | --- | --- |
| `RigidRgbdTracker` | RGB feature correspondences, forward/backward optical flow, measured RGBD landmarks, RANSAC rigid fitting against a fixed reference | Lost masks/depth, repeated or insufficient texture, poor feature spread, inconsistent motion, replay, reference age >30 s; no global re-identification |
| `MovingRgbdTracker` | Current whole measured appearance components and declared geometry/motion profile | Lost/ambiguous associations latch; diagnostic candidates, not calibrated global identity |
| `MultiViewRgbdTracker` | Continuous head/wrist association within one source-synchronized capture; explicit complete-prior fragment and lighting profiles | Cached/misaligned views, global loss, competing or incompatible geometry; ideal shared-snapshot diagnostic, not hardware timing qualification |
| `RobotViewGuard` | Reject camera association if an original winning depth ray may hit motor-derived rigid tool/camera outer geometry | Rejection only: never trims pixels, labels self, frees cells or proves absence of interference outside its named scope |
| `ContinuousRgbdSource` | Process original camera captures in a background producer independently of tool/LLM reads; publish immutable candidate/frame pairs | Stale, changed/replayed, lost or invalidated evidence stops the source; does not assemble the support/background/collision evidence required by `ObservedNativeAdapter` |
| `RgbdCaptureGeometry` | Retain original camera calibration, motor-derived EE and conservative gripper volumes with each immutable RGBD view; assemble visual-effect frames | Mixed captures, changed geometry profiles and world-point mismatches reject; explicit simulation assumptions do not qualify hardware or a complete collision scene |
| `ObservedBackground` | Historical environment/free-space evidence with original per-cell capture ages and geometry-qualified self exclusion | Unqualified excluded endpoints invalidate cached free cells; occupied evidence survives. False self labels remain environment when qualification is requested. Longer validity requires a bounded static-scene assumption |
| `ShapePoseEvidence` | Explicit primitive solid equivalence with external pose-error bounds | Unknown material yaw remains unknown; generic unique-pose consumers reject |
| `GripperGeometryEnvelope` | Full tool subtree over unmeasured joint intervals, anchored at motor-derived rigid EE | Conditional model bounds; no passive-joint sensor claim or full-arm certificate |
| `StretchGeometryEnvelope` | Whole collidable robot over motor-conditioned telescope marginals and passive-joint intervals | Outer geometry only; mechanical limits and localization margins remain explicit assumptions, never inner material or free space |
| `PayloadGeometryEvidence` | Complete conservative volume in an explicit tracked reference frame | No visible-point AABB promoted to a complete shape; geometry priors remain declared inputs |
| `ObservedNativeCollisionChecker` | Robot-only FK, payload geometry and optional captured-material solid queries | Missing workspace, stale geometry, scene revisions, unresolved payload geometry or unqualified candidate bounds |
| `CapturedRobotMaterial` / `RobotGeometryBound` | Immutable captured inner material and model/calibration/profile-bound candidate envelopes | Future-pose ownership, relocated/undersized links, clock/calibration changes and uncertain clearance shells |
| `ObservedTargetContact` | Complete capture-bound target envelope, exact fingertip geom IDs and bounded penetration in named motion phases | Whole-body exemptions, deep penetration, vanishing target cores, stale/replaced profiles; never exempts environment or unknown space |
| `ObservedSupportContact` | Original finite support depth/mask witness, explicit solid-layer/motion-error priors, bounded payload contact and fixed lift-departure floor | Mixed/unattributed cells, unknown space, robot contact, hidden backside without a solid prior, stale or replaced capture/calibration |
| `VisualEffectFrame` | One tracked pose with capture interval, error bounds, bracketed robot feedback, full calibrated gripper volumes | Receive time alone, matching joint endpoints alone, missing alignment bounds, changing calibration, excessive uncertainty |
| `ObservedVisualVerifier` | Fresh repeated visual lift and post-retreat placement effects | Height-only lift, static/slipped object, occlusion, replay, unsupported/outside/unsettled placement, gripper volume still near object |
| `ObservedRegionTransferProvider` | Complete observed role/region predicates and the existing task search/session interface | Missing roles, unsupported/held objects, ambiguous occupancy, unknown destination space |

The feature tracker supplies measured relative rigid motion from material features; it
does not infer a canonical object yaw from a symmetric/untextured shape. Such
objects return unknown in the feature tracker. The separate shape-aware path
below preserves their declared geometric symmetry. Registration residuals are
diagnostics, not uncertainty estimates: position/orientation admission bounds
must come from independent calibration. Automatic current-instance mask
production is experimental and bounded by colour/geometry continuity;
long-term identity recovery and default live source assembly are not supplied.

The multi-view diagnostic defaults to strict single-component association.
`component_policy="complete_prior_union"` explicitly permits fitting the whole
union of reachable fragments against a declared primitive; it never selects a
subset by fit quality. `allow_clipped_views=True` retains measured boundary
points but still requires pose observability. `appearance_model="neutral_specular"`
is an opt-in saturated-colour lighting profile; default `"chromaticity"` behavior
is unchanged. It removes neutral highlights rather than relaxing geometry or
motion admission. These declarations are recorded with each diagnostic result.
The [head/wrist contact repeat](../experiments/observed_handoff_20261030.md)
retains one track through all 81 captures while privileged native control
completes. This qualifies neither the live collision source nor hardware error
bounds. Different real camera exposures require independently bounded alignment;
the development producer uses one frozen simulator snapshot.

The paired diagnostic now archives the same-snapshot eight motor positions,
original depth image/intrinsics, depth-to-RGB transform, winning source pixel
index and continuous projected UV. `register_depth(..., return_provenance=True)`
returns these mappings alongside the unchanged aligned-depth image; holes use
index `-1` and NaN UV, and equal-depth ties choose the first row-major source.
Registration does not establish RGB/depth surface ownership. Camera world poses
in this diagnostic are labelled simulator localization debug, not observed
navigation or hardware calibration. `--capture-period-s` defaults to `0.25`;
larger values delay real acquisitions without changing their timestamps or the
tracker's existing 1.5-second maximum gap. Setup/processing may make actual gaps
longer than that requested minimum. See the
[November 16 progress and retained failures](../experiments/observed_transfer_progress_20261116.md).

`bound_stretch_robot(model, estimate, now=..., position_error_m=...,
rotation_error_rad=...)` consumes motor feedback and the estimated localized base
pose. It intersects each telescope's mechanical interval with measured total
extension and retains uncertain intermediate-link volumes. After checking serial
ancestry and parallel axes, the distal rigid frame uses the measured sum; it does
not claim the individual split is sensed. Wheels, passive fingers, rubber and the
navigation-camera hinge retain their model intervals. Supplied pose errors dilate
all output volumes. The motor/model signature includes named tendon terms and
coefficients; changing that calibration invalidates the capture. These bounds
cannot by themselves justify deleting robot-labelled depth endpoints.

`scripts/eval_stretch_encoder_geometry.py --robot-envelope --output DIR` audits
all collidable geometry against separate simulator state during native contact
controls. `--gripper-envelope` retains the earlier tool-only scope. An outer-bound
escape or whole-robot interval violation fails the robot audit; sampled
containment still does not certify mechanical/hardware bounds or task execution.

## Provider assembly

`SynchronizedRgbdView.geometry` optionally retains immutable `RgbdCaptureGeometry`.
Its camera unprojection must match the view's world points. Motor feedback and
gripper geometry must describe that exact original capture; a later robot read
cannot substitute. Tracking pins camera/model/error profiles, and duplicate
handling compares the retained geometry as well as pixels. `visual_frame(...)`
uses the stored fit and robot brackets without reacquiring either.

The explicit schema-1
[`observed_stretch_geometry_debug.json`](../../configs/eval/observed_stretch_geometry_debug.json)
supplies **simulation assumptions**, not measured hardware calibration or errors
inferred from fit residuals. Only the ideal shared-snapshot simulator source can
use its zero temporal-alignment term. Robot FK/localization errors remain
nonzero and propagate separately through relative retention, rotation, EE lift
height and retreat checks. Rotation uncertainty includes the EE-to-object lever
arm. Gripper envelopes retain their pose-error budgets so they cannot silently
be paired with another profile. Retreat reserves robot pose error conservatively,
including when a source's boxes already contain a reserve; unproven separation
is unknown, not evidence of actual contact. Historical references validate at
the later of their original image receipt and robot bracket time, without retiming.

The [coherent-frame integration record](../experiments/observed_visual_frames_20261117.md)
separates live visual-frame assembly from the remaining background/material/
contact-source/refiner and actual observed pick/place gates.

The camera source has a separate candidate-level contract:
`ContinuousRgbdSource(tracker, initial_views=..., read_views=...).start()` calls
a bounded-I/O reader independently of consumers. `snapshot()` returns the current
candidate and its original immutable views. `collect(after=..., min_span_s=.2,
timeout_s=2.)` returns two fresh original captures strictly after the barrier.
It skips stale history instead of retiming it. Identical retransmissions can have
a later transport receipt, but retain the original published packet and freshness.
Changed source intervals or contents under the same ID reject. Tracking loss,
staleness and `invalidate()` latch; an explicit new reference/source is required.
Call `close()` and successfully join the producer before closing its camera or
client. A blocked reader produces a shutdown error, not a successful close.
This source is not directly interchangeable with the full adapter source below.

`RobotViewGuard.from_registered_depth(...)` binds a capture's original winning
depth rays, RGB-grid unprojection, calibrated camera pair and explicit pose/error
profile. Any possible rigid housing/camera interference rejects that entire
camera; another synchronized camera may still supply valid evidence. Whole-view
or all-view loss never silently restarts identity. Outer visual/convex geometry
is used only for rejection. Passive fingers, other robot links, depth ownership
and workspace free space remain unqualified by this guard.

The [November 17 camera-source diagnostics](../experiments/observed_camera_source_20261117.md)
preserve normal-cadence association failures as well as delayed-camera failures.
Continuous processing is necessary plumbing, not a demonstrated fix for them.

The subsequent [same-capture transport investigation](../experiments/observed_rgbd_transport_20261117.md)
isolates the live first-update failure to JPEG RGB colour corruption: original
RGB passes with either original or ordinary transported depth. An explicit
lossless Stretch wire profile (`EMET_SIM_RGB_CODEC=png`, legacy JPEG default)
then passes two stationary live camera diagnostics, including one without the
optional raw-provenance channel. This closes that transport/cadence diagnostic,
not the distinct private-snapshot robot-interference, collision-source or full
observed-transfer gates. Sensor/model uncertainty budgets remain required.

`ObservedNativeAdapter(robot, source=..., refine=..., tasks=...)` is deliberately
explicit. `source.input_mode` must be `observed`.

Configure `robot_name="stretch"` explicitly for that calibrated robot when
using a named selector. Both primitive and task discovery normalize case and
whitespace. An unspecified identity accepts an omitted selector but cannot
claim to match a named robot merely from a profile string.

The source provides:

| Method | Contract |
| --- | --- |
| `snapshot(object_ref, support_ref)` | A fresh `ObservedNativeSnapshot`: temporally qualified `VisualEffectFrame`, background `PlacementScene`, coherent scene revision, observed support bounds |
| `current_revision()` | Revision of the current background geometry; changes during a segment stop execution |
| `collect(object_ref, support_ref, after=...)` | A bounded batch of at least two distinct fresh frames captured strictly after the requested client-monotonic barrier, spanning the required settling interval |
| `invalidate()` | Clear old/in-flight tracking and geometry evidence after external base/head/view motion |

A source must separate robot and target occupancy by provenance, without erasing
nearby obstacles or declaring occluded space free. `ObservedWorkspace.from_depth`
provides conservative 3D depth carving, but **does not** perform that separation,
fuse views, bound the whole object, or qualify sensor timing. These are remaining
source-assembly requirements, not inferred behavior of the adapter.

An explicitly configured snapshot may include `admission=ObservedMaterialAdmission(...)`.
Its background geometry and captured robot material must match the snapshot and
robot model. Each candidate `RobotGeometryBound` must contain the nominal geom
plus the declared displacement and collision reserves; the checker validates
its model, geom, calibration and configuration-profile identity. Every refresh
keeps that profile pinned. Generic GT placement behavior is unchanged.

An opt-in `target_contact=ObservedTargetContact.from_frame(...)` snapshot binds
the complete payload prior and its external pose-error bounds to exact calibrated
fingertip collision geoms. Supply the same policy when constructing the checker.
Only grasp, close, lift, preplace, place and release permit bounded fingertip
contact. Pregrasp and retreat retain ordinary clearance; wrist/housing geometry
always retains it. Contact allowances cannot erase background occupancy or
certify unknown space. A target with no provably nonempty eroded core rejects.
After release the checker checks both the original release hypothesis and fresh
visual target geometry, so a displaced object remains an obstacle during retreat.

This policy requires an explicit `gripper_close_configuration` (finite motor
joint targets) and `command_gripper_joints` mapping on the robot-only executor.
Missing configuration returns `gripper_close_geometry_unavailable`. Planning
certifies the sampled closing path before any skill dispatch; execution refreshes
feedback and revalidates each segment, sends only its finite target, and requires
measured arrival and settling. The sample increment and arrival tolerance are
0.001 in model joint units (metres for slides, radians for hinges). Passive
joint uncertainty still requires outer geometry bounds; a motor acknowledgement
does not establish a grasp. The new command uses the same possible-dispatch
receipt boundary as arm commands. Full-close presets remain on the legacy path.
This is tested control plumbing, not qualification of a real Stretch closure,
calibration, contact-force bound or live observed transfer. The local support
policy below and a concrete live source/refiner still need live qualification.

An optional `support_contact=ObservedSupportContact(...)` on both snapshot and
checker binds one object/support instance to a finite `ObservedSupportPatch`,
its exact historical capture, original depth, support mask, intrinsics/extrinsics
and calibration ID. Source/profile replacement stops before command dispatch.
Only the named payload during lift, place or release can use this contact policy;
robot and wrist geometry retain their ordinary environment checks.

`ObservedBackground` now preserves every contributor to retained occupied cells,
including excluded foreground labels in mixed cells. Occupied updates union owners
and retain their oldest acquisition age; missing legacy provenance is background.
The ownership layer participates in the geometry revision. A newer foreground
observation cannot relabel an old obstacle as support. An immutable bounded capture
history pins the witness's complete original interval, not merely its ID.

Support-only occupied cells can be refined by bounded subdivision against the
original ray witness. The whole subquery must project into valid support pixels
inside the finite measured footprint. Regions before the nearest possible return
are measured free. Any refinement behind that return additionally requires
`backing_depth_m` and a nonempty `backing_prior_id`: an explicitly qualified finite
solid top layer covering the admitted band, including waived planning padding.
Depth endpoints alone cannot establish that material. Missing backing leaves it
blocked. Other/mixed occupied owners never consume this exception. A coarse
unknown subvolume can be certified strictly in front of every original selected
depth return, without changing its cell to free. Only an eligible shallow
payload contact can additionally consume the finite backing-layer prior there:
the material layer is continuous across voxel boundaries. A robot or noncontact
query can use only the original free-ray proof, never the backing material.
Missing pixel coverage, shadowed space and geometry behind the declared layer
remain blocked. No occupancy cell becomes free because contact is allowed.

The worst physical payload corner is checked against the uncertain **unit** plane
with at most 1 mm penetration. Payload pose error and the mandatory explicit
`payload_motion_error_m` remain in that physical bound; the latter must cover
encoder/FK/base/retention displacement over the motion hypothesis. Zero is used
only by declared ideal fixtures. Ordinary collision padding is separate and remains
active against unrelated geometry. Lift cannot deepen beyond one measured initial
clearance; IK/RRT query order cannot renew that floor. Budgets larger than the
permitted contact depth can correctly prevent admission; these tests do not
calibrate those budgets or justify relaxing the penetration limit.

After release, both the frozen release hypothesis and current measured object
geometry must pass. This does not prove settled support: the visual effect checker
still requires fresh repeated post-retreat evidence. Missing/replaced current
geometry stops execution, preserving possible-progress feedback after dispatch.

`capture_arrays` now preserves depth precision during observed unprojection,
avoiding the generic camera helper's float16 rounding at metre-scale distances.
This removes extra numeric quantization; it does not improve sensor accuracy or
resolve RGB/depth ownership. Historical tracking results are not rescored.
See the [support mechanics evidence](../experiments/observed_support_contact_20261117.md).

`ObservedBackground.from_capture/update(..., robot_material=...)` validates every
self-labelled endpoint against captured inner material before excluding it.
Missing or inconsistent proof preserves environment evidence. Without that
qualification, excluded endpoints cannot retain old free status. Hidden cells
without new endpoints retain their original age; a static assumption cannot
restore cells invalidated by unqualified labels. See the
[qualification plan and unfinished live gates](../plans/2026-11-18_tamp_agent_closeout.md).

Payload records carry position/angular error bounds for unique rigid poses and
shape equivalence alike. A visual frame must cover those bounds; attachment and
current-capture uncertainty both contribute to retention. Planning-only failures
return no-actuation receipts, while missing dispatch feedback after execution
remains possible progress. Public ambiguity/calibration errors are finite codes;
private provider exception details remain hidden.

Canonical payload vertices are kept separate from pose uncertainty. Collision
queries include object error once and the two EE captures' attachment error
once each. Retention uses each capture's relative-pose bound once; angular
displacement is already included in `attachment_error_m`. All candidate grasp
levers contribute to the reserved attachment limit, which is checked at actual
attachment and subsequent retention measurements. Requested placement regions
must contain the physical uncertainty envelope, excluding ordinary clearance.
Release and hypothetical-state restoration retain these bounds.

`RgbdCaptureGeometry.from_stretch(..., original_rays=True)` requires original
registration provenance matching the wire RGB, quantized depth, calibration and
capture intervals. It retains the original raw-depth grid/K/pose for map carving
and the exact winning original endpoints for RGB-associated tracking. Rounded
RGB pixel centres are not substituted for the original depth rays. This opt-in
path uses unquantized simulated sensor returns; it is a separate source profile
from the millimetre wire-depth reconstruction, not a hardware accuracy claim.
The cadence script exposes `--original-depth-rays`, requiring
`--registration-audit`, `--robot-only-model` and `--geometry-profile`.

The [mapped contact diagnostic](../experiments/mapped_contact_20261117.md) runs
public tools with a provided simulated RGBD background scan and **privileged**
native state. It is a separate assisted control, not observed-input acceptance.
Native collision checkers accept optional conjunctive `additional_scene_checks`
that return a strict boolean and receive the current checker/model/private data.
Native contacts always win; held/opening/released-fall clones preserve the
guards, so release planning cannot accidentally drop the collision map. Default
native checks retain an empty guard tuple.

The diagnostic's `--map-resolution` chooses 2 cm, 1 cm or 5 mm (default 5 mm);
`--placement-region` takes JSON world-XY bounds. `--fault` selects an explicitly
labelled occupied-map preflight control or loss of evidence after grasp. Raw
artifacts stay in `.tamp-runs/`; the compact experiment index records outcomes.
`ObservedEnvironmentAdmission` shares the strict environment/ray/support query
engine without captured robot material. It cannot grant self ownership or pass
the observed executor's material-admission type check.


The initial grasp reference now requests a genuinely post-motion frame through
`source.collect(..., after=barrier)`, just as lift/place verification does.
`snapshot()` may still return the latest cached fresh frame; it need not block
for a new acquisition. Empty collection, an acquisition before the barrier
(even with a new ID), changed clock/geometry, and incomplete visual evidence
remain failures. Acquisition timestamps are never renewed at receipt time.

`refine(task, snapshot, placement_region=...)` is planning-only and returns a
robot-specific `PhysicalPickPlaceExecutor` with a robot-only calibrated model,
measured-joint synchronization, normal actuator commands, open-gripper geometry,
and three grasp / three placement pose targets. `placement_region` is passed only
when requested. It must search/calibrate grasp targets independently of object
names. No controller calls may occur in this callback.

The adapter certifies those targets through shared IK/dense motion planning,
binds plans to the robot model, object/reference/geometry and scene revision,
and rechecks before execution. It expands the supplied payload envelope by the
initial admitted pose-error bounds; an increase in those bounds stops execution.
Scene refresh uses original capture receive times. New captures do not excuse a
changed scene during a path. Controller/gripper calls mark the actuation boundary;
pre-command observation failures remain `execution_started=false`, while failures
after a possible command return `partial`. Plans are single-use.

`execution_started=true` means **possible dispatch/progress**, not confirmed
actuation. The adapter marks entry into joint/gripper transport before calling
it. An ordinary `TimeoutError` cannot distinguish failure before send from send
followed by lost acknowledgement, so it returns `command_outcome_unknown` once
that boundary was entered. Only failures before transport entry can assert
no actuation under the current transport contract. Marking after a return would
incorrectly classify some sent commands as no-actuation failures.

`VisualEffectFrame` explicitly requires conservative `alignment_error_m` and
`alignment_error_rad` from qualified camera/encoder clock and motion bounds.
Two equal joint samples do not establish motion between those samples. A new
image ID or fresh receive time is insufficient. `CaptureInterval.calibration_id`
identifies the qualified clock/calibration bundle, not an arbitrary availability
flag. Current generic sensor observations do not automatically supply these
bounds; no adapter constructs them by substituting the local current time.

## CHAT and multi-step binding

Programmatic CHAT startup accepts
`run_agent_with_robot(..., observed_native_factory=factory)`. The factory receives
`(robot_client, executor)` and returns the explicitly configured adapter. Startup
requires physical mode with observed inputs and validates the returned robot
identity; errors close the client. There is no new CLI flag that fabricates an
observed source from the existing discovery provider.

A factory can call:

```python
adapter.bind_regions(
    regions,  # tuple of TransferRegion values grounded on one observed support
    robot_profile_id=calibrated_profile_id,
    reset_id=episode_reset_id,
)
```

This binds a `TaskSession` to the same adapter. CHAT installs it alongside the
primitive tools. Region role names come from the configured semantic task names;
all roles must be freshly observed and visually supported/settled. A destination
is free only when its entire required 3D volume is known free. Missing objects
cannot become `outside`, and missing map cells cannot become `free`. The existing
greedy/beam/MCTS implementations, protected-goal checks and failure stop policy
remain shared. Native articulation and autonomous physical recovery remain
unsupported.

Direct tool-registry callers can supply `context['observed_native_adapter']` and,
for multi-step tools, its `task_session`. No adapter is constructed merely by
setting an input label or providing a discovery proposal.

## Evidence and next gate

See [integration regressions](../experiments/observed_execution_integration_20261026.md)
and the [execution plan](../plans/2026-10-26_observed_native_execution.md).
The next gate is a concrete live source with independent geometry/clock
qualification, followed by held-out stationary Stretch single transfers,
multi-step plans and explicit recovery-boundary failures using Caliban inference.
Only afterward should navigation/exploration or another robot enter the scored
observed-task denominator.

### Repeatable live readiness audit

The existing camera diagnostic accepts `--execution-readiness`. It uses Caliban
for semantic initialization, refreshes current surfaces through the real registry,
and audits the production tracker's reference feature admission, depth carving,
missing source prerequisites and default execution refusal. Capture files retain
original receive times and image-timing metadata. Job completion means the sensor
and refusal diagnostics passed, not that observed execution is qualified; inspect
`execution_readiness.qualification` and each gate in `result.json`.

The [October 27 live run](../experiments/observed_live_readiness_20261027.md)
passed 5/5 surface refreshes but rejected the small object surface for insufficient
pose-tracking features. Clock/geometry/provenance assembly gates remain open.

### Scan geometry and camera fidelity

`ObservedWorkspace.fuse_static(views)` combines aligned captures of an explicitly
static scene; occupied evidence wins over free evidence. It supplies no capture
freshness or scene-stability certificate. Old scans must be discarded when the
scene changes. `partition(points_world, labels).excluding(object_refs)` retains
mixed target/obstacle cells and leaves exclusive removed endpoints **unknown**,
never free. Unattributed occupied cells stay background. This is the first target
endpoint provenance consumer; it does not supply a full robot solid-volume or
phase-specific contact policy.

`ObservedBackground.from_capture(...)` and `.update(...)` retain measured free
cells when exclusively robot/target-owned endpoints occlude them later. Each cell
keeps its original acquisition time until a new ray observes it. Exclusion never
creates free evidence. Mixed robot/background cells remain occupied. A free/occupied
contradiction rejects the update; the owning source must discard the map and
invalidate plans. `placement_scene()` supplies the existing native checker with
a whole-volume free predicate that checks per-cell age on every query. Without a
`StaticBackgroundAssumption`, evidence expires after at most two seconds. An
explicit assumption names the scene epoch and a validity interval of at most
300 seconds; it extends validity, never capture timestamps. It requires an actual
prior scan and does not establish one from robot geometry. This is a tested query
component, not an assembled live background provider or a contact exception.

For simulator view qualification, `EMET_STRETCH_HEAD_CAMERA_SCALE=2` renders
848×480 raw head samples with matching intrinsics/registration.
`EMET_SIM_HEAD_RENDER_ROBOT=1` keeps robot self-occlusion. Both are opt-in to
preserve existing controls. Proposal normals use a local measured-plane fit to
tolerate millimetre depth quantization at finer pixel pitch; depths/holes and
tracking thresholds are unchanged.

### Simulator head-camera calibration change in #226

The simulated head intrinsics now match MuJoCo's pinhole render frustum rather
than copying the physical SE3-3044 focal calibration. At 424×240 and 42° vertical
FOV, the renderer uses `fx=fy=312.6107` px, versus the former physical values
`304.24/304.07` px (about 2.75%/2.81%). At 848×480 the render focal length doubles.
Pixel centers use `(width-1)/2`, `(height-1)/2`, then the server rotates the matrix
clockwise with the image. Hardware calibration settings are unchanged.

This intentionally changes simulator unprojection and downstream metric geometry;
it is not a numerically identical camera refactor. Do not silently rescore or
compare historical metric results across the change without recording the source
version and rerunning the relevant experiment. Registration uses the same raw
render intrinsics before the published image rotation. The actual-model regression
checks the camera-creation settings, MuJoCo GL frustum, registration K, constructor
`head_K`, projected rays, baseline offset and unfilled registration holes at both
resolutions, without opening a GPU context.

The wrist stream now uses the same render-derived convention: its 480×270
frame and 58° vertical FOV determine the focal length and pixel centers. Both the
server's `ee_K` and camera manager use rendered dimensions, rather than mixing
physical-camera focal calibration with 1280×720 sensor metadata. The actual-model
test compares these matrices with MuJoCo's GL frustum. This changes simulated
wrist unprojection; hardware calibration settings are unchanged. Sensor noise,
range limits and hardware timing/pose-error bounds remain unqualified.

Simulator images now retain separate RGB/depth acquisition intervals under the
physics/render lock. `local_sim_capture_interval` maps their union only when the
Linux boot/time-namespace clock identity matches the client. Other hosts,
missing intervals and hardware clocks are rejected. The helper does not infer
encoder capture times, camera/object stillness across the stereo gap, or zero
alignment error. Hardware and full robot/camera alignment remain separate gates.

### Measured support and full-extent admission

`fit_observed_support` and `support_mask_from_seed` build a partial planar support
patch from calibrated depth plus explicit semantic pixel evidence.
`refresh_support_patch` remeasures its visible footprint without assuming a whole
receptacle is observed; holes shrink/reject the patch. Measurement margins are
explicit inputs and must be calibrated before hardware admission.

`enclose_observed_object` constructs a world-frame bound only when measured free
space closes the target's whole possible connected occupied/unknown component.
Boundary-reaching components, including unseparated support contacts, are
unbounded and rejected. It does not produce a tracked pose or automatically turn
single-view points into `PayloadGeometryEvidence`. The connection assumption,
scan revision, original oldest receive time and capture IDs are explicit.

The camera diagnostic's optional `--support-query` exercises the support producer
separately from object-only task proposals. An unmet requested support diagnostic
fails the process; passing it still does not qualify manipulation. See
[October 28 evidence](../experiments/observed_support_enclosure_20261028.md).

### Geometry tracking and co-motion diagnostics

`GeometryTracker` is an alternative to material-feature tracking for freshly
segmented 3D surfaces. It fits declared geometry using bounded multi-start
registration, validates every original point against the fitted surface, and
retains distinct candidate poses and local unobservable directions. Previous
fits seed search only. A failed update clears the current fit; replayed captures,
old receive times, scene changes and exhausted work budgets reject the update.
The caller still supplies current instance segmentation and identity. The live
diagnostic currently reassociates a stationary reference surface; arbitrary
moving-object segmentation is not supplied by that harness.

Two input tiers share the same registration interface:

| Model | Input and limitation |
| --- | --- |
| `PrimitiveModel` | Explicit sphere/cylinder/box dimensions with `construction=known_model_prior`. This is a declared shape prior, never inferred from object names or hidden scene poses. Sphere rotation, cylinder yaw/axis sign and box symmetries stay ambiguous. |
| `MeasuredSurfaceModel` | Reference-frame measured surfaces plus a separate free-space enclosure certificate. A visible-points box cannot replace that certificate. Static multi-view aggregation retains original capture IDs and the oldest receive time; incompatible scene revisions reject fusion. |

`GeometryFit.as_dict()` includes candidate poses, surface residuals, local
nullspaces, center spread and the union of candidate model volumes. These are
search diagnostics, **not calibrated confidence bounds or a complete posterior**.
`require_unique` needs independently calibrated position/rotation bounds and
rejects unresolved orientation. A symmetric model is therefore not silently
converted to a unique rigid pose. The explicit `require_shape` path below is a
separate contract for shape-aware native consumers.

`check_geometry_comotion` compares all retained pose pairs against measured
end-effector motion, accounting for declared shape symmetry. Lift checks also
require object motion and support clearance. Its `consistent` state is geometric
evidence only: `execution_qualified` remains false. Capture/encoder alignment,
repeated temporal evidence, gripper clearance and contact policy are still
required by the execution interface.

The camera diagnostic accepts `--geometry-model PATH`, a JSON file containing
`schema_version: 1`, `construction: "known_model_prior"`, `kind` and
`dimensions_m`. Dimensions are a sphere radius, cylinder radius/half-height or
three box half-extents, in metres. The fixture prior lives in
`src/test/fixtures/geometry/cylinder_prior.json`. A requested fit failure fails
the diagnostic; successful registration does not qualify manipulation. Caliban
is used for initial semantic selection only; geometry refreshes add no VLM calls.

See the [live geometry diagnostic and figure](../experiments/geometry_tracking_20261028.md)
for both attempts and their distinct discovery/registration outcomes.

### Synchronized simulator inputs and motor-only FK

The opt-in `EMET_SIM_SYNCHRONIZED_RGBD=1` renderer copies a scene under the physics
lock and renders all images from that private snapshot. Sampling bounds describe
the copy, not later rendering/publication. Physics continues while rendering;
this avoids freezing the live scene for a complete camera batch. Published camera
poses describe those pixels rather than a later robot posture.

Two channels are deliberately separate:

- `motor_sample` contains eight actual motor-position readouts: lift, total arm
  extension, wrist yaw/pitch/roll, gripper slider, head pan and head tilt. The
  `actuator_length` calibration is validated against the Stretch transmission
  model. It does not expose passive fingertip angles or individually measured
  telescope lengths. `read_local_sim_motor_capture` rejects hardware/other-clock,
  asynchronous, stale, missing and extra-channel packets.
- `robot_sample` is **`simulator_robot_kinematics_debug`**. It contains robot-only
  configuration, including passive joints, for independent diagnostics. It never
  contains environment joint state. `RobotPoseSample` retains its source, and
  `VisualEffectFrame.require` rejects this debug source with
  `robot_feedback_unqualified`. A producer cannot use it to qualify observed
  execution by simply supplying zero alignment bounds.

`estimate_stretch_ee_from_encoders` consumes only `motor_sample` and the localized
camera pose. It reconstructs the rigid grasp-frame FK without passive fingertip
truth. Equal distribution of total extension preserves the collinear chain's EE
pose; it does not measure the four individual link lengths. The estimator checks
model/camera/transmission identity and rejects unmeasured joints on the EE/camera
ancestry. Its output explicitly leaves gripper geometry and execution unqualified.
Conservative link/fingertip volumes, physical calibration and sensor/localization
error bounds still belong to the live source; motor FK alone does not supply them.

`scripts/prepare_observed_stretch_fixture.py --output DIR` provisions the declared
development scene, separate robot-only MJCF, source hashes and a JSON launch
configuration. The diagnostic's `--synchronized-robot-model` option exercises
motor readouts and compares rigid EE FK against the debug channel in the evaluator
only. See [sampling/encoder qualification evidence](../experiments/observed_encoder_sampling_20261029.md)
for all attempts and the remaining first-transfer gate.

## Moving masks, shape equivalence and gripper intervals

`MovingRgbdTracker` accepts coherent RGBD/world points, a fresh grounded reference
mask, an explicit geometry model and a declared velocity/gap profile. It does not
accept commanded object transforms or simulator masks. Defaults are 0.25 m/s,
1.5 s maximum gap, 5 mm association margin, 0.03 chromaticity distance and a
0.75 s update budget. It uses interior pixels only to learn reference colours;
current measured components remain whole. Every reachable appearance component
competes, including small fragments. Multiple candidates, loss, nonobservable
translation, stale/replayed captures, changed epoch and exhausted budgets latch
loss. Reinitialize with genuinely fresh semantic evidence and a new tracker;
do not retime old references. Returned candidates remain `execution_qualified=false`.
This bounded colour/geometry association is not a general re-identification model.

`GeometryFit.require_shape(calibration_id=..., position_error_m=...,
rotation_error_rad=...)` can return `ShapePoseEvidence` for an explicit complete
primitive prior when centers and observable axes agree within externally supplied
bounds. Fit residuals do not supply those bounds. A cylinder preserves unknown
axial yaw; a sphere has no material orientation; a box uses only its true discrete
shape symmetries. Generic `read_object_pose` still rejects this as ambiguous.
Only explicitly shape-aware payload/retention/effect consumers accept it. The
payload must be `known_model_prior` and its convex geometry must enclose the
declared shape's reference box. Visual support bounds concern that declared
complete solid, enlarged by pose error; planning envelopes may be looser.
This cannot certify task predicates involving a handle, label or material frame.

Successful public plan/execute receipts include `payload_geometry_source` and
`payload_pose_scope` (`unique_rigid_pose` or `declared_shape_equivalence`). They do
not publish private model or calibration identifiers. Scene, geometry, calibration,
shape model and error-bound changes still invalidate stored plans/evidence.

`bound_stretch_gripper(model, estimate, ...)` uses the eight captured motors and
rigid EE estimate with the complete tool subtree, including housing and D405.
Unmeasured fingers/rubber use analytic full model-joint intervals; unlimited
hinges cover a full turn. Supplied interval overrides require `calibration_id`.
Declared EE position/angular errors enlarge all boxes; zero defaults are only an
ideal model diagnostic, not a physical calibration. `robot_brackets(...)` checks
the original capture, motor values, EE pose and model signature before supplying
those volumes to visual effects. It does not establish clock/alignment error or
qualify the assumptions. The source must do that independently. Model bounds and
`execution_qualified=false` are explicit; no passive-joint truth becomes a sensor.
Individual telescope-link uncertainty is still required for complete arm planning.

See [rendered motion, contact audit, retained failures and videos](../experiments/observed_moving_geometry_20261029.md).

## Bounded local collision-map lifecycle

`ObservedWorkspace.from_depth` builds its projection geometry in batches
(`chunk_cells=4096`) rather than allocating corners for the entire workspace.
`max_cells` retains its existing 250,000 default. The additional defaults are
`max_output_bytes=64*1024**2` and `max_working_bytes=64*1024**2`.
Scalar budgets and estimated float conversion/sanitation costs are checked
before full-frame conversion and grid allocation. Optional `build_stats` reports
the cell/output count, conservative array-work estimate and maximum batch size.
These are map-construction budgets, not a process-RSS limit: input owners,
MuJoCo models, renderers and other application memory remain separate. Array-like
inputs may already require storage when first inspected; callers should provide
bounded sensor arrays. Chunking preserves the free/occupied/unknown proof,
including missing returns and all pixels in each projected cell rectangle.

`LocalSceneStore` is an opt-in owner of local collision evidence, separate from
navigation mapping. Its defaults admit one scene with at most 64 MiB of counted
retained storage. `load(key, builder, reserve_bytes=...)` retires older entries
before building and checks the scene's deduplicated Python/NumPy storage before
publication. The builder must bound its own scratch. Builders and consumers
must not retain scenes outside their lease borrow; arbitrary external references
and callback-owned allocations are not controlled by this store.

Plans and collision hooks retain `LocalSceneLease`, an opaque weak reference and
unique generation. Replacing even the same key invalidates old leases. Expired
captures retire their scene; failed reacquisition leaves the old scene invalid.
A borrow serializes a geometry query against eviction. Check the same expected
lease again at command dispatch, and during ongoing control. Cancellation must
remain available after retirement. This is evidence-lifetime enforcement, not a
claim that the scene stayed unchanged: owners must explicitly invalidate on a
scene change and clear their pending plan handles. Acquisition time is never
renewed by a lease check.

The development runner `scripts/eval_mobile_mapped_contact.py` exercises this
lifecycle in simulated Stretch. It navigates unloaded, builds a source map,
uses the public primitive tools, reacquires at a stopped loaded handoff, checks
and executes loaded wheel transport, reacquires at the destination, places and
retreats, then remaps the released object before navigating away. Only private
planning data receives hypothetical poses; live base motion uses ordinary wheel
velocity actuators. It retains the native adapter's existing 0.3 s endpoint
stillness check; controller and release thresholds remain unchanged.
Loaded sweeps preserve measured attachment, explicit grasp
contact allowances and map hooks, and execution checks measured retention and
arrival. The unchanged 5 mm/0.025 rad sweep spacing and 10 mm/0.05 rad tracking
envelope are sampled checks, not continuous collision certificates.

This runner explicitly uses native robot/object state, native full-scene geometry
and a provided external RGBD scanner. Completely contained solids also face the
local map's strict unknown-space checks; solids outside or straddling its bounds
retain the declared native guard. This is an assisted integration control. It
does not enable default OVMM loaded transport, qualify onboard-only mapping,
run Habitat/RoboCasa, or establish a real-robot sensor/controller contract.
The [mobile-map record](../experiments/local_map_mobile_20261117.md) separates
those limits from the measured memory and task evidence.

The native bounded release search now keeps an already certified measured pose
without moving. If adjustment is required, it compares all existing finite
candidates and maximizes the smaller support margin across the certified
commanded/lowered delivery interval and the maximum drop bound. It retains the
same raise/drop limits, sampled opening/retreat checks and two-adjustment budget.
This reduces dependence on a first feasible pose with almost zero support margin;
it does not certify future controller error from the measured tracking reserve.
Internal `release_candidate` events distinguish IK/path, opening, support bounds,
retreat and acceptance. Public tool codes and partial-execution receipts retain
their existing schema; private geometry stays in evaluator logs.

`scripts/eval_mobile_mapped_contact.py --case-config CASE.json` accepts declared
initial conditions: `initial_base_x`, `initial_object_xy`,
`initial_distractor_xy`, `placement_region`, `mobile`, `approach_m`, and
`departure_m`. Values are validated before model creation. The initial overrides
run before discovery/commands. With `mobile=false`, default approach/departure
are zero and transport is stationary; the record explicitly labels that control.
The default configuration preserves the previous mobile episode.

The fixed development matrix is
`configs/eval/mobile_map_regression_20261117.json`. Run
`scripts/run_mobile_map_regression.py --fixture FIXTURE --output OUT` through
`emet jobs run --no-gpu-exclusive`. `--lane N --lanes K` partitions cases across
separate managed jobs on disjoint safe CPU allocations; defaults are lane 0 of 1,
with a 900 s per-case timeout. `--case ID` selects exact case IDs and is repeatable.
Scaled geometry and constant-density mass are compiled from copied fixture XML
before physics; source fixtures are unchanged. Task failures, negative controls,
crashes and timeouts remain separate in each JSON lane manifest.

`scripts/tamp_release_diagnostics.py` saves evaluator-only measured-state/actuator
snapshots at release decisions. `scripts/inspect_tamp_release_states.py --fixture
FIXTURE --states DIR... --output RESULT.json` audits those snapshots offline.
These private MuJoCo contacts/forces never become robot sensor inputs or tool
acceptance evidence. See the
[release investigation](../experiments/mobile_release_20261117.md).

`report_mobile_map_regression.py --runs OUT --output SUMMARY.json` summarizes all
nine declared cases, including cases not yet launched. It counts the seven task
cases separately from two negative controls, preserves crashes/timeouts/failures,
and rejects duplicate or inconsistent case receipts. These development counts
are not a qualified benchmark denominator. The fixed suite is the default;
`--suite PATH` explicitly selects another declaration.

### Reacquire after search, before publishing a plan

`NativeTableAdapter(..., reacquire_scene=callback)` opts into a source handoff;
the default remains unchanged. Search records an inert proposal with immutable
nominal joint paths, phase start states, FK endpoints and payload transforms.
It cannot be executed through a public handle. Finalization consumes older
handles and retires the original map lease before calling the source provider.
The callback must acquire a new scan after that barrier and return
`RefreshedNativeScene(lease, checks)`. Capture identity, calibration identity,
non-overlapping capture times, object/support identity and lease generation are
checked. Reusing or retiming an old capture cannot authorize the proposal.

The callback is a trusted provider seam: its checks must actually query the
returned lease's geometry and preserve support-ray/unknown-space checks. The
callback tuple is copied and its identities are bound to the finalized handle;
bound guards exposing a `lease` must own that same lease. An arbitrary callback
cannot be independently certified by this interface. The provider must bound
its acquisition I/O and construction scratch.

Finalization replays all eight nominal joint segments (gripper opening,
pregrasp, grasp, lift, preplace, place, release opening and retreat), plus any
selected loaded straight base sweep, on private MuJoCo data. It checks fixed
edges and FK endpoints against fresh native and local-map geometry without
additional IK search. Identical coarse grids are insufficient: original support
rays remain part of the geometry query. The existing measured grasp/placement
refinement, checked closing and bounded release search still run during execution;
nominal replay does not certify their future contact outcomes.

This first handoff requires the full measured scene to stay stationary to
`1e-8` qpos difference during acquisition/revalidation, a numerical equivalence
gate for this assisted synchronous simulator. It rejects proposals with either
an unloaded base approach or an arm approach action. Loaded transport remains
supported. These restrictions are not a hardware tracking tolerance. Model,
policy and plan changes also invalidate the proposal. Fixed-path validation has
a separate 240-second wall budget; source lifetime and IK/search budgets are
unchanged, and the source lease must still be current when validation finishes.
Only successful finalization publishes a fresh executable handle.

Before the first possible transport send, execution checks the finalized binding
and current lease. Each subsequent primitive send checks its current phase lease.
An expired source during planning reports `execution_started=false`; a transport
timeout after entering a possible send reports partial execution with
`command_outcome_unknown`. Both consume the handle. Transport providers must
preserve lease guards when acquiring later carry/destination scenes.

Both mobile-map scripts accept `--refresh-source` to enable this experiment.
The regression runner forwards it to task workers only; negative controls retain
their declared paths. See the [source-finalization record](../experiments/source_finalization_20261117.md)
for validation, retained failures and source-tier limits.

### Bounded release destination alternatives

The opt-in native bounded-release policy first checks the measured pose and its
existing vertical corrections. If none certifies opening the whole hand, it
also searches horizontal destination offsets. `ReleaseLimits.lateral_offsets_m`
defaults to `(0.01, 0.02, 0.03)` metres, with the four cardinal
`lateral_directions`. An empty offset tuple disables this fallback and permits
an empty direction tuple. Up to three
horizontal magnitudes and four unit directions are supported. With the default
three height increments this is at most 36 lateral candidates per planning pass.
These are generic geometry candidates, independent of object labels.

Every candidate checks closed-hand arm/payload motion, commanded and lowered
opening geometry, sampled released fall, and open-hand retreat. The original
support/goal-region footprint checks and 1 mm contact tolerance remain mandatory.
Horizontal target/FK displacement is bounded from the original measured release
position across both corrections, rather than granting a new 3 cm radius after
each move. This bounds proposed destinations; sampled paths and future tracking
remain separately checked, with no new controller-error guarantee.

Executed lateral corrections shift the subsequent placement/retreat targets by
the same amount. The internal selected goal follows the certified correction;
the caller's requested region stays unchanged. The same two-adjustment limit
covers vertical and lateral motion together, and a new current-pose certificate
is still required before opening after the last adjustment.

Native release refinement charges IK to the stored primitive's remaining call
budget. Complete planning passes, including dense collision/opening/fall/retreat
queries, charge its remaining planning time; a late candidate cannot authorize
an adjustment or opening. Loaded refinement also carries its consumed calls/time
forward. Controller execution remains outside the planning-time charge and keeps
its existing measured-motion deadlines. Exhaustion after grasp returns partial
execution with `completed_ops=["grasp"]`, `failed_stage=release` and inspection
recovery, consuming the handle without sending the failed opening.

Legacy native policy has no release refiner. This change adds no tool, fictitious
sensor, attachment, live pose write, free-space assertion or relaxed retention
threshold. [The lateral-release record](../experiments/lateral_release_20261118.md)
keeps geometry diagnostics, complete assisted tasks and failures separate.
