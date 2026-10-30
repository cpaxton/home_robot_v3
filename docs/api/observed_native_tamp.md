# Observed native TAMP integration

The production CHAT tools can now use an explicitly injected
`ObservedNativeAdapter`. It shares `scene_tasks`, `plan_pick_place`,
`execute_pick_place_plan`, `pick_place`, and optionally `plan_task` /
`execute_task_plan` with the privileged control. No simulator body names, object
pose writes, scene snapshots or contacts are used by this adapter.

**Status:** implementation and synthetic integration regressions pass. A complete
live RGBD source is not assembled or qualified by default. The usual RGBD
`scene_tasks` provider remains read-only. This is not an observed-task benchmark,
a contact-execution result, or a hardware sensor-availability claim.

**Independent planning blocker:** a self-occluding head-camera depth map marks
robot surfaces occupied and their interiors/occlusions unknown. The current
collision checker requires every robot-geom AABB to be known free, so that raw
map rejects the starting arm state. Even perfect robot endpoint segmentation
leaves unknown cells; it does not certify free robot space. No self-occupancy
or phase-specific robot/target/support-contact policy is implemented here.
The adapter is an integration seam, not a working depth-only arm executor.
Its synthetic transfer/swap fixture supplies `known_free=lambda _: True` to test
tool orchestration. This is not evidence that a real self-occluding map admits
those paths. This planning blocker is separate from moving-mask visibility loss.

## Components

| Component | Responsibility | Rejects / limits |
| --- | --- | --- |
| `RigidRgbdTracker` | RGB feature correspondences, forward/backward optical flow, measured RGBD landmarks, RANSAC rigid fitting against a fixed reference | Lost masks/depth, repeated or insufficient texture, poor feature spread, inconsistent motion, replay, reference age >30 s; no global re-identification |
| `MovingRgbdTracker` | Current whole measured appearance components and declared geometry/motion profile | Lost/ambiguous associations latch; diagnostic candidates, not calibrated global identity |
| `ShapePoseEvidence` | Explicit primitive solid equivalence with external pose-error bounds | Unknown material yaw remains unknown; generic unique-pose consumers reject |
| `GripperGeometryEnvelope` | Full tool subtree over unmeasured joint intervals, anchored at motor-derived rigid EE | Conditional model bounds; no passive-joint sensor claim or full-arm certificate |
| `PayloadGeometryEvidence` | Complete conservative volume in an explicit tracked reference frame | No visible-point AABB promoted to a complete shape; geometry priors remain declared inputs |
| `ObservedNativeCollisionChecker` | Robot-only FK, complete link/payload boxes, self collision and whole-volume known-free queries | Missing workspace, stale geometry, scene revisions, unresolved payload geometry |
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

## Provider assembly

`ObservedNativeAdapter(robot, source=..., refine=..., tasks=...)` is deliberately
explicit. `source.input_mode` must be `observed` and it provides:

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
resolutions, without opening a GPU context. Wrist-camera calibration is a separate
qualification gate.

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
