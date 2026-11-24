# Observed native manipulation: changes and transfer recipe

The PandaOmron RoboCasa work uses the ordinary CHAT tools (`scene_tasks`,
`plan_pick_place`, `execute_pick_place_plan`) and the shared observed executor.
The robot moves through native joint-position control; objects are never attached
or teleported. Camera depth, camera calibration and measured encoders supply
geometry. Simulator object/fixture state is confined to the independent scorer.
A known object-local CAD prior and explicit diagnostic settings are assistance,
not evidence of general perception or physical-robot readiness.

## Changes by layer

### Design requirement: observed geometry first

Core goal, reaffirmed by the user on 2026-11-23: build a general-purpose agent
harness with minimal assumptions. Let the agent acquire information, choose
tools, act, inspect outcomes, and recover. Do not require benchmark-specific
knowledge, fixed task sequences, manually supplied regions, object assets, or
perfect perception as prerequisites for ordinary manipulation. Keep physical
robot capabilities and calibrated sensing explicit; expose missing information
and uncertainty so the agent can resolve them. Benchmark adapters translate
sensor/control interfaces and score outcomes; they must not supply hidden task
solutions. Scripted task routing is a diagnostic control, not agent qualification.

User decision, 2026-11-23: the general manipulation harness must not require
supplied object CAD, asset identity, or a complete material object pose to plan,
execute, or verify ordinary pick-and-place. Use VLM semantic hints, measured
point clouds, local feedback, and explicit uncertainty about unseen surfaces.
Robot kinematic/gripper models remain necessary and are distinct from object CAD.

Object CAD may be an optional known-object input or diagnostic control. Keep
such assistance explicit in evaluation receipts; assisted success does not
qualify the default observed method. Do not let diagnostic shortcuts become
mandatory shared interfaces. The live source now supports an approximate box
without CAD. Its tracking path and the separate VLM one-shot diagnostic remain
under development; neither constitutes general agent or hardware qualification.

| Layer | Main files under `src/emet/` | Reason and resulting behavior |
| --- | --- | --- |
| Agent decisions | `agent/loop.py`, `agent/tamp_session.py` | Permit at most two fresh perception/planning recovery rounds after explicitly unstarted planning failures. Partial execution requires inspection; consumed handles cannot be replayed. |
| Public task interface | `controller/task/tamp/api.py`, `observed_adapter.py` | Use shared semantic task handles, observed planning, one-shot execution, structured failure receipts and measured visual effects. Preserve verified progress on later failures. |
| Robot transfer seam | `controller/task/tamp/observed_arm.py`, `observed_candidates.py` | Separate robot model, native commands, grasp calibration and scene provider. Support an explicitly symmetry-aware current grasp heading. Generate placement orientations before spending the candidate budget on offsets; derive optional transit height from measured source support and payload clearance. |
| Sensing and transport | `core/capture_clock.py`, `core/zmq_obs_codec.py`, `controller/manipulation/capture_clock.py`, `controller/generic_zmq_client.py` | Preserve original acquisition intervals, lossless depth and capture-aligned encoders. Receiver time cannot renew stale measurements. |
| Panda binding | `robots/pandaomron/client.py`, `observed.py` | Reconstruct robot-only FK from measured base/arm feedback. Use separate angular and linear arrival tolerances; fast servo feedback is separate from capture-owned geometry. Cache self-surface qualification only within an immutable capture. Expose both benchmark scene cameras and the wrist camera with one acquisition interval; optionally hold measured torso height. |
| Live scene | `controller/task/tamp/live_transfer.py`, `observed_source.py`, `transfer_grounding.py` | Ground source, object and destination from RGB-D. Track selected camera views independently of initialization view; retain thin visible fragments for geometric validation. Fuse static background separately from current payload/robot state. |
| Geometry tracking | `controller/manipulation/geometry_tracking.py`, `multiview_rgbd_tracking.py`, `continuous_rgbd.py` | Fit calibrated intrinsic geometry, respect shape symmetry, support opt-in qualified single-camera fallback and reject conflicting views. Rejected fits cannot renew old timestamps. |
| Collision representation | `motion/local_manipulation_map.py`, `local_geometry_distance.py`, `observed_contact.py`, `surface_contact_patches.py`, `controller/manipulation/observed_collision.py` | Use measured surfaces and local native solid distances for manipulation rather than treating inflated voxel cells as physical contact. Keep raw measured samples separate from map cell centers. Convex/box contact uses a single-thread halfspace feasibility solve, retaining conservative subdivision on unresolved solves. |
| Departure and handoff | `motion/contact_departure.py`, `motion/arm_rrt.py` | Sample exits from invalid starts with fixed initial contact patches, bounded overlap/travel and new-contact rejection. Retain the original clearance context through recovery, goal connection, later planning phases and runtime revalidation. Do not create expanding contact exceptions. |
| Motion generation | `controller/manipulation/physical_pick_place.py` | Add bounded redundant IK starts, checked three-leg Cartesian staging and Cartesian loaded transport. Translate out before changing tool orientation; retain loaded orientation until the destination-front waypoint. Initialize the original measured clearance context for Cartesian paths as well as joint searches. Separate long snapshot planning from current execution clocks. Share the wall budget across IK proposals. Optional waypoint coalescing removes only collinear points with bounded angular/linear commands; recovery retains fine steps. A validated grasp path may be reused from a measured open, matching start, with live segment checks. |
| Contact transitions | `controller/manipulation/physical_pick_place.py`, `motion/observed_support_contact.py`, `controller/task/tamp/placement_confirmation.py` | Predict first nominal gripper contact, distinguish relative fingertip geometry from world pose uncertainty, associate measured support across views and support guarded placement probing. |
| Simulator/scorer | `simulation/pandaomron_server.py`, `pandaomron_probe.py`, `pandaomron_environment.py`, `eval/pandaomron_live_transfer.py`; `scripts/eval_pandaomron_sink_agent.py` | Run the pinned benchmark through the same robot client. Score full-footprint sink containment, two-pad lift, final settling, unintended external/self contacts and allowed actuation independently of tool claims. Save initial and final camera images. |

## Porting to another arm

1. Register the robot's native client, controlled joint names, limits, FK model,
   end-effector/gripper bodies, open configuration and controller tolerances.
   Linear tolerance is in meters; angular tolerance is in radians.
2. Bind `ObservedArm` with fresh measured synchronization and native command
   callbacks. Supply capture-aligned encoders for the perception reader, even
   when the control loop has a faster independent feedback stream.
3. Supply a scene reader and object/source/destination selection. Keep intrinsic
   CAD dimensions and grasp calibration separate from observed world poses.
   Match the benchmark's declared intrinsic asset scale.
4. Calibrate robot inner/outer solids, sensor/model error, named contact pads and
   support uncertainty. Prefer measured local geometry to coarse global voxels
   for arm manipulation. Unknown space, perception loss and contact assumptions
   need explicit policies; they are not solved by a different embodiment.
5. Generate calibrated grasp and placement candidates through the shared factory.
   Use robot-native IK/control; validate any staging proposal and every connection.
6. Test sensing, free motion, invalid-start departure and its clearance handoff,
   grasp/retention, loaded transport, release and independent final task success.
   Keep scripted CHAT routing distinct from a real model-selected run.

## Diagnostic settings and limits

Simulation overrides are explicit in the profile and tool receipt. Generic
support penetration remains 1 mm; an explicit simulator tolerance can request
up to 10 mm and applies to nominal payload/support penetration, with other
obstacles checked separately. The development profile uses 5 mm. Contact
clearance reserve is distinct from physical penetration. Robot visual association
above the ordinary band, unknown-space optimism, contact-terminated close and
local payload contact each remain separately identified assistance.

`simulation_best_fit_pose` chooses the minimum-residual observed hypothesis while
retaining all alternatives in diagnostics. It preserves original timestamps,
search-boundary and local observability checks; it does not calibrate confidence
or prove competing poses impossible. The simulator factory rejects this setting
on non-simulation sessions.

`simulation_ignore_environment` is a positive control with collision scope
`robot_self_only`. Its physics score remains useful, but even a successful
transfer cannot qualify environment collision planning. Never combine that result
with collision-aware runs into a single success count.

The named `held-out-shaker` snapshot has been used repeatedly for development.
It is now explicitly a development case. Freeze fresh layouts/assets/seeds before
claiming held-out performance, then run the real model CHAT test and independent
robot ports. See [experiment history](../experiments/harness_assessment_20261119.md)
for retained attempt artifacts and outstanding failures.

## Reproducing the development attempt

The compact profile is
[`configs/eval/pandaomron_sink_observed_development.json`](../../configs/eval/pandaomron_sink_observed_development.json).
It is specific to the frozen shaker development snapshot and its robot-model
fingerprint. Reground and recalibrate for another scene; its pixel selections are
not reusable semantic perception. Environment bypass is **off** in that profile.
A positive control must explicitly set it on in a separate copied profile.

The frozen full-transfer positive control is
[`pandaomron_sink_assisted_transfer_development.json`](../../configs/eval/pandaomron_sink_assisted_transfer_development.json).
Copy that file instead in the recipe below to reproduce the assisted transfer.
It explicitly enables environment bypass and the simulation-only phase carry
model. It uses a wider measured destination region, front-floor proposals and
an aligned hand heading to fit between the rim and faucet. This does not qualify
collision-aware transfer or automatically grounded perception.
It also declares `simulation_placement_settling_tolerance_m=0.02`: the translation
bound includes both captures' 5 mm errors. The ordinary 10 mm bound leaves no
noise allowance with those errors. The simulator option is limited to 10–30 mm,
reported in tool receipts, and does not change support, footprint, orientation
stability or retreat checks. This is a task acceptance tolerance, not improved
measurement accuracy.

The development trial `logs/pandaomron-finish-bounded-stop-Wn4g/report.json`
passes observed execution and independent physics: full transfer and final visual
placement verification, 100.97 mm two-pad lift, zero unintended touches or scored
contacts. This result uses scripted public-tool selection rather than a model
choosing actions. Collision-aware transfer, fresh held-out cases, unknown-model
perception and physical-robot qualification remain open.

Use RoboCasa 1.0.1 at commit
`7bfc60c9d26c02506d4265cbe0e86b35e6e25d42`, robosuite 1.5.2 and MuJoCo 3.3.1
for the benchmark server, with the benchmark assets installed. The planner/client
requires MuJoCo 3.5 (`mj_copyData`). Keep those Python import paths separate;
`--sim-pythonpath` applies only to the server subprocess. Native mesh compilation
is serial on benchmark resets. Fixture XML is prepared before compiling the frozen
asset configuration; native startup failures are recorded separately from task failures.

Generate the intrinsic prior next to a copy of the profile:

```bash
mkdir -p /tmp/panda-transfer-profile
cp configs/eval/pandaomron_sink_observed_development.json /tmp/panda-transfer-profile/profile.json
PYTHONPATH=src python scripts/calibrate_transfer_payload.py \
  --asset /path/to/robocasa/robocasa/models/assets/objects/lightwheel/salt_and_pepper_shaker/SaltShaker012/model.xml \
  --visual-mesh SaltShaker012_vis --exclude-annotation-geom reg_bbox \
  --calibration-id salted-shaker-isolated \
  --output /tmp/panda-transfer-profile/shaker-prior.json
```

The source model XML SHA256 is
`78220fc52e9442f4264c305dab0e681a81d26fabeadf3a853b96d0b02bfa9831`.
The profile scales this isolated prior by the frozen asset's intrinsic 1.05
scale. This uses known shape, never live object pose.

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
  python scripts/eval_pandaomron_sink_agent.py \
  /path/to/PickPlaceCounterToSink-held_out-1 /path/to/new-output \
  --case held-out-shaker --scene-camera right --port-offset 5240 \
  --observed-profile /tmp/panda-transfer-profile/profile.json \
  --diagnostic-torso-offset-m .002 \
  --sim-pythonpath /path/to/server-numpy:/path/to/mujoco331:/path/to/robocasa:/path/to/robosuite:src
```

The 2 mm initial torso offset is an explicit diagnostic initial condition, not a
runtime action. CPU trials hide CUDA and pin six CPU cores; render-thread counts
and CPU affinity should be recorded alongside results. Add a model endpoint only
after the scripted routing run is physically and observationally verified.


Both scene cameras are RoboCasa's standard benchmark cameras; the wrist camera
is mounted on the hand. This three-camera setup does not establish that a
particular physical robot has equivalent visibility. Additional named RGB-D
views preserve arrays, calibration and timing in `Observations.auxiliary_rgbd_views`.
Native local-distance kernels are warmed before continuous capture processing;
that warmup is reported separately and cannot refresh old sensor evidence.

The profile's placement headings are proposals. The native reference normalizes
the destination wrist heading; the observed recipe prioritizes that same heading
without importing its simulator object pose or trusting its route. Grasp and
place feasibility, actual native execution and independent physics remain gates.


## Native reset and capture corrections

RoboCasa settles the scene during reset with zero actions. Its neutral delta
command is an absolute zero-joint target under our native absolute controller.
`simulation/pandaomron_environment.py` now translates **reset-only** neutral
actions into a hold of the native reset posture. Normal episode commands retain
absolute semantics. Measured joint 4 is now approximately -2.25 rad, rather than
the erroneous nearly straight -0.15 rad. This changed wrist visibility, so the
development profile grounds the shaker and surrounding counter in `scene_alt`;
the destination remains in the wrist view. Offline RGB-D replay established the
source region needed to observe support around the occluded object footprint.
No support predicate was relaxed for this region correction.

The Panda bridge owns alternate and wrist RGB-D rendering from the primary
snapshot. Its server-local generic camera list contains only the primary camera,
preventing a second, generic stereo wrist capture from overwriting the shared
acquisition interval. Session capabilities report all three cameras. The runner
saves every auxiliary RGB-D array and calibration in `sensor.npz` for replay.

Robot-only model loading and scene mesh compilation both request serial native
compilation; the temporary robot compiler binding is restored even on failure.
This is a scoped mitigation for observed native startup crashes, not a proven
fix of their underlying cause. Current diagnostic runs use `PYTHONMALLOC=malloc`
and retain startup failures separately from task results.

An explicit simulator diagnostic may widen visual self-association to 30 mm,
subject also to the calibrated overall surface band. The ordinary 3 mm visual
fallback remains unchanged. A wider band can absorb nearby external returns;
it is labeled assistance and has not qualified real-robot self-segmentation.

`simulation_best_fit_view` is a separate opt-in diagnostic requiring single-view
fallback. If fused tracking fails and independently valid camera estimates
conflict, it chooses the lowest-residual view, retaining all independently fitted
view candidates and a selection counter in the report. Ordinary tracking still
rejects conflicting cameras. Every selected fit retains its actual acquisition
interval and must pass the same local observability, residual, search-boundary,
identity and age checks. This is a best guess, not calibrated pose confidence.
It is simulator-gated and listed separately in assistance receipts.

With `debug_planning`, `observed_source.native_motion_events` retains command and
measured joint endpoints, phases, residuals and monotonic event times. Path-start
events distinguish planned and commanded waypoint counts. These records augment
the actuation audit and independent physics trace.

Optional `execution_shortcut_angle_rad` and `execution_shortcut_slide_m` simplify
nearly collinear joint paths within declared per-joint deviations (defaults zero,
limits 0.005 rad/1 mm). Every skipped original sample must stay within the bound
of its proposed chord, and commands retain the 0.1 rad angular cap. The slide span defaults to 5 mm;
`execution_max_slide_step_m` can request 2–20 mm with the same adaptive direct-edge checks.
Each resulting direct segment is checked against fresh geometry before it is sent;
this simplification is not an alternative collision certificate. Departure
prefixes retain their fine execution waypoints. The diagnostic trial uses
0.002 rad/0.2 mm and reduced one pregrasp from 364 samples to 101 commands.

`simulation_retention_rotation_rad` may explicitly raise the normal 0.1 rad
retention threshold (up to π rad in an orientation-unconstrained simulator control) for scored simulator diagnostics. The same
configured angle is used by runtime retention and visual lift verification; the
2 cm relative translation limit, measured co-motion, support clearance and
independent physical lift score remain. The current diagnostic uses 0.2 rad.
Its receipt records both assistance and the numeric threshold.


The separate `configs/eval/pandaomron_sink_assisted_lift_development.json` retains
the exact assisted profile that physically lifted the shaker 7.2 cm with two-pad
contact. It enables environment bypass and best-view assistance. It stopped on
visual geometry mismatch before completing lift verification/transport; it is
not the collision-aware development profile and does not demonstrate full transfer.

The attachment-envelope calibration includes the same 2 cm relative-motion
allowance as runtime retention when computing the maximum lever radius. This
keeps angular error propagation conservative while avoiding a hidden smaller
lever gate. The original attachment remains fixed; subsequent drift is still
compared to it and cannot rebaseline itself.

Rejected tracking packets are retained for diagnostic replay only. The benchmark
writes `tracking-rejection-NN.npz` (RGB, depth, world points, foreground masks and
seed poses) and a JSON file with original acquisition intervals and the rejection
reason. A rejected packet never replaces accepted evidence or refreshes its age.

## Capture-aligned retention versus current control feedback

`PayloadGeometryEvidence.captured_world_from_ee` optionally carries the tool pose
from the same qualified observation as the object pose. The shared adapter supplies
it from `VisualEffectFrame.robot_after`. Runtime retention compares object and tool
from that capture, rather than subtracting a newer servo pose from older imagery.
This is essential during motion: camera delivery latency can otherwise appear as
payload slip. Both object and tool retain the observation's actual age.

Actual post-closure attachment may explicitly use that capture-aligned transform.
Hypothetical grasp/lift planning continues to use the candidate tool pose, so a
capture of the robot at home cannot replace the planned grasp transform. Legacy
providers without a capture-owned tool pose retain their original interface.
Regression tests cover asynchronous servo movement, matching camera/tool motion,
true relative slip, stale evidence, immutable transforms and hypothetical planning.

`simulation_retention_translation_m` can explicitly request 20–50 mm instead of
the ordinary 20 mm limit. Runtime retention, visual lift verification and the
fixed attachment lever calibration use the same value. This is recorded
separately from angular retention assistance. A diagnostic is now testing
30 mm / 0.35 rad; defaults, measured lift, support clearance and independent
physical scoring remain unchanged.

`simulation_fit_residual_m` is another explicit simulator diagnostic (3–10 mm,
default 3 mm). It applies consistently to reference and updated geometry fits.
The existing validation still checks every measured point: the 95th percentile
must stay within the configured value and the maximum within three times that
value. No points are trimmed. These fit residuals do not calibrate pose confidence.

Geometry-generated tracking windows now intersect the original measured
appearance model before fitting; proximity to the prior alone cannot turn robot
pixels into object identity. An exact retransmission of a rejected capture is
skipped as a duplicate, without another fit or renewed evidence. Changed data
under that rejected capture ID still raises replay, and future real captures
must pass normal fitting and age checks.


A simulator-only retention angle of π is explicitly labeled
`simulation_retention_orientation_unconstrained`. It tests native carry/placement
without requiring a fixed observed material orientation inside the fingers.
Relative-position, identity, visual co-motion, support and independent physical
scores still apply. This control cannot qualify a rigid payload orientation
assumption for collision-aware planning. Ordinary retention remains 20 mm/0.1 rad.

## Phase-dependent carry evidence

`simulation_rigid_carry_prediction` is an explicit simulator assistance tier.
After native finger closure settles, the arm stays still while the source obtains
fresh post-closure captures. The attachment reference is frozen once from those
captures. Capture-owned tool motion then proposes object orientation; current
visible depth fits translation within a 20 mm search. All points still receive
residual validation, translation must be locally observable, and contradictory
or missing evidence can stop the action. Captured finger aperture must remain
blocked above its commanded closed endpoint and close to its settled reference.

This replaces an unconstrained visual orientation fit during carry with a
conditional mechanical model. `GeometryFit.conditioning` records
`rigid_attachment_orientation`; it cannot claim independently observable material
orientation. Tool receipts identify the assistance. The model is disabled before
release, and attachment references cannot drift during transport. Ordinary
tracking and independent physical lift/contact/placement scoring remain separate.
This is a simulator control, not calibrated hardware retention confidence.

Placement proposals can take `placement_xy_fractions`: an ordered list of up to
nine normalized offsets on both axes of the *measured feasible footprint region*.
Each component is within ±0.5; the full uncertainty-expanded payload footprint
remains inside the selected support. The default longer-axis proposals are
unchanged. This lets a caller select a visible interior/front portion of a sink
instead of repeatedly targeting its center near a faucet. Arm/gripper clearance
still belongs to motion validation; a payload-fitting goal alone does not prove
the hand fits through the opening.

The scorer separately records `unintended_touch_contacts` / `unintended_touch_frames`
for shallow negative-distance contacts. The existing 1 mm penetration score is
unchanged. Zero scored penetration frames must not be described as zero physical
touches. Those diagnostics are scorer-only and are not agent observations.

The current assisted carry implementation still needs at least 25 visible depth
samples and locally observable translation. Fully occluded carry using bounded
belief and qualified force/contact evidence remains future work.

## Pose requirements and remaining generalization work

Continuous, independently observable material 6D pose is not the desired contract
for every phase. Grasping needs a reachable contact region and a usable approach;
carry needs evidence of retention with bounded slip; placement needs destination
clearance, support and release evidence. Symmetry and task tolerances should decide
which orientation components matter. Tight insertion or pouring can require much
more orientation information than dropping an upright shaker into an open sink.

The current implementation still represents these hypotheses as poses and uses a
known CAD prior. Its simulation-only rigid-carry model reduces visual observability
requirements, but does not qualify an occluded hardware grasp. General deployment
still needs partial-view geometry without a supplied CAD model, task-dependent
pose uncertainty, qualified contact/force or bounded-belief carry, and guarded
placement recovery. These are explicit capability gaps, not solved by the pilot.

Historical effect references retain their original acquisition times. They are
validated at their original receive time and against current identity/geometry,
without an arbitrary task-duration cutoff. Current placement samples still need
fresh, ordered post-retreat captures and a measured settling interval. Extending
a transport must not renew old images or invalidate an already observed lift.

`ObservedSupportContact.measured_surface_bounds` bounds plane exclusions using
all original measured witnesses, rather than an unbounded plane or an inscribed
placement rectangle. Carry foreground association additionally checks proximity
to the known measured model surface; an occluding pixel inside the coarse payload
enclosure is insufficient. The Panda evaluation settles the restored fixture for
30 native-control steps while holding the measured absolute arm/torso posture
before publishing the initial scene. Independent scoring records that warmup.

The shared arm validator admits only 0.1 mm slide / 0.1 mrad hinge overshoot at
nominal joint stops. Native soft-stop feedback can exceed a nominal endpoint by
micrometers; rejecting that measured start blocked pre-release retreat. Collision
checks use the actual unmodified state, while IK/sampling retain nominal ranges.
The diagnostic and boolean validators share this band. Regression coverage
checks return from a 50 micrometer overshoot and rejection at 200 micrometers.

## Runtime VLM diagnostic and instruction controls

`configs/eval/pandaomron_vlm_one_shot_development.json` contains only native
finger targets. The one-shot path skips the old tracking-profile factory. With
`--open-loop-transfer --model-endpoint URL --model MODEL
--open-loop-object-description DESCRIPTION --open-loop-destination DESTINATION`,
`eval_pandaomron_sink_agent.py` asks the VLM for image regions and closing direction,
uses depth for metric targets, and executes native pick/carry/release/retreat.
This is scripted phase routing with VLM grounding, not production agent routing.
It bypasses environment collision checks and must remain simulation-only.

Coarse object hints seed measured depth components; missed hints can snap only
to a nearby clearly preferred component. Source support is measured beside the
object rather than accepting the object's own top as a counter. Destination
floor growth stays connected to the VLM-selected region. The front placement
point uses multi-view depth, spatially balanced samples and estimated floor axes.
No complete object pose, asset identity or object CAD is required. Approximate
preplace acceptance is limited to 35 mm / 0.15 rad, recorded explicitly; it does
not change the final target or independent physical scoring.

Run `scripts/eval_vlm_instruction_pair.py SENSOR_NPZ NEW_OUTPUT --endpoint URL
--model MODEL --object DESCRIPTION --destination LEFT_REQUEST --destination
RIGHT_REQUEST` to compare decisions against exactly the same recorded image and
depth, without robot motion. Its report includes the image hash, raw decisions,
depth failures and metric goals. Different goals are evidence of instruction
sensitivity, not proof of physical task completion or general performance.

User decision, 2026-11-23: contact is acceptable. Success is completed execution
and the task's independent predicate, not zero contact. The one-shot diagnostic
therefore reports `task_passed` separately from `zero_contact` and the historical
`clean_transfer_passed` statistic. Contacts remain telemetry. Incomplete execution,
missing lift, wrong containment, an unsuccessful official predicate or an unsettled
object still fails the task score. Preserve older raw scores when rescoring policy.

### Codex-supervised tool assessment

The [2026-11-24 assessment](../experiments/codex_tool_assessment_20261124.md)
uses `scripts/codex_observation_gateway.py` to expose the same image/text requests
and JSON decisions as the local VLM service. It does not issue robot commands.
`src/emet/eval/native_pose_targets.py` supplies aspect-correct image/plane
conversions, gradual wrist rotation and known-hand support bounds. The native
waiter can verify the requested Cartesian grasp endpoint instead of accepting
joint arrival alone. Crop refinement preserves aspect ratio; floor-plane growth
preserves the selected semantic placement region.

The opt-in `pandaomron_atomic_transfer_tolerant_development.json` profile plans
visually before grasp and runs loaded motion with native grip monitoring,
without blocking loaded inference calls. It uses a 2 mm grasp-command arrival
tolerance and releases above an observed open sink. Source-counter height is a
sink-rim proxy in this development mode. Physics and gripper force limits remain
unchanged. This is simulation diagnostic routing with environment collision
certification bypassed, not production CHAT or hardware qualification. Keep
long-hold retention, official benchmark success and the additional full-box
containment diagnostic separate in reported results.


Native diagnostic scoring version 2 uses the official benchmark predicate plus
verified lift, completed execution and settling for `task_passed`. The stricter
registration-box check is reported as `strict_box_transfer_passed` and
`whole_box_contained`; it no longer overrides benchmark task success. Historical
version-1 receipts remain unchanged.
