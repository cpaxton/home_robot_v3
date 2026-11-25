# Shared public-tool native transfer controls, 2026-11-24

These controls call canonical `pick_place` through the production CHAT registry,
shared observed adapter and physical executor. Outer tool selection is scripted;
Qwen3.5-9B supplies fresh visual grounding. Object CAD, actor state and saved
semantic regions are absent from the actor. Benchmark metadata initializes and
independently scores the fixture. The development simulation assists are
reported in each receipt; these runs do not qualify hardware collision safety.

Artifacts live at repository-root
`logs/shared-harness-public-transfer-20261124/`. Native drivers and profiles are
retained there. Every retry has its own output directory and free port offset.
`integration-control-summary.json` retains outcomes, including orchestration
failures. There is no model-only or frozen success-rate claim.

| Trial | Commands | Result |
| --- | ---: | --- |
| pear-v1 | 0 | `not_dispatched` |
| pear-v2 | 0 | `support_patch_unavailable` |
| pear-v3 | 0 | `support_patch_unavailable` |
| pear-v4 | 0 | `support_plane_unqualified` |
| pear-v5 | 0 | `geometry_tracking_no_match` |
| pear-v6 | 0 | `task_object_not_settled_on_support` |
| pear-v7 | 0 | `geometry_tracking_no_match` |
| pear-v8 | 0 | `scene_changed_replan` |
| pear-v9 | 0 | `no_observed_transfer_candidate` |
| pear-v10 | 27 | `geometry_tracking_no_match` |
| pear-v11 | 27 | `geometry_tracking_no_match` |
| pear-v12 | 0 | `visual_evidence_incomplete` |
| pear-v13 | 28 | `geometry_tracking_no_match` |
| pear-v14 | 0 | `collision` |
| pear-v15 | 27 | `geometry_tracking_no_match` |
| pear-v16 | 55 | `operation_failed` |
| pear-v17 | 41 | `payload_not_retained` |
| pear-v18 | 61 | `operation_failed` |
| pear-v19 | 59 | `approximate_surface_overlap_lost` |
| pear-v20 | 38 | `joint_bounds` |
| pear-v21-cli | 38 | `joint_bounds` |
| pear-v22-cli | 64 | `payload_not_retained` |
| pear-v23-cli | 65 | `payload_not_retained` |
| pear-v24-cli | 78 | `payload_not_retained` |
| pear-v25-cli | 44 | `visual_effect_unknown` |

Changes driven by these controls:

- Controller-owned phase/payload state gates model queries. Empty state comes
  from reset or verified placement, never model assertion. Dispatch failures
  preserve unknown load state.
- The scripted runner now actually dispatches a single canonical call, and its
  scorer recognizes the canonical execution receipt. Physical criteria remain.
- Support expansion follows connected measured plane returns, stays local for
  the source and excludes object pixels. Seed bounds retain grown source and
  destination masks instead of recropping them to semantic hints.
- Connected object search tolerates coarse boxes while imposing a 15 cm metric
  bound. Synchronized local, appearance-consistent above-plane surfaces from
  other cameras can augment the CAD-free registration reference.
- Approximate pickup hypotheses tolerate 20 mm footprint changes from partial
  views. Approximate preplanning orientation changes are bounded by 10 mm of
  geometry displacement and 0.2 rad. Contact/collision patches and final-place
  verification are unchanged; exact geometry retains its original check.
- Tracking records per-camera residual witnesses and excludes capture-aligned
  measured robot returns using the existing mapping association.

Trials v10 and v11 reached 27 actor commands and approximately 0.70 m measured
end-effector displacement, with zero unintended contacts recorded by the scorer.
They stopped during pregrasp; no pickup or completed transfer was verified.
The reference-camera occlusion/handoff remains the next native validation gate.

The first full-frame self-association trial (v12) reduced pose capture rate to
roughly one frame per 1.4–1.6 seconds and timed out gathering evidence. The
Panda binding now offers the same ray/triangle association for candidate pixels
only. Full-frame mapping qualification remains separate and lazy. Trial v13
restored tracking throughput and reached 28 commands, then rejected a partial
surface match just outside the 6 mm diagnostic fit limit. Approximate boxes
now use a 10 mm residual floor; exact models retain their existing limits.
The actual tracking limit is recorded in source diagnostics, and approximate
pose scope is explicitly labeled rather than described as unique rigid pose.

Trial v14 refused gripper-close certification for self collision before motion.
The next control removes the driver's zero-aperture override and uses the robot
profile's calibrated closure. No collision checker was disabled for that refusal.

Validation: the final related regression suite passed **725 tests** with two
existing dependency warnings. A focused test process suffered the known native
import segmentation fault; its log and the successful full-suite rerun with
`PYTHONMALLOC=malloc` are retained beside the trial artifacts. Ruff and
whitespace checks pass. No completed shared transfer is claimed.

Trial v15 used calibrated gripper closure, certified all six route phases and
reached 27 commands / 0.701 m end-effector displacement. It still stopped in
pregrasp on `geometry_tracking_no_match`; no pickup or transfer was verified.
Its remaining camera had median reference distance 1.65 mm but 95th-percentile
distance 13.91 mm. The wider residual policy did not resolve this view handoff.
The evidence therefore supports a structural next step: separate bounded
approximate task geometry from mandatory continuous rigid registration, with
observed grasp/lift/place verification at action boundaries. Do not keep
increasing residual limits or score this as a model reasoning failure.

## Follow-up: approximate geometry instead of mandatory rigid fits

The shared multiview tracker now treats `approximate_observed_box` as a task
estimate. It first estimates translation from majority measured surface
correspondences, retaining reference orientation as a convention. If a new
camera sees a previously unobserved face, a bounded approximate-envelope
association may supply a translation estimate instead. It requires fresh
appearance-associated, self-filtered depth compatible with the approximate
volume; it does not complete a hidden shape or declare a unique rigid pose.

The surface route uses at least 60% correspondence within the configured
residual band, a 30 mm 99th-percentile distance and 60 mm hard cap. The envelope
route projects translation into the interval compatible with visible extent,
requires less than 20 mm correction per reference axis, and limits envelope
residual to 10 mm at the 95th percentile / 30 mm maximum. These are declared
approximation policies, not calibrated confidence bounds. Diagnostics retain
association method, true surface-overlap fraction, maximum surface distance,
RMS scope, original capture identity and time. Neither translation nor
orientation observability is claimed by this estimator. Exact geometry keeps
its rigid-fit path. No stale pose is republished as a fresh measurement.

A rejected estimate retains the existing bounded transient-rejection policy;
it cannot extend evidence age. Planning failures discard the stopped track and
require fresh grounding. Execution failures do not silently restart perception
or restore a known-empty payload state.

Native progress with live Qwen grounding and scripted public-tool selection:

- `pear-v16`: 55 commands, completed pregrasp and began descent; stopped on
  association loss. No pickup verified.
- `pear-v17`: completed pregrasp/descent/closure, then stopped during lift when
  the measured retention bound reached 20.6 mm against a 20 mm limit. The
  independent scorer recorded 37 mm lift with both pads, below pickup criteria.
- `pear-v18`: explicit simulation retention allowance 35 mm, then measured
  gripper-aperture loss during lift. No pickup verified.
- `pear-v19`: approximate support-to-visible-top grasp height, then association
  loss at closure. No pickup verified.
- `pear-v20`: bounded-envelope association active; gripper closure refused for
  measured joint bounds. Detailed joint witnesses were added to closure logs.

Approximate top-down candidates now estimate interior height halfway between
measured support and observed object top, then apply the robot's calibrated
hand/tool offset. A visible-surface median biased the grasp toward the top.
This modifies proposals only: native route checks and observed lift/placement
verification remain required. The 35 mm retention experiment is a reported
simulation profile setting, not a changed hardware default.

The benchmark CLI now supports `--grounding-endpoint` and `--grounding-model`
with `--observed-profile`. In scripted mode it calls canonical `pick_place`
once through the production CHAT registry. `--model-endpoint` / `--model`
remain separate outer-supervisor options. Runtime grounding rejects CAD-bearing
profiles and cannot be combined with the alternate open-loop diagnostic.
Example, after exporting a robot-only profile without semantic regions:

```sh
python scripts/eval_pandaomron_sink_agent.py FIXTURE NEW_OUTPUT \
  --case expansion-pear --case-spec FROZEN_CASE.json \
  --observed-profile ROBOT_ONLY_PROFILE.json \
  --grounding-endpoint http://127.0.0.1:8765/v1 \
  --grounding-model Qwen/Qwen3.5-9B --port-offset FREE_OFFSET
```

Use the existing native simulator dependency path and software EGL environment
recorded by the retained drivers. These are development integration controls,
not a Codex/Qwen success-rate qualification. Full transfer, native drawer,
Stretch and frozen multi-episode qualification remain outstanding.

The first normal-CLI run (`pear-v21-cli`) reproduced the gripper refusal with
an actionable witness: finger joint 1 was at 0.040105 m for a 0.040000 m upper
limit. The existing bound tolerance was 0.1 mm, so this exceeded it by only
0.005 mm. Explicit simulated contact-terminated closure now permits up to
0.5 mm of measured start overrun on a linear inward path. Goals keep nominal
bounds, the actual measured start is collision-checked, and the path cannot
move farther outside the limit. Ordinary/hardware closure keeps zero additional
slack. The simulation receipt reports this start-bound allowance. Tests cover
actual-start checking, monotonic return, oversized overruns, outward goals and
colliding starts. No joint reading is clipped or rewritten.

The shared box-corner helper also preserves input endpoints exactly; its old
midpoint reconstruction could shrink a bound by floating-point roundoff and
reject points used to construct that same measured envelope.

Validation after these changes: **738 related tests passed**, with two existing
dependency warnings; Ruff and whitespace checks pass. This includes stale-age
protection under approximate association rejection, planning-only perception
reset, explicit public failure codes, new-face volume association, rejection of
a stationary object under an 80 mm moving-tool prior, and generic grasp-height
proposal behavior. Native qualification is recorded separately from unit tests.

`pear-v22-cli` completed a roughly 90 mm two-pad lift; the independent scorer
reported retained pickup, but a fixed 4 mm aperture-change condition stopped
the observed controller before it accepted lift. The explicit simulation carry
condition now permits seating by half the original remaining finger travel,
clamped to 4–10 mm. It still refuses jaws near the empty-close goal and larger
changes, and requires fresh depth. The actual per-finger tolerance is recorded.

`pear-v23-cli` completed **observed lift verification** and began preplace.
Its maximum two-pad lift was about 98 mm. During preplace the grip continued
seating/closing and the carry condition stopped the task; the final scorer no
longer confirmed retained pickup. This is verified grasp/lift progress, not a
completed sink transfer. No physical score or placement verification was relaxed.
The next control uses a narrower valid finite gripper target (6 mm per finger,
instead of 12 mm) to test closure reserve; that profile is retained separately.

`pear-v24-cli` retained roughly 84 mm two-pad pickup in the scorer but stopped
on asymmetric per-finger seating. The seating budget now bounds total absolute
jaw travel, retaining per-jaw blocked-empty checks. This permits an object to
center without granting more total motion. The associated regression includes
asymmetric centering and still rejects empty or excessive travel.

`pear-v25-cli` reached about 77 mm maximum two-pad lift but lost the object
before observed lift verification completed; the jaws approached the finite
empty-close goal and the scorer no longer found retained pickup. This loss
must not be hidden by further tolerance changes. No complete sink transfer
passed in this batch. The best shared-controller progress remains v23's
observed lift verification and start of preplace.

### Next task-level gap

The shared grounder currently deliberately requests only regions
(`estimate_grasp=False`, `estimate_grasp_site=False`). Thus these controls still
use the robot's calibrated fixed top-down grasp. The existing magnified VLM
refiner can already propose a solid body site, closing direction and height,
and shared `GraspProposal` / `calibrate_observed_grasps` primitives already exist.
The next implementation should connect those proposals to the common observed
candidate generator, with capture/frame binding, depth validation and native
hand/tool calibration. Codex-provided richer grasp hints must reach that same
seam. Do not add another Panda executor or assume a CAD shape. Then repeat
matched native controls before broader task/robot qualification.

Single-view and multiview tracking now call the same approximate estimator;
there is no Panda-specific tracking policy. Reference initialization also marks
all pose axes unqualified: fitting a reference to itself establishes a coordinate
convention, not measured rigid observability.

Final checkpoint validation: **743 related tests passed**, with two existing
dependency warnings. Ruff and whitespace checks passed. The complete test log
and all native reports, requests, decisions, sensor images and failure witnesses
are retained under the artifact root. This validates the shared integration and
its refusal semantics; full transfer and frozen benchmark rates remain open.

## Shared semantic grasp integration

The common source now realizes the magnified VLM's site, closing direction and
world-height fraction as `GraspProposal` values. Pixel hints select local measured
material; projecting through an unseen low plane had displaced the first site's
XY by several centimetres. The native binding owns the hand/tool transform and
aperture limit. The shared support-candidate generator calibrates these proposals
and preserves the binding's approach/lift/retreat settings. Known hand geometry
and the support measurement margin can raise preferred targets; no object CAD
or new model query while carrying was introduced.

Codex and other supervisors can supply the same seam through capture-bound
`hints.grasp.world_from_hand` plus required aperture. Strict mode refuses a
required adjustment; preferred mode reports requested/applied values. Primitive
hand geometry is supported alongside meshes/boxes, and unnamed geometries use
IDs. The API contract is documented in `docs/api/tamp.md`.

| Trial | Outcome |
| --- | --- |
| v26 | VLM proposals reached native motion; projected low-plane site led to descent tracking failure. |
| v27 | Measured-site proposals passed observed lift, then lost grip during preplace; counter contacts were recorded. |
| v28 | Tighter joint tolerance could not settle; that experimental binding change was reverted. |
| v29, v30 | Simulator exited before binding (signal 4); task unmeasured, failure metadata retained. |
| v31 | Traceback-enabled startup succeeded; descent tracking failed with torso near lower stop. |
| v32 | Existing fixed-torso binding lifted about 84 mm and moved the pear about 39 cm into the sink. Upstream goal and whole-box checks returned true; tool remained partial after grip loss before planned release. One unintended contact frame and 91 touch frames were recorded. |

These are development controls with scripted outer selection and live Qwen
visual grounding. v32 is evidence of physical goal achievement, not controlled
placement or full observed-tool success. No failure was relabeled success, and
no matched Codex/Qwen rate is established. Next, stabilize carry/release and
verify the task goal from observations after an unexpected release.

Validation: **759 related tests passed**, plus focused strict/preferred grasp,
reference-binding and architecture checks. Ruff and whitespace checks passed;
CI YAML and shell syntax were checked locally. The existing TAMP workflow now
runs targeted shared-grasp lint, grounding/phase/geometry tests and an AST import
boundary guard. Broad CPU CI and pre-existing legacy layering inversions remain
separate follow-ups. The living implementation plan was reconciled rather than
extending its superseded per-trial history.

## Obstacle-enabled carry diagnosis

The v32 independent physics trace identifies a forearm strike on the faucet
spout at simulation time 50.35 s, followed by loss of both pad contacts around
52.75 s. This diagnostic uses scorer fixture labels only to explain the failure;
fixture coordinates and object state are not supplied to the runtime planner.
The hand remained upright. Disabling all environmental checks therefore hid a
real carry obstacle; gripper tolerance alone does not explain that run.

| Trial | Outcome |
| --- | --- |
| v33 | Dense certified execution instead of coalesced commands still lost grip during preplace; upstream goal false, 44 unintended contact frames. |
| v34 | Restored local environmental collision checks; preflight refused with no motion. Approximate target-envelope and local-surface blockers exposed. |
| v35/v36 | Scoped approximate target contact and retained cross-view object associations. No motion; a hand-adjacent measured return still blocked planning. Preparing private ray-query data did not remove it. |
| v37 | Debug sites hidden from sensor rendering. The hand-adjacent phantom blocker disappeared; a remaining near-object surface blocked preflight. No commands sent. |

Frozen robot-only replay established the sensor defect: pixel (401, 220) in
v35's alternate camera measured 0.931780815 m; rendering the capture-aligned
robot alone reproduced 0.931780279 m. Diagnostic segmentation attributed it to
`gripper0_right_grip_site_cylinder`, a translucent debug site with 10 m half-length.
Hiding sites removes that return; physical geometry groups are unchanged.
See artifact `debug-site-depth-replay.json`. Renderer/scorer identities were used
only for this diagnostic, never for agent object grounding or motion planning.
The shared RoboSuite sensor renderer now excludes debug sites by default.

Tracking interior erosion is useful for fitting, but its omitted object edges
must not become background obstacles. Capture-bound views now retain a separate
full measured association mask. Mapping uses that association, bounded by the
approximate object's local volume, while fitting retains its interior subset.
Unassociated pixels, including other surfaces inside the approximate box, remain
obstacles. Bootstrap retains all contributing cameras' associations. No CAD
completion or blanket volume erasure was added.

Approximate boxes are explicitly distinguished from known-model solid material
in the contact contract. Existing explicit contact-terminated simulation mode
can approach the approximate target with the named hand in grasp/close only;
measured surrounding surfaces, arm links and self collision remain checked.
Default/hardware and known-model target policies are unchanged. Contact identity
includes this distinction, and a known-model snapshot cannot claim approximation.

Robot surface ray queries now prepare a private position-complete MuJoCo data
copy. Mapping and tracking cannot share mutable native ray-query scratch or rely
on a capture that has only had forward kinematics updated. A regression verifies
caller-state preservation and separate scratch ownership across calls.

v38 retained full associated edge masks, with local obstacle checks enabled.
It still refused before motion (`ik_failed`); remaining hand/world witnesses lie
near the pear, for example [1.923, -0.477, 0.987] m. The frozen sensor replay finds
nearby pear-colored returns in both scene cameras, with different illumination.
Thus the full-association fix removes erosion as an ownership loss mechanism but
does not establish complete cross-view object segmentation. Do not clear the
whole approximate box to hide that remaining uncertainty. Next, qualify measured
object ownership across views/lighting, then retry obstacle-aware carry/release.
The v38 scripted outer selection is not a new autonomous-agent benchmark result.

Checkpoint validation: **778 related tests passed**, with two existing dependency
warnings. Focused rendering/concurrency checks passed (9 tests); Ruff and whitespace
checks passed. TAMP CI includes the robot-ray and sensor-scene regressions. All
v33–v38 reports and failures remain under the artifact root; no new complete
transfer success is claimed.
