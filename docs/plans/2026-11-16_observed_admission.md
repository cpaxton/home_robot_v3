# Observed TAMP execution repair

Branch `fix/tamp-observed-admission`, based on main `6386e9d3` after the native
agent stack landed. Tracks [issue #245](https://github.com/cpaxton/home_robot_v3/issues/245).
Keep physical hardware disabled and native privileged controls separate. A
robot-only model, RGBD, encoders and declared calibration/geometry priors are
permitted runtime inputs; simulator object/contact state is evaluator-only.

Integrated the consolidated merged-stack corrections from
[PR #246](https://github.com/cpaxton/home_robot_v3/pull/246), runtime `5a891c59`,
after the `3b1fd9ad` observed checkpoint. Conflicts retain both the qualified
background/contact policies and the post-barrier observation/robot-selector
fixes. The combined admission/contact/release/API/provider regression batch
passes 390 tests, zero failures/errors/skips (`20261116_162854_3643a6`). This
does not complete any live observed-transfer gate below.

PR #246 is now merged as `2ccf390c`. This branch integrates that main commit at
`63dceb73`; all runtime/scripts/configs bytes are unchanged from the passing
390-check integration. Squash-ancestry conflicts preserve the tested combined
implementation and import the final evidence documents. No new task score or
observed/hardware qualification follows from this ancestry-only integration.

Implemented so far:

- Maximum elapsed capture-interval time for bounded mask reachability. This is
  not a fix for the historical RGB/depth silhouette-ownership disagreement.
- Symmetry-aware observed task settling and revision comparison; unique and
  shape-aware retention retain both captures' external error bounds. A frame
  cannot downgrade payload uncertainty.
- Planning-only exceptions/malformed plan results return no-actuation receipts.
  Execution remains outside that catch; missing dispatch flags remain unknown.
  Safe observation errors survive the JSON/task boundary.
- Opt-in captured-material admission with real primitive geometry and explicitly
  declared convex material; no AABB or hollow-mesh ownership inference. Inner
  material is eroded by supplied uncertainty; candidate envelopes must contain
  the actual nominal robot geom and its declared error reserve. Candidate poses
  never construct ownership. Geometry/model/calibration/configuration/clock
  profiles remain pinned across capture refreshes.
- Existing environment/mixed occupancy wins over ownership. Unknown queried
  material is rejected unless captured material or genuinely measured free
  space covers it. Subdivision exhaustion rejects. No workspace cell is marked
  free by ownership.
- Geometry qualification checks self-attributed endpoint error balls before
  exclusion. False/unproved labels remain background when qualification is
  requested. Unqualified excluded endpoints invalidate historical free cells;
  old occupied cells remain occupied. Hidden non-endpoint evidence retains its
  original acquisition age. A bounded static assumption extends validity only.
- Exact fingertip target-contact policy, pinned to the current visual capture,
  complete prior, pose error and robot model. Nonfinger clearance stays active;
  empty target cores reject. Fresh released-object geometry and the retained
  release hypothesis both constrain retreat. Environment/unknown checks remain
  independent. Local support-contact admission is still missing.
- Finite observed gripper closing configuration and motor-command mapping:
  certify the sampled complete sweep in preflight, refresh/revalidate segments
  before dispatch, check measured arrival and settling. Missing configuration
  rejects instead of sending a full-close preset. No contact/force sensor is
  introduced. This does not qualify a Stretch controller or passive geometry.

The source-backed MuJoCo raycast regressions distinguish raw start rejection,
exact captured-material start admission, unseen-shadow rejection and motion
through a real historical scan. These are geometry tests, **not a live Stretch
transfer**. Nonzero uncertainty/clearance shells still need measured free space.
No whole-free callback or automatic mesh-material declaration is used by the
new admission tests.

Completed gates: 275 affected geometry/background/API/provider tests pass
(`20261116_135501_a4683d`), and the full motion suite passes all 481 tests
(`20261116_140002_b0273c`), with zero failures/errors/skips in either run.
The expanded task/CHAT run passed 425 of 427 tests; two failures exposed a
public error-code compatibility regression. After restoring the canonical
`object_state_unknown` mapping to `observation_unavailable`, all 125 focused
API/provider tests pass (`task-public-codes.xml`). The subsequent expanded
task/CHAT run passes all 437 tests (`20261116_144121_f38b46`). Earlier failed attempts are
retained under `.tamp-runs/observed-admission-20261116/`: cube fixture collection,
private planner mutation expectation, missing public ambiguity code and the
unsafe unqualified-self-mask prescan expectation were diagnosed separately.
The public-provider-code regression remains part of the required checks.
The subsequent contact batch passes 208 tests with zero failures/errors/skips
(`20261116_144121_b819f2`). It includes the original affected seams plus contact,
closing-sweep and partial-receipt regressions. Earlier attempts are retained:
191 tests exposed an incomplete custom-close test fixture; the corrected fixture
and release geometry passed 202 tests before closing-sweep additions. The latest
full motion check passes 506 tests (`20261116_144227_c21a8b`). Review then added
final-settling target convergence and an immediate pre-dispatch budget check;
all 210 focused contact/integration tests pass after those fixes
(`20261116_144622_47b5f9`). These suites overlap and are not task success metrics.

Independent archive inspection attributes the first tracking failure to two
head pixels at the RGB target silhouette whose registered depths describe a
different surface. One pixel is target-labelled even by the independent RGB
scorer, so stronger colour selection or removing a small component would not
establish depth ownership. The archive lacks capture-bound motor feedback and
depth reprojection provenance; its final privileged qpos cannot supply them.
Collect those inputs on a new diagnostic before implementing an ownership
filter. Preserve unknown/mixed endpoints and keep scorer masks evaluator-only.

Remaining gates, in order:

November 16 continuation: same-sample eight-motor and original-depth provenance
are now archived, and the new whole-robot outer envelope passes the 620-check
motion/capture/adapter regression batch. Two separate physical controls contain
all 175 sampled whole-robot volumes, including 75 in target contact. This remains
conditional model/audit evidence, not qualified self ownership. A two-frame
capture at a valid 1.351-second gap reproduces the old tracking rejection:
independent rays identify both suspicious depth returns as `link_d405`, even
though one RGB pixel is target-labelled. A fresh normal 81/81 repeat is therefore
not a fix for timing-dependent association. See
[results and exact cohort limits](../experiments/observed_transfer_progress_20261116.md).

The committed geometry audit then exposes two native CPU `_structs` crashes;
the faulthandler reproduction identifies the bulk contact-field read in the
existing scene collision checker. A scalar-snapshot mitigation, `f34380ac`,
passes 110 physical/contact/payload checks and a source-frozen 2/2 native-control
replay with 175 contained geometry samples. Both failed attempts remain recorded.
This native-runtime mitigation is separate from observed-task acceptance and
does not close the source/contact gates below.

1. Local support-contact rules and full-source validation of the new fingertip
   and finite-closing policy. Preserve unrelated obstacles, wrist/housing checks
   and deep-penetration rejection; no whole-support or unknown-space exemption.
   Retain support-only versus mixed/unattributed occupancy provenance. Compare
   the full uncertain physical payload against the local plane and unchanged
   1 mm penetration limit; strip only planning clearance. Lift departure uses
   one fixed initial penetration floor, independent of planner query order.
   Initial and destination patches can differ: a fresh view cannot fill the
   support hole under the resting object. Preserve original scan ages or declare
   a separately qualified finite planar-continuity prior.
   November 17: the local policy, historical owner/capture binding, measured
   departure anchor and snapshot/checker integration are implemented. Corrected
   focused checks pass 161 tests; support/path audits pass 31. Source-backed
   MuJoCo ray fixtures admit shallow table contact and reject a visible small
   peg; lift/place paths pass an independent dense collision replay. These are
   mechanics checks, not a live Stretch transfer. The source must still qualify
   masks, backing thickness, registration and motion-error budgets. See
   [the current support record](../experiments/observed_support_contact_20261117.md).
   Broader motion/TAMP/capture regressions pass 988 tests. After preserving the
   same attachment's original floor during measured-pose replanning, 188 final
   physical/interface checks pass. Independent lift/place replay records 110
   samples per path and 0.3 mm peak penetration against the unchanged 1 mm bound.
2. Full calibrated Stretch outer link/gripper geometry over unmeasured joints,
   with inner material justified independently. Do not substitute passive debug
   joints or telescope equal-split estimates for actual sensors.
   The motor-conditioned whole-robot interval implementation is now tested;
   qualification of its model/pose bounds and separate inner material is open.
3. Concrete live source/refiner assembly using original synchronized captures,
   measured support/background scans, qualified self attribution and visual
   effects. No default execution enabled from read-only discovery alone.
   Target self-occupancy also needs a separately typed, capture-bound material
   proof for a declared solid prior. Excluded target cells remain unknown;
   captured robot material cannot admit the payload's own occupied volume.
   November 17: the independent camera producer and whole-camera rigid-tool
   interference rejection are implemented; 79 focused checks pass. Live original
   and changed-tilt views still reject their first update at normal 0.266-second
   gaps. Publish original raw registration provenance on the live ZMQ path next;
   continuous processing alone does not fix incompatible depth returns. See
   [the camera-source evidence](../experiments/observed_camera_source_20261117.md).
   Subsequent same-capture replay isolates the live failure to JPEG RGB colour
   corruption, not depth quantization. Optional original provenance publication
   is implemented. Explicit PNG transport passes 41 captures at the original
   view and 43 at the changed view (the latter without provenance inputs), each
   through four paused-consumer windows. This resolves the live transport gate;
   complete source/refiner/ownership assembly below remains open. See
   [the causal transport record](../experiments/observed_rgbd_transport_20261117.md).
   Same-capture camera/motor/gripper evidence is now retained with tracked views.
   The live debug profile assembles six visual frames across 27 tracked captures;
   declared FK/localization error reaches lift, retention and retreat checks.
   Historical reference validation uses original image receipt as well as robot
   bracket time. This is a source component, not the complete collision source
   or observed task gate. See [the visual-frame integration audit](../experiments/observed_visual_frames_20261117.md).
4. Resolve moving-mask foreign components with independent geometry/provenance
   evidence; preserve tiny unresolved competitors, complete-union controls,
   latched loss and explicit fresh reacquisition. Do not retime captures, shrink
   motion bounds or drop points by fit quality.
5. First stationary public-tool observed transfer, independent physical scoring,
   pre-command stale/unknown rejection and post-dispatch loss/consumed-handle
   controls. Then actual Caliban orchestration, held-out layouts and multistep
   tasks. Do not report those gates complete from the synthetic adapter tests.

November 17 mapped-contact integration now completes a **separate assisted
control**: both a cube and cylinder pass public-tool grasp/lift/place/release/
retreat against a provided 5 mm RGBD map, with native robot/object state.
Original depth-ray geometry, continuous backing contact across voxel boundaries,
canonical payload uncertainty, and map guards through held/released checker
clones are implemented. Pre-command occupied-map and after-grasp evidence-loss
controls stop correctly; the latter preserves partial progress and sends no
further command. Full motion/TAMP regression passes 1,044 tests; subsequent
environment-query and immutable-cache batches pass 192 and 129 overlapping
checks. The [experiment record](../experiments/mapped_contact_20261117.md)
retains source profiles, negative cases, cancelled infeasible goals and a native
crash separately. This establishes collision-map integration, **not completion
of gates 2–5 above**. The observed source/refiner and relative contact geometry
remain the next qualification work; no fictitious passive sensor was added.

### November 17: local map resources and mobile control

The next OVMM-shaped control adds batched workspace construction and an opt-in
single-owner local scene store with generation leases. Old maps/plans invalidate
before reacquisition; capture ages and unknown occupancy remain unchanged.
An isolated 2.304-million-cell comparison preserves the exact grid checksum and
reduces peak RSS from 1,877,245,952 to 136,491,008 bytes. The initial mobile
control reaches grasp/lift but outlasts source evidence during loaded preflight;
reacquisition now occurs at the stopped loaded handoff. See the
[mobile experiment record](../experiments/local_map_mobile_20261117.md).
This is provided-map/native-state assistance, not default OVMM or full observed
source qualification. One cube control now completes navigation/carry/place/departure;
the cylinder completes carry but rejects release with persistent measured
tracking shortfall. Both already use the native endpoint-settling gate; a
redundant proposed setting and its duplicate runs were discarded. Release and
collision bounds remain unchanged.
