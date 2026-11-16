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

1. Local support-contact rules and full-source validation of the new fingertip
   and finite-closing policy. Preserve unrelated obstacles, wrist/housing checks
   and deep-penetration rejection; no whole-support or unknown-space exemption.
2. Full calibrated Stretch outer link/gripper geometry over unmeasured joints,
   with inner material justified independently. Do not substitute passive debug
   joints or telescope equal-split estimates for actual sensors.
3. Concrete live source/refiner assembly using original synchronized captures,
   measured support/background scans, qualified self attribution and visual
   effects. No default execution enabled from read-only discovery alone.
4. Resolve moving-mask foreign components with independent geometry/provenance
   evidence; preserve tiny unresolved competitors, complete-union controls,
   latched loss and explicit fresh reacquisition. Do not retime captures, shrink
   motion bounds or drop points by fit quality.
5. First stationary public-tool observed transfer, independent physical scoring,
   pre-command stale/unknown rejection and post-dispatch loss/consumed-handle
   controls. Then actual Caliban orchestration, held-out layouts and multistep
   tasks. Do not report those gates complete from the synthetic adapter tests.
