# Observed camera source: interference and independent acquisition

November 17, branch `fix/tamp-observed-admission`, after support checkpoint
`61ac6adf`. These are development camera diagnostics, not task trials. **No
observed pick/place success or live collision-source qualification is claimed.**
Compact [machine-readable record](observed_camera_source_20261117.json); raw
archives remain under `.tamp-runs/observed-camera-20261117/` in the owning checkout.
Runs record implementation hashes because the checkpoint had uncommitted changes.
The camera cohorts below differ in initial conditions and must remain separate.

## Changes and checks

`RobotViewGuard` reconstructs original winning depth rays and separately binds
the RGB-grid world points consumed by the fitter. Its rigid tool/camera outer
geometry comes from eight motor readings and a robot-only model. A possible
intersection rejects the entire camera association. It never deletes a target
pixel, labels an endpoint self, clears a map cell or certifies untested links.
The model, camera pair, capture, coordinate arrays and declared error budgets
remain bound. Signed-zero FK roundoff affects only the calibration fingerprint,
not the measured geometry. Passive fingers remain outside this guard's scope.

`ContinuousRgbdSource` processes original frames in a background producer rather
than waiting for a tool/LLM consumer. It publishes immutable candidate/frame
pairs, collects two fresh post-barrier captures, and latches loss/staleness.
Identical retransmissions retain the original published freshness despite later
receipt; changed source intervals or contents reject. Shutdown joins the reader
before its shared client closes, and blocked readers produce explicit failure.
The diagnostic counts arrivals during the consumer pause before collection can
wait for more frames. This candidate source still lacks the support, background,
ownership and visual-effect assembly required by `ObservedNativeAdapter`.

Final focused CPU regressions: **79 passed, zero failures/errors/skips**,
`20261117_085410_ff96e6` (`final-regressions-corrected.xml`). This includes real
robot-model guard tests, source lifetime/replay/collection tests, cross-camera
rejection, gripper/encoder contracts and depth registration. Earlier overlapping
checks passed 67 and 47 tests. A preliminary final command used the wrong encoder
test path and collected no tests; its failed job is retained. Ruff is clean.
These checks are not task success rates.

## Paired private-snapshot camera cohort

The native development fixture relocates its target during setup. Stationary
three-capture diagnostics use a requested minimum 1.35-second acquisition period,
same-snapshot motors and original depth registration provenance. Both leave only
their initial candidate valid; neither attempts a transfer.

| Head pan/tilt | Job | Result |
| --- | --- | --- |
| -1.57 / -0.7 | `20261117_082352_b78997` | 1/3 candidate captures; head rejects `rgbd_robot_interference`, wrist loses association, then global loss remains latched |
| -1.57 / -0.95 | `20261117_082352_19af92` | 1/3 candidate captures; the complete union fails geometry, then loss remains latched |

Independent archive inspection identifies the original suspicious returns as the
rigid `link_d405` camera housing. One RGB pixel is target-labelled even by the
independent scorer. The changed view instead admits three target-coloured
background singletons into the unchanged elapsed-time reachability region.
Neither stronger colour thresholds nor dropping small components proves ownership.
These early recordings precede the final guard world/profile binding fixes and
are retained as diagnostic evidence, not final source qualification.

## Live ZMQ camera cohort

The public live fixture retains its original target placement. It uses an
operator red interior seed, explicit cylinder prior (radius 0.02 m, half-height
0.04 m), same-sample eight-motor capture timing and simulator camera localization.
No object poses, segmentation or contacts enter tracking. CPU OSMesa renders the
ordinary camera stream; no local GPU or VLM is used. Raw registration provenance
is not published by this ZMQ stream, so these runs explicitly have no
`RobotViewGuard`. Setup sends posture, velocity and head commands; the audit
records no forbidden command. No manipulation transfer is dispatched.

| Attempt | Job | Result |
| --- | --- | --- |
| Initial launch | `20261117_084947_00af2b` | Setup command refused because the launch omitted the required physical audit path; no tracking assessment |
| Audited original view | `20261117_085044_d2e952` | Initial candidate, first update rejects `geometry_tracking_no_match`; 0 completed consumer-pause rounds |
| Original view with input archives | `20261117_085212_3d5d9e` | Same failure at a **0.266493 s** original capture gap; 0 completed rounds |
| Explicit tilt -0.95 | `20261117_085455_ca0fab` | Same failure at a **0.266698 s** original capture gap; 0 completed rounds |

The archived original-view update contains a 238-pixel connected component plus
three isolated returns incompatible with the declared cylinder. Its seed fit
has a 0.181 mm RMS diagnostic residual; that residual does not qualify a physical
pose-error bound. The complete union correctly rejects instead of trimming
unresolved returns. Thus a long LLM/tool pause is **not sufficient to explain
the live failure**. Continuous processing has passed its orchestration tests,
but neither live view completes its pause gate. The alternate-view attempt
includes the reviewed duplicate, collection-accounting and cleanup corrections.

## Next gate

Publish original raw registration provenance alongside the live stream, with
the original capture/motor ticket intact. Diagnose each unresolved return with
that source ray and independently measured scene/support geometry. Unknown or
mixed correspondence must remain unknown; no target-fit-based deletion or
automatic free-space claim is justified. Reacquisition must use an explicit new
reference after loss. Then repeat the live camera gate before assembling the
support/background/robot/contact evidence into the observed execution source.
Only a complete observed transfer can close the task gate; privileged native
successes and camera candidates remain separate controls.
