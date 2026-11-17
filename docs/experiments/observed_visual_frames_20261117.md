# Coherent observed visual-frame integration

November 17, `fix/tamp-observed-admission`, after `bab3a405`. Raw diagnostics are
under `.tamp-runs/observed-frame-20261117/`. The schema-1 simulation profile declares
object error (3 mm, 0.02 rad) and robot pose error (1 mm, 0.005 rad). These are
explicit development assumptions, not measured hardware bounds or fit confidence.
**No manipulation transfer or complete observed executor is qualified here.**

The [compact machine-readable record](observed_visual_frames_20261117.json)
contains final test counts and the remaining gates.

## Issues found and repaired

- Tracked views discarded original camera calibration and encoder geometry.
  A later read could therefore mix captures when assembling effects. Immutable
  `RgbdCaptureGeometry` now retains camera K/pose, the same eight motors,
  motor-derived EE and conservative gripper volumes. Pixels/world points,
  capture identity and camera/model/error profiles remain bound.
- `capture_arrays` requested float64 output from a helper that still computed
  with float32 intermediates. Genuine float64 unprojection fixes the numerical
  mismatch without widening the geometry comparison tolerance.
- Declared robot pose uncertainty only inflated gripper volumes; visual lift
  acceptance treated EE pose as exact. Robot position/angular errors now reach
  relative retention, relative rotation and EE lift-height checks. Angular error
  includes its displacement over the EE-to-object lever arm. Retreat checks
  conservatively reserve pose uncertainty; uncertain separation is unknown.
  The added regression also preserves rejection of empty gripper geometry.
- A same-snapshot motor bracket ends before RGBD transport receipt. Historical
  reference validation used that earlier bracket time and incorrectly rejected
  the assembled reference as stale. It now uses the maximum of the two original
  receipts and rejects future receipts. No acquisition timestamp is renewed.
  An assembly-to-verifier regression reaches semantic static-object rejection
  instead of a spurious stale-reference result.

## Evidence

| Attempt | Job | Outcome |
| --- | --- | --- |
| Initial contracts | `20261117_102528_7742d7` | Eight new-fixture errors: missing timing availability/domain fields; existing tests run separately in the same batch |
| Initial live binding | `20261117_102528_ea7530` | Rejects `rgbd_geometry_capture_mismatch`; exposes intermediate-precision bug; no commands beyond setup |
| Corrected contracts/effects | `20261117_102900_5f758a` | 157 checks pass |
| Added retreat/precision checks | `20261117_103125_e4c3f6` | 160 checks pass |
| Corrected live binding | `20261117_103125_b1bdb3` | 27 tracked captures; three paused-consumer windows pass with 8, 9, 9 updates; six visual frames assembled |
| Final contact/integration | `20261117_103516_d25f95` | 218 passed, zero failures/errors/skips; includes historical-reference regression and broader support/admission checks |

The live run uses PNG through the ordinary ZMQ client, a separate robot-only
model, same-snapshot eight-motor feedback and simulator localization debug.
Every assembled visual frame carries eight motor values and eleven conservative
gripper volumes. It never reads the privileged passive-joint packet. No extra
provenance channel, VLM or local GPU is required. The ideal frozen sample bounds
temporal misalignment, while the separate declared robot pose errors remain.
The live run precedes the final historical-reference consumer correction; the
assembly-to-verifier regression covers that additional integration path.

Read-only review caught the omitted robot errors and historical-reference bug;
both are addressed. The final review finds no remaining concrete coherence or
error-bound blocker in this slice. Tests overlap; capture/frame totals are not
task trials, and assembly is not a verified lift or placement.

Remaining: qualified observed background and robot/target material ownership,
support/contact evidence bound into the complete source/refiner, then an actual
stationary observed pick/place with independent scoring and failure feedback.
Read-only discovery must not enable execution from these camera checks alone.
