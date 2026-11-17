# Live RGBD transport cause and lossless repair

November 17, branch `fix/tamp-observed-admission`, after `1c10e344` with the
transport/provenance changes uncommitted during these diagnostics. Raw artifacts:
`.tamp-runs/observed-depth-20261117/`. Compact
[machine-readable evidence](observed_rgbd_transport_20261117.json).

**The reproduced live first-update failure is caused by JPEG RGB colour
corruption.** It is not explained by a long LLM pause or by the measured depth
quantization in this recording. Explicit lossless RGB transport fixes the
stationary live camera gate with unchanged tracker/model/motion thresholds.
This is not a complete observed transfer or a general depth-ownership certificate.

## Causal check on identical measurements

The new optional registration packet archives original RGB, raw depth, aligned
depth, camera calibration and original acquisition intervals. It contains no
object IDs, scene segmentation or contact state. The ordinary tracker still
uses the decoded ZMQ observation. The recorded camera pose remains simulator
localization debug; no new hardware sensor is inferred.

`20261117_093749_31e804` reproduces the ordinary JPEG failure and archives both
original and wire measurements. `20261117_093911_43db76` replays the same two
captures, identical seed pixels, cylinder prior and tracker thresholds:

| RGB source | Depth source | First update |
| --- | --- | --- |
| Ordinary wire JPEG | Ordinary wire depth | Rejects `geometry_tracking_no_match` |
| Ordinary wire JPEG | Original aligned depth | Rejects `geometry_tracking_no_match` |
| Original RGB | Ordinary wire depth | Candidate accepted; 268 mask pixels |
| Original RGB | Original aligned depth | Candidate accepted; 268 mask pixels |

Original capture gap: 0.290001 seconds. Maximum depth transport difference is
0.00099993 m. JPEG changes 67,233 RGB pixels in that frame. For example, an
extra target-colour candidate has original RGB `[143, 101, 55]` but wire RGB
`[135, 100, 96]`; its depth changes only from 0.9910 to 0.9900 m. Another becomes
`[115, 89, 90]` from `[108, 86, 116]`. The candidate colour corruption introduces
incompatible surfaces into the complete reachable union. The replay changes
RGB and depth independently; it does not discard small components or loosen fits.
This particular result does not establish that JPEG is the cause of every
historical tracking failure, including the separate raw-camera robot returns.

## Repair and live checks

`EMET_SIM_RGB_CODEC=png` opts the Stretch full and servo publishers into lossless
RGB transport. The existing image decoder accepts PNG. Legacy JPEG remains the
default; depth transport, render geometry, capture timing, native dynamics and
motion/geometry bounds are unchanged. This preserves captured sensor pixels;
it does not obtain privileged object geometry. Hardware and other robot publisher
profiles are not qualified by this change.

| Cohort | Job | Tracked captures | Consumer-pause windows |
| --- | --- | --- | --- |
| Original tilt -0.7; provenance archived for audit | `20261117_094228_2ef7af` | 41 | 4/4 pass; 10, 11, 9, 10 updates during the respective 3-second pauses |
| Changed tilt -0.95; provenance channel disabled | `20261117_094430_eecfb4` | 43 | 4/4 pass; 10, 11, 10, 11 updates during the respective 3-second pauses |

Each initial candidate counts in the tracked-capture total. The metrics are
separate development source cohorts, **not 84 task trials**. No post-pause
collection wait contributes to the idle-window counts. All eight retained PNG
frames in the audited cohort are bit-identical to original RGB; depth still has
its ordinary less-than-1-mm transport quantization. The unaudited repeat shows
that lossless RGB works without the extra diagnostic channel. Setup sends only
posture, velocity and head commands; both audits show zero forbidden actions and
no manipulation transfer. There are no source/client shutdown errors in results;
server teardown logs include multiprocessing broken pipes after termination.

## Validation and reusable harness scope

The final affected transport/camera/core-codec/tracking/source batch passes
**96 tests, zero failures/errors/skips**, including malformed-channel and
replay-reference regressions (`20261117_094851_f43292`). The earlier overlapping
batch passes 94 tests. The source packet
round-trips through JSON and validates depth reprojection. Replacing either head
channel invalidates old provenance. The replay tool rejects long bounded archive
tails missing the original reference, rather than pairing old seed pixels with
a different capture. Independent read-only review finds no remaining blocker.

The paper now discusses TAMPAgent and expands Agentic GenTAMP's relevance to
reusable task interfaces and development/deployment splits. PDDLStream was
already cited. `paper/build.sh` passes after materializing 6.3 MB of existing
cached LFS figures (`20261117_094333_00c344`). The first build's libpng error was
an unmaterialized figure pointer; no figure contents were changed.

Next: assemble the live observed support/background, qualified robot/target
material and contact/effect evidence into the existing source/refiner interface,
then require an actual observed pick/place. Keep the separate raw-camera
robot-interference failure, observed navigation/exploration, complex household
adapters and hardware qualification open. Frozen programs and other planners
should be able to use the same task/reset/tool/result boundary; numerical
baseline reproduction is a separate optional comparison.
