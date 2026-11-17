# Observed transfer: capture provenance and whole-robot bounds

Branch `fix/tamp-observed-admission`, continuation of `b0ff18ad`. No complete
observed-input pick-and-place succeeds in this record. Controls use privileged
native planning and independent physical scoring; tracking receives RGBD,
calibration, an operator red-appearance seed and a declared cylinder prior.
No VLM, local GPU or physical hardware is used.

Raw files remain under `.tamp-runs/observed-transfer-20261116/`. The accompanying
JSON retains compact results and evidence hashes instead of per-frame arrays.
These are dirty-source development diagnostics, not a clean-commit benchmark or
an independent reliability population. The normal transfer and physical envelope
audit precede the final tendon-signature/base-validation guards; those guards
have separate regression coverage. Their physical outcomes are not retroactively
rescored at the later source.

The committed-source audit at `060cdb69` subsequently crashes in native code:
`robot-contact-final` completes only the cube control, then child exit `-11`;
the unchanged-source `robot-contact-faulthandler` retry crashes during its first
placement before publishing a completed control. Both source-integrity records
report unchanged Python sources. These attempts do **not** pass qualification,
and the earlier development 2/2 is not substituted for them. See the native
incident below.

| Check | Result | Limit |
| --- | --- | --- |
| Initial capture/encoder/multiview regressions | 45 pass, zero failures/errors/skips | Before whole-robot addition |
| Initial whole-robot/encoder/gripper regressions | 20 pass, zero failures/errors/skips | Before final calibration guards |
| Full motion + capture + observed adapter | 620 pass, zero failures/errors/skips | Overlaps the focused batches; not task trials |
| Final source-depth reconstruction/calibration checks | 33 pass, zero failures/errors/skips | Includes the last added source-to-UV test; overlaps the full regression batch |
| Final whole-robot ancestry checks | 9 pass, zero failures/errors/skips | Pins nonserial-arm rejection after the last ancestry guard |
| Normal synchronized native transfer | Physical control passes; tracking 81/81 | Unchanged tracker; timing-dependent failure remains |
| Requested 1.5 s minimum capture gap | Physical control passes; tracking 1/89 | Actual first interval 1.5011 s exceeds unchanged 1.5 s maximum |
| Requested 1.35 s minimum capture gap | Physical control passes; tracking 1/83 | Startup/processing makes first interval 1.7692 s; not an inside-limit ownership test |
| Separate stationary pair | Tracking 1/2; `geometry_tracking_no_match` at 1.35137 s | Real inside-limit reproduction; no task execution |
| Whole-robot physical audit | Both native controls pass; 175 samples, zero corner escapes/interval violations | 75 contact samples; conditional mechanical/model bounds, not hardware certification |
| Scalar-contact committed replay (`f34380ac`) | Both native controls pass; 175 samples contained, zero interval violations | Separate replay after both preserved native crashes; runtime sources unchanged throughout |
| Post-mitigation motion/capture/native/observed adapters | 690 pass, zero failures/errors/skips | Final source `f34380ac`; overlaps earlier unit suites |

The first renderer launch fails before simulation because its environment lacks
the existing user-space OSMesa library path. `head-wrist/` preserves that failure;
the corrected CPU renderer uses the existing software libraries. All later
failed tracking runs and their successful privileged controls remain separate.
The loss stays latched; no implicit identity reset or threshold changes occur.

## Concrete tracking cause

The stationary pair archives eight motor readings at the original shared
snapshot interval and both cameras' original depth/calibration. Nearest-depth
provenance identifies head source pixels `(115,308)` and `(116,308)` for registered
RGB pixels `(115,316)` and `(116,316)`. Independent evaluator rays identify both
source surfaces as the wrist-camera body `link_d405`. RGB is respectively
`[196,143,143]` and `[177,128,128]`; the second is also target-labelled by the
independent RGB scorer. Thus RGB target colour does not establish depth ownership.

The scorer identities never enter tracking. The source has no passive finger
angles, telescope split measurements, object pose or contact channel. Camera
world poses are explicitly ideal simulator localization debug. An outer robot
envelope is insufficient to remove these points as proven robot material;
uncertain registration/ownership must remain unknown or trigger a fresh view.
The normal 81/81 repeat does not resolve the delayed-source case.

## Geometry and review

`StretchGeometryEnvelope` covers all collidable robot geoms using the eight motor
channels, a rigid localized base and declared mechanical intervals. Telescope
marginals preserve uncertainty in intermediate links; checked serial collinear
kinematics make the distal sum exact. Passive fingers/rubber/wheels remain
intervals, not sensors. Localization errors explicitly dilate the bounds.

Independent review found a missing tendon-coefficient binding in the kinematic
signature. The fix adds canonical named wrap terms/types/coefficients. Mutation
tests cover coefficients and arm reference positions, plus rigid base validation
and retracted/extended motor endpoints. Follow-up review finds no concrete blocker
in these additions. These are outer geometry and provenance, not inner material,
free-space evidence or observed task success.

## Next acceptance gate

Implement finite local payload/support contact with support-only provenance,
qualified target self-occupancy, and the concrete live source/refiner. Then run
`scene_tasks → plan_pick_place → execute_pick_place_plan` using observed inputs,
score the physical effect independently, and exercise pre-command stale/unknown
and post-dispatch loss/consumed-handle feedback. Preserve the 1 mm penetration
bound, unknown-space rejection and existing retention/tracking thresholds.

## Native CPU collision-accessor incident

Jobs `20261116_232907_fb9824` and `20261116_233200_b02240` preserve the two failures.
The kernel identifies the Python `_structs` extension in the first crash; the
retry's faulthandler stack locates `MujocoSceneCollisionChecker.configuration_collides`
at `data.contact.geom` during an RRT placement edge check. The installed/runtime
MuJoCo is 3.5.0. The older kernel record mentioning 3.17.0 belongs to a different
earlier process and is not this runtime. Rendering is disabled and CUDA devices
are hidden in both jobs; this evidence does not identify a GPU failure.

The targeted mitigation copies scalar `geom1`, `geom2`, `dist` from fresh indexed
contacts, then retains the existing body/depth filtering and exact exceptions.
An empty contact set returns immediately. No solver parameters, collision
thresholds, contact exclusions or retention limits change. Tests prohibit both
bulk fields and per-contact geometry-array reads, alternate real contact counts
through repeated copy/collision calls, and preserve body/geom exceptions. This
bypasses the implicated binding path; the underlying native memory fault remains
unproven until independently investigated.

The committed `f34380ac` scalar-only replay, job `20261116_233748_a47132`, completes
both physical controls and all 175 geometry samples (75 in contact), with zero
corner escapes or interval violations. The raw manifest pins Python/native
MuJoCo 3.5.0 and source hashes; its integrity check reports unchanged source and
child exit 0. All 110 focused physical/contact/payload tests pass, including the
guarded scalar-access regression. This is successful mitigation evidence for
this replay, not proof that arbitrary native corruption is cured or that an
observed task can execute. The broader post-change regression, job
`20261116_233748_c3c620`, passes all 690 tests with zero failures/errors/skips.

Read-only review finds no concrete semantic regression in the mitigation. Its
per-contact Python copies add a constant cost; performance in furnished scenes
remains unqualified and is not inferred from these small control fixtures.
