# Shared grounding: bounded cross-task pilot

## September 13: separated-neighbor pickup and carry controls

`default_table_stretch_clearance.yaml` changes only the blue neighbor's x
position (-0.02 → -0.25 m). The original fixture is untouched; a model-equality
test checks geometry, inertia, limits and all other initial coordinates.

| Job | Source | Physical pick/place | Finding |
| --- | --- | --- | --- |
| `20260913_074731_6dfa15` | `336956a3` | true / false | Wrist reaches 0.163 m and lifts the cylinder, but loaded lift sags 2.33 cm, outside the unchanged 2 cm completion gate |
| `20260913_075853_3d9d40` | `63f7ed1b` | true / false | Lift completes; cylinder is lost during wrist folding; placement then crashes on missing shared `manipulation_radius` property |
| `20260913_081154_ea1dcd` | `4e1f7df2` | false / false | Fixed 30 cm pregrasp requests -5.8 cm extension after precision alignment; correctly rejects motion |
| `20260913_081801_d64db7` | `7e3d4d43` | true / false | Reachable 20 cm pregrasp, pickup, carry, receptacle navigation and place controller complete; released cylinder briefly touches cube then falls to table |
| `20260913_083212_06fb47` | `c6b6b761` | true / false | Cylinder slips during carrying; final visual check correctly rejects absent object and prevents empty release |
| `20260913_084739_a09908` | `424db036` | false / false | Calibrated 3D servo reaches ~1.4 cm error, but 12 mm base corrections are discarded inside the 2 cm arm-base deadband |
| `20260913_085624_0c7f62` | `a29d664d` | true / false | Fine base corrections reach ~5.7 mm grasp-center error and verified pickup; object slips during later transport; visual release check rejects the absent payload |
| `20260913_091308_642f8d` | `9714349f` | true / false | Preserved wrist/lift posture retains pickup and retraction; cylinder gradually moves toward pad edges and loses contact during base-turn braking |
| `20260913_092427_1e2e65` | `b91d761e` | true / false | First contact-offset pilot moves sideways because its calibration used the wrong diagnostic frame convention; still loses the payload |
| `20260913_093103_771bbf` | `69d8e478` | true / false | Corrected insertion direction improves initial depth but still slips during transport; closure tolerance and loaded arm displacement remain relevant |
| `20260913_093656_0b9dbd` | `292e84df` | true / false | World-frame handoff reaches the receptacle and starts placement; payload has already slipped |
| `20260913_094303_496e65` | `0a539b30` | true / false | 25 mm contact pilot retains payload through transport; loses it during a placement approach distorted by a height outlier and an unintended base turn |
| `20260913_095547_9ec2d2` | `c1468a11` | true / false | Payload remains held through all placement corrections; falsely low observed object bottom makes alignment raise the object and refuse release |
| `20260913_100708_0f6983` | `076fa458` | true / false | Registered depth fixes the camera contract, but minority mask-edge depth still drives oscillating placement corrections; payload remains held |
| `20260913_102057_336037` | `ea16a60f` | true / false (cancelled) | Receptacle reacquisition fails before visual placement; navigation repeatedly executes the same truncated prefix without progress; cancelled and independently scored |
| `20260913_103007_684ea5` | `89581c70` | **true / true** | Complete receptacle route, bounded visual placement corrections, released cylinder stably supported by cube at end; agent process also exits zero |
| `20260913_103442_018239` | `89581c70` | true / false | Unchanged repeat takes a longer route; payload slips during final turn, and empty-gripper placement is correctly rejected |
| `20260913_104630_1f3338` | `f7c024d3` | true / false | Payload retained, but Qwen accepts a mixed cube/cylinder receptacle mask; wrong support geometry and final nonconvergence prevent placement |
| `20260913_110132_edd836` | `cfb61c3d` | **true / true** | Coherent support geometry, longer transport retained, verified final release onto cube; process exit zero |

Artifacts: `~/runs/emet/grasp-separated-neighbor-control` and
`~/runs/emet/grasp-loaded-carry-control`, including physical traces/results and
accepted/rejected wrist captures. These are diagnostics, **not acceptance passes**.
Both establish sustained physical pickup, not successful carrying or placement.

`63f7ed1b` preserves the gripper command through simulator posture changes,
matching the real bridge, and increases lift feedback/damping without increasing
the 70 N force limit. Static payload controls at 0, 0.5 and 0.75 kg pass the
existing position gate. This does not establish general grasp reliability.

Replay from the second run's held state (sim time 66.18 s), with identical
physics and arm retraction, reproduces payload ejection on a direct pitch target:
peak measured wrist speed is 20.90 rad/s. `bb8685e9` adds a physics-time
0.8 rad/s wrist reference limiter. The matched replay retains finger contact,
with peak measured speed 1.04 rad/s. These are conservative simulator settings,
not calibrated hardware limits or a guarantee on measured speed/acceleration.
The captured fixture and old-controller negative replay are regression tests.

`4e1f7df2` exposes reach on the common controller, shares precision arm-facing
alignment/reacquisition between pick and place, reads fresh copied joint state,
and prevents release after failed approach/IK. Release and retreat failures
remain failures; confirmed release clears held-object state even if retreat
fails. `7e3d4d43` replaces the fixed pregrasp with six bounded standoffs, largest
first, down to 20 cm; infeasible IK is rejected, not clipped at a negative joint.

The next live control retains the cylinder throughout carry/navigation. At
release, however, its center is offset about 2–3 cm from the nominal end-effector
origin. It briefly contacts the cube then falls onto the table. The tool reports
controller completion with an explicit physical-unverified disclaimer; the
independent scorer rejects placement. Artifacts are
`~/runs/emet/grasp-reachable-place-control`.

`c6b6b761` adds Qwen-verified current head RGB-D before release, without instance
creation or oracle state. It corrects observed payload XY alignment and visible
lower-surface height over the freshly grounded static support, with at most
three 5 cm moves, and abstains on missing freshness/identity/near-gripper support,
invalid IK or failed motion. Each alignment frame/mask/prompt is retained. This
is a local placement prototype: visible geometry may not reveal a hidden object
bottom, the support is assumed static, and it is **not real-robot acceptance**.
The first trial of that check correctly rejects a lost payload. Its saved head
frame shows only the receptacle; private tracing confirms the cylinder fell
before placement. No missing-object detection should be counted as a VLM error.

`424db036` adds a **separate** `query_geometry_manipulation_pilot.yaml` preset:
the same Qwen-associated wrist surfaces, but robust 5th/95th-percentile 3D bounds
and the measured grasp-center pose replace the fixed pixel/depth closure gate.
Corrections are bounded to 5 cm and require valid IK and completed motion.
The old preset remains the control. Placement also trims depth-edge outliers.
The geometry trial stabilizes at about 12 mm lateral / 7 mm vertical error;
actuator traces show no base response to the remaining 12 mm requests.

`a29d664d` targets 5 mm for the simulated arm's base-joint component, leaving
ordinary exploration/precision navigation policies and the 12 mm 3D grasp gate
unchanged. A new no-progress guard stops after three completed corrections
without measured end-effector movement. Job `20260913_085624_0c7f62` reaches
about 5.7 mm error and sustained pickup, but loses the cylinder during later
transport. The empty-hand navigation posture lowers the lift and folds the
wrist despite a potentially held payload.

`9714349f` captures lift/wrist posture after pickup and preserves it through
transport and placement preparation, retracting the arm using existing joint
commands. Confirmed opening clears the constraint; failed carry transition
retains conservative held-object state. This constraint is not holding proof.
Matched stationary replays (`20260913_090821_557461`) retain contact in **both**
postures: relative translation drift is 9.2 mm folded versus 5.1 mm preserved,
rotation drift 0.140 versus 0.093 rad. They do not establish the cause of loss
under navigation. The earlier scratch replay's absolute-height predicate was
invalid for comparing intentionally different lift heights; use contact and
relative drift, not that predicate. Live job `20260913_091308_642f8d` still fails
placement: verified pickup at 53.33 s, contact loss at 70.05 s during base-turn
braking. The last wrist frame correctly centers the intended red object.
Private contact replay places its center at pad-local x=-16 mm after lifting,
drifting to -27 mm before loss (pad half-width 20 mm). This suggests insufficient
contact depth rather than missing visual identity.

Matched recorded-command replay `20260913_091820_d1cead` loses the payload with
both unchanged wheel references and references limited to 8 transmission-rad/s².
Smoothing delays but does not eliminate loss; no braking workaround was added.
Grasp-depth counterfactual `20260913_092110_b42b80` tests this contact-geometry
hypothesis from the saved preclosure state, without changing physics. With the
same close/lift/retract and 35 s stationary hold, 0 mm insertion loses the object,
while 15/25 mm insertion retains it within 14.7/13.5 mm of the nominal EE link.
This is a diagnostic grasp-depth intervention, not an agent success count.

`b91d761e` provides a bounded grasp-frame `contact_offset_m` calibration and a
separate `query_geometry_contact_pilot.yaml` preset (15 mm toward the palm from
the nominal grasp link). The desired robot contact point, transformed by the
measured EE orientation, is aligned with observed object geometry. There is no
object-label rule, physics/friction change, or relaxed closure tolerance. This
is not a hardware calibration or proof of cross-object reliability. Full-task
job `20260913_092427_1e2e65` still fails: its executed grasp is shifted sideways,
not deeper. The simulator's massless `link_grasp_center` marker used intrinsic
MuJoCo Euler angles copied from fixed-axis URDF RPY, rotating its axes about
120 degrees relative to the published URDF frame. The diagnostic calibration
mistakenly used marker +Y instead of published-frame -X (toward the palm).

`69d8e478` expresses the fixed marker using the URDF-equivalent quaternion,
checks orientation at three wrist pitches (residual below 3 mrad), and corrects
the experimental offset to `[-0.015, 0, 0]`. It changes no collision geometry,
mass or inertia, and does not alter the already-URDF-based published pose.
Job `20260913_093103_771bbf` still loses the payload at 73.11 s. The commanded
arm extension at closure is 0.173 m versus 0.167 m in the nominal run: the
executed depth improvement is smaller than the 15 mm static intervention.
Keep the 12 mm closure gate, visible-surface bias and loaded arm displacement
in view; corrected axes alone do not establish robust holding. Original
geometry-only centering remains an explicit control. `ea26ace8` saves the
private preclosure fixture and matched retention tests for reproducibility.

The subsequent coordinate audit finds grasp/place absolute geometry using
episode-relative `get_base_pose()` with world-frame point clouds. `292e84df`
uses `get_base_pose_world()` and explicitly world-frame alignment goals;
nonzero-origin grasp/placement tests pass (62 focused tests). This is a contract
fix, not an offset tuned for the tabletop origin. Physical regression job
`20260913_093628_3c845e` passes all 34 tests (114 s), including matched retention,
frame agreement, loaded lift, wrist profiling, command atomicity and wheel
transmission. Live retry `20260913_093656_0b9dbd` starts afterward under the
exclusive experiment lock. It reaches placement, but the payload has slipped.

`0a539b30` advances only the experimental contact preset to the replay-tested
25 mm offset. Job `20260913_094303_496e65` retains the object through transport
and receptacle reacquisition (contact through 94.23 s), then loses it during
the placement approach. A mixed-depth point at z=0.749 m raises the requested
EE height to 0.849 m although the cube top is 0.56 m. Its short 4.3 cm arm-base
command also starts a general SE(2) correction: the base turns while the arm is
extended, and the client reports joint-position completion before base yaw
settles. The saved final image shows the dropped red object on the table beside
the blue cube; the near-gripper check correctly prevents an empty release.

Three separate fixes follow: `f5ac318e` uses robust support height and avoids
mutating sampled cloud points; `c508574d` executes the arm base component as a
signed translation with heading hold and requires 100 ms of quiet measured
joint/base velocities before completion; `c1468a11` reuses the existing
physics-time limiter for arm (0.1 m/s) and lift (0.15 m/s) references. Default
navigation stays pose-based. No force limits, object geometry, friction or
physical scoring gates change. Ninety focused tests pass, including motion
contracts and wrist/translation reference tests. Full-task retry
`20260913_095547_9ec2d2` runs before neighboring precision navigation control
`20260913_095730_685498`; the shared evaluation checkout remains frozen until
both finish. The manipulation retry keeps the object held to the end, but its
four alignment frames contain 10–15% support/table depth inside the correctly
selected red mask. The resulting z correction stays +0.10 m despite repeated
upward moves. The final object is still held: this failure is not a slip or
Qwen identity mistake. Navigation completes all ten route moves with no
failures; overall health remains `incomplete_telemetry`, not full acceptance.

The depth contamination has a source bug: the MJCF head depth eye is 15 mm
from RGB, while the bridge publishes color intrinsics/pose and advertises
registered pixels. `076fa458` moves only the virtual aligned-depth viewpoint to
the color optical center in both Stretch assets. It does not change physical
objects. Render control `20260913_100547_d6990c` passes four tests, reproducing
incorrect object-pixel depth with the old offset and verifying agreement with
registered rendering. Full-task retry `20260913_100708_0f6983` still refuses
release: the selected mask includes minority support-depth pixels, and its
four z corrections are +9.9, -5.5, +2.1 and +10.0 cm. Correct registration alone
does not guarantee clean semantic-mask depth. The object remains held.

`ea16a60f` trims minority depth tails in the already verified placement mask
using a median/MAD band, retaining at least 80% of valid pixels or abstaining.
It preserves raw masks/depth and records filtering statistics. This assumes
dominant coherent object support, not arbitrary multimodal geometry; no motion
or physical success tolerance changes. Twenty-eight focused tests pass.
The matched learned retry `20260913_102057_336037` never reaches this check:
after pickup it loses the receptacle view and repeatedly executes the same
short navigation prefix. It is cancelled after about five minutes; preserved
trace scoring confirms pickup but not placement. Do not count it as a depth
filter outcome or omit it from complete-system failures.

The navigation audit exposes search/execution disagreement: eight-connected
search allows diagonal edges through blocked corners that the execution LOS
check rejects. The safety filter then discards the unsafe suffix but retains
the original arrival marker and hides the rejection. `89581c70` shares
no-corner-cutting neighbors across both searches and reachability, rejects
unsafe routes without a false arrival marker, and stops chunk continuation
without measured translation. Forty-five focused tests pass, including
single/multi-goal detours and disconnected diagonals. The fresh learned retry
`20260913_103007_684ea5` uses the reproducible driver with explicit contact and
separated-neighbor presets. It passes independent pickup and final placement:
pickup at 59.134 s, final supported state at 117.584 s, no gripper/other contact.
Final cylinder center is approximately (-0.2632, -0.5612, 0.5996) m. Placement
corrections have z components -3.39, +1.64 and +1.46 cm; raw captures and depth
support statistics are retained. Agent process exits zero after 165 s. The
combined focused suite passes 168 tests. This is the **first separated-neighbor
diagnostic pass**, not the original-clutter or cross-task gate. An unchanged
repeat, `20260913_103442_018239`, fails: pickup at 55.834 s, last finger contact
at 95.960 s during the later turn, then object falls to floor. Final grounding
correctly finds no held target. Current repaired configuration is **1/2**, not
reliable. Do not advance to original clutter yet. Private matched replay
`20260913_104239_40cb0e` compares the recorded commands, wheel-reference slew
limits and the simulator's full-close endpoint; these are diagnostics, not
extra agent successes or changes to production grip limits.
All four replays retain the object, **including the recorded-command control**;
the 10 Hz snapshots with reconstructed velocity do not reproduce the live loss.
Do not use them to claim a causal benefit for smoothing or stronger closure.
The replay figures confirm both live endpoints (cube support versus dropped
object); they do not replace the original traces.

The wheel contract audit separately finds independent actuator saturation:
requested geared wheel targets exceed their limits and clip to equal speeds,
driving straight despite a turn request. The next fix scales both targets by
one feasible fraction, preserving the requested curvature without raising any
speed/force limit. Twenty-nine wheel/translation tests pass. No new grip force
or acceleration tuning is promoted from the inconclusive replay.
`7ee502f3` implements this wheel-target scaling. Learned retry
`20260913_104630_1f3338` and subsequent ten-move navigation control
`20260913_104659_c274cb` share frozen `f7c024d3` (the extra commit only hides
rangefinder debug rays in offline figures). Both finish. The navigation control
completes ten moves with no failures, but health remains `incomplete_telemetry`.
The learned run retains the payload but fails placement. Initially the residual
vertical correction suggests loaded-joint tracking bias; inspecting the raw
receptacle mask reveals a more important error: Qwen selects a combined cube
and held-cylinder proposal, producing a false support top at about 0.82 m
(real cube top 0.56 m). Do not tune release tolerance against that geometry.

`cfb61c3d` applies the existing 8 cm depth-adjacency boundary to external masks
as well as raw RGB-D proposals. The saved mixed mask separates into cube support
at roughly 0.92–0.99 m camera depth and held-object support at 0.58–0.63 m.
No color/category rule is added; smooth sloped and textured objects remain
connected. Coplanar/touching mixed objects still need semantic/multiview checks.
Thirty-four focused tests pass. Learned retry `20260913_110132_edd836` passes
pickup (57.634 s) and final placement (130.088 s), with process exit zero after
187 s. It retains the payload on the longer route and grounds the real cube
support. One visual correction precedes release; final z error is 1.48 cm,
within the unchanged 1.5 cm gate. Unchanged repeat `20260913_110659_839e2c`
is pending. No accumulated-motion workaround or looser release gate was added.

The independent IK audit also fixes a joint-layout contract: full eleven-joint
seeds must convert to nine solver joints once and return a full configuration,
preserving passive joints. Both IK entry points pass an actual zero-error FK/IK
round trip, alongside 41 focused tests. This later fix is not in that running
retry. Future private traces also retain measured velocity, activation and
solver warm-start; sampled traces are still not exact command replays.

Rendering caveat: the existing simulator hides robot geometry from head RGB-D
to avoid mapping self-obstacles; wrist images retain it. This removes real
head-camera self-occlusion during placement. Do not infer real perception
robustness from these passes. Validate robot-visible observations with proper
mapping self-filtering before hardware claims. Offline replay figures explicitly
show robot visual geometry and are labelled reconstructed views, not agent
observations.

`query_geometry_aperture_pilot.yaml` is an additional, not-yet-live-tested
clearance ablation. It measures both identified finger markers in calibrated
RGB-D and narrows at standoff toward the observed target extent plus a 6 cm
marker/thickness/uncertainty margin. Missing markers/depth/standoff stop the
grasp. This is not an inner-jaw calibration or collision-free guarantee. Keep
geometry-only and aperture variants separate in reporting. Original-clutter
clearance, room OVMM and learned TAMP gates remain pending.

`query_geometry_contact_aperture_pilot.yaml` combines the same contact setting
with that aperture margin for the next original-clutter diagnostic. The two
component controls remain unchanged. This tests a combined system, not an
isolated contact-depth effect.

## September 13: manipulation command delivery audit

The next battery and stop gates are in [bounded acceptance](manipulation_acceptance.md).
These diagnostics do **not** establish a passing manipulation gate.

| Job | Frozen source | Task outcome | Evidence |
| --- | --- | --- | --- |
| `20260913_012127_e3d28d` | `afc32857` | Pickup failed; place skipped | Two accepted wrist captures; stopped on failed servo motion |
| `20260913_013027_fc29a4` | `aa2739d4` | Pickup failed before wrist approach; place skipped | Private physical score false/false; zero gripper-contact samples; tool failure correctly retained |
| `20260913_013657_dc2c7d` | `1a9959bb` | Pregrasp and nine wrist steps pass; approach stalls; place skipped | Private physical score false/false; replay identifies right fingertip contacting neighboring cube |

Artifacts are `~/runs/emet/grasp-command-identity-retry/evidence` and
`~/runs/emet/grasp-precision-physical-retry/hybrid_learned_pick_place`. The latter
contains `physical_trace.jsonl`, `physical_result.json`, `process.log`, and
`evidence/manipulation_outcomes.jsonl`. The agent process exits zero, but the
driver exits one on physical failure. This is intentional, not an infrastructure
exception. Broken-pipe errors during teardown remain a separate cleanup issue.

The first run's short base correction requested 5.2 cm, while the dynamic motor
tolerance stopped at roughly 2.6 cm error, outside the arm client's 2 cm gate.
`60fb8652` applies the existing precision policy to manipulation base goals.
It does not relax client, geometry or grasp acceptance. Retransmissions now
retain command identity, and the grasp loop honors failed motion/invalid depth.

The second run exposed a separate lost-command bug: the server logs a lift goal
of 0.643286 m, but the private actuator trace stays at 0.600 m and the measured
lift stays near 0.593 m. This is not a failure to physically follow the requested
actuator setpoint: **the setpoint itself never changed**.

![Requested lift versus actual actuator target and measured joint position](lost_lift_command.png)

Figure source: that run's private trace, wall-time interval
`1789277543.5–1789277548.0`; `ctrl[2]` and `qpos[9]`, checked against the frozen
default-scene model (48 qpos). Dashed line is the request in the server log.

The physics consumer reads a serialized command snapshot, clears triggers, then
writes the entire snapshot back. Writers previously held a different lock, so
this acknowledgement could erase a concurrently submitted joint command.
`1a9959bb` shares the interprocess lock across read/modify/write on both sides,
and releases it before waiting for stop consumption. A deterministic concurrent
writer/consumer test and stop-wait test pass (25 focused simulation tests).
Matched live retry `20260913_013657_dc2c7d` on `1a9959bb` passes pregrasp and
nine wrist association steps, approaching from 0.368 to 0.220 m. Unlike the
previous run, lift and arm requests appear in the actuator trace. It then stops
on a genuinely stalled approach, reports pickup failure and skips place. Private
physical scoring is false/false with zero red-cylinder gripper-contact samples.
Artifacts: `~/runs/emet/grasp-atomic-command-retry/hybrid_learned_pick_place`.

Offline `mj_forward` replay of the frozen model with recorded `qpos` and `ctrl`
isolates a collision with the *neighbor*: `rubber_tip_right` contacts blue cube
`object1`, penetrating about 1.65 mm at wall time `1789277958.553609` and 2.10 mm
at `1789277961.4929755`. The red cylinder remains supported on the table. These
are replayed contact geometries, not saved live contact-force measurements.
The red-only live contact trace cannot by itself establish robot-to-neighbor
collisions; the replay supplies that additional evidence.

![Last accepted wrist view: red target between fingers, neighboring blue cube at the right finger](grasp_neighbor_clearance.png)

Unedited final accepted wrist capture `grounding-7d6f1abc90184d659eecc60b8109b14c`,
recorded at wall time `1789277959.3625605`. Contact attribution above comes from
the state replay, not solely from this image.

Next diagnosis: a predeclared separated-neighbor fixture control, then
geometry-aware grasp aperture/approach clearance. Do not silently remove the
blue cube from the original task, widen association thresholds, or call a
smaller graph a manipulation improvement. The tabletop gate still fails, so
room OVMM and learned TAMP have not advanced. A neighboring precision navigation
control `20260913_014326_489ec4` on `1a9959bb` succeeds on six route moves,
then fails move seven: XY error 0.02120 m, yaw error 1.08045 rad, stop confirmed,
reason `navigation stalled`. The remaining three moves are not run. Frozen
pre-lock control `20260913_014724_bf28d2` on `aa2739d4` passes all ten moves
(max XY 0.01723 m; max yaw 0.02901 rad). Its overall health status is still
`incomplete_telemetry`, because posture/actuator telemetry is missing.

This pair blocks a no-regression claim. One possible contributor is the added
manager-lock RPC overhead in the physics loop; a separate controller/progress
issue near the 2 cm XY boundary also needs consideration. `a244ac25` retains
atomicity with a native spawn-context lock, avoiding per-tick manager lock RPCs.
An actual spawned-process exclusion test passes. The same-route native-lock
control `20260913_015157_35365a` passes all ten moves, zero corrections, max XY
0.01958 m and max yaw 0.02874 rad. As with the baseline, the full health probe
is `incomplete_telemetry`, not full robot acceptance. This is one matched control,
not evidence that all navigation regressions are ruled out or that IPC overhead
alone caused the preceding failure. Keep that failed run in the record.
The final combined focused suite passes 161 tests (command lifecycle, navigation,
grasp handoff, query memory, physical scoring and simulator command atomicity).
This includes the native-lock spawned-process test.

## Earlier frozen pilot

Interpretation: [environment acceptance progression](../environments/README.md),
[Habitat search scope](../environments/habitat.md),
[simple-sim controls](../environments/simple_sim.md).

This freezes the September 10 candidate, not a new default. The earlier verified
red/blue table finds are integration evidence, not OVMM or manipulation success.

Launch: managed job `20260910_213038_d86e1a`, frozen source `b194395a` at
`/tmp/emet-cross-task-20260910-frozen`. Artifacts:
`/home/cpaxton/runs/emet/shared-grounding-cross-task-20260910`.
This first launch is invalid for comparing strategies: all four OVMM cases and
six EQA cases encountered SAM2's missing `iopath` dependency during model
construction. OVMM serialized the exception despite exit zero. The absent-object
find then reached its 360-second cap while exploring; learned pick/place never
started. Retain these artifacts as infrastructure failures and a search timeout,
not a model-quality score.

September 11 retry: `20260911_185541_56a4c4`, frozen `eed1d868`, artifacts
`/home/cpaxton/runs/emet/shared-grounding-habitat-retry-20260911`. Installed
`iopath==0.1.10` and `portalocker==3.2.0` in Habitat, without changing Torch or
model weights. The runner now constructs SAM2 and executes synthetic box
inference before episodes, and supports `PHASE=habitat|sim|all` (default all).
This retry selects Habitat only, with the original cases, budgets and presets.
`process_status.tsv` records exits, not task acceptance. All ten retry processes
completed without infrastructure exceptions:

| Strategy | EQA q15 / q16 / q25 | OVMM object + receptacle, two scenes |
| --- | --- | --- |
| Hybrid | correct / correct / wrong (2/3) | 0/4 |
| Qwen-box | wrong / correct / correct (2/3) | 0/4 |

These three questions do not establish no regression against the earlier 3/3
smoke. The strategies disagree on two questions despite equal aggregate scores.
Hybrid scene 00025 returned an object localization 3.615 m from the target;
Qwen-box returned a receptacle localization 5.184 m from its evaluator target.
Neither is a task success. Other phases returned no localization.

Manual review of hybrid scene 00025's accepted support
`grounding-965b2a585dd64549851ab3ce1af623dd-surface-0.png` shows bedding/fabric,
not an identifiable lamp. The proposal query is `nearest bed`, while the support
selector receives the full question about the lamp nearest a bed. Qwen accepts
the bedding while claiming a small object on it resembles a lamp. This is a
concrete target/anchor confusion and false semantic acceptance; metric distance
alone would not reveal it. Preserve identity versus relationship as separate
verification concerns in the next fix, without handing context pixels back the
authority to validate an incorrect mask.

The pending learned Stretch pick/place diagnostic was launched separately as
`20260911_194952_1415c5` on the same `eed1d868`, with a 600-second cap, velocity
navigation, hybrid preset and no oracle scene plans. It does not rerun the known
absent-query timeout first. The tool failed after 195.9 seconds with
`Command 101 failed: terminal command outcome is immutable`; the agent reported
the failure rather than claiming success. No successful learned pick/place is
established. The command lifecycle exception needs diagnosis; process completion
is not physical task acceptance.

Lifecycle follow-up `efba8d7f`: reproduced the exception when a goal had already
failed with stop unconfirmed, then a later cancellation confirmed stopping and
attempted to rewrite the terminal status to cancelled. The fix retains the failed
outcome/reason and records later stop confirmation separately, releasing motion
ownership only on confirmed stop. Both core/deploy copies match; 37 focused
command, trajectory and adapter tests pass. Same-preset bounded sim retry:
`20260911_200332_5b6840`, artifacts `~/runs/emet/lifecycle-manip-retry/evidence`.
The retry reached manipulation without the lifecycle exception, but failed
pregrasp with arm extension -0.180 m and subsequently lost current-frame target
support. No pick/place success. This run did not necessarily exercise stop recovery.

Manipulation handoff follow-up `40288111`: the verified find pose faced the
object with the camera, while Stretch's side-grasp expects the target along -Y.
The grasp adapter now owns a measured target-facing side rotation and fresh
reacquisition; find remains unchanged. Failed/nonfinite pregrasp IK now returns
failure instead of falling through into visual servo. 23 focused tests pass.
Bounded same-preset sim check: `20260911_202359_853d5b`, artifacts
`~/runs/emet/grasp-handoff-retry/evidence`. The retry completed orientation and
reacquisition, then reached pregrasp (IK arm -0.039 m, clamped by the legacy 5 cm
allowance). Wrist tracking rejected absent/ambiguous current-frame support;
pickup failed and place was skipped. This is not manipulation acceptance.

`3b2ce750` additionally propagates arm-motion failure rather than treating valid
IK as execution success (25 focused tests pass). `a7c45096` retains wrist RGB,
depth, calibration and grounded world points on tracking rejection under the
existing episode evidence opt-in. Serial diagnostic `20260911_202818_77c40c`
uses that frozen source; artifacts `~/runs/emet/wrist-tracking-audit/evidence`.
No tracking-tolerance relaxation or oracle fallback was introduced.

The wrist audit completed with failed pickup, not a task success. Manual review
of `wrist_tracking/grounding-054937277d1849de892afab78332be86.png` shows the red
cylinder clearly. Its stored world median projects to wrist pixel (160.3, 216.1),
with expected camera depth 0.380 m versus observed 0.370 m. This argues against
a gross camera-frame error in this observation. Replaying the current expanded
world bounding box gives two components of 4,726 and 2,005 pixels, triggering the
intended ambiguity rejection. Bounding-box-only association is insufficient in
this close view; object-specific wrist support is needed. Neither largest-mask
fallback nor looser margins is justified by this evidence.

September 12 candidate fix shares `ground_vlm_frame` between head and wrist:
same configured provider and semantic selector, followed by the original
grounded-target spatial mask association. Wrist verification does not ingest a
new instance. Positive wrist depth is accepted below the navigation-map minimum
because grasp views can be closer than mapping views; zero/invalid depth remains
excluded. The former box-only connected-component gate is not a semantic mask.
63 focused tests pass, including semantic abstention and wrong-world-target
rejection. Added per-frame VLM work may increase servo latency and must be
measured; no real-robot or learned-manipulation acceptance is claimed.

Offline red-cylinder, blue-cube and absent-banana queries on the exact saved
wrist view were submitted as `20260912_091644_fa11a1`, then cancelled before
inference: NVML reports a driver/library mismatch (loaded kernel 595.84,
userspace 595.91). GPU repair is required before cached-model and live validation.
The three inputs all reference the previously listed wrist capture; they are
diagnostic queries on one view, not an expanded independent test set.

Post-reboot replay `20260912_153104_dafcde`, frozen `dd29d217`, completed after
kernel/userspace driver alignment to 595.91.07 and successful full SAM2 inference
preflight. Artifacts are `~/runs/emet/wrist-postboot-{proposals,baseline,support}`.
YOLOE→SAM2 produced one red-cylinder proposal, one blue-cube proposal, and zero
absent-banana proposals. Support-only Qwen accepted red and blue (selector times
1.23 s and 0.99 s respectively); banana abstained without a model call.
Manual inspection of `wrist-postboot-support/support_only/0-panel-0.png` confirms
the red support depicts the cylinder. Reprojecting the saved calibrated depth
and applying unchanged original-red-target association accepts 1,397 red pixels
and rejects the blue mask. This validates the intended distinction on this
saved frame, not robustness during motion or physical pickup. The focused
63-test suite also passes post-reboot. Live bounded pick/place retry
`20260912_153738_fb4a22` uses the same frozen source and preset; evidence goes to
`~/runs/emet/wrist-live-postboot/evidence`. The tool failed after 40.3 s:
wrist semantic/world association completed, then a legacy shape check accessed
`servo.semantic.shape`, despite head semantic labels being absent on this path.
No pickup or place succeeded. Process exit zero is not task success; teardown
also emitted multiprocessing broken-pipe/reset errors. Fix `2db1b757` validates
the selected wrist mask against wrist world geometry instead, with tests for
absent head labels and mismatched wrist resolution (11 handoff tests pass).
Same-preset bounded retry `20260912_201500_bb5284` retains all model and control
settings; artifacts `~/runs/emet/wrist-shape-retry/evidence`. The focused suite
passes 65 tests. The retry completes the wrist approach from 0.369 to 0.166 m,
closes the gripper, and advances to finding the blue cube. It then fails
placement navigation at the first waypoint: command 186, `motion deadline
exceeded`, `stop_confirmed: true`. Tool duration is 182.3 s; overall pick/place
fails. There is no independent confirmation of a held object. Post-grasp head
image `grounding/grounding-6b9317d388ee4aa09d068ca40238fbdf.png` shows the blue
cube and a small red region beyond the table, not a verified in-gripper cylinder.
Do not score gripper closure as physical pickup. Next isolate physical grasp
verification and post-grasp navigation; do not loosen tracking tolerances or
launch broader acceptance pilots yet. Multiprocessing teardown errors remain.

### Placement-navigation timeout: wheel transmission units

The failing placement waypoint is approximately an in-place 175-degree turn,
not a long drive. Isolated baseline `20260912_232620_453a49` (source `933fa725`)
reproduces it without perception models or manipulation. The nominal 10 s
command budget is stretched by the existing sim-time ratio: failure occurs at
12.27 wall seconds, measured yaw -2.369 versus goal -3.05 rad (0.681 rad short).
Pose samples show continued slow rotation, not a stationary obstacle stall.
The earlier diagnostic `20260912_232447_3fc9ac` never launched simulation because
its temporary route YAML was malformed; exclude it from navigation results.

Stretch's MJCF wheel velocity actuators use `gear=3`. The bridge previously
sent wheel-joint rad/s directly to actuator controls, and interpreted actuator
velocity as wheel-joint velocity in feedback. MuJoCo's [transmission semantics](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-general-gear)
scale actuator velocity by gear. Fix `76ebf9a1` multiplies commands by the model's
actual gear and reads joint `qvel` for wheel-speed telemetry. No MJCF friction,
limits, navigation deadline, arrival tolerance, or speed-boost changes.
15 tests cover command/feedback units with unit, positive non-unit and negative
gears; the broader adapter/command suite passes 44 tests.

Matched fixed diagnostic `20260912_232858_58901d` succeeds: first positioning
command 1.35 s, half-turn 7.01 s, return turn 6.46 s. Terminal half-turn error
is 0.055 m / 0.142 rad, within unchanged exploration tolerances 0.07 m / 0.15 rad.
These are exploration-contract passes, not precision-navigation acceptance.
Both runs use the existing `probe_rby1_camera.py` on seed-0 default-table Stretch,
with a temporary wrapper selecting the unnamed contract and nominal 10 s timeout,
and logging pose samples. Route: `(0.13, 0.0085, 0.006)`,
`(0.13, 0.0085, -3.05)`, `(0.13, 0.0085, 0.006)` in episode coordinates.
Artifacts: `~/runs/emet/nav-turn-{trace-v2,geared}`, including observations and
images; managed job logs retain timestamped `NAV_SAMPLE` and `NAV_RESULT` lines.
Integrated same-preset learned retry `20260912_233137_f012c2` uses frozen
`76ebf9a1`; artifacts `~/runs/emet/manipulation-wheel-units/evidence`. Initial
navigation completes without timeout, but fresh target grounding rejects
`target absent or ambiguous` before pickup (tool 17.7 s); place is skipped.
This run does not exercise post-grasp navigation, and does not establish
manipulation success. Keep that integration gate open despite the isolated
navigation fix passing. Existing multiprocessing/EGL teardown errors persist.

Follow-up `20260912_233819_e45145`, source `94149560`: one repeat of the existing
`configs/benchmarks/navigation_acceptance.yaml` using `probe_rby1_camera.py`
(`--route-repeats 1`), default-table Stretch, no perception models. All ten
commands succeed under the named precision contract (2 cm / 0.03 rad), including
±10° and ±30° turns and 20 cm forward/return translation. Maximum terminal
errors: 0.01939 m and 0.02953 rad; zero corrective redispatches. Artifacts:
`~/runs/emet/nav-geared-precision`. The probe summary is `incomplete_telemetry`,
not full acceptance: Stretch omits the requested base-up/actuator-target fields.
This is one route repeat on one robot/environment, not cross-robot robustness.

Manual comparison of the final two frames from `manipulation-wheel-units`
shows the red cylinder on the extreme lower-right edge of accepted frame
`grounding-4e5d1b8424ba4aeda2f56cb0e358c07f.png`, and no visible cylinder in
rejected `grounding-8a445c194f364a029016ffcaa02cb195.png`. The rejection has no
supported proposals; Qwen is not rejecting a clearly visible cylinder there.
The sweep permits an early return within 0.35 rad of commanded head pose,
even during motion. Residual camera motion is a hypothesis to measure before
changing verification. Camera-only trace `20260912_234053_eb1405` confirms the
mechanism: after the soft-wait handoff, saved camera rotation changes another
8.9145 degrees; base displacement is under a micrometre and yaw change under
0.003 degrees. `~/runs/emet/head-handoff-trace/handoff/head_0.png` shows the
cylinder and cube, while `head_3.png` shows neither. First sample is already
0.20 s after soft-wait return; this is a lower bound on total post-return drift.
The full observations have `image_timing: null`; do not infer capture latency
from joint-state sampling alone.

Fix `d9bb2ce3` uses the robot adapter's blocking head move and existing newer-frame
wait for verification callbacks only; ordinary mapping retains its soft sweep.
91 focused head-sweep/query/grounding/manipulation tests pass. This does not
upgrade the underlying head-motion API to a verified success contract; clients
that silently time out remain a follow-up. Serial job `20260912_234501_ffeb03`
checks the fixed camera handoff, then the same bounded learned command. Artifacts
`~/runs/emet/head-handoff-settled` and `~/runs/emet/settled-gaze-retry/evidence`.
Fixed camera handoff rotates just 0.0037 degrees across the corresponding saved
views after a 1.05 s head/frame wait (versus 8.9145 degrees before). Integrated
retry passes arrival verification and immediate fresh grounding, then rejects
after the subsequent side-grasp rotation/head movement: final
`grounding-930a1f12ee68467e86625b9a83089794.png` shows floor and table edge.
Tool duration 30.1 s; no pickup or place. The grasp adapter did not wait for the
RGB-D stream to advance after that second movement. Candidate `4f946d1d` adds
the existing post-motion frame wait there, without changing mask acceptance;
46 targeted tests pass, including head → frame wait → grounding order.
Bounded retry `20260912_234917_6b6fdf`, artifacts
`~/runs/emet/grasp-fresh-frame-retry/evidence`, completes with failed pickup
(tool 62.1 s). Both head-camera handoffs pass and wrist servo approaches from
0.368 to 0.218 m. Final calibrated wrist capture
`wrist_tracking/grounding-006bc3656f6440b19ab073405e0573ec` shows the cylinder
partly behind a gripper finger and clipped at the lower image boundary. Qwen
selects candidate 4, but its resulting instance-0 mask fails original-target
world association: only 1,339 / 3,385 valid pixels (39.6%) lie in the original
bounds plus the unchanged 5 cm margin, versus the required 80%. Selected support
median `(0.118, -0.465, 0.476)` differs from original target median
`(0.079, -0.539, 0.523)`. Do not relax the threshold on this example. Distinguish
object displacement/contact, mixed mask support, and camera/pose timing before
changing tracking. Several servo-center depths are zero and manipulation base
goals are repeatedly reissued; inspect these control paths next. No gripper
closure or place success in this retry. The full 91-test focused suite still
passes; these are correctness repairs, not long-horizon manipulation acceptance.

## Fixed comparison

- Hybrid: `query_detector_segmented_pilot.yaml`, YOLOE boxes → SAM2.
- Reference: `query_segmented_support_pilot.yaml`, Qwen boxes → SAM2.
- Both: support-only Qwen3-VL-8B-Instruct int4 verification, lazy query memory,
  unchanged fusion and task budgets. This tests proposal strategies, not SAM2
  versus YOLOE segmentation, and not lazy memory versus DynaMem.
- Habitat OVMM: nearest-v2 scenes 00006 and 00025, seed 0, 12 rounds/8 nav steps.
- EQA: questions 15, 16, 25, 20 planning/10 movement steps, no semantic sensor
  labels or enriched GT hints. The CLI does not pin EQA RNG seeds: these are
  matched questions/settings, not a deterministic causal no-regression test.
- Hybrid-only Stretch table: one absent-object find and one learned pick/place.
  Velocity navigation; no teleport or oracle manipulation. This is a multistep
  integration diagnostic, **not the PR #160 TAMP battery**. The oracle TAMP
  control previously passed; learned TAMP acceptance remains outstanding.

All twelve processes run serially. Stop after a timeout to inspect cleanup.
Each has a wall cap, not an extended task budget. The runner records process
completion separately from benchmark correctness. Report abstentions, grounding
errors, navigation failures and manipulation failures separately; do not turn
tool `ok` or exit zero into physical success.

## Reproduction

From a frozen checkout with the SAM2 small, YOLOE-L and MobileCLIP checkpoints
available at the usual loader paths:

```bash
OUT=/absolute/new-artifact-directory \
HABITAT_BIN=/absolute/habitat-env/bin/emet-habitat \
AGENT_PY=/absolute/agent-env/bin/python \
SAM2_SOURCE=/absolute/pinned/segment-anything-2 \
emet jobs run --name shared-grounding-pilot --need-mib 16000 \
  --cpu-safe --gpu-exclusive -- bash scripts/run_shared_grounding_pilot.sh
```

`EMET_CONFIG` already feeds the unified preset to both Habitat runners through
`get_parameters`; no new benchmark-only config API is needed. Preflight verified
that the EQA and OVMM profiles retain the selected query options and model.
SAM2 source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4` for this run.
Using it through `PYTHONPATH` avoids changing the shared Habitat installation.
Habitat's installed executable sets its required C++ library path.

Review the saved grounding RGB/masks/poses, agentic traces, navigation receipts,
and EQA map/video bundles. Inspect final accepted support manually, especially
any positive absent-query response. A new contamination or wrong-object
acceptance blocks promotion even if aggregate correctness improves. Retain
zero-result failures and infrastructure exclusions. No full sweep is authorized
by this pilot; use its failures to select the next shared-mechanism fix.
