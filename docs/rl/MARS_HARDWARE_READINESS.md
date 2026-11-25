# Mars learned-controller hardware readiness

## Direct arm-and-base WBC trial

The direct path gives WBC ownership of wheel forward/yaw velocity and all five
arm joint velocities. The base driver accepts a velocity command. The audited
Innate 0.8 arm callback accepts six position targets, and writes position goals
in the motor loop. Its motor SDK exposes a goal-velocity register. For this
trial, keep the existing position servo and convert the WBC arm rates to
short, lead-limited position updates from measured joint state. Hold the
gripper and use one command owner for both actuator paths.

The guarded session now converts the normalized WBC action to SI and supports an
explicit translation envelope around the measured start. Its default holds XY.
A MuJoCo loopback test executes one timed 8 cm forward-plus-elbow trajectory
through that interface. In a matched comparison at seed 1900000, feedback
traveled 7.7 cm and moved the elbow 3.3°. The shared actor traveled 7.9 cm and
moved the elbow 4.1°. Both issued simultaneous base and arm commands, completed
and confirmed a simulator stop. The test uses MuJoCo actuators. The hardware
adapter for moving waypoints remains open.
See the [direct command regression](../../src/test/robots/test_mars_control_session.py).

An attended hardware hold used a temporary robot-side endpoint with exclusive
arm-command and wheel-teleop input ownership. The analytic WBC streamed 32
position updates across 2.05 s. The wheel mux echoed zero, and odometry showed
zero travel and rotation. After the workstation stopped sending updates, the
endpoint expired the stream in 202 ms and confirmed a measured hold. Joint
error relative to the hold anchor peaked at 0.18°. The elbow read one encoder
count above its published limit at the end, with an in-range applied-target
echo. This establishes a stationary hardware hold and stream-expiry test.
Stopping from nonzero wheel commands and tracking moving waypoints remain
open. The local trace is `logs/herman-wave/live-wbc-motor-hold-v2.jsonl`.

For the physical trial, give the direct adapter exclusive access to the two
command paths. Verify arm and odometry units, live publisher ownership,
deadline-to-stop behavior and measured wheel/arm settling. From a validated
clear posture, run a retimed forward-plus-elbow waypoint schedule and record
actual travel, hand tracking and stop. Use the same schedule for feedback and
learned WBC.

## Folded pose simulation

On 2026-11-22, the captured-scene model's forearm box overlapped the chassis by
17.5 mm at the measured, in-range sock-trial fold. The convex hull of the same
forearm CAD mesh installed on Herman gives 1.43 mm of modeled clearance. Herman
replays now use that hull and the captured head angle with Herman's published
head limits. A deliberately intersecting joint configuration still produces
forearm–chassis contact. Archived training models retain their original geometry
and fingerprints.

The old forearm box aborted the captured-start pregrasp at the first tick. With
the CAD hull, the same WBC waypoint schedule completed in 500 ticks with 1.52 mm
of final hand error, 0.53 mm of base drift, and a confirmed simulator hold. The
trace recorded zero contacts beyond wheel support. The local paired traces are
`logs/herman-wave/sock-folded-legacy-matched-v1.json` and
`logs/herman-wave/sock-folded-cad-matched-v1.json`. Gripper closure, the lift,
and physical execution were outside that pregrasp run.

## Sock pickup rehearsal

A fresh stereo and joint capture again showed the sock on the box and the same
in-range folded arm. The simulator now runs the complete fixed-base task through
the WBC waypoint runtime. It uses a rigid 20 g proxy for the accessible fold of
the sock and the measured gripper range from the captured URDF. The executor
checks the pinch pose before closing, requires 250 ms of bilateral jaw contact
before lifting, and cancels on loss of that contact during lift.

The wider opening let both jaws reach the proxy. During the best lift, the proxy
rose about 3.5 cm and slipped toward the fingertips. The runtime confirmed a
hold on cancellation. Some narrower or deeper approaches instead provoked a
wheel-lift stop during descent. Faster lift and reduced squeeze variants did
not retain the proxy. The traces record simulated grip position and contact
forces. Herman's real sock is deformable, and the installed arm configuration
describes a current-capped position servo for the jaw. The proxy experiment
does not measure the real sock's deformation, jaw force, or retention.

Replay the recorded loss with `scripts/rehearse_mars_pick_waypoints.py` from the
fresh capture using `--pickup --target .2745 -.0107 .0757 --box .3045 -.0107 .032
.084 .090 .032 --swing-radius .26 --duration 20 --pitch 15 --stand-off -.04
--clearance .09 --grip-open .67 --pinch-offset .019 .018 --lift-duration 4`.
Local evidence is under `logs/herman-wave/` in `sock-pickup-current-capture` and
`sock-pickup-fresh-retention.json`. The gripper and lift were exercised only in
simulation. The hardware WBC hold trial above remains the verified direct path.

The next live pickup check found a latched overload on elbow servo 3. The
encoder read 91.93° against a 90.00° upper limit, about 22 encoder counts beyond
the limit. The corrected CAD hull shows 1.08 mm of modeled overlap at this new
pose. A static inward sweep over those 22 counts gives 1.22 mm of modeled
clearance at the limit. The sweep carries no load or recovery qualification.
Herman had `auto_rest=true`, and the installed targeted error reset enables
torque before it reads the hold position. A reviewed recovery should suppress
the idle fold and establish a verified holding target before enabling torque.
This check sent zero motion, torque, or restart commands. Evidence is in
`logs/herman-wave/sock-live-wbc-before/` and
`logs/herman-wave/sock-live-folded-overrun-audit/report.json`.

The attended recovery then cleared the overload by rebooting only servo 3.
Firmware 50 left the goal at the measured encoder count while torque was off.
With motor PWM verified at zero, the second attempt was able to enable the
servo register and verify an in-range goal of 1024 before allowing limited motor
output. The elbow remained at count 1002 for all 14 samples. Its load proxy
rose from 0 to 92 raw units (`9.2%` on the XL430 register) over 0.42 s. The
watchdog stopped the attempt and confirmed servo 3 torque was off. This
establishes a stall under the tested output cap. The load proxy cannot identify
the mechanical cause. The other arm joint positions matched the start in the
before and after captures, and no base or gripper commands were sent.

Read-only joint, head and camera telemetry was restored with the stock motion
driver and automatic fold inactive. The elbow remains at 91.93°. Evidence and
stereo recordings are under `logs/herman-wave/elbow-guarded-recovery-live/`,
`logs/herman-wave/elbow-guarded-recovery-resume/`, and
`logs/herman-wave/elbow-guarded-recovery-after/`.
[ROBOTIS XL430 output limiter](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#goal-pwm100).

After an operator adjustment, the passive capture read elbow encoder 1000
(92.11°), two counts farther out than the earlier 1002 reading. Innate's
`jointRad` reverses the elbow sign, so counts rise as the elbow unfolds.
The motor's native minimum **goal** count is 1023. The published 90° upper
joint limit maps to count 1024, which Emet uses for ordinary motion admission.
Present position can remain outside the command range while elbow torque is
off. The raw reading and the published coordinate conversion agree.

A static base-fixed replay from the new capture found about 0.24 mm of modeled
forearm clearance at the best small shoulder-yaw angle if the elbow held its
angle. With one degree of additional passive elbow rotation into the fold,
none of the sampled yaw offsets cleared the chassis. This does not establish a
usable powered unloading move with the elbow unpowered. The diagnostic is
`logs/herman-wave/elbow-after-manual-adjustment/joint-limit-diagnostic.json`.
[ROBOTIS position goal limits](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#minmax-position-limit52-48).

The arm recovered once the operator supported the forearm at 82.1°. The driver
read back count 1114 as both the measured position and the target with output
at zero. It enabled limited output and confirmed the same encoder count
throughout the hold. The replacement driver adopted the existing holds without
initializing the servos or issuing the startup fold. A fresh four-second
capture confirmed healthy torque, the elbow at 82.1°, and all published joints
in range. The prop still carried the forearm during that check. Automatic fold
is disabled and the motion publishers remain paused. Evidence is under
`logs/herman-wave/elbow-supported-hold-v2/`.

Innate's normal trajectory planner admits a parked start by using
`insideLimitsKeepingPitch()`. It projects the measured state into the command
range and passes each pitch joint's excess rotation to the next joint. This
preserves wrist pitch where the following joints have room. Its first command
can still move the arm from the measured parked posture to that projected
start. In the recorded 92.11° example, the projection lowered the elbow by
2.11° and raised wrist pitch by 2.11°. The kinematic sweep moved the hand
5.06 mm from a pose with 1.29 mm of modeled forearm overlap. It is recorded
in `logs/herman-wave/elbow-after-manual-adjustment/innate-start-projection.json`.
Emet's bridge already starts for observation with a parked arm. Add
the pitch-preserving projection as a candidate for a separately requested
rest-to-hold transition, with checks on the physical sweep, servo health,
measured progress, and the stop. Use this transition to reach a legal hold
before accepting WBC waypoints.

Emet now computes that candidate in the captured-scene rehearsal, retaining
the measured reset, compensating pitch, and leaving two encoder counts of
margin when it has to correct a joint. It samples the entire transition for
contact. The 92.11° capture still starts in modeled contact, so its proposed
transition needs contact review. The projection and sweep are available in
`report.json` from `scripts/rehearse_herman.py`. Hardware execution of a
parked transition remains separate from the planner.

After the prop was removed, the arm held at 83.4° and reported load on the
elbow. In the recorded pickup trial, the analytic WBC sent 17 arm setpoints
for a straight-lift pregrasp before a velocity or load-proxy stop, and the
driver confirmed that stop. A new opening waypoint based on an unfolded
joint posture passed in simulation at 1.85 mm final hand error with a
confirmed hold. On the robot it moved the shoulder by 1.76° before the
fixed-base watchdog stopped it. The stop receipt failed because it repeated
the base envelope check while attempting to latch the arm hold. A subsequent
measured-hold check still saw elbow drift; passive follow-up showed stationary
in-range feedback at 85.9°, healthy torque, and no arm command publisher.
The pickup did not reach the sock. Logs and stereo recordings are under
`logs/herman-wave/sock-live-unfold-entry/`.

## Stereo-guided sock grasp

A fresh stereo pair located the raised fold near `[0.320, 0.023, 0.082]` m in
`base_link`. The left image is rectified. The right image was rectified
using its captured calibration before matching the fold in both images.
Local patch matches agreed with the selected landmarks. Built-in depth had
no valid samples at the sock. Perturbing the landmark pixels by two pixels
in each image changes the forward estimate by about 23 mm and the height
estimate by about 14 mm, so the point remains a grasp hypothesis.

The planned fixed-base motion opens the arm out of the fold, arcs above
the box, approaches the raised fold at a 12° downward hand pitch, closes
after reaching the pinch pose, and lifts 50 mm while leveling the wrist.
Two higher pinch poses held both sides of the 20 g rigid sock proxy through
lifts of 49.4 and 49.8 mm. Both completed with a confirmed simulator hold.
The contact trace recorded only the two jaw–proxy pairs for these runs.
The lower nominal pinch reached the object but contacted the modeled box,
so it was superseded by the higher poses. The best run finished with 2.37 mm
of hand position error and 1.95 mm of base drift.

Local evidence is in
`logs/herman-wave/sock-grasp-estimate-live/sock-landmarks.json` and
`logs/herman-wave/sock-estimated-crest-raised-020.json`. The replay includes
the exact task-space waypoints and command parameters. A live attempt needs
an independent measured hold after a base-motion stop. This rehearsal sent
zero hardware motion commands.

The subsequent live readiness check tried Innate's scheduled-gain hold.
The driver acknowledged the measured target, while the elbow settled ten
encoder counts (0.88°) lower under gravity. Restoring the earlier inward
hold target did not recover the expected measured position. The check
stopped on the hold envelope. The follow-up showed healthy torque,
automatic rest disabled, no arm command publisher, and the elbow stationary
near 87.9° with a reported load proxy of 18.0%. No pickup command was sent.
The scheduled request and follow-up telemetry are under
`logs/herman-wave/sock-pick-scheduled/`.

The next retry found the elbow back at 84.6°. The fresh capture's simulated
pickup lifted the proxy 49.4 mm and confirmed a hold. In the live opening,
Innate's scheduled controller moved the shoulder 4.1° and the elbow about 1°
inward. The elbow settled near 84.1°, short of the 80.5° goal. Raw servo
readback showed goal count 1131, position count 1091, and output 160 against
the recovery cap of 160. In a monitored retry at the same goal, raising the
cap to 210 led to measured output 195 and a load proxy of 22.0%, with no change
in elbow count. The monitor stopped there, restored cap 160, and restored
healthy arm feedback with automatic fold disabled. This isolates a live
tracking stall even below the raised cap. It does not establish the source
of the resistance. No gripper or base motion command was sent.

The verified physical recovery from an out-of-range, torque-off elbow is
still to support the forearm, bring the elbow inside range, verify the
stationary goal at zero output, and adopt the resulting powered hold with
automatic fold disabled. The parked-pose planner now accepts a measured
settling reserve with `--parked-settling-reserve-deg`. A 2° example places the
projected elbow goal at 87.82° and preserves hand pitch. The old overrun still
needs contact review. Evidence is under
`logs/herman-wave/elbow-bounded-output-live/` and
`logs/herman-wave/emet-parked-start-settling-reserve/`.

The current-pose reach check fixes the elbow at its measured 84.1°, inside
the native limits. At the original sock position, the closest hand position
is 104 mm short with the base fixed, even with hand orientation relaxed. In a
straight base approach, shifting 95 mm forward still leaves an estimated
17 mm position error and 11° orientation error at the pinch. The next sampled
candidate at 105 mm introduces modeled box contact. These results use the
approximate box and stereo estimate from the capture.

Placing the fold at the near edge of the box around `[0.225, -0.035, 0.076]`
m in `base_link` gives a staged approach using shoulder and wrist motion
with the base and elbow held. With the pinch about 7 mm higher than the
initial pose, the rigid sock proxy lifted 34 mm and settled with both jaws
in contact. The simulation sent zero hardware commands. See
`logs/herman-wave/sock-pick-available-joints/available-joints-staged-pick-replay-0.040.json`.
The measured elbow stall is indexed in the
[single-arm incident replay manifest](../../configs/benchmarks/mars_arm_sock_replays_v1.json).

The installed Innate configuration sets servo 3's PWM ceiling to 885. The
recovery procedure set its active Goal PWM to 160. `adoptAppHolds()` reads
the existing goals and starts the driver without normal servo initialization,
so that temporary cap carried into the scheduled opening trial. The on-robot
retry saturated at 160, then settled at 195 after the cap was raised to 210.
The elbow stayed at encoder count 1091 throughout that retry, so increasing
the cap alone did not demonstrate adequate tracking. PWM is motor duty, and
the XL430 load register is an estimate derived from internal output. Neither
identifies why the elbow failed to track. A later passive read showed a
stationary elbow at 82.18°, healthy torque, and no arm command publisher.
This read cannot establish what moved it between samples.

Keep Innate's motor transport and position loop. Give the recovery handoff
an explicit transition to the normal operating profile while holding the
measured pose. Read back the active goal, PWM cap, profile and gains. For a
bounded repeat at the same pose, record the servo's internal position
trajectory, input voltage, temperature, hardware status and encoder, first
with the forearm supported and then with it carrying its own weight. This
lets us separate an incomplete handoff or weak position loop from supply
sag and resistance at the elbow. Preserve the no-fold handoff and measured
stop on both paths. [ROBOTIS XL430 control table](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#control-table).

After a reboot, the stock driver came up with normal configuration and held
the elbow at encoder count 1023 (90.09°). With automatic folding disabled,
an unsupported 1.67° upward Innate scheduled request produced a 3-count
drop into the fold within 0.21 s. The driver changed from teleop to scheduled
gains. The elbow's inferred load reading decreased from 27.2% to 12.9% as it
dropped. The motion guard preempted the trajectory. It did not confirm its
hold within the test window. A separate read confirmed healthy torque and a
stationary elbow with its requested and measured angles both at 90.00°.
This run used the stock driver and did not read live PWM or supply voltage.
Preserve hold through gain-mode changes and run the supported comparison
before attributing the sag to a particular gain or mechanical cause. The
recorded run is under `logs/herman-wave/elbow-stock-stroke-live/`.

## Innate arm activation and first lift

On revision `4be109a5`, Innate configures the servos, enables torque and
then synchronizes the goal to the measured position. Its targeted repair
reboots only servos reporting an error, reapplies their configuration and
enables torque before writing their measured hold. A scheduled move starts
from `insideLimitsKeepingPitch()`, which transfers pitch error from a
clamped joint to the next pitch joint.

Innate's idle fold first moves shoulder and elbow together to −0.9 and
+0.9 radians over 2.5 seconds. It holds yaw, wrist and grip for this stage.
The wrist levels in the second stage before the arm folds in the third.
Keep idle folding disabled on Herman and evaluate the first lift separately.
A captured-scene static sweep from the in-range boundary to the first
waypoint showed no modeled contact. The first tenth raised the hand 11.8 mm
with a 1.8° shoulder change and a 3.8° elbow change. It is a candidate for
a bounded test after preserving the powered hold through the gain handoff.
Evidence is in `logs/herman-wave/elbow-after-restart-recheck/stock-lift-stage-audit.json`.

`run_safe_start()` stages the first tenth of that lift, confirms the
measured hold, then advances to the first fifth and reserves at least 5°
at the elbow. The rehearsal report exposes these waypoints under
`audit.safe_start`. The live adapter still needs to preserve motor output
when changing gains and confirm stationary feedback for each hold.

## Earlier hardware audit and simulator evidence

Updated 2026-11-18 from source and simulation; the passive robot audit below is
from Nov 17 and has not been repeated in this update.
Herman is reachable after power-on; source revision is `0d4c350d`, ROS Humble with
Zenoh/domain 0. Existing arm/odom publications arrive at about 200/30 Hz. No
motion commands, driver/service starts or active deployments were performed.
A patched driver package was built only in a temporary copy; it was not executed.
[Current audit and offline patch validation](../experiments/herman_feedback_audit_20261117.md).

## Current position

Connectivity and passive telemetry checks now pass. Shadow-policy evaluation
still requires the unit/validity and joint-mapping audit. Learned
whole-body motion and hand-contact tests are not ready to enable yet. There
are two independent tracks: verified robot-side execution/feedback, and acceptable
policy behavior in the specific initial test envelope. Perfect full TAMP success
is not required before a constrained free-space test, but correct stop behavior
and bounded motion in that selected envelope are required.

| Capability | What is implemented/observed | What remains before physical use |
| --- | --- | --- |
| Simulator policy | Fresh cases improve 12/24 → 16/24, safety flags 4 → 1, no longer-trained wheel lift in this set. | Coupled/goal-transition misses, a contact stop and large jerk peaks remain; checkpoint is not promoted. |
| Experimental corrective-data fit | Original spline pool 23/24; fresh splines 19/32 vs teacher 31/32. All 110 spline/general/TAMP stops confirm. | One fresh spline contact stop, speed/path misses and broader task failures remain. Experimental weights are outside deployment bundles; this does not replace the retained controller or authorize hardware. |
| Mixed fitting follow-up | General hand/goal-change pools improve to 14/16 each; unloaded grasp pose 6/6. | Base 2/8, splines 13/32, three wheel-lift episodes and two unconfirmed actor stops. This candidate is not qualified. Source base-feedback restoration is under simulator evaluation. |
| Robot-side execution | `GuardedControlSession` defines SI base/arm velocity callbacks and command slew. | No attached verified hardware driver. Existing bridge is not this whole-body executor. Validate actual servo modes and supported command semantics. |
| Measured feedback | Local bridge changes retain source/arrival ages, complete arm state and six-component odom twist, with child-to-world planar velocity mapping. | Not deployed or ROS-build-validated. Structural validity is separate from verified units and successful motor acquisition. Current/load remains a proxy. |
| Units / read validity | Current robot source hashes retain the known velocity conversion and zero-on-failure defects. Updated patch passes offline C++ checks and a full native ROS build in an isolated temporary workspace. | Review/install the fix and verify live units, read validity, signs and capture ages. Source inspection is not proof of compiled-binary behavior. |
| Fault/stop boundary | Loopback-tested latched origin, stale-state/overspeed/stream-gap checks and stop-confirmation interface. | Demonstrate real arm+base stop and independent robot-side watchdog under dropped commands, stale telemetry and client loss. |
| Wheel support / tilt | Simulator checks wheel gaps/loads at physics cadence. | Identify and verify real tilt/support sensing and its stop response; simulator support measurements are not hardware sensors. |
| Contact compliance | Simulator force/wheel monitors and bounded disturbances exist elsewhere. | Calibrate current/load signals and force response. Tracking error alone is not a force limit; current/load proxies are not automatically Nm or external N. |

The old bridge dropped odom twist and filled base velocity slots with zeros.
Local changes now publish `mars-source-feedback-v1` validity/ages and measured
world planar velocity without claiming verified units or bus acquisition. Cached
state cannot acquire a new source capture just because the bridge republishes.
The guarded session now requires independent arm/odom capture ages, verified
SI/read-validity capabilities, and finite six-component base twist; it checks
all base velocity axes at startup and before dispatch. These are locally tested
software changes, not hardware verification or deployment.

The **passive operator console is live on localhost:8770**, with head stereo,
arm/odom and hand TF. Wrist supplied no frames. Its pose edits are local previews,
not executed motor commands; the stop control is visibly unavailable. See the
[responsibility division, SSH forwarding and deployment sequence](CONTROLLER_DEPLOYMENT.md).

The published current arm pose is outside the latest narrow training reset
neighborhood. Validate joint/model mapping and select a tested initial envelope;
do not stream the current policy from an arbitrary parked posture.

The [captured-scene rehearsal](../experiments/herman_rehearsal_20261117.md) now
makes that gap concrete. Reported joint1/joint2 exceed both published URDF and
model bounds by 1.318°/10.019°; the conservative link3 proxy intersects the chassis.
The as-recorded state is static-FK-only. From a separate admitted synthetic start,
teacher succeeds 6/6 while both learned actors succeed 0/6; all 18 simulator stops
confirm. Small hand errors coexist with continued motion. Reconcile the stow
envelope/geometry and validate posture-specific stable tracking before deployment.
This is also distinct from the outstanding driver/stop verification.

## First physical sequence

1. **Passive audit and shadow inference:** verify installed driver/profile,
   q/dq/base twist units and timestamps; record actual feedback cadence and
   policy runtime while sending no policy commands.
2. **Driver and stop verification:** connect the real executor to the guarded
   interface; test robot-side timeout, command loss, invalid/stale feedback,
   low limits and measured whole-body stop in a supervised bring-up procedure.
3. **Small free-space test:** selected nearby reachable arm targets, held base,
   conservative speed/acceleration, clear workspace, operator stop available.
   Log measured versus commanded state and settle/drift/jerk. This can precede
   full manipulation qualification once the preceding checks pass.
4. **Bounded whole-body hand tracking:** exercise coordinated motion within
   the agreed fixed-origin envelope. The default target holds XY and allows
   yaw ±45°. The measured boundary adds 5 cm of translation tolerance and 2°
   of yaw tolerance by default, with odometry age ≤200 ms. A trial can
   explicitly choose a nonzero XY radius and its separate measured tolerance.
   The origin stays fixed throughout the session. A violation stops, reports
   confirmation and exits.
5. **Gentle contact and objects:** only after calibrated load response and
   stopping are demonstrated. Hand pushing, grasp/lift/carry and full TAMP are
   separate later tests, not consequences of a free-space success.

The guarded callback contract requires an independent robot watchdog ≤200 ms and
bounded stop confirmation (currently requested with a 0.75 s timeout). Those are
software interface requirements, not measured stopping performance on Herman.
No reliable motion-test ETA is justified until the deployed driver audit and
stop/feedback work are complete. Continuing PPO alone does not finish that work.

References: [guarded session](../../src/emet/robots/innate_mars/control_session.py),
[boundary](../../src/emet/robots/innate_mars/motion_boundary.py),
[passive feedback audit](../experiments/herman_feedback_audit_20261023.md),
[bring-up runbook](../robots/innate_mars_hardware.md),
[current training mixture](TRAINING_MIXTURE.md).
