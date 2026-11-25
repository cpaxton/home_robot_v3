# Herman sock pickup handoff

Last checked November 23, 2026 (New York). The sock remains on the box.
Herman made a measured reach step on hardware. The next test is a
feedback-driven pregrasp, followed by a small lift with visible retention.

## Last verified hardware state

- The last capture recorded arm angles of `[82.97, -61.96, 82.00, -27.86,
  24.52, 3.34]°` for joints 1–6. Base odometry reported zero twist. The
  elbow stayed at 82.00° during the reach. The captured hand origin in
  `base_link` was `[0.1040, 0.0928, 0.1386]` m.
- The stock Innate driver was running and automatic idle folding was
  disabled at the time of the test. A restart can re-enable folding.
  Recheck arm health, the supported pose, the base, and command authority
  before sending targets. Herman was on wired power. Keep the base
  stationary for the arm-only test. For a separate coupled test, keep total
  measured travel within 20 cm and rotation within 90°, including stopping.
- The wrist test achieved 10.4° pitch and 24.3° roll. Its roll missed the
  test's joint tolerance by less than a degree, and a measured hold
  confirmed. The later reach achieved about 6.1° of yaw and 7.3° at the
  shoulder. The hand origin moved 15 mm closer to the estimated sock fold.
  The remaining hand-origin distance was 281 mm.

## Target and unresolved blocker

The latest stereo estimate for the raised fold is
`[0.331, -0.064, 0.085]` m in `base_link`. Use the nominal approach
of 15° right and 12° down as a seed. Reacquire the fold after motion.
A two-pixel disparity change shifted the forward estimate by about 12 mm.
Use measured jaw-to-fold error and a grasp region for closure, then check
that the sock follows the jaws during a small lift.

The elbow stalled at 82° even during the small supported lift streamed
with the gains already active. This reproduced the problem without the
scheduled gain switch. The earlier stock scheduled test also sagged
during a gain change, and servo 3 later reported a latched overload.
The supported targeted reset restored a verified hold. The cause of the
stall under load remains unresolved. Do not use successful tracking at
the wrist as evidence of available elbow motion.

The closest fixed-base hand pose from the previous relaxed reach check
fell about 10 cm short of the previous pinch hypothesis. The newer stereo
estimate is farther right. Recompute reach for that estimate and the
latest measured pose with the elbow held before taking a full approach.
An earlier rehearsal reached a relocated fold near the edge of the box
and lifted a rigid sock proxy 34 mm. Treat that as an alternate setup,
with fresh stereo localization after moving the sock.

## Resume

1. Use the `feat/wbc-grasp-primitive` worktree at
   `.worktrees/wbc-grasp-primitive`. The session's code and documents are
   uncommitted. Run `git status --short` in that worktree before editing.
2. Take a fresh passive capture with the robot attended and the arm area
   clear. From the worktree, use a new output path.

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/capture_herman_rehearsal.py \
     --ssh jetson1@192.168.1.43 \
     --output ../../logs/herman-wave/sock-next-capture --seconds 4
   ```
3. Check servo 3 health. If overload has latched, the verified recovery was
   to support the forearm at about 82°, confirm idle fold is off, and use
   the targeted Innate reset. Recheck the powered hold before continuing.
   If the elbow stays healthy but cannot track, use its measured angle
   as a fixed constraint for the next approach.
4. Start from the measured hand and sock poses. Replan a reachable,
   fixed-base pregrasp and stream short corrections through a single
   robot-side owner. Use the hand-to-fold error to decide approach
   progress and verify grip and retention from feedback. Stop if the
   available joints cannot reduce the error or the arm loses its hold.

The live reach was a monitored Innate position stream. Closed-loop
stereo feedback through WBC and the live safe-start hold adapter remain
open. The [safe-start planner and sequence runner](../../src/emet/robots/innate_mars/startup_recovery.py)
are available for rehearsal and adapter integration. See the
[hardware readiness notes](MARS_HARDWARE_READINESS.md#innate-arm-activation-and-first-lift)
and the Mars entries in [TODO.md](../../TODO.md).

## Local evidence

Paths are relative to the main workspace and live under
`logs/herman-wave/`.

| Evidence | Path |
| --- | --- |
| Latest stereo fold estimate | `sock-closed-loop-before/sock-stereo-pose.json` |
| Last reach and measured settle | `sock-reach-stage-live/motion.log` |
| Post-reach stereo, hand transform, and encoder capture | `sock-reach-stage-after/` |
| Recorded reach | `sock-reach-stage-live/stereo.jsonl` |
| Last supported recovery health | `sock-recovery-go-ahead/post-health.log` |
| Supported lift using unchanged gains | `supported-stream-lift-retry/motion.log` |
| Simulated near-edge pickup | `sock-pick-available-joints/available-joints-staged-pick-replay-0.040.json` |

See [Herman RL lessons](HERMAN_RL_LESSONS.md) for transfer and policy qualification.

## Two-camera capture

Pass `--astra-device /dev/video0` to the passive capture command to record
local Astra RGB alongside the robot stereo, hand transform and telemetry.
Use Herman's calibrated stereo to estimate the fold in `base_link` and the
Astra side view to independently review whether the sock follows the jaws.
The capture records host start/end times for the Astra clip. Frame-level
clock alignment and an Astra-to-base transform still need calibration for
metric feedback from the external camera.

For a bounded test, run the trial recorder from the feature worktree.

```bash
PYTHONPATH=src .venv/bin/python scripts/record_herman_trial.py \
  --output ../../logs/herman-wave/new-trial --expected-seconds 20 \
  -- .venv/bin/python path/to/monitored_test.py
```

A trial includes the Astra clip, timestamped stereo and robot telemetry,
command output, and lifecycle events. The monitored test must report its
own motion outcome and stop receipt. A command exit code alone does not
establish successful motion or a confirmed hold. Camera timing is recorded
for review, with frame alignment still unverified.

### Shoulder limit during yaw changes

The stock Innate driver clamps the shoulder lower bound as a function of
arm yaw. At yaw 53°, the shoulder cannot be commanded below −0.5 rad
(−28.65°). Include this curve in planned paths and measured envelopes.
A requested yaw change with a fixed, more negative shoulder target had
produced apparent shoulder overshoot.

A driver-aware reach moved the hand from 237 mm to 221 mm from the sock.
The shoulder and yaw followed the new path. The elbow remained near
78.4° on the way to a 77° target, so the progress guard stopped the move
and confirmed a hold. The Astra clip, command trace and robot feedback
are under `logs/herman-wave/sock-driver-limit-trial-v2/`.

### Mars WBC action interface

`MarsWBCInterface` runs one policy update on the supplied observation and
sends its action through `GuardedControlSession`. The first two action
channels become forward velocity and yaw rate. The next five become
positions relative to the latest measured arm pose for one control
period. The adapter defers yaw when it would force an unrequested
shoulder motion under Innate's coupling. The active gripper target stays
with the task executor.

The bridge has an opt-in `create_wbc_transport()` binding. It publishes
arm positions and base twist on the Innate topics after checking sole
publisher ownership, compatible subscribers and verified feedback. The
robot-local ROS timer expires commands at 200 ms or less, sends a zero
base twist and requests a measured arm hold when fresh feedback and
ownership are present. Stop confirmation requires newer quiet motor and
odometry samples.

The transport and action interface pass loopback tests. Herman's native
source revision `4be109a501aa8fd5fb69d11394b878592e53789e` uses the
`mars_bot/mars_arm` layout. The
[Mars feedback patch](../patches/innate_os_mars_feedback_4be109a5.patch)
applies to that revision. It converts the signed Present Velocity register
from 0.229 rpm per unit to rad/s, aligns head velocity with its position
sign, and timestamps both joint messages at acquisition. A failed or
incomplete motor read exhausts the existing retries and skips publishing
that cycle, allowing the WBC watchdog to expire the last valid sample.
The XL430 elbow's effort channel remains a signed load estimate in percent.
[ROBOTIS XL430 register definitions](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#present-load126).

Before enabling the live executor, build and install the patch, compare
reported joint speed against position changes during a supervised motion,
and verify that dropping motor feedback causes the watchdog to send zero
base twist, reject the expired arm sample and report the actual stop
outcome. Keep the WBC capability disabled until those checks pass on Herman.

The patch compiled on Herman's Jetson in an isolated colcon overlay using
the installed ROS Humble workspace as an underlay. The stock driver stayed
running. A read-only ROS probe using `rmw_zenoh_cpp` received joint states
and reported healthy, torque-enabled servos. At the time of the probe,
`manipulation_server`, `arm_sdk_server`, `skills_action_server`, `mars_app`
and `udp_leader_receiver` advertised the arm command topic, and
`mars_app` advertised the base teleop topic. Commissioning requires a
controlled handoff from those publishers to WBC before streaming motion. The native
`cmd_vel_mux` subscribes to `/cmd_vel_teleop` with reliable QoS, which the WBC
binding now offers. The stock driver had `auto_rest=false` at the time of
the probe. Set that parameter explicitly at launch when commissioning the
patched driver because a restart otherwise restores the stock setting.



The robot-local deterministic executor uses the same waypoint sampler as
the simulator. Start the bridge with
`--local --wbc-config-factory emet.robots.innate_mars.wbc_commissioning:create_executor`
and set `EMET_MARS_WBC_CONFIG` to a local JSON record. The record requires
`robot_description` pointing to the captured URDF, relative to the record.
Add `feedback_si_units_verified`, `motor_read_validity_verified` and
`elbow_load_percent_verified` only after the corresponding hardware checks
succeed. Omitted fields keep motion unavailable. Use a fixed-base waypoint
for the first motion test. The default bridge does not start an executor.
The executor supports consecutive streams after a fresh, confirmed stop.
Within a stream, each waypoint updates the same command identity until
finish, cancellation, or lease expiration. Head integration and physical
commissioning remain pending.

## Next supervised session

Use the commissioning archive in `logs/herman-wave/wbc-commissioning-20261124/`
at the workspace root. It contains the Python overlay, the reviewed C++
patch, the last captured robot description, and the exact actor tested in
simulation. The extracted configuration selects the deterministic
controller with motion disabled pending feedback verification.

1. Copy the archive to the Jetson, extract it to a new directory, and run
   the commands below from that directory. The isolated driver build prints
   its install prefix. The probe starts the executor for six seconds and
   reports feedback and command publishers without sending targets.

   ```bash
   bash run_herman_wbc.sh install-runtime
   bash run_herman_wbc.sh build-driver
   bash run_herman_wbc.sh probe
   ```

2. With Herman attended, check the arm area, current joint bounds and wired
   base position. In the `ros_nodes` tmux session, identify current panes
   from their running commands. The last graph mapped the competing arm
   publishers to `mars_control app.launch.py`, `manipulation behavior.launch.py`,
   `brain_client brain_client.launch.py` and
   `mars_control udp_leader_receiver.launch.py`. Stop those specific launches
   cleanly and record the commands for restoration. Leave the base driver,
   odometry, cameras, and `cmd_vel_mux` running. Make sure there are no active
   native arm actions or other mux inputs commanding the base. Recheck the graph
   after discovery converges.
3. Replace only the native arm launch with the isolated patched binary
   using the installed arm YAML and an explicit `auto_rest:=false` override.
   With the stock ROS environment sourced by the launcher, the direct command
   is below. Set `arm_overlay` to the exact prefix printed by the build.
   Keep the stock launch command available for rollback.

   ```bash
   arm_overlay=/absolute/path/to/native-build/install/mars_arm
   "$arm_overlay/lib/mars_arm/arm" --ros-args -r __node:=mars_arm \
     --params-file "$HOME/innate-os/ros2_ws/install/mars_arm/share/mars_arm/config/arm_config.yaml" \
     -p auto_rest:=false
   ```

4. Verify arm health, measured position, command echo and odometry after
   restart. Compare corrected signed velocity with encoder position changes
   and qualify the motor read failure and elbow load reporting before
   recording the three verification fields. Run the probe again. Expect a
   sole bridge publisher for each WBC input, the native arm and mux subscribers,
   and `cmd_vel_mux` as the sole final base publisher.
5. Run `bash run_herman_wbc.sh start` in the bundle. Send the measured
   end-effector pose first through `open_pose_stream(frame="odom", free_base=False)`
   with the base omitted. Confirm a measured stop before opening the next
   stream for one small hand waypoint. Exercise stream loss and verify the
   reported stop outcome before chaining more waypoints. Preserve the wired
   envelope of 20 cm and 90° for a later coupled test.
6. For the learned runtime, the factory accepts an `actor` directory and
   a matching `verified_actor_weights_sha256` in the commissioning record.
   The loader checks the bundle, action contract and source-model fingerprint.
   The archive includes a candidate for simulation and shadow use. Install
   a Jetson-compatible CPU PyTorch if needed and qualify the candidate in the
   selected live envelope before recording its hash as verified.

To roll back after a confirmed stop, exit the WBC bridge and patched arm
process, restore the stock arm launch, then restore the recorded publisher
launches. Verify arm health and publisher ownership after restoration.

### Reproduce the simulation

From this worktree, point `ACTOR` at the extracted actor directory or the
training checkpoint used to build the archive.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/smoke_mars_pose_stream.py --actor "$ACTOR"
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/smoke_mars_pose_stream.py --actor "$ACTOR" --coupled
```

The smoke test advances the simulator between waypoint submissions and
checks the terminal stop receipt. Its JSON includes separate teacher and
learned results. The fixed-base run passed for seeds 1900000 and 1900001.
The learned actor's final hand error was about 0.22 mm and final base drift
about 0.72 mm in both runs. Neither had an unsafe contact, wheel lift, or
unconfirmed stop. These small free-space trials do not qualify sock pickup,
arbitrary starting postures, or the physical watchdog.

The separate coupled smoke requested a 40 mm base translation plus an arm
change, with the simulator's fixed-base episode contract disabled. The
teacher completed the stream and confirmed its stop. The learned actor
timed out and confirmed its stop. Its final hand and base position errors
were 2.1 mm and 2.6 mm. Keep the actor out of the first coupled hardware
trial. The full receipts are in the same evidence directory.

### Innate OS simulator

Run the headless Innate simulation smoke to exercise the ROS handoff
with the real WBC binding and upstream MuJoCo plant. Use a local Innate
OS checkout containing `ros2_ws/src/maurice_bot/maurice_sim` and an extracted
commissioning archive. The launcher builds the simulator-only status message
in a scratch directory. It requires ROS Humble and uses localhost-only
Fast DDS discovery on domain 87 by default.

```bash
bash scripts/run_innate_ros_wbc_sim.sh ~/src/innate-os /path/to/robot_description.urdf
```

The test uses the upstream physics and control callbacks, adds arm status,
command echo and child-frame odometry, and routes velocity through a simple
ROS relay registered as `cmd_vel_mux`. It verifies that a competing arm
publisher blocks readiness, then runs a hold and a wrist waypoint and
checks the measured stop. The relay exercises topic ownership and routing.
Innate's full mux arbitration and Jetson Zenoh behavior remain separate
integration checks. Use the [Innate OS setup](../robots/innate_mars.md)
for its full GUI stack and the MuJoCo smoke above for policy regressions.
