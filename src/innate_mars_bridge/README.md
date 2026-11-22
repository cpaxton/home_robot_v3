# innate_mars_bridge

ROS2 bridge for the **Innate Mars** robot ([innate-os](https://github.com/innate-inc/innate-os)). Exposes observations and **Nav2 navigation** over ZMQ (same contract as `stretch_ros2_bridge`).

**Upstream pin:** `main` @ [innate-inc/innate-os](https://github.com/innate-inc/innate-os) — verify topics with `uv run python scripts/audit_innate_os_topics.py`.

## Capabilities

- **Pose:** base `(x, y, theta)` from `/odom`, TF for cameras and `ee_link`
- **Proprioception:** `/mars/arm/state` → 10-DoF Emet joint vector (base + arm + gripper mimic)
- **Cameras:** head stereo (`/mars/main_camera/left|right/...`), wrist `/mars/arm/image_raw`
- **Navigation:** ZMQ `xyt` → Nav2 `navigate_to_pose` (innate-os `maurice_nav`)
- **Depth:** not on ZMQ; clients use DA3 (`dynav_innate_mars.yaml`)

## Prerequisites

- ROS2 Humble
- innate-os Mars stack (`maurice_bringup`, `maurice_cam`, `maurice_arm`, `maurice_nav`)
- `emet-core` (from this repo)

## Build

```bash
# Symlink or copy src/innate_mars_bridge into a ROS workspace src/
colcon build --packages-select innate_mars_bridge
source install/setup.bash
```

## Run

```bash
ros2 launch innate_mars_bridge server.launch.py
# or
ros2 run innate_mars_bridge server --local
```

Default ZMQ ports: send **4401**, recv **4402**, state **4403**, servo **4404**.

## Emet client

```bash
uv run emet run dynagraph --robot innate_mars --robot-ip <MARS_IP> \
  --dynav-config dynav_innate_mars.yaml --export runs/mars_hw
```

See [docs/robots/innate_mars.md](../docs/robots/innate_mars.md) and [docs/robots/innate_mars_hardware.md](../docs/robots/innate_mars_hardware.md).

## Named arm and gripper targets

Use an existing connected Emet client and the protocol v2 motion helpers.
Targets are radians in the vendor joint convention. The gripper is `joint6`;
omitted arm joints hold their measured positions from goal acceptance.

```python
from emet.core.command_client import send_reactive_goal, wait_reactive_goal

command = send_reactive_goal(client, arm={"joint6": desired_gripper_rad}, timeout=10.0)
reached = wait_reactive_goal(client, command, timeout=11.0)
```

The bridge sends bounded position increments directly to the Mars driver and
uses applied-setpoint and joint-state feedback to confirm completion. Base
goals and named arm goals share the Emet motion slot. Use
`cancel_reactive_goal(client, command)` for a measured-position hold.

Before sending an arm goal, assign `/mars/arm/commands` exclusively to the
bridge and leave vendor arm services idle. An additional ROS publisher, stale
joint feedback, or unhealthy servo status blocks the motion. Deploy the
bridge and core together using `emet deploy --robot innate_mars`.

### Single-joint limit recovery

Use `send_arm_joint_recovery(client, joint=..., target_rad=...,
start_arm_rad=...)` to request an isolated return of one out-of-range joint.
Pass all six measured positions from the previously checked starting pose.
The bridge compares that pose with fresh stationary feedback, monitors the
other five joints, and uses the normal Emet motion slot and cancellation
receipt. The destination must be inside the selected joint's limits. A reviewed
clearance stage can pass `stage_only=True` to move an in-range joint while
another joint is outside its Emet limits.

The driver must expose `/mars/arm/recover_joint` as a dedicated
`control_msgs/action/FollowJointTrajectory` action. The bridge sends a
trajectory containing exactly one joint with the measured start and the
in-range destination using the live URDF limits. The driver owns path
admission, applied setpoint limits, preservation of the non-selected holds, and a confirmed halt
before reporting any terminal status. A normal full-arm trajectory action
cannot serve as this endpoint. The Emet session advertises
`isolated_joint_recovery` when the dedicated action is available.


The [Innate 0.8.0 driver patch](../../docs/patches/innate_os_0.8.0_isolated_joint_recovery.patch)
targets release commit `4be109a5` and adds the dedicated action. The driver
enables recovery-only mode at startup when given
`isolated_recovery_joint`, `isolated_recovery_start_rad`, and
`isolated_recovery_target_rad` for one reviewed segment. The start is the
complete six-joint pose in vendor order and the target is in radians.
The target must be at least two encoder counts inside the selected servo's
native limits. The driver rejects starts more than one count outside those
limits. For a start one count outside, the first setpoint and any immediate
cancellation hold use the nearest legal count.

Recovery-only mode opens the bus and verifies that the servos are already
holding. It publishes feedback, omits ordinary arm and head command handlers,
and uses individual writes to the selected servo for the recovery operation.
Use it as the sole arm driver on the serial bus. The bridge requires its
recovery action and arm feedback to come from the same node and checks the
ordinary arm command subscribers. The Innate manipulation server's passive
LeRobot readback subscription is allowed. Innate nodes that publish
ordinary commands can stay running during recovery. The action also rejects
unexpected start poses, checks other joint positions during the move, and
reports a distinct error if a stop cannot be confirmed. Choose the configured
segment using independently checked clearance for the exact start pose.

For offline verification, build the patched `mars_arm` on ROS 2 Humble with
`control_msgs`, then run the serial-bus test against the built executable.

```sh
python3 scripts/test_innate_arm_recovery.py \
  --arm-binary "$INNATE_BUILD/mars_arm/arm" \
  --config "$INNATE_OS/ros2_ws/src/mars_bot/mars_arm/config/arm_config.yaml" \
  --driver-log /tmp/mars-recovery-driver.log
```

The test runs ROS on loopback with an emulated serial device. It covers
successful single-servo writes, cancellation, a one-count recovery, rejection
of larger native-limit overruns, and a motor write error that latches recovery off.

### Native Innate recovery session

The same patch adds `adopt_existing_holds` for Innate 0.8.0. Start this mode
with `auto_rest=false`, the serial device in `isolated_recovery_device`,
and a reviewed `adopt_recovery_joint`, six-joint `adopt_recovery_start_rad`,
and `adopt_recovery_target_rad`. Startup verifies existing torque, servo goals,
and the stationary pose before exposing any motion service. The process must
be the only owner of the serial device.

Emet uses Innate's scheduled trajectory through `/emet/mars/arm/goto_js_v2`
for that reviewed stroke. The lease keeps ordinary arm receivers offline.
The driver writes position and gain settings only to the selected servo.
An explicit Emet joint recovery goal is required. Check
`emet mars status --connection herman` for session readiness. Repeat the
serial-bus test with `--native` to exercise adoption and cancellation.
