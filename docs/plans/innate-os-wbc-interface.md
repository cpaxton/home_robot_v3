# Innate OS changes for torque/contact whole-body control

Status: **interface review; no innate-os changes merged.** Inspected
`innate-inc/innate-os` at `4953aaa3` (the Mars arm code is unchanged from the
pinned 0.8.0 commit `4be109a5`). This document covers what Mars must expose so
Emet's multi-arm torque contract can drive it, and what should stay outside
innate-os.

See [`whole-body-rl-control.md`](whole-body-rl-control.md) for the Emet-side
schema and adapter.

## Bottom line

- Mars is a **position-controlled arm**. The current `mars_arm` driver writes
  six joint positions and reads position, velocity, and a load/current proxy.
  It has no per-joint torque command, no contact-force state, and no
  machine-readable capability manifest.
- Only the **wrist and gripper servos** have current-control hardware. The
  shoulder yaw, shoulder pitch, and elbow are XL430/XC430-class servos without
  current control. Full-arm joint torque is **not possible** on the current
  Mars without a hardware change.
- The fastest software path is a **robot-side position-servo torque adapter**
  on the existing 200 Hz control loop: Emet sends torques, the driver converts
  them to bounded position offsets and current/PWM limits, and publishes the
  applied torque and saturation state. This is the same `PositionServoAdapter`
  design as Emet, moved to where the feedback is fresh.
- Do **not** turn `mars_arm` into the multi-robot abstraction. Keep it
  Mars-specific and expose a standard command/state/capability interface. The
  multi-arm/multi-sim abstraction belongs in Emet; other robots provide their
  own backend.

## What the current stack does

### Arm driver

`ros2_ws/src/mars_bot/mars_arm` owns seven Dynamixel servos:

| Servo | Motor | Current control | Role |
|---|---|---|---|
| joint1 | XL430-W250 | no | shoulder yaw |
| joint2 | XC430-W150 | no | shoulder pitch |
| joint3 | XL430-W250 | no | elbow |
| joint4 | XC330-M288 | yes (modes 0/5) | wrist |
| joint5 | XC330-M288 | yes | wrist roll |
| joint6 | XC330-M288 | yes (mode 5) | gripper |
| joint7 | XC330-M288 | yes | head |

The node runs a 200 Hz control loop (config), a 100 Hz trajectory writer, and
publishes `/mars/arm/state` at 50 Hz, `/joint_states` at 25 Hz, and health at
0.2 Hz. Commands are a six-element `std_msgs/Float64MultiArray` on
`/mars/arm/commands`; every value is a joint position. There is no command TTL
on the arm, no per-joint mode, and no contact/force state.

`/mars/arm/state.effort` is **not a consistent unit**: it is mA for joints with
`current_limit > 0` (the x330 servos) and percent load for the x430 servos.
That is a blocker for any multi-robot or contact-aware consumer.

`arm_config.yaml` carries motor types, position limits, PWM limits, control
modes, gain schedules, and `goal_current` for the gripper. It does **not**
carry torque constants, gear ratios, identified `kp_eff`, or per-joint torque
limits. The driver has `setGoalCurrent` but no per-tick torque/PWM command
path; `Robot::setGoalPos` is the only command writer in the control loop.

### Simulator

`ros2_ws/src/mars_bot/mars_sim_driver` is a MuJoCo Mars twin. It publishes the
same position-only arm interface and simulates the arm with a PD servo
(`KP_JOINT`, `KD_JOINT`, `EFFORT_LIMIT=50 N*m`) applied through
`qfrc_applied`. The URDF declares `effort="2.0"` for each joint, so the sim's
torque ceiling and the model's declared limit do not agree. The sim has no
torque/current command mode and no contact-force API.

### LeRobot bridge

The bridge is hard-coded to six joint positions plus base velocity. Its state
header has `jointN.pos`, `cmd.jointN.pos`, base, head, and `busy`; it has no
effort, current, temperature, gripper force, or contact. The bridge clamps the
gripper position and relies on the gripper's current-based position mode for
force. A torque/contact dataset cannot be recorded through this schema.

### Robot info / capabilities

`/robot/info` carries identity (name, color, hardware revision), not
capabilities. There is no way for Emet to discover arm count, joint names,
DOF, control modes, torque limits, or whether a joint has current feedback.

## Required interface

### 1. Typed command and state

Replace the position-only `Float64MultiArray` with a typed message pair. Keep
the old topic as a compatibility adapter while consumers migrate.

```text
ArmCommand
  header
  lease_id: string
  ttl_s: float
  joints: JointCommand[]
    name
    mode: POSITION | VELOCITY | EFFORT | CURRENT | PWM
    position, velocity, effort, current, pwm
    max_effort, max_current, max_pwm
    kp, kd                      # optional impedance terms
    valid
```

```text
ArmState
  header
  joints: JointStateExt[]
    name
    position, velocity
    effort_nm                   # estimated joint torque
    current_a                   # measured motor current, when available
    load_percent                # servo load proxy, when current is unavailable
    pwm
    temperature_c
    mode
    valid
  contact[]                    # optional force/wrench or contact flags
  gripper: position, current, force_proxy, contact, slip
```

Rules:

- `effort` is always **N*m**; current is always **A**; load is always `%`.
  Never mix units by joint type in one field.
- Timestamps and source ages are explicit. A consumer must be able to reject
  stale state without guessing.
- The fast state topic (or a `state_fast` sibling) runs at 100–200 Hz for WBC.
  Keep a downsampled topic for UI/RViz.
- The command carries a lease/TTL. On expiry, stale feedback, overcurrent, or
  thermal fault, the robot performs a measured hold/stop. A raw publisher must
  not be able to fight the active controller indefinitely.

### 2. Robot-side torque adapter

The arm node already runs at 200 Hz with fresh encoder state. Put the torque
mapping there:

```text
tau_total = clip(tau_cmd + gravity(q) + friction(qd), +/-tau_limit)
q_cmd     = clip(q_meas + clip(tau_total / kp_eff, +/-max_lead), position_limit)
current   = |tau_total| / (kt * gear_ratio)      # when current control exists
pwm_limit = clip(current * kt * gear / stall_torque, 0, 1)  # x430 approximation
```

- x330 joints (4, 5): use `CURRENT_CONTROLLED_POSITION` and per-tick
  `Goal Current` with `GroupSyncWrite`; this gives a real torque/current cap.
- x430 joints (1, 2, 3): use position offset plus a bounded `Goal PWM`/PWM
  limit. This is **impedance-like**, not true torque control. The current
  feedback is a load proxy, not calibrated torque.
- Gripper (6): keep mode 5, expose `goal_current` as a force command and
  measured current/contact as state.
- `kp_eff`, `max_lead`, `torque_limit`, `kt`, `gear_ratio`, and gravity/friction
  terms become `arm_config.yaml` fields. They are identified on a bench, not
  guessed from a datasheet.
- Publish the applied torque/current, saturation flags, and whether the command
  was clipped. A policy must see when the actuator could not deliver the
  requested torque.

### 3. Capability manifest

Publish a machine-readable manifest, e.g. `/robot/capabilities` as JSON or a
typed message, at startup and on change:

```json
{
  "schema": "wbc-capabilities-v1",
  "robot": "innate_mars",
  "backend": "innate_os",
  "base": {"kind": "planar", "dof": 2},
  "arms": [
    {
      "name": "primary",
      "joint_group": "primary_arm",
      "dof": 5,
      "ee_link": "ee_link",
      "gripper": {"name": "primary_gripper", "dof": 1, "mode": "position"}
    }
  ],
  "joints": {
    "joint1": {"motor": "XL430-W250", "modes": ["position", "pwm"], "has_current": false},
    "joint2": {"motor": "XC430-W150", "modes": ["position", "pwm"], "has_current": false},
    "joint3": {"motor": "XL430-W250", "modes": ["position", "pwm"], "has_current": false},
    "joint4": {"motor": "XC330-M288", "modes": ["position", "current", "pwm"], "has_current": true},
    "joint5": {"motor": "XC330-M288", "modes": ["position", "current", "pwm"], "has_current": true}
  },
  "sensors": {"ft": false, "contact_proxy": "current_or_load"}
}
```

Emet can build its `EmbodimentProfile` from this instead of hard-coding Mars.
Other robots publish the same manifest with their own arms, torso, and
grippers; the manifest is the multi-robot boundary, not the Mars topic names.

### 4. Multi-arm and multi-sim

Keep `mars_arm` Mars-specific. For a dual-arm or torso robot, use a backend
that owns its own joints and publishes the same manifest/command/state
contract. Do not stretch `mars_arm` into a generic N-arm controller; that
duplicates Emet and the robot-specific kinematics.

For the long term, the cleanest common interface is **ROS 2 Control**:

- `SystemInterface` for the real Dynamixel bus, with `position`, `velocity`,
  `effort`, and custom `current`/`temperature` state interfaces.
- `JointTrajectoryController` for position, `EffortController`/`ForwardCommandController`
  for torque/current, `JointStateBroadcaster`, and a force/torque sensor
  broadcaster.
- MuJoCo, Gazebo, and Isaac implement the same interfaces. Emet then talks to
  one contract across sims and robots.
- `mars_arm` becomes either a `SystemInterface` plugin or a bridge to one.

This is a larger migration than the typed-message path. Do the typed
command/state first so Emet can integrate; move to `ros2_control` when a second
robot or second sim needs the same interface.

### 5. Simulator parity

`mars_sim_driver`/`VirtualMars` must implement the same command and state
semantics:

- Add `set_joint_command(name, mode, value)` with position, velocity, effort,
  current, and PWM modes.
- In effort/current mode, apply `qfrc_applied` directly with per-joint effort
  limits; do not route through the position PD.
- Expose actuator torque, `qfrc_bias`/constraint load, and contact forces.
  `data.contact` + `mj_contactForce` is enough for a contact API; group by
  body/geom for training and evaluation.
- Publish position, velocity, effort, current, temperature, and mode in the
  same `ArmState`.
- Align the URDF effort limit (`2.0 N*m`) with the sim and hardware limits.
  The current `EFFORT_LIMIT=50` is not a Mars torque limit.
- Support namespaced robots/arms if the sim hosts more than one; otherwise
  keep the sim Mars-specific and let Emet launch separate instances.

### 6. Data and LeRobot schema

The LeRobot bridge is a separate consumer, but it should not fossilize the
position-only schema:

- Version the protocol. Keep `jointN.pos` for v1 datasets.
- Add `jointN.vel`, `jointN.effort`, `jointN.current`, `jointN.temperature`,
  `gripper.force`, `gripper.contact`, and a contact flag/list.
- Add action modes: `jointN.pos`, `jointN.torque`, `jointN.current`, with a
  per-joint mode or a typed action block.
- For multiple arms, use namespaced keys (`left.joint1.pos`) or a versioned
  vector schema; do not assume six joints.
- Record commanded and measured torque/current separately. A contact-rich
  dataset needs both.

## Validation against the external MuJoCo sim

The external innate-os `VirtualMars` plant has been exercised with the Emet
`PositionServoAdapter`; the harness lives in this repository but imports the
external checkout and never vendors it. See
[`innate-sim-validation.md`](innate-sim-validation.md).

Current smoke result:

- adapter-driven motion is within `0.0024 rad` and `0.0062 rad/s` of direct
  torque across all five arm joints for a 0.5 N*m step;
- a hardware-like 2 N*m torque limit and 0.05 rad position lead clip correctly;
- gravity-feedforward home drift is `0.0015 rad`;
- a cube–gripper pad contact remains measurable (`3.65 N` after) while a wrist
  torque is applied;
- the Mars action vector is `1 arm × 5 joints + 1 gripper = 8`.

This validates the mapping and the contact telemetry, not full-arm torque or
hardware identification. The x430 joints still need the position-offset
adapter; the wrist/gripper can use current control.

## Prototype branch

A fork exists at `cpaxton/innate-os`, branch `feat/torque-wbc-interface`
(commit `a6b77861`), with a draft PR into the fork's `main`:

- **Fork PR:** https://github.com/cpaxton/innate-os/pull/1
- **Branch:** https://github.com/cpaxton/innate-os/tree/feat/torque-wbc-interface

This is an external prototype, not vendored into Emet, and not for upstream
merge yet. The fork PR exists to keep the interface visible and reviewable.

It includes:

- `ArmCommand`, `JointCommand`, `ArmState`, and `JointStateExt` messages;
- a per-joint `position`/`velocity`/`effort`/`current`/`pwm` command callback
  with `kp_eff`/`max_lead` torque-to-position mapping;
- `/mars/arm/state_ext` at 100 Hz, `/robot/capabilities`, and unit-consistent
  effort/current/load fields;
- `VirtualMars` effort mode, `joint_state_ext`, and `contact_forces`;
- `mars_sim_driver` `ArmCommand`/`state_ext`/capabilities topics;
- a `test_driver_core.py` torque/contact check.

The C++ `mars_msgs` and `mars_arm` packages build with `colcon` against the host
ROS 2 Humble + a local Dynamixel SDK; the Python sim test passes. It has **not**
been run on hardware, and the actuator constants in `arm_config.yaml` are
placeholders. Do not treat the branch as a hardware qualification.

## Hardware limits and options

| Joint | Servo | Position | Current/torque | Practical contact control |
|---|---|---|---|---|
| 1 | XL430-W250 | yes | no | position offset + PWM limit (impedance-like) |
| 2 | XC430-W150 | yes | no | position offset + PWM limit; gravity-dominated |
| 3 | XL430-W250 | yes | no | position offset + PWM limit; current elbow stall |
| 4 | XC330-M288 | yes | yes | current-controlled position / current |
| 5 | XC330-M288 | yes | yes | current-controlled position / current |
| 6 | XC330-M288 | yes | yes | gripper force (mode 5) |
| 7 | XC330-M288 | yes | yes | head; not part of the arm action |

Consequences:

- **Full-arm torque is not available.** If shoulder/elbow torque is required
  for contact-rich work, replace joints 1–3 with current-capable servos or use
  a different arm. A six-axis F/T sensor at the wrist would also help, but it
  does not give joint torque on the x430 joints.
- The elbow stall at ~82–84° is a mechanical/thermal/current-limit problem, not
  something a torque command can push through. Fix the folded-pose clearance
  and characterize the servo under supported and unsupported load before
  training a contact policy from that posture.
- A wrist F/T sensor is the strongest hardware addition for contact-rich
  manipulation. Current/load proxies are gravity-dependent and coarse.

## Priority

**P0 — unblock Emet**

1. Typed `ArmCommand`/`ArmState` with explicit units, modes, validity, and
   timestamps.
2. High-rate state (100–200 Hz) and a command lease/TTL/measured hold.
3. Capability manifest with joint names, DOF, modes, current availability, and
   torque limits.
4. Robot-side position-servo torque adapter with per-joint `kp_eff`,
   `max_lead`, and current/PWM limits.
5. Gripper force/contact command and state.
6. Sim parity: same command/state, torque mode, contact forces, effort limits.

**P1 — data and multi-robot**

7. LeRobot protocol v2 with effort/current/contact and variable DOF.
8. Mode-switch service with a safe hold transition.
9. Namespaced multi-arm support or a second backend.
10. Bench identification: torque constants, gravity, friction, thermal limits.

**P2 — hardware / long-term**

11. ROS 2 Control `SystemInterface` and sim plugins.
12. Current-capable shoulder/elbow servos or a different arm.
13. Wrist F/T sensor and compliant gripper.

## Files to change

| Area | Path |
|---|---|
| Messages | `ros2_ws/src/mars_bot/mars_msgs/msg/ArmCommand.msg`, `ArmState.msg`, `ArmStatus.msg` |
| Driver | `ros2_ws/src/mars_bot/mars_arm/mars_arm/{arm_control,arm_services,arm_config,robot,dynamixel}.cpp` |
| Driver headers | `ros2_ws/src/mars_bot/mars_arm/include/mars_arm/{arm_node,robot,dynamixel,arm_types}.hpp` |
| Config | `ros2_ws/src/mars_bot/mars_arm/config/arm_config.yaml` |
| Description | `ros2_ws/src/mars_bot/mars_description/urdf/mars.urdf` (effort limits, transmissions, ros2_control tags) |
| Sim core | `ros2_ws/src/mars_bot/mars_sim_driver/mars_sim_driver/{core,node,world}.py` |
| LeRobot bridge | `ros2_ws/src/brain/manipulation/manipulation/lerobot_bridge.py` |
| LeRobot client | `lerobot/lerobot_robot_mars/{schema,wire,mars}.py` |
| Robot info | `ros2_ws/src/mars_bot/mars_control/mars_control/app.cpp` (`/robot/info` -> capabilities) |

## Non-goals / anti-patterns

- Do not send raw torques/currents to `/mars/arm/commands`; it is interpreted
  as six positions.
- Do not switch servo operating modes while powered without a verified hold.
- Do not put mA, percent load, and N*m into the same `JointState.effort` field.
- Do not treat `Present Load` as calibrated torque; identify it per joint.
- Do not make `mars_arm` the multi-robot abstraction. Publish a capability
  contract and let Emet own the schema/adapters.
