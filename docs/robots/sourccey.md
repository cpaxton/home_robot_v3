# Sourccey Mk.V (Vulcan Robotics)

Emet uses the official full-body Mk.V assembly from
[vulcan-forge/sourccey-simulation-mujoco](https://github.com/vulcan-forge/sourccey-simulation-mujoco).
Its source URDF matches `URDF/FullBody/SourcceyMkV.urdf` in the
[hardware repository](https://github.com/vulcan-forge/sourccey-hardware).
The previous simplified chassis and mirrored standalone arms have been replaced.

## Pinned sources

Verified against upstream HEAD on 2026-10-09:

| Repository | Commit | Purpose |
|---|---|---|
| `sourccey-hardware` | `dbb4bfd72108d7a79b1b27a87619aa260e71a953` | Full-body URDF cross-check |
| `sourccey-simulation-mujoco` | `c1059b0d15200fb91bd38ac276a7ac47758e4b26` | Native MJCF, full URDF, 171 meshes, camera/lidar metadata |
| `lerobot-robot-sourccey` | `d4f10db4fdf2786f8c87ec5b3562a812f93b1a9f` | Hardware SDK/protocol reference; installed separately |
| `lerobot-vulcan` | `330152dc9df45930f9d10ec991f8e6fe9741e5c6` | Vendor diagnostics and setup reference |

The asset `upstream/manifest.json` records source revisions and SHA-256 hashes.
`upstream/models/sourccey.xml` is the untouched native model; `sourccey.xml` is
Emet's generated planar adaptation. Regeneration needs no network or vendor SDK.

## Emet simulation

| Interface | Convention |
|---|---|
| Base | World-aligned `base_x`, `base_y`, `base_yaw`, velocity actuators; robot +X forward, +Y left |
| Elevator | Emet `lift` aliases upstream `linear_actuator`; meters, −0.315 to −0.0142 (upstream operating upper stop); only shoulder mounts/arms move |
| Arms | Actual independent left/right chains; five rotational joints plus gripper each; radians |
| Limits | Arm limits from hardware URDF, including asymmetric shoulder lifts; no upstream simulation-only −135° elbow extension |
| Grippers | −5° closed, +60° open on both sides, configured through `ArmChain` |
| IK target | `Gripper_Base_v1_1` / `Gripper_Base_v1_2` wrist-roll origins, independent of gripper opening |
| Cameras | `front_left`, `front_right`, `bottom`, `wrist_left`, `wrist_right`; upstream poses and individual FOVs |
| Home | Upstream raised-arm startup, elbows clamped to the hardware URDF limits; not a verified collision-free navigation tuck |
| Footprint | Conservative 0.50 m square chassis envelope; arm clip guards add clearance |
| TAMP | Standard `front` approach; kinematic manipulation advertised for simulation |

The adapter freezes wheel joints, replaces the floating base with three planar
joints, preserves CAD frames/inertials/visuals, and removes robot contact hulls.
This supports Emet's kinematic pick/place and scene placement; it does not model
wheel traction or physical grasp forces. The native upstream model retains those
geometries, but mecanum traction also requires upstream's Python simulation loop.
The lidar mount and metadata are retained; Emet does not yet publish its ray scans.

Upstream explicitly reports **169.042 kg CAD inertials**, pending corrected material
properties. We preserve these rather than claiming measured dynamics. Camera poses
are provisional fits, not physical calibration; the wrist views can be substantially
occluded by the grippers in the raised-arm pose. The generated model's stable steps
and matching forward kinematics do not establish sim-to-real accuracy.

```bash
# Default table, no downloaded kitchen assets required
uv run emet serve mujoco --config configs/sim/default_table_sourccey.yaml --headless

# RoboCasa kitchen
uv run emet serve mujoco --config configs/sim/robocasa_pick_place_sourccey.yaml --headless

# Connect an agent to the Emet simulation server
uv run emet run agent --robot sourccey --robot-ip 127.0.0.1 \
    --config configs/agent_sourccey.yaml --headless
```

## Motion and mapping smoke test

```bash
MUJOCO_GL=egl uv run python scripts/smoke_sourccey_mapping.py --out /tmp/sourccey-mapping
```

This runs a full scan and a square driving loop through the production navigation
controller, without teleporting, and feeds rendered front-camera RGB-D into
DynaMem's geometric mapper. It checks all waypoints, map growth, and detection of
the known table in the occupancy grid. Outputs: `drive.mp4`, `map.png`, `map.npz`,
and `report.json`. The test uses simulator pose/depth and prescribed clear
waypoints; it does not validate SLAM, autonomous exploration, or hardware sensing.

## When the robot arrives

The real host is **not** an Emet ZMQ server. `SourcceyBackend.create_client()` speaks
Emet's simulation protocol. Pointing it at `sourccey-host` does not provide hardware
control. The current vendor SDK is
[`lerobot-robot-sourccey`](https://github.com/vulcan-forge/lerobot-robot-sourccey),
version `0.2.4.dev14` at the revision above; it requires Python 3.12 or 3.13.
Keep it in a separate environment from Emet and install matching revisions on
host and desktop using the vendor's
[robot setup](https://github.com/vulcan-forge/lerobot-robot-sourccey/blob/d4f10db4fdf2786f8c87ec5b3562a812f93b1a9f/docs/01-setup/robot/README.md)
and [desktop setup](https://github.com/vulcan-forge/lerobot-robot-sourccey/blob/d4f10db4fdf2786f8c87ec5b3562a812f93b1a9f/docs/01-setup/desktop/README.md).

1. Confirm the delivered revision, device names, servo calibration and camera streams
   using the vendor setup and hardware diagnostics. Start `sourccey-host` on the robot.
2. Stop teleoperation and other observation clients: the vendor PUSH stream distributes
   packets between receivers rather than broadcasting each packet to everyone.
3. From the vendor environment, run this checkout's observation-only check:

   ```bash
   python /path/to/home_robot_v4/scripts/sourccey_preflight.py \
       --ip ROBOT_IP --seconds 5
   ```

   It opens only the PULL observation port (5556), checks the installed protocol
   version, arm/base submessages, finite values and all five nonblank, changing
   camera feeds. It exits nonzero for missing data, incompatible protocol, frozen
   images or timeout. It never opens a command socket. Passing means the observation
   stream is usable, not that calibration or motion is validated.
4. Use the vendor SDK/teleop for initial hardware operation. Before Emet autonomous
   operation, implement and validate a protocol adapter with measured servo zero/sign
   mapping, lift calibration, base velocity scaling, localization and depth input.

Do not apply a linear URDF-range conversion to vendor commands: the default servo
mode is normalized, optional degree mode depends on host configuration, lift targets
are −100…100, and base axes are −1…1 **uncalibrated** commands. The wheels have no
encoders in the referenced driver. Emet's SI commands require measured mappings;
simulation base coordinates are not wheel odometry. No hardware motion or calibration
has been tested by this update.

## Refreshing assets

```bash
# Use clean local clones; fetch/check out the desired upstream revisions first.
python scripts/robot_assets/sync_sourccey.py \
    --simulation /path/to/sourccey-simulation-mujoco \
    --hardware /path/to/sourccey-hardware
uv run python scripts/robot_assets/assemble_sourccey.py
MUJOCO_GL=egl uv run python -m pytest src/test/robots/test_sourccey_robot.py
```

The sync rejects differing hardware/simulator URDFs and unresolved Git LFS pointers.
Review the manifest and generated diff; run the RoboCasa/navigation tests when those
assets are installed. The simulator snapshot includes its MIT license. See the
asset [NOTICE](../../src/emet/assets/robot/sourccey/NOTICE.md).

### Wave emote and kinematics checks

Sourccey now implements `wave` (left arm), `wave_left`, and `wave_right` through
its registered emote backend and the shared robot-client joint interface:

```python
from emet.controller.task.emote.emote_task import EmoteTask

# Existing agent connected to the Emet Sourccey simulation server; base at goal.
success = EmoteTask(agent).get_task("wave").run()
```

The gesture raises the selected arm, rotates its wrist through three ±0.5 rad
waves, and returns to the initial arm configuration. Quintic interpolation limits
commanded arm speed to 0.8 rad/s. The lift, other arm, and grippers hold their
initial measured positions; the planar base receives zero velocity commands.
Small measured joint-limit violations (up to 0.03) are clipped to the model limits.
Missing/invalid feedback or excessive tracking error fails the operation and
stops advancing the trajectory. It finishes in manipulation mode. Commands use
Emet's radians/metres, not the vendor SDK's normalized units.

Reproduce the actuator-driven test and save a video, joint trace, and report:

```bash
MUJOCO_GL=egl uv run python scripts/smoke_sourccey_wave.py --out /tmp/sourccey-wave
uv run pytest src/test/robots/test_sourccey_wave.py -q
```

The smoke test runs `EmoteTask`, `GenericZmqClient` command serialization, the
production simulation action handler, and MuJoCo dynamics for both arms. Transport
is in-process and simulator time replaces sleeps; it does not test network delivery.
The regression tests compare analytic Jacobians against central finite differences,
check 24 nearby FK→position-IK round trips to 1 mm (including translated/rotated
base and lowered lift), reject unreachable targets, and check limits, speed,
held joints, and failure reporting. Position IK does not constrain orientation.

These gestures are simulation-tested, not collision-planned: robot collision
meshes are disabled in the current adaptation. Use clear space in simulation;
physical execution still needs the calibrated hardware bridge and commissioning.
