# Sourccey (Vulcan Robotics)

Sourccey is an open-source home robot by [Vulcan Robotics](https://vulcanrobotics.ai/specs):
a wheeled mobile base with a vertical linear lift, a dome head with stereo cameras, and
dual 5-DOF + gripper arms (6 revolute joints per arm).

| Property | Value |
|----------|-------|
| Mobility | 4 mecanum wheels (omnidirectional) |
| Lift | 12 V 100 N linear actuator |
| Arms | 2× (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) |
| Actuation | Feetech STS3215 / STS3250 serial servos |
| Cameras | `front_left`, `front_right` (dome), `wrist_left`, `wrist_right` |
| Footprint | 414 mm diameter, 1030 mm tall, 15.88 kg |
| Hardware | https://github.com/vulcan-forge/sourccey-hardware (STEP CAD) |
| Software | https://github.com/vulcan-forge/lerobot-vulcan (LeRobot fork) |

## Emet support

Emet supports **Sourccey in simulation** (no real-hardware ZMQ bridge yet — `create_client`
returns a `GenericZmqClient` stub so joint/gripper plumbing has a target).

| Stack | Status |
|-------|--------|
| Vendored MJCF | `src/emet/assets/robot/sourccey/sourccey.xml` |
| Backend / registry | `emet.robots.sourccey.SourcceyBackend` → `ROBOT_REGISTRY["sourccey"]` |
| MolmoSpaces | merge + spawn metadata (`molmospaces_spawn.json`) |
| RoboCasa | strip-replace (`PandaMobile` placeholder → vendored MJCF) |
| ZMQ serve | `emet serve mujoco --robot sourccey` (RobosuiteZmqServer) |

### Launch examples

```bash
# plain MuJoCo serve
uv run emet serve mujoco --scene ithor --robot sourccey --headless

# RoboCasa kitchen
uv run emet serve mujoco --config configs/sim/robocasa_pick_place_sourccey.yaml --headless

# MolmoSpaces merge
uv run emet serve mujoco --config configs/sim/molmospaces_ithor_train_sourccey_0.yaml --headless
```

## Model notes

- **Arm kinematics / inertials** are the exact `Arm.urdf` chain from `lerobot-vulcan`
  (joint frames, axes, limits, and mass/inertia tensors, scaled to a realistic dual-arm
  mass so the full robot lands near the real 15.88 kg). The left arm is the X-mirror
  of the right (sagittal reflection), so left/right joint targets are opposite sign.
- **Base / dome / wheels / lift** are simplified pragmatic geometry assembled from the
  STEP CAD parts (see `scripts/robot_assets/`). Cosmetic detail is trimmed; the base
  carries a single box collider.
- **All geoms are visual-only** (contype=0), matching innate_mars: Robocasa planar
  autoplace stays O(1) (the first-candidate hint is accepted instantly). Spawn safety
  comes from the planar clip guards + footprint; motion-planning collision is delegated
  to external planners. Self-collision at the `sourccey_home` keyframe is clean.
- **Planar base**: `base_x` / `base_y` / `base_yaw` slides + yaw on `base_root` driven by
  velocity actuators (like innate_mars / xlerobot); a nav P-controller converges to a
  world goal in a few seconds.
- **Cameras**: `front_left`/`front_right` are a stereo pair on the dome (forward, 20°
  down), `wrist_left`/`wrist_right` look outward along the arms. FOV 70°. Rendered RGB +
  depth are verified consistent (`assert_zmq_observation_frames_consistent`).
- **Home keyframe** `sourccey_home` tucks the arms for navigation and is collision-free.

## Regeneration

All assets are generated — never hand-edit `sourccey.xml` or `meshes/`. See
[`scripts/robot_assets/README.md`](../../scripts/robot_assets/README.md) and the
asset NOTICE at `src/emet/assets/robot/sourccey/NOTICE.md`.

## License

Sourccey hardware is released under [CERN-OHL-S-2.0](https://github.com/vulcan-forge/sourccey-hardware/blob/main/LICENSE).
Converted meshes/MJCF preserve upstream notices; the LeRobot fork URDF is Apache-2.0.
See `src/emet/assets/robot/sourccey/NOTICE.md`.
