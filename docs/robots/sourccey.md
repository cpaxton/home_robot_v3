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
returns a `GenericZmqClient` stub so joint/gripper plumbing has a target). The sim model
uses the **updated official `sourccey-hardware` arm URDF** (`URDF/ArmLeft/ArmLeft.urdf`,
vendored under `src/emet/assets/robot/sourccey/urdf/`); the right arm is the code-side
X-mirror of the canonical left arm. Kinematic pick/place (`mcts` manip mode) is advertised.

| Stack | Status |
|-------|--------|
| Vendored MJCF | `src/emet/assets/robot/sourccey/sourccey.xml` (official ArmLeft URDF arm) |
| Vendored URDF | `src/emet/assets/robot/sourccey/urdf/ArmLeft/ArmLeft.urdf` + `ArmRight/` (STL meshes only; no Unity sidecars) |
| Backend / registry | `emet.robots.sourccey.SourcceyBackend` → `ROBOT_REGISTRY["sourccey"]` |
| Kinematic model | `create_model` → `SpecRobotModel`; declarative left/right `arm_chains` |
| Kinematic manip | `advertise_kinematic_manip=True` → `mcts` OVMM pick/place (e.g. `robocasa_sourccey_counter_to_cab_mcts`) |
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

# Agent (connect to the sim server, then run with an LLM endpoint)
uv run emet serve mujoco --robot sourccey --scene ithor --headless          # terminal 1
uv run emet run agent --robot sourccey --robot-ip 127.0.0.1 \
    --config configs/agent_sourccey.yaml --headless                          # terminal 2
```

## Model notes

- **Arm kinematics / inertials / meshes** are the exact `URDF/ArmLeft/ArmLeft.urdf`
  chain from the **updated** `vulcan-forge/sourccey-hardware` repo (joint frames, axes,
  limits, masses, and Unity-exported meshes). The right arm is the code-side X-mirror
  of the left (the two official `ArmLeft`/`ArmRight` exports are asymmetric, so one
  canonical arm is mirrored to keep the robot symmetric).
- **Arm reach**: the 6-DoF arm reaches outward/sideways (workspace bottoms out around
  z≈0.36 m below the shoulder), so table/counter-top objects are reachable but **floor
  objects are not** — use counter-based `mcts` episodes (e.g.
  `robocasa_sourccey_counter_to_cab_mcts`), not the floor pick/place row. TAMP
  `plan_pick_place` uses `RobotSpec.tamp_approach="side"` (left yaw=+π/2) rather than
  the Galaxea front standoff; a front-facing rby1 pose misses by ~0.6–1.2 m.
- **Base / dome / wheels / lift** are simplified pragmatic geometry assembled from the
  STEP CAD parts (see `scripts/robot_assets/`). The body is a 3-level pyramidal shell
  (250 → 207 → 183 mm plates/walls), 4 mecanum wheels with holder brackets at the
  corners, a vertical linear lift, and a rounded dome head with stereo cameras. Cosmetic
  detail is trimmed; the base carries a single box collider.
- **All geoms are visual-only** (contype=0), matching innate_mars: Robocasa planar
  autoplace stays O(1) (the first-candidate hint is accepted instantly). Spawn safety
  comes from the planar clip guards + footprint; motion-planning collision is delegated
  to external planners. Self-collision at the `sourccey_home` keyframe is clean.
- **Planar base**: `base_x` / `base_y` / `base_yaw` slides + yaw on `base_root` driven by
  velocity actuators (like innate_mars / xlerobot); a nav P-controller converges to a
  world goal in a few seconds.
- **Cameras**: `front_left`/`front_right` are a stereo pair on the dome (forward, 20°
  down), `wrist_left`/`wrist_right` look outward along the grippers. FOV 70°. Rendered RGB +
  depth are verified consistent (`assert_zmq_observation_frames_consistent`).
- **Home keyframe** `sourccey_home` tucks the arms for navigation and is collision-free.

## Regeneration

All assets are generated — never hand-edit `sourccey.xml`, `arm_frag.xml`, or `meshes/`. See
[`scripts/robot_assets/README.md`](../../scripts/robot_assets/README.md) and the
asset NOTICE at `src/emet/assets/robot/sourccey/NOTICE.md`.

## License

Sourccey hardware is released under [CERN-OHL-S-2.0](https://github.com/vulcan-forge/sourccey-hardware/blob/main/LICENSE).
Converted meshes/MJCF preserve upstream notices.
See `src/emet/assets/robot/sourccey/NOTICE.md`.
