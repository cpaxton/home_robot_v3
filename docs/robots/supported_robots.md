# Supported robots (sim + learning)

Maintainer inventory for which robot IDs work in **MolmoSpaces**, **RoboCasa**, **ZMQ serve**, and **dataset / recording** tooling.

Regenerate the machine-readable table:

```bash
uv run python scripts/audit_robot_support.py
uv run python scripts/audit_robot_support.py --json
```

MolmoSpaces static list (wrapper not required):

```bash
emet molmospaces list-robots
```

## Mobile manipulation (learning experiments)

These robots have vendored MJCF, merge into MolmoSpaces scenes, and run on the ZMQ stack for navigation / exploration recording. **Agentic sim pick/place** on MolmoSpaces defaults to ZMQ `sim_set_body_pose` (teleport). Live **kinematic latch** (`capabilities.kinematic_manip`) is opt-in via `RobotSpec.advertise_kinematic_manip` (currently **rby1** / **galaxea_r1**, **innate_mars**, **nori**) — a resolvable `ArmManipProfile` alone is not enough. See [molmospaces.md](../molmospaces.md) and [ovmm_full_benchmark.md](../ovmm_full_benchmark.md).

**Kinematic pick/place profiles** are auto-discovered from each robot's spec + vendored MJCF via `ArmManipProfile.discover_from_spec()` ([arm_manip_profile.py](../../src/emet/motion/arm_manip_profile.py)) — no per-robot table needed for sourccey, xlerobot, innate_mars, franka_fr3; rby1/galaxea_r1 keep an explicit shared profile. Offline IK tests stay on for robots that do **not** advertise latch. Coverage is enforced by the end-to-end discovery → IK → gripper-contact tests in `src/test/motion/test_arm_manip_profile.py` (see [motion_planning.md](../motion_planning.md#armmanipprofile-discovery-no-hardcoded-table)).

| Robot | Aliases | Base type | MolmoSpaces | RoboCasa | ZMQ client | Recording |
|-------|---------|-----------|-------------|----------|------------|-----------|
| **stretch** | `hello_stretch` | planar (wheels) | yes (default) | strip-replace | `StretchZmqClient` | `molmospaces-explore`, LeRobot LfD |
| **rby1** | `galaxea_r1`, `rb_y1` | freejoint | yes | strip-replace | `GenericZmqClient` | `molmospaces-explore` |
| **innate_mars** | `maurice` | planar | yes | strip-replace + spawn guards | `GenericZmqClient` | `record_innate_mars_episode.py`, dynagraph |
| **xlerobot** | `xlerobot_dual` | planar / diff-drive | yes | strip-replace | `GenericZmqClient` | `molmospaces-explore`, DynaMem/Dynagraph nav baselines (`--robot xlerobot`); ZMQ `head_to`, `gripper_left`/`gripper_right` |
| **sourccey** | — | planar (wheels) + lift | yes | strip-replace + spawn guards | `GenericZmqClient` (stub) | [sourccey.md](sourccey.md) — Vulcan Robotics; vendored MJCF from STEP CAD + `lerobot-vulcan` `Arm.urdf` |
| **nori** | `nori_a3` | freejoint (diff-drive wheels visual) | yes | no (MolmoSpaces merge only; RoboCasa strip-replace is a follow-up) | `GenericZmqClient` | [nori.md](nori.md) — Nori A3 bimanual, 19-DoF; vendored URDF-derived MJCF (CC BY-NC-SA); curated per-arm `ArmChain`s |

## Tabletop manipulation (MolmoBot data alignment)

| Robot | MolmoSpaces | RoboCasa | Notes |
|-------|-------------|----------|-------|
| **franka_fr3** | yes (fixed base) | lower priority | MolmoBot-Data Franka pick/place configs |

## Listed but not end-to-end

| ID | Issue |
|----|-------|
| `franka_droid`, `franka_cap`, `floating_rum`, `floating_robotiq`, `rby1m` | In `MOLMOSPACES_ROBOT_IDS` only; no vendored MJCF in emet |
| `mobile_aloha`, `yor` | Registry stubs (`NotImplementedError`) |

## Extension checklist (new mobile robot)

1. Vendor MJCF under `src/emet/assets/robot/<name>/` (meshes beside XML).
2. Add `get_robot_mjcf_path("<name>")` in [`src/emet/utils/assets.py`](../src/emet/utils/assets.py).
3. Implement `RobotBackend` + `RobotSpec` (joints, actuators, cameras, footprint, spawn guards).
4. Register in [`ROBOT_REGISTRY`](../src/emet/robots/__init__.py).
5. RoboCasa: add to `_uses_strip_placeholder_robot()` if using strip-replace ([`robocasa_gen.py`](../src/emet/simulation/stretch_mujoco/robocasa_gen.py)).
6. MolmoSpaces: `emet molmospaces write-spawn-metadata --robot <name>` → commit `molmospaces_spawn.json`.
7. Smoke: `timeout 90 uv run emet serve mujoco --scene ithor --robot <name> --headless`.
8. Update this doc and run `scripts/audit_robot_support.py`.

See also: [molmospaces.md](../molmospaces.md), [molmospaces_spawn_metadata.md](../molmospaces_spawn_metadata.md), [datasets/molmobot.md](../datasets/molmobot.md).

## Manipulation data sources

| Source | Format | Emet tooling |
|--------|--------|--------------|
| MolmoBot-Data (HF) | H5 + MP4 | `emet dataset molmobot inspect`, export-lerobot, replay |
| MolmoSpaces explore | `metadata.jsonl` + RGB | `emet run molmospaces-explore` |
| Stretch teleop | dobbe folders | hello-robot LeRobot fork ([learning_from_demonstration.md](../learning_from_demonstration.md)) |
