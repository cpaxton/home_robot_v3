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

These robots have vendored MJCF, merge into MolmoSpaces scenes, and run on the ZMQ stack for navigation / exploration recording. **Agentic sim pick/place** on MolmoSpaces uses **rby1** + ZMQ `sim_set_body_pose` (teleport); see [molmospaces.md](../molmospaces.md) and [ovmm_full_benchmark.md](../ovmm_full_benchmark.md).

| Robot | Aliases | Base type | MolmoSpaces | RoboCasa | ZMQ client | Recording |
|-------|---------|-----------|-------------|----------|------------|-----------|
| **stretch** | `hello_stretch` | planar (wheels) | yes (default) | strip-replace | `StretchZmqClient` | `molmospaces-explore`, LeRobot LfD |
| **rby1** | `galaxea_r1`, `rb_y1` | freejoint | yes | strip-replace | `GenericZmqClient` | `molmospaces-explore` |
| **innate_mars** | `maurice` | planar | yes | strip-replace + spawn guards | `GenericZmqClient` | `record_innate_mars_episode.py`, dynagraph |
| **xlerobot** | `xlerobot_dual` | planar / diff-drive | yes | strip-replace | `GenericZmqClient` | `molmospaces-explore`, DynaMem/Dynagraph nav baselines (`--robot xlerobot`); ZMQ `head_to`, `gripper_left`/`gripper_right` |
| **sourccey** | — | planar (wheels) + lift | yes | strip-replace + spawn guards | `GenericZmqClient` (stub) | [sourccey.md](sourccey.md) — Vulcan Robotics; vendored MJCF from STEP CAD + `lerobot-vulcan` `Arm.urdf` |

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
