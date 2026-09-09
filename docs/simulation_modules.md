# Simulation modules (maintainer reference)

Python modules added or extended for **Robocasa serve**, **MolmoSpaces spawn**, and **registry-robot ZMQ sim** (`RobosuiteZmqServer`). End users normally use [`emet serve`](cli.md); this page is for contributors tuning MJCF, assets, and load behavior.

Related user docs: [simulation.md](simulation.md), [molmospaces.md](molmospaces.md), [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md).

## Overview

| Module | Purpose |
|--------|---------|
| [`fall_detection`](../src/emet/simulation/fall_detection.py) | Detect tipped/fallen base from MuJoCo `xmat` (body +Z · world +Z); red error in Stretch + Robosuite ZMQ servers. |
| [`mujoco_stationary_control`](../src/emet/simulation/mujoco_stationary_control.py) | Build full-length stationary `ctrl` from joint transmissions + spec hold buffer (used by Stretch and registry robots). |
| [`robosuite_stationary_control`](../src/emet/simulation/robosuite_stationary_control.py) | Re-export shim → use `mujoco_stationary_control`. |
| [`robosuite_load_utils`](../src/emet/simulation/robosuite_load_utils.py) | Post-load helpers for `RobosuiteZmqServer`: home keyframe, free-joint addrs, diagnostics. |
| [`mujoco_home_tune`](../src/emet/simulation/mujoco_home_tune.py) | Interactive GUI to tune robot **home** pose and print MJCF `<key ctrl="..."/>` snippets. |
| [`robocasa_assets_check`](../src/emet/simulation/robocasa_assets_check.py) | Preflight: basic fixtures, LightWheel registry, objaverse `reg_bbox` before `emet serve robocasa`. |
| [`robocasa_registry_sync`](../src/emet/simulation/robocasa_registry_sync.py) | Sync LightWheel meshes into `fixture_registry/*.yaml` (`fixtures/` + `objects/lightwheel/` accessories + cabinet door/handle mesh ids). |
| [`robocasa_objaverse_bbox`](../src/emet/simulation/robocasa_objaverse_bbox.py) | Post-process objaverse MJCFs to add `reg_bbox` geoms (wraps Robocasa `calc_object_bb_reg`). |
| [`molmospaces_spawn_metadata`](../src/emet/simulation/molmospaces_spawn_metadata.py) | Runtime load of per-robot `molmospaces_spawn.json`. |
| [`write_molmospaces_spawn_metadata`](../src/emet/app/write_molmospaces_spawn_metadata.py) | Offline measure + write spawn JSON (see [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md)). |
| [`chase_camera`](../src/emet/simulation/chase_camera.py) | FREE chase cam off `base_link` for `--record-mp4` / `EMET_SIM_THIRD_PERSON` (raised lookat; avoids torso clip). Nadir overhead via `EMET_SIM_OVERHEAD`. |
| [`scene_base_spawn`](../src/emet/simulation/scene_base_spawn.py) | Re-export shim for spawn helpers (original import path). |
| [`spawn_geom`](../src/emet/simulation/spawn_geom.py) | Floor rays, collision clip, occupancy, exterior-tongue gates. |
| [`spawn_settle`](../src/emet/simulation/spawn_settle.py) | Free-joint Z settle / restore / foot-clearance probes. |
| [`spawn_debug`](../src/emet/simulation/spawn_debug.py) | ASCII/PNG occupancy maps when `EMET_MOLMOSPACES_SPAWN_DEBUG` is on. |
| [`spawn_molmospaces`](../src/emet/simulation/spawn_molmospaces.py) | MolmoSpaces free-joint XY search (`find_molmospaces_freejoint_xyz`). |
| [`spawn_planar`](../src/emet/simulation/spawn_planar.py) | Planar slide+yaw spawn (`find_planar_base_xyt`, `write_planar_base_xyt`). |
| [`spawn_robocasa`](../src/emet/simulation/spawn_robocasa.py) | Robocasa free-joint autoplace (`find_robocasa_freejoint_xyz`). |

Runtime wiring: [`robosuite_server.py`](../src/emet/simulation/robosuite_server.py), [`mujoco_serve_argv.py`](../src/emet/simulation/mujoco_serve_argv.py), [`scene_base_spawn.py`](../src/emet/simulation/scene_base_spawn.py) (re-exports `spawn_geom` / `spawn_settle` / `spawn_debug` / `spawn_molmospaces` / `spawn_planar` / `spawn_robocasa`).

---

## MuJoCo home pose tuning (`mujoco_home_tune`)

Tune the **home** keyframe for registry robots (Galaxea R1 / rby1, innate_mars, etc.): open a MuJoCo viewer, pose the robot, close the window, copy the printed `<key name="home" ctrl="..."/>` line into the robot MJCF.

Uses [`compute_stationary_ctrl_vector`](../src/emet/simulation/mujoco_stationary_control.py) and [`robosuite_load_utils`](../src/emet/simulation/robosuite_load_utils.py) for initial pose (`default` / `zeros` / `home`).

**Requires a display** (not for headless CI).

```bash
# Physics Simulate (full dynamics UI)
uv run python -m emet.simulation.mujoco_home_tune \
  src/emet/assets/robot/galaxea_r1/galaxea_r1.xml \
  --base-body base_link

# Kinematic viewer only (no mj_step; drag joints safely)
uv run python -m emet.simulation.mujoco_home_tune path/to/merged.xml \
  --kinematic --initial-pose home

uv run python -m emet.simulation.mujoco_home_tune --help
```

| Flag | Meaning |
|------|---------|
| `mjcf` | Robot or merged scene MJCF path (positional). |
| `--base-body` | Free-joint base body (default `base_link`). |
| `--initial-pose` | `default`, `zeros`, or `home` before tuning. |
| `--tune-base-z` | Hoist floating base to this world Z before freezing (default `0.38`). |
| `--kinematic` | Passive viewer + `mj_forward` only (no physics). |

Programmatic API: `build_tune_model()`, `run_tune_home_gui()`, `print_home_keyframe_snippet()`.

Tests: [`test_robosuite_load_utils.py`](../src/test/config/test_robosuite_load_utils.py) (`build_tune_model`, `format_key_ctrl_attr`).

---

## Stationary control (`mujoco_stationary_control`)

When the ZMQ server steps physics, non-nav joints need a **stationary** `ctrl` vector so arms/torso do not collapse. This module:

- Fills `model.nu` from actuator transmissions and current `qpos`
- Zeros wheel / velocity actuators
- Overlays `RobotSpec` hold targets from `RobosuiteZmqServer`

Per-robot policies: [`galaxea_r1/sim_stationary.py`](../src/emet/robots/galaxea_r1/sim_stationary.py), [`stretch/sim_stationary.py`](../src/emet/robots/stretch/sim_stationary.py).

---

## Robosuite load helpers (`robosuite_load_utils`)

Used during `RobosuiteZmqServer` startup:

| Function | Role |
|----------|------|
| `apply_home_keyframe_preserving_base` | Apply MJCF `home` keyframe; keep merged-scene base pose and **scene freejoints** (table cubes) when Molmo autoplace ran. |
| `apply_home_keyframe_preserving_planar_base` | Same for planar `base_x/y/yaw` robots (Sourccey / xlerobot); restores object `<freejoint/>` qpos after `mj_resetDataKeyframe`. |
| `apply_zero_joint_pose_preserving_base` | Zero articulated joints; preserve base. |
| `freejoint_qpos_qvel_addrs` | Locate free joint on `base_link` (Molmo merge). |
| `log_post_load_diagnostics` | Optional velocity / contact logging (`EMET_ROBOSUITE_POST_LOAD_DEBUG=1`; see [environment_variables.md](environment_variables.md)). |
| `probe_max_qvel_unforced_steps` | Short dynamics probe after load. |

---

## Robocasa asset tooling

### Preflight (`robocasa_assets_check`)

`emet serve robocasa` calls these before building a kitchen:

- `basic_fixtures_present` — base fixtures zip extracted
- `lightwheel_registry_ok` — `fixtures_lw` + `Sink025` in registry
- `fixture_registry_layout_ok` — required registry YAML stems exist
- `objaverse_reg_bbox_ok` — objaverse models have `reg_bbox` geoms

### Registry sync (`robocasa_registry_sync`)

The `fixtures_lw` zip adds meshes but often **replaces** `fixture_registry/` with a slim subset. Restore vendored YAML from git, then sync LightWheel entries:

```bash
git -C third_party/robocasa checkout -- robocasa/models/assets/fixtures/fixture_registry/
uv run python scripts/sync_robocasa_lightwheel_registry.py
```

Module: `sync_lightwheel_fixture_registry()`, `missing_required_registry_stems()`.

### Objaverse bbox (`robocasa_objaverse_bbox`)

One-time post-process after objaverse download:

```bash
uv run python scripts/process_robocasa_objaverse_reg_bbox.py
```

Wraps Robocasa `calc_object_bb_reg` over `third_party/robocasa/.../objaverse/`. `download_robocasa_assets.py` runs this automatically when objaverse is fetched.

---

## MolmoSpaces spawn metadata

Runtime: [`molmospaces_spawn_metadata.py`](../src/emet/simulation/molmospaces_spawn_metadata.py) loads optional `molmospaces_spawn.json` next to each robot MJCF.

Offline measurement: **`emet molmospaces write-spawn-metadata`** — full guide in [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md).

---

## Tests

| Test file | Covers |
|-----------|--------|
| [`test_robosuite_load_utils.py`](../src/test/config/test_robosuite_load_utils.py) | Load utils, home keyframe, `mujoco_home_tune` |
| [`test_robocasa_assets_check.py`](../src/test/config/test_robocasa_assets_check.py) | Asset preflight helpers |
| [`test_molmospaces_spawn_metadata.py`](../src/test/config/test_molmospaces_spawn_metadata.py) | Spawn JSON load + `resolve_serve_robot` |

```bash
uv run emet test src/test/config/test_robosuite_load_utils.py \
  src/test/config/test_robocasa_assets_check.py \
  src/test/config/test_molmospaces_spawn_metadata.py -q
```
