# MolmoSpaces spawn metadata (`molmospaces_spawn.json`)

Optional per-robot JSON files tune MolmoSpaces **Z placement** (foot clearance above the walkable floor). Runtime spawn in [`scene_base_spawn`](../src/emet/simulation/scene_base_spawn.py) / [`molmospaces_spawn`](../src/emet/simulation/molmospaces_spawn.py) loads them via [`molmospaces_spawn_metadata`](../src/emet/simulation/molmospaces_spawn_metadata.py). If a file is missing, spawn uses built-in heuristics.

This is **maintainer tooling**, not required for `emet serve molmospaces` to start.

## File location

For registry robots with a vendored MJCF (`get_robot_mjcf_path`), the default path is:

```text
src/emet/assets/robot/<robot_dir>/molmospaces_spawn.json
```

Examples checked into the repo:

| Robot | Path |
|-------|------|
| stretch (default MolmoSpaces serve) | [`robot/molmospaces_spawn.json`](../src/emet/assets/robot/molmospaces_spawn.json) next to [`stretch.xml`](../src/emet/assets/robot/stretch.xml) |
| rby1 / galaxea_r1 | [`galaxea_r1/molmospaces_spawn.json`](../src/emet/assets/robot/galaxea_r1/molmospaces_spawn.json) |
| innate_mars | [`innate_mars/molmospaces_spawn.json`](../src/emet/assets/robot/innate_mars/molmospaces_spawn.json) |

`RobotSpec.spawn` on rby1 / galaxea_r1 can mirror the same fields (see [`RobotSpawnSpec`](../src/emet/robots/base.py)).

## JSON fields

| Field | Type | Used for |
|-------|------|----------|
| `schema_version` | int | Version tag (currently `1`). |
| `molmospaces_target_foot_clearance_above_floor_m` | float | Target gap between lowest robot collision geom and floor during `settle_free_base_z_to_floor`. |
| `molmospaces_nominal_base_height_above_floor_m` | float | Optional reference base-link height above floor (for future tuning / QA). |
| `requires_floating_base_spawn_settle` | bool | Reserved; not required for basic clearance tuning. |
| `notes` | string | Free text (e.g. how the file was generated). |

## When to update

- After changing a robot MJCF (base collision geoms, foot/wheel layout).
- When adding a **new** vendored robot used on MolmoSpaces merge paths.
- When spawn QA shows the base floating or clipping on iTHOR-style floors for that robot.

Use a **merged** scene+robot MJCF (not the standalone table scene). The measurement runs a short kinematic settle at a seed `(x, y)` on the scene floor. If `(0, 0)` fails (common on iTHOR clutter), merge beside the robot MJCF (so meshes resolve), pick a walkable point from autoplace or occupancy QA, and pass **`--seed-x` / `--seed-y`**.

## Workflow

### 1. Merge scene + robot

```bash
emet molmospaces merge-scene --scene ithor --split train --index 0 --robot stretch \
  -o src/emet/assets/robot/ithor_stretch_merged.xml --install-if-missing
emet molmospaces merge-scene --scene ithor --split train --index 0 --robot rby1 \
  -o src/emet/assets/robot/galaxea_r1/ithor_rby1_merged.xml --install-if-missing
```

Use the same `--robot` you serve with. Write **`-o`** next to the robot MJCF (e.g. `src/emet/assets/robot/` for stretch, `galaxea_r1/` for rby1) so MuJoCo can load meshes during measurement; `/tmp` often breaks mesh paths.

### 2. Measure and write JSON

**CLI (recommended):**

```bash
emet molmospaces write-spawn-metadata --robot rby1 --mjcf /tmp/ithor_rby1_merged.xml
```

Help:

```bash
emet molmospaces write-spawn-metadata --help
```

**Module (same logic):**

```bash
uv run python -m emet.app.write_molmospaces_spawn_metadata \
  --robot rby1 --mjcf /tmp/ithor_rby1_merged.xml
```

**Options:**

| Flag | Meaning |
|------|---------|
| `--robot` | Robot id (required); must match the merge. |
| `--mjcf` | Path to merged MJCF (required). |
| `-o` / `--output` | Override output JSON path. |
| `--no-merge` | Replace file instead of merging into existing JSON. |
| `--base-body` | Base body with free joint (default `base_link`). |
| `--seed-x`, `--seed-y` | World XY for measurement pose (default `0`, `0`). |

### 3. Commit the JSON

Check in `molmospaces_spawn.json` next to the robot MJCF. CI includes [`test_molmospaces_spawn_metadata.py`](../src/test/config/test_molmospaces_spawn_metadata.py) asserting the file exists for supported merge robots.

### 4. Verify in sim (optional)

```bash
emet serve molmospaces --robot rby1 --headless --debug-molmospaces-spawn
```

Compare autoplace / floor alignment logs with and without the updated clearance value.

## Implementation reference

| Component | Role |
|-----------|------|
| [`write_molmospaces_spawn_metadata.py`](../src/emet/app/write_molmospaces_spawn_metadata.py) | Offline measure + write (app layer, like `build_molmo_occupancy_map`). |
| [`molmospaces_spawn_metadata.py`](../src/emet/simulation/molmospaces_spawn_metadata.py) | Runtime load of JSON into `MolmospacesSpawnMetadata`. |

Other simulation internals (home tuning, stationary control, Robocasa asset checks): [simulation_modules.md](simulation_modules.md).
| `emet molmospaces write-spawn-metadata` | Click wrapper in [`cli.py`](../src/emet/cli.py). |

## Related docs

- [MolmoSpaces](molmospaces.md) — install, merge, serve, agent/ZMQ.
- [MolmoSpaces environment variables](molmospaces_environment_variables.md) — autoplace toggles (`EMET_MOLMOSPACES_AUTOPLACE`, etc.).
