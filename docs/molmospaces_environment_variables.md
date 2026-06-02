# MolmoSpaces environment variables

Optional toggles for MolmoSpaces merge, spawn, navigation, tests, and tooling. Set them in the shell before `emet serve mujoco`, `emet run dynamem`, or pytest.

**Related (not Molmo-specific):** [Environment variables index](environment_variables.md) lists other `EMET_*` knobs used with simulation and ZMQ clients.

## Paths and wrapper

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLSPACES_ASSETS_DIR` | `~/.cache/molmospaces/assets` (via `ensure_molmospaces_assets_dir_env`) | MolmoSpaces scene/object assets root. Must not equal or nest inside `MLSPACES_CACHE_DIR`. |
| `MLSPACES_CACHE_DIR` | `~/.cache/molmospaces/resource_cache` | Molmo resource cache. |
| `MOLMOSPACES_PYTHON` | (unset → wrapper venv or `emet` delegation) | Python executable for `emet molmospaces` subprocess merge/install. |
| `EMET_MOLMOSPACES_AUTO_INSTALL` | (unset) | Wrapper: when `1`/`true`, auto-install missing Molmo scene assets on demand (`packages/emet_molmospaces`). |

## Spawn and autoplace

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMET_MOLMOSPACES_AUTOPLACE` | `1` | Free-joint base placement after MJCF load. `0`/`false`/`no`/`off` disables. `extended` or `2` also enables heuristics for some renamed merges (e.g. `FloorPlan` / `ithor` in basename). |
| `EMET_MOLMOSPACES_SPAWN_DEBUG` | (unset) | Verbose spawn logging; also set by `emet serve mujoco --debug-molmospaces-spawn`. |
| `EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG` | (unset) | Path for spawn debug top-down PNG; `0` skips default PNG (ASCII still logs). |
| `EMET_MOLMOSPACES_OCC_MAP` | `1` | iTHOR: use orthographic occupancy for XY spawn sampling. `0`/`false` disables. |
| `EMET_MOLMOSPACES_OCC_SEED` | `0` | RNG seed for occupancy free-point subsample. |
| `EMET_MOLMOSPACES_OCC_PRIORITY_MAX` | (internal default) | Cap occupancy priority sample count (clamped 200–12000). |
| `EMET_MOLMOSPACES_ANNULUS_N_RADII` | (heuristic default) | Override annulus spawn radii count. |
| `EMET_MOLMOSPACES_ANNULUS_BASE_ANGLES` | (heuristic default) | Override annulus base angle count. |
| `EMET_MOLMOSPACES_GRID_STEP_M` | (heuristic default) | Grid spawn step (meters). |
| `EMET_MOLMOSPACES_GRID_MAX_POINTS` | (heuristic default) | Max grid spawn candidates. |

## Navigation (ZMQ server)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMET_MOLMOSPACES_NAV_TELEPORT` | `1` | On merged MolmoSpaces sessions, use **free-joint teleport** for `xyt` / `rotate_in_place` goals. `0`/`false`/`no`/`off` forces wheel / `set_goal_pose` drive (experimental; unreliable yaw on iTHOR floors). Session `teleport_base` capability follows this flag. |

Implementation: `emet.simulation.molmospaces_env.molmospaces_nav_teleport_enabled()`.

## ZMQ client / slow Molmo loads

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMET_ZMQ_STARTUP_TIMEOUT` | `60` (client) | Seconds to wait for first ZMQ obs+state after connect. Increase for slow Molmo merges (e.g. `120`). |

## Mapping / nav grid debug (MolmoSpaces tests)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMET_NAVGRID_ASCII` | (unset) | Print cropped ASCII nav grid to stderr after mapping updates. |
| `EMET_NAVGRID_MAX_SIDE` | `320` | Longest edge (cells) for ASCII downsampling; `640` for full detail. |
| `EMET_NAVGRID_CONTEXTS` | (unset) | Comma-separated hook labels allowed to print (e.g. `rotate_in_place,explore`). |
| `EMET_NAVGRID_MIN_EXPLORED_IOU` | `0.25` | Multi-robot navgrid similarity test threshold. |
| `EMET_NAVGRID_MIN_OBSTACLE_IOU` | `0.20` | Multi-robot navgrid similarity test threshold. |

## Pytest / CI only

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMET_MOLMOSPACES_ORIENTATION_N` | `10` | iTHOR base-settle test: number of scenes. |
| `EMET_MOLMOSPACES_SETTLE_STEPS` | `900` | Physics steps after spawn before orientation check. |
| `EMET_MOLMOSPACES_ORIENTATION_MAX_DEG` | `8` | Max \|roll\|/\|pitch\| after settle. |
| `EMET_MOLMOSPACES_MIN_UP_DOT` | `0.92` | Min body z · world z after settle. |
| `EMET_MOLMO_DYNAMEM_PORT_OFFSET` | `100` | ZMQ port offset for Stretch Molmo Dynamem floor-map test. |
| `RUN_MOLMOSPACES_TESTS` | (unset) | Enable optional MolmoSpaces CLI integration tests. |
| `RUN_MULTI_ROBOT_NAVGRID` | (unset) | Enable multi-robot navgrid similarity test. |

## Boolean parsing

For flags documented as on/off, these values are recognized:

- **On:** `1`, `true`, `yes`, `on` (and non-empty other strings for `env_flag` defaults)
- **Off:** `0`, `false`, `no`, `off`

Shared helper: `emet.simulation.molmospaces_env.env_flag`.
