# Environment variables

Optional process-environment toggles for simulation, ZMQ clients, and MolmoSpaces. Most apps read these at startup; export in the shell before `emet serve` / `emet run`.

## MolmoSpaces

**[MolmoSpaces environment variables](molmospaces_environment_variables.md)** — spawn, autoplace, occupancy map, navigation teleport (`EMET_MOLMOSPACES_NAV_TELEPORT`), asset paths, and related test knobs.

**[MolmoSpaces spawn metadata](molmospaces_spawn_metadata.md)** — checked-in `molmospaces_spawn.json` per robot and `emet molmospaces write-spawn-metadata` (offline; not an env var).

See also [MolmoSpaces](molmospaces.md) for install and CLI usage.

## ZMQ and simulation (general)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_ZMQ_STARTUP_TIMEOUT` | ZMQ clients, `emet run molmospaces-explore` | Seconds to wait for first observation (default 60). Documented in [molmospaces_environment_variables.md](molmospaces_environment_variables.md). |
| `EMET_NAVGRID_ASCII` | Dynamem / Dynagraph mapping | Terminal nav grid; see [dynagraph.md](dynagraph.md). |
| `EMET_NAVGRID_MAX_SIDE` | Nav grid ASCII | Default 320. |
| `EMET_NAVGRID_CONTEXTS` | Nav grid ASCII | Limit which hooks print. |
| `EMET_ROBOSUITE_POST_LOAD_DEBUG` | `robosuite_load_utils` | Post-load velocity / contact diagnostics after `RobosuiteZmqServer` startup. Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG` | `robosuite_server` | Log stationary `ctrl` apply cycles (first N steps, then periodic). Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG_VERBOSE` | `robosuite_server` | Full per-actuator `ctrl` lines (use with `EMET_MUJOCO_CTRL_DEBUG=1`). |
| `EMET_ROBOSUITE_AUTOPLACE` | `scene_base_spawn` | Planar base autoplace on Robocasa / default-table merges (default `1`). `0`/`false`/`no`/`off` disables. |
| `MUJOCO_GL` | MuJoCo rendering | e.g. `egl` for headless GPU cameras on Linux. |

See also [simulation_modules.md](simulation_modules.md) for maintainer-oriented module notes.

Add new cross-cutting `EMET_*` variables here or in a topic-specific doc and link from [simulation.md](simulation.md) / [README.md](../README.md) as appropriate.
