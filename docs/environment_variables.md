# Environment variables

Optional process-environment toggles for simulation, ZMQ clients, and MolmoSpaces. Most apps read these at startup; export in the shell before `emet serve` / `emet run`.

## MolmoSpaces

**[MolmoSpaces environment variables](molmospaces_environment_variables.md)** — spawn, autoplace, occupancy map, navigation teleport (`EMET_MOLMOSPACES_NAV_TELEPORT`), asset paths, and related test knobs.

**[MolmoSpaces spawn metadata](molmospaces_spawn_metadata.md)** — checked-in `molmospaces_spawn.json` per robot and `emet molmospaces write-spawn-metadata` (offline; not an env var).

See also [MolmoSpaces](molmospaces.md) for install and CLI usage.

## Benchmarks

| Variable | Where used | Notes |
|----------|------------|-------|
| `SQA3D_DATA_DIR` | `emet sqa3d`, `emet eval-sqa3d` | Root for SQA3D Zenodo JSON (`sqa_task/`, optional `localization_task/`). Default `~/.cache/sqa3d/data`. See [sqa3d.md](sqa3d.md). |
| `SCANNET_ROOT` | `emet sqa3d run-episode` | ScanNet v2 download root (`scans/<scene_id>/…`). Default `~/.cache/scannet`. |
| `SCANNET_DOWNLOAD_SCRIPT` | `scripts/download_scannet_data.py` | Override path to `download-scannet.py`. |
| `RUN_SQA3D_SCANNET_TESTS` | pytest | Set `1` to run embodied ScanNet integration smoke. |

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
