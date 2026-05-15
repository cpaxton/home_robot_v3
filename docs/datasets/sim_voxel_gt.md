# Sim voxel + ground-truth object dataset

This document describes artifacts produced by **`emet dataset capture-sim-episode`** (or **`uv run python -m emet.app.capture_sim_dataset_episode`**) and how to extend the pipeline.

**Dataset product id:** `sim_voxel_gt_episode` — one short sim run that writes a DynaMem voxel map plus **simulator** object poses (MuJoCo bodies), not a separate “generic” dataset format.

**Visible sim window:** when this tool spawns the MuJoCo server, pass **`--show-sim`** or **`--no-headless`** so the subprocess does not get `--headless` (you need a working **DISPLAY** / GLFW for windowed GL). Add **`--show-viewer-ui`** to forward the server’s viewer side panels when supported (see `emet serve mujoco --help`).

**Kinematic sim (default for packaged table + rby1):** `configs/sim/default_table_rby1.yaml` sets **`physics_mode: kinematic`** so the non-Stretch server advances with **`mj_forward` only** (poses snap; no `mj_step` contact dynamics). Override per run with **`--kinematic-sim`** (forces kinematic) or edit the YAML. `dataset_manifest.json` records **`sim_physics_mode`**. For agents: **`emet run agent --start-sim --sim-kinematic`**. For `emet serve mujoco`: **`--kinematic-sim`**.

## Artifacts (per episode directory)

| Path | Description |
|------|-------------|
| `dynamem/` | DynaMem memory directory (point cloud, frames, optional `graph.json` when GT objects exist). Same layout as other DynaMem saves; see `emet.memory.backend` / `emet.app.create_and_print_memory`. |
| `ground_truth.json` | Simulator **object** poses from MuJoCo bodies (not open-vocab detections). Includes `dataset_product: "sim_voxel_gt_episode"`; list under `objects` with `body_name`, `pos_xyz`, `quat_wxyz`, optional AABB. |
| `dataset_manifest.json` | `dataset_product`, description, git SHA (if available), robot id, scene kind, sim YAML path, timestamps, `sim_spawn_headless` / `sim_spawn_show_viewer_ui` when this tool started the sim, **`sim_physics_mode`** (`dynamic` \| `kinematic`), artifact paths. |
| `gt_trajectory.jsonl` | Optional: one JSON object per logged step (`--gt-trajectory`) with `robot_xyt` and `objects` for alignment experiments. |
| `_agent_workspace/` | Internal RobotAgent log prefix used during capture (can be ignored or deleted after a run). |

**CI / default tests:** heavy sim and capture are skipped in normal pytest runs. Use `emet test --no-sim` for the default fast path. Backend smoke tests for memory formats live in `src/test/memory/test_memory_backends_smoke.py` and `docs/plans/TESTING_BACKENDS.md`.

## How GT is produced

MuJoCo ZMQ servers attach **`emet_gt_objects`** (see `emet.core.zmq_protocol.EMET_ZMQ_GT_OBJECTS_KEY`) to each full observation. Entries are built in-process from `mjModel` / `mjData` via `emet.dataset.mujoco_gt.extract_gt_object_dicts` (body names matching `object*` by default, or an explicit allowlist in code).

The capture script reads the latest observation from the robot ZMQ client and writes **`ground_truth.json`**. The same list is converted to a **`GraphBlob`** and passed as `extra_graph` into `get_memory_backend("dynamem", ...).save(...)` so `dynamem/graph.json` matches simulator bodies (not YoloE / SceneGraph perception).

During capture, **`emet.dataset.sim_health.check_robot_sim_stable`** runs after agent startup, after exploration steps, and before save. If joint velocities, camera origin, or GPS/base hints look like a **physics blow-up** (robot flying away / NaNs), the tool prints a red error and **exits without writing** the episode dataset.

## Quick start examples

### Packaged table + mobile robot (`default_mujoco`)

The default table is `scene_environment.xml` merged with the robot MJCF. The server applies **`snap_packaged_table_robot_to_scene_floor`** (`emet.simulation.default_table_spawn`) so the free-flying base is placed beside the table and **above the floor** (avoids the origin-through-table / floor clip you get from a raw merge).

```bash
uv run emet dataset capture-sim-episode --output-dir ./ep_table --robot rby1
# With a visible MuJoCo window (spawned subprocess):
uv run emet dataset capture-sim-episode --output-dir ./ep_table --robot rby1 --show-sim
```

YAML defaults: `configs/sim/default_table_rby1.yaml` (or `default_table_stretch.yaml` when `--robot stretch` and no `--sim-config`).

### Robosuite / Robocasa kitchen (`--source robosuite`)

CLI **`--source robosuite`** maps to the packaged **Robocasa** sim YAML (`configs/sim/robocasa_pick_place.yaml`). The ZMQ server is still `emet.simulation.mujoco_server` with **`--use-robocasa`**; the robot runs on **`RobosuiteZmqServer`** (MuJoCo from the generated kitchen), not the small table merge above.

**Prerequisites:** sim install path (`emet install sim` / `uv sync --extra sim`), `third_party/robosuite` + `robocasa`, and kitchen assets (see `docs/simulation.md` and repo install scripts).

```bash
uv run emet dataset capture-sim-episode \
  --source robosuite \
  --robot PandaOmron \
  --output-dir ./data/kitchen_ep0

# Same recipe with an explicit YAML (edit robocasa_task / layout / style there):
uv run emet dataset capture-sim-episode \
  --sim-config configs/sim/robocasa_pick_place.yaml \
  --robot PandaOmron \
  -o ./data/kitchen_ep1
```

To change the kitchen task, copy `configs/sim/robocasa_pick_place.yaml` and set **`robocasa_task`**, **`robocasa_style`**, **`robocasa_layout`**, then pass **`--sim-config`** (or extend `--source` in `capture_sim_dataset_episode.py`).

## Adding a new scene

1. Add a sim launch YAML under `configs/sim/` using `kind: default_mujoco`, `robocasa`, or `molmospaces` (see `emet.config.sim_launch_config`).
2. Ensure objects you care about use MuJoCo **body names** matching the extractor globs (`object*` default) or extend `extract_gt_object_dicts` / server wiring to pass a scene-specific **allowlist** (library change).
3. Run capture with `--sim-config path/to/your.yaml` or add a new `--source` branch in `capture_sim_dataset_episode.py` if you want a stable short name.

## Adding a new robot

1. Start the sim with the same `--robot` as **`emet serve mujoco`** (see `emet.app.robot_cli.create_robot_client_from_cli` and `emet.robots.ROBOT_REGISTRY`).
2. The dataset script only consumes what the server publishes on the observation socket (RGB-D, intrinsics, poses) plus **`emet_gt_objects`**. New robots using `RobosuiteZmqServer` or Stretch `MujocoZmqServer` pick up GT automatically.

## Multi-camera and extrinsics

The MVP records the **primary** head RGB-D used by DynaMem, plus camera pose / `camera_K` in saved frames as today. To extend:

- Record additional camera names and extrinsics in `dataset_manifest.json` (custom fields) and optionally save per-camera RGB in a sidecar layout.
- Keep a single DynaMem map from the primary stream unless you fork the mapper for multi-rig fusion.

## Multi-episode / multi-room

Loop over episodes in your driver script:

1. Choose a unique `output_dir` per shard (e.g. `data/sim_gt/run_000/ep_012/`).
2. Optionally aggregate manifests into a top-level `index.json` listing each episode’s `dataset_manifest.json` path, robot, scene seed, and split name.

## Gated integration test

With a MuJoCo ZMQ server already listening on **127.0.0.1:4401** (default ports):

```bash
export RUN_SIM_DATASET_CAPTURE=1
uv run emet test src/test/dataset/test_capture_sim_dataset_smoke.py -q
```

The test discovers **`emet_robot_id`** from the live server so it matches Stretch vs rby1 automatically.
