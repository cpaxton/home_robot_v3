# Rerun live visualization

## World-frame policy (required)

All **map and scene geometry** must be logged in the **navigation / voxel world frame** (same as `navigation_origin_xyt` + fused voxel map):

| Entity path | Frame |
|-------------|--------|
| `world/point_cloud`, `world/obstacles`, `world/explored` | World |
| `world/objects`, `world/scene_graph/*`, `world/dynagraph/*` | World |
| `world/head_camera/points` | World (from `get_xyz_in_world_frame()`) |
| `world/robot` | World pose (`gps` + `compass` + session origin) |
| `world/robot/mjcf_visual/*` | **Base-relative** vertices; composed via `world/robot` transform |

**Do not** set the live `Spatial3DView` blueprint `origin` to `world/robot`. That re-expresses the whole scene in the robot frame, so the map and voxels **appear to rotate** when the base turns. Use `origin="world"` with `contents="world/**"` (see `spatial3d_view_world()` in `src/emet/visualization/rerun.py`).

The robot still moves under `world/robot`; only the **view coordinate system** stays fixed to world. Optional `spatial3d_view_robot()` keeps the camera on the base but co-rotates map layers (debug only).

## ZMQ sim observation frames

Sim servers (Stretch `mujoco_server_stretch`, Robosuite `robosuite_server`) publish:

| Field | Frame |
|-------|--------|
| `gps` / `compass` | Episode-relative to `emet_session.navigation_origin_xyt` |
| `camera_pose` | **Absolute MuJoCo world** (OpenCV cam-to-world) |

MJCF Rerun replay (`mjcf_rerun_robot.apply_zmq_obs_to_mujoco_data`) drives planar-base standalone MJCF from episode-relative `gps`/`compass`; `world/robot` composes to world. Regression tests: `src/test/simulation/test_zmq_observation_frame_contract.py`, `test_robosuite_zmq_frame_contract.py`.

## Load / stability

Defaults in `rerun:` YAML (`src/emet/config/agents/default_rerun.yaml`):

- **Off**: per-frame head RGB-D point cloud (`show_camera_point_clouds: false`) — use voxel `world/point_cloud` instead.
- **Strided**: voxel map, dynagraph panel, MJCF mesh re-upload (`voxel_map_stride`, `dynagraph_stride`, `mjcf_mesh_stride`).
- **Off by default**: dynagraph crop images and edge lines (`dynagraph.log_crops`, `dynagraph.log_edges`).

Increase strides or disable MJCF mesh if the viewer still OOMs/crashes.

## Motion plans

When the agent plans a path (find / navigate / explore), Rerun logs under ``world/nav/``:

| Entity | Content |
|--------|---------|
| ``world/nav/plan`` | Green ``LineStrips3D`` polyline of the **executed** waypoint chunk |
| ``world/nav/plan_full`` | Dimmer blue full A* path when longer than the 8-waypoint exec chunk |
| ``world/nav/waypoints`` | Waypoint dots (``wp0`` …) |
| ``world/nav/arrows`` | Segment directions |
| ``world/nav/summary`` | Markdown: localize source (graph / voxel / frontier / …), path length, chunked? |
| ``world/object`` / ``world/xyt_goal`` | Target object + base goal pose |

DynaMem still executes at most **8 waypoints per chunk**. Mapping / empty-text
explore used to drop leftover and pick a new frontier; it now **resumes the same
goal until arrival** (look-at yaw on the last hop). Agentic ``investigate`` /
``navigate_to_target_pose`` hop the same way instead of capturing from the 8-wp
midpoint. You will still see ``(chunk)`` in Discord/terminal on long A* paths.

With ``emet run agent --confirm-nav`` (or ``EMET_CONFIRM_NAV=1``), each plan also posts a 2D crop to Discord / ``world/nav/plan_map`` and waits for **y/n** before ``execute_trajectory``.

## 2D camera panels

Blueprint ``head_rgb`` / ``ee_rgb`` panels use origin ``world/head_camera/rgb`` and ``world/ee_camera/rgb`` (not the parent ``world/head_camera`` entity). The parent path also receives optional ``DepthImage`` streams; binding the panel to the parent made Rerun show depth colormap instead of RGB. Live head depth is off by default; set ``EMET_RERUN_HEAD_DEPTH=1`` to log ``world/head_camera/depth``.

## Troubleshooting `capacity overflow` (Rerun viewer panic)

Likely causes in this stack (besides the full graph tree `TextDocument`, now off by default):

| Source | Risk | Mitigation |
|--------|------|------------|
| **`world/obstacles` / `world/explored`** | Internal 2D grid is up to **1024×1024**; logging every occupied cell can be **hundreds of thousands** of points per update. | `rerun.max_map_2d_points` (default 25000, uniform subsample). |
| **`world/point_cloud`** | Full voxel cloud | Capped by `max_displayed_points_per_camera` + `voxel_map_stride`. |
| **`world/semantic_memory/pointcloud`** | Uncapped instance semantic cloud | Now subsampled in `log_custom_pointcloud`. |
| **`world/<instance>_…` per-object clouds** | One dense cloud per detection | Subsampled in `update_scene_graph`. |
| **Head / EE camera clouds** | H×W points every ZMQ step | Off by default (`show_camera_point_clouds: false`). |
| **MJCF mesh re-upload** | Full mesh buffers each step | `mjcf_mesh_stride` (default 3). |
| **`robot_monologue` text** | Long EQA answers | Truncated at 48k chars. |
| **Dynagraph gallery markdown** | Large `TextDocument` | Off by default (`dynagraph.log_gallery: false`). |

## CLI (dynagraph / dynamem)

Rerun is **on by default**. Use `--no-rerun` to disable. See `.cursorrules` for the full flag table vs `emet run agent`.
