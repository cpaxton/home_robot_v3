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

The default live `Spatial3DView` uses `origin="world/robot"` with `contents="world/**"` (`spatial3d_view_robot()` in `src/emet/visualization/rerun.py`) so the 3D camera stays centered on the base. Map and voxel layers remain visible; they **co-rotate** with the robot when the base turns in place.

For a fixed world-frame camera (map stays put while the robot moves), use `spatial3d_view_world()` (`origin="world"`, same `contents`).

## Load / stability

Defaults in `rerun:` YAML (`src/emet/config/agents/default_rerun.yaml`):

- **Off**: per-frame head RGB-D point cloud (`show_camera_point_clouds: false`) — use voxel `world/point_cloud` instead.
- **Strided**: voxel map, dynagraph panel, MJCF mesh re-upload (`voxel_map_stride`, `dynagraph_stride`, `mjcf_mesh_stride`).
- **Off by default**: dynagraph crop images and edge lines (`dynagraph.log_crops`, `dynagraph.log_edges`).

Increase strides or disable MJCF mesh if the viewer still OOMs/crashes.

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
