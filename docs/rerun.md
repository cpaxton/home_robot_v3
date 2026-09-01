# Rerun live visualization

Emet logs maps, cameras, robot meshes, and graph overlays through [rerun-sdk](https://www.rerun.io/) (`>=0.21.0,<0.23.0`). Implementation: [`src/emet/visualization/rerun.py`](../src/emet/visualization/rerun.py) (`RerunVisualizer`, `spatial3d_view_world()`), MJCF mesh/skeleton in [`mjcf_rerun_robot.py`](../src/emet/visualization/mjcf_rerun_robot.py), YAML overlay in [`rerun_config.py`](../src/emet/config/rerun_config.py).

Default viewer URL:

```text
http://127.0.0.1:9090?url=ws://127.0.0.1:9877
```

The `?url=ws://…` query is required. Do not open `https://app.rerun.io`. Remote / SSH: [debug.md](debug.md).

## Open the viewer

| Mode | How | Ports |
|------|-----|--------|
| **Web (default)** | `rr.serve` — browser at `:9090`, stream on websocket `:9877` | 9090 HTTP, 9877 WS |
| **Native desktop** | `--rerun-native` / `RERUN_NATIVE_VIEWER=1` (needs `DISPLAY` / Wayland) | TCP `:9876` (Rerun default) |
| **Headless** | `--headless` / `RERUN_HEADLESS=1` — web server only, no auto-open browser | same as web |
| **Bind all** | `--rerun-bind` / `RERUN_BIND_ALL=1` — listen on `0.0.0.0` | same + LAN hostname URL |

**Native and web are exclusive.** Spawning the native app skips `rr.serve`; doing both would send logs only to the websocket sink and leave the native window empty. Use web *or* native, not both.

`--rerun-show-panels` un-collapses the blueprint / selection tree (default is a simplified time panel). `--rerun-debug` prints obs/servo/step counters on the ZMQ client.

## Which CLI enables Rerun

| Command | Default | Enable / disable |
|---------|---------|------------------|
| `emet run agent` | **Off** | `--rerun` |
| `emet run dynamem` | **On** | `--no-rerun` to disable. `--rerun` is a no-op compatibility alias. |
| `emet run dynagraph` / `emet run lazy-graph` | **On** | Same as dynamem (`graph_nav_cli.py`). |
| `emet run scene-graph` | **On** | Same flags as dynamem. |
| `emet run graph-eqa` | **On** | Always starts the viewer (no `--no-rerun`). |
| `emet stream` | **On** | Cameras + mesh; mapping loop if `--backend` (or remote default dynamem). |
| `emet capture --backend …` | On unless `--no-rerun` | `--rerun-hold-s` keeps the viewer open after the one-shot map step. |
| `emet show-memory PATH` | Memory playback | `memory_view=True` blueprint (frame timeline). |
| `emet show FILE.rrd` | Replay a recording | `--web` for browser. |
| `emet ovmm find` / `full` / `sweep` | **Off** | `--rerun` or `EMET_EVAL_RERUN=1`. One viewer per box. |
| `emet-habitat run-episode` / `emet habitat run-episode` | **Off** | `--rerun` or `EMET_EVAL_RERUN=1`. Paper H2H stays off unless you opt in. |

`--rerun-native` and `--headless` cannot be combined. Dynagraph/LazyGraph/EQA also have `--save_rerun` / `--SR`: `maybe_save_rerun_recording()` writes `logs/…/data_N.rrd` on rotate / navigate / EQA. If a live `RerunVisualizer` is already streaming, that helper only calls `rr.save` (it does **not** `rr.init`, which would empty the websocket). Offline (`--no-rerun`) still inits then saves. DynaMem `--output-path DIR` writes `DIR/rerun_log.rrd` from the visualizer itself.

`world/explored` (white) is the 2D nav mask: observed voxel columns **plus** a small start-pose `local_radius` disk (default 0.25 m). Dynamem stamps that disk on the **first** observation (not every rotate-in-place frame). Open/close with `filters.smooth_kernel_size` (default 3) fills 1-cell voxel speckle so `map_topdown` / A* are not full of holes; multi-meter gaps between Stretch `look_front` cones stay unexplored until those rays exist. Restore pans with `EMET_FORCE_HEAD_SWEEP=1`. Innate Mars keeps `local_radius: 0.85`.

Shared when Rerun is on: `--headless`, `--rerun-native`, `--rerun-show-panels`, `--rerun-debug`, `--rerun-bind`.

## Config (`rerun:` YAML)

Loaded from unified config (`configs/emet/default.yaml` → `package://emet/config/agents/default_rerun.yaml`) and merged in `build_rerun_visualizer_kwargs()` / `open_live_rerun_visualizer()`.

```yaml
rerun:
  headless: false
  native_viewer: false
  bind_all: false
  show_panels: false
  mjcf_show_visual_mesh: true
  mjcf_show_skeleton: false
  server_memory_limit: "4GB"
  show_camera_point_clouds: false
  max_displayed_points_per_camera: 4096
  max_map_2d_points: 25000
  voxel_map_stride: 2
  dynagraph_stride: 2
  mjcf_mesh_stride: 3
  dynagraph:
    log_crops: false
    log_edges: false
    log_summary: false
    log_gallery: false
```

**Precedence** (`resolve_rerun_bool`): CLI flag **on** wins, then YAML if the key is present, then environment, then the dataclass default. CLI flags only force options *on* (`--headless`, `--rerun-native`, `--rerun-show-panels`); omitting them does not force off.

`EMET_DYNAGRAPH_RERUN_CROPS=1` / `EMET_DYNAGRAPH_RERUN_EDGES=1` **force those two channels on** in `build_rerun_visualizer_kwargs()` even when YAML is `false`. They cannot force them off if YAML already enabled them. There is no env toggle for `log_summary` / `log_gallery`.

YAML strides/crops/mesh options apply for every live ZMQ client via `open_live_rerun_visualizer()` — **`StretchZmqClient`** (URDF mesh) and **`GenericZmqClient`** (MJCF when the robot spec has a mesh). `EMET_STRETCH_GENERIC_ZMQ=1` still routes Stretch through `GenericZmqClient`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `RERUN_HEADLESS` | Web server only, no native spawn / auto-open browser |
| `RERUN_NATIVE_VIEWER` | Native desktop viewer (if a display exists) |
| `RERUN_BIND_ALL` | Bind HTTP/WS to `0.0.0.0` (`RERUN_SERVER_HOST` / `RERUN_SERVER_WS_HOST`) |
| `EMET_RERUN_HEAD_DEPTH` | Log `world/head_camera/depth` on the live stream (off by default) |
| `EMET_DYNAGRAPH_RERUN_CROPS` | Per-node crop images + mosaic |
| `EMET_DYNAGRAPH_RERUN_EDGES` | Graph edge `LineStrips3D` |
| `EMET_EVAL_RERUN` | Opt in to live Rerun during OVMM / Habitat / other evals (default off) |
| `EMET_CONFIRM_NAV` | Gate `execute_trajectory` on y/n + 2D plan image (`world/nav/plan_map`) |

Also listed in [environment_variables.md](environment_variables.md).

## World-frame policy (required)

All **map and scene geometry** is logged in the **navigation / voxel world frame** (`navigation_origin_xyt` + fused voxel map):

| Entity path | Frame |
|-------------|--------|
| `world/point_cloud`, `world/obstacles`, `world/explored` | World |
| `world/objects`, `world/scene_graph/*`, `world/dynagraph/*` | World |
| `world/head_camera/points` | World (`get_xyz_in_world_frame()` / mapping depth) |
| `world/robot` | World pose (`gps` + `compass` + session origin via `nav_xyt_to_world_xyt`) |
| `world/robot/mjcf_visual/*` | **Base-relative** vertices; composed via `world/robot` |
| `world/robot/mesh/*` | Stretch URDF links (when no MJCF spec) |
| `world/sim_nav/spawn_origin`, `world/sim_nav/walkable_clip` | World (once per session, Robocasa / MuJoCo) |

**Do not** set the live `Spatial3DView` blueprint `origin` to `world/robot`. That re-expresses the whole scene in the robot frame, so the map and voxels **appear to rotate** when the base turns. Use `origin="world"` with `contents="world/**"` (`spatial3d_view_world()`). Default `contents` of `$origin/**` would hide map layers if origin were `world/robot`.

The robot still moves under `world/robot`; only the **view coordinate system** stays fixed to world. Optional `spatial3d_view_robot()` keeps the camera on the base but co-rotates map layers (debug only).

## Blueprints (what you actually see)

`RerunVisualizer.setup_blueprint` is a 3D world view + `head_rgb` / `ee_rgb` / `world/memory/text`. Controllers **replace** that layout:

**DynaMem / scene-graph** (`DynamemController.setup_custom_blueprint`): world 3D, `robot_monologue` + `/observation_similar_to_text`, head/EE RGB + `world/map_snapshot/topdown`, `world/scene_graph/summary`.

**Dynagraph / GraphEQA / LazyGraph** (`send_graph_memory_rerun_blueprint`): world 3D, monologue + relevant image, cameras + map, then a **graph** column: 3D graph (`origin=world/dynagraph`), always-on **Context (VLM)** (`world/dynagraph/context`), and **EQA images** mosaic (`world/dynagraph/context/mosaic`). That context panel is the debug surface for what the model actually received (SCENE_GRAPH prompt, Image 1…N, router evidence cards). `--ground-truth` labels the 3D view «Graph (ground truth)». `--compare-to-gt` stacks «Sim GT (reference)» (`world/dynagraph/ground_truth/`) in the same column. Four Horizontal columns always get `column_shares=[3,1,1,1]` so the graph/context column is not dropped.

**Saved memory** (`emet show-memory`): 3D world + `world/frames/current` + up to 16 `world/frames/{i}` + `world/memory/text`. Scrub the **frame** timeline.

2D RGB panels must bind to `world/head_camera/rgb` and `world/ee_camera/rgb`, not the parent `world/head_camera` entity (that parent also gets optional `DepthImage`, which made Rerun show a depth colormap instead of RGB).

## Entity catalog (live)

Logged from `RerunVisualizer.step` (ZMQ thread), `update_voxel_map` / `log_dynagraph_state` (controller `update()`), and nav/manip helpers.

| Path | Type | Notes |
|------|------|--------|
| `world/point_cloud` | Points3D | Voxel RGB-D cloud; capped by `max_displayed_points_per_camera`; strided by `voxel_map_stride` |
| `world/obstacles` / `world/explored` | Points3D | 2D occupancy / explored (voxel columns **plus** the small start-pose `local_radius` disk; small holes closed). Subsampled to `max_map_2d_points` (default 25000) |
| `world/semantic_memory/pointcloud` | Points3D | Instance semantic cloud (same cap via `log_custom_pointcloud`) |
| `world/map_snapshot/topdown` | Image | Cropped 2D map (blueprint `map_topdown`) |
| `world/head_camera/rgb` | Image | Every ZMQ step |
| `world/head_camera/depth` | DepthImage | Only if `EMET_RERUN_HEAD_DEPTH=1` |
| `world/head_camera/points` | Points3D | Only if `show_camera_point_clouds` (or memory view) |
| `world/ee_camera/rgb` | Image | When EE RGB is present |
| `world/ee_camera/depth` | DepthImage | When EE depth is present (not gated by `EMET_RERUN_HEAD_DEPTH`) |
| `world/robot` | Transform3D | Base in world |
| `world/ee` | Transform3D | Skipped if pose is missing or identity (Robosuite placeholder) |
| `world/xyz`, `world/map_box` | static axes / placeholder box | Logged at visualizer init |
| `world/dynagraph/bboxes` | Boxes3D | Object AABBs when `bounds_3d` / `extent_half` exist |
| `world/dynagraph/nodes` | Points3D | Object nodes **without** boxes |
| `world/dynagraph/frontiers` | Points3D | Frontier clusters |
| `world/dynagraph/viewpoints` | Points3D | LazyGraph / viewpoint nodes |
| `world/dynagraph/edges` | LineStrips3D | Off by default |
| `world/dynagraph/crops/…`, `…/crops_mosaic` | Image | Off by default; still written on `--export` |
| `world/dynagraph/summary`, `…/gallery` | TextDocument | Off by default (`log_summary` / `log_gallery`); summary truncated at 64k, gallery at 48k |
| `world/dynagraph/context` | TextDocument | **Always on** — last EQA prompt, Image-N → obs_id map, router state, graph health |
| `world/dynagraph/context/mosaic` | Image | Numbered thumbnails of the last VLM attachments |
| `world/dynagraph/ground_truth/*` | Points / boxes / text | `--compare-to-gt` / GT overlay |
| `world/objects`, `world/scene_graph/*` | Instances / open-vocab graph | Cleared while Dynagraph is the object source of truth |
| `world/graph/nodes`, `world/graph/edges` | MemoryState graph | Saved-memory playback |
| `world/nav/plan` | LineStrips3D | Green executed waypoint chunk |
| `world/nav/plan_full` | LineStrips3D | Dimmer blue full A* when longer than the exec chunk |
| `world/nav/waypoints`, `world/nav/arrows` | Points / arrows | `wp0`… |
| `world/nav/summary` | TextDocument | Localize source, path length, chunked? |
| `world/nav/plan_map` | Image | Confirm-nav 2D crop |
| `world/object`, `world/xyt_goal`, `world/robot_start_pose` | Goals | Target object / base goal / start |
| `world/direction` | Arrows3D | Legacy alias of nav arrows |
| `world/manip/ee_path`, `world/manip/targets` | Manip overlays | Pick/place |
| `robot_monologue` | TextDocument | Plan/EQA markdown + live map stats (truncated at 48k) |
| `/observation_similar_to_text` | Image | EQA “relevant image” patch |
| `task/plan` | TextDocument | Manipulation plan text |

`log_dynagraph_state` / `update_voxel_map` honor `dynagraph_stride` / `voxel_map_stride` (`force=True` after lifelong checkpoint load). MJCF triangle meshes re-upload every `mjcf_mesh_stride` steps (default 3).

## ZMQ sim observation frames

Sim servers (Stretch `mujoco_server_stretch`, Robosuite `robosuite_server`) publish:

| Field | Frame |
|-------|--------|
| `gps` / `compass` | Episode-relative to `emet_session.navigation_origin_xyt` |
| `camera_pose` | **Absolute MuJoCo world** (OpenCV cam-to-world) |

MJCF Rerun (`mjcf_rerun_robot.apply_zmq_obs_to_mujoco_data`) drives planar-base standalone MJCF from episode-relative `gps`/`compass`; `world/robot` composes to world. Tests: `src/test/simulation/test_zmq_observation_frame_contract.py`, `test_robosuite_zmq_frame_contract.py`.

Once per session, `log_sim_nav_reference_geometry` draws the walkable clip + spawn origin so you can pan the 3D view to the kitchen (Robocasa has no scene mesh in Rerun).

## Motion plans

When the agent plans a path (find / navigate / explore), Rerun logs under `world/nav/` (see catalog). DynaMem still executes at most **8 waypoints per chunk**; if the A* path is longer you will see `(chunk)` in Discord/terminal and another plan after the next look-around. That is planner execution, not lifelong memory restore. NaN finish-marker rows are skipped (`finite_nav_waypoints`). Plan rows are planar `xyt`; yaw is not used as height, so `wp*` arrows stay on the floor (object height is `world/object`).

With `emet run agent --confirm-nav` (or `EMET_CONFIRM_NAV=1`), each plan also posts a 2D crop to Discord / `world/nav/plan_map` and waits for **y/n** before `execute_trajectory`. Scripted `-c` auto-accepts.

Mapping / empty-text explore used to drop leftover and pick a new frontier; it now **resumes the same goal until arrival** (look-at yaw on the last hop). Agentic `investigate` / `navigate_to_target_pose` hop the same way instead of capturing from the 8-wp midpoint.

## Load / stability

Defaults in `rerun:` YAML (see above):

- **Off**: per-frame head RGB-D point cloud (`show_camera_point_clouds: false`) — use voxel `world/point_cloud` instead.
- **Strided**: voxel map, dynagraph 3D panel, MJCF mesh re-upload.
- **Always on (cheap):** `world/dynagraph/context` markdown + Image-N mosaic after an EQA/router call.
- **Off by default**: dynagraph crops, edges, full tree `summary`, and gallery markdown.

Increase strides or set `mjcf_show_visual_mesh: false` if the viewer still OOMs/crashes. Export (`--export DIR`) still writes crops and `gallery.md` on disk when live Rerun does not.

## Troubleshooting `capacity overflow` (Rerun viewer panic)

Likely causes in this stack:

| Source | Risk | Mitigation |
|--------|------|------------|
| **`world/obstacles` / `world/explored`** | Internal 2D grid is up to **1024×1024**; logging every occupied cell can be **hundreds of thousands** of points per update. | `rerun.max_map_2d_points` (default 25000, uniform subsample). |
| **`world/point_cloud`** | Full voxel cloud | `max_displayed_points_per_camera` + `voxel_map_stride`. |
| **`world/semantic_memory/pointcloud`** | Instance semantic cloud | Subsampled in `log_custom_pointcloud`. |
| **`world/<instance>_…` per-object clouds** | One dense cloud per detection | Subsampled in `update_scene_graph`. |
| **Head / EE camera clouds** | H×W points every ZMQ step | Off by default (`show_camera_point_clouds: false`). |
| **MJCF mesh re-upload** | Full mesh buffers each step | `mjcf_mesh_stride` (default 3); or `mjcf_show_visual_mesh: false`. |
| **`robot_monologue` / dynagraph gallery / context** | Long markdown | Truncated at 48k chars (EQA prompt slice 12k). |
| **Dynagraph `summary` (`print_memory`)** | Full graph tree | Off by default (`log_summary: false`); 64k cap if enabled. Use `--export` or stdout. |

Headless / bind / empty viewer: [debug.md](debug.md). World-frame “map spins with the robot”: this page, World-frame policy.

## Eval (OVMM / Habitat / GraphEQA)

Live Rerun is **off** on paper evals so a batch does not fight the operator viewer or extra GPU/CPU load. Opt in with **`EMET_EVAL_RERUN=1`** or **`--rerun`**:

```bash
uv run emet ovmm find \
  --episodes configs/ovmm/find_phase_episodes.yaml \
  --tier S0 --episode-id default_table_s0 \
  --backend dynagraph --rerun
# Habitat uses .venv-habitat (no flash-attn wheel). Allow SDPA, or pass --mock-llm to skip Qwen.
EMET_ALLOW_SDPA_ATTN=1 uv run emet habitat run-episode --question-id 11 --method dynagraph --rerun
```

That attaches `open_live_rerun_visualizer()` even when there is no ZMQ `_rerun` thread (Habitat). The same **Context (VLM)** column is used as live `emet run dynagraph`. Head RGB is logged from `update()` when the robot has no ZMQ Rerun thread. `--rerun` does **not** dump `.rrd` files (that is still `--save_rerun` / `--SR`). Overnight H2H / `--via-jobs` should leave this **off** unless you are sitting at the viewer.

`NullVisualizer` lives in `emet.visualization.null_visualizer` so CPU tests never import `rerun-sdk` native extensions.

## Tests

- `src/test/config/test_rerun_config.py` — YAML overlay, kwargs vs `RerunVisualizer` signature (AST, no native import), env crop force-on, `EMET_EVAL_RERUN`
- `src/test/controller/test_save_rerun_recording.py` — `--save_rerun` does not `rr.init` while live
- `src/test/visualization/test_nav_plan_rerun.py` — waypoint NaN skip / path length
- `src/test/visualization/test_dynagraph_rerun.py` — crop paths, gallery markdown, labels (imports `emet.visualization.rerun` / rerun-sdk; pin to CPUs `0-7,12-31`)
- `src/test/visualization/test_dynagraph_context.py` — VLM-context markdown / mosaic (mock-only, no `rr.init`)
- `src/test/visualization/test_mjcf_rerun_robot.py` — MJCF mesh / planar base
- `src/test/simulation/test_zmq_observation_frame_contract.py` — gps vs `camera_pose` frames

See also [dynagraph.md](dynagraph.md) (graph channels + export), [cli.md](cli.md) (`emet stream` / `emet show` / `emet ovmm`), [AGENT_RUN.md](AGENT_RUN.md) (agent `--rerun` / `--confirm-nav`).
