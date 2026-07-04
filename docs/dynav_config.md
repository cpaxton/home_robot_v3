# DynaMem / mapping configuration (`mapping` section)

DynaMem, GraphEQA, Dynagraph, and `emet run agent` load navigation and mapping parameters from the unified config **`mapping:`** section. Default: [`configs/emet/default.yaml`](../configs/emet/default.yaml) (see [Unified EMET configuration](emet_config.md)).

Legacy flat files still work:

```bash
emet run dynamem --config configs/emet/default.yaml
emet run agent --config configs/agent_innate_mars.yaml
# Deprecated aliases: --dynav-config, --agent-config
```

Basenames like `dynav_config.yaml` resolve under `src/emet/config/` and auto-wrap under `mapping:`.

Innate Mars depth tuning is in **`robots.innate_mars.mapping`** (default config); [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) is a thin `extends:` alias. Sim with ZMQ depth should use **`depth_source: sensor`** (stretch default). See [Innate Mars / sim depth](robots/innate_mars.md).

Related docs: [Unified config](emet_config.md), [Dynamem](dynamem.md), [Dynagraph](dynagraph.md), [Simulation configs](sim_configs.md), [Agent run](AGENT_RUN.md).

---

## File layout (overview)

| Section | Purpose |
|--------|---------|
| `voxel_size`, `obs_*`, `pad_obstacles` | 2D/3D voxel map resolution, height bands, obstacle density |
| `map_boundary` | Optional grid-edge obstacle ring on the Dynamem 2D map |
| `depth_source`, `da3_*` | Sensor vs Depth Anything 3 for mapping |
| `detection`, `instance_memory` | Open-vocab detection and instance memory |
| `filters` | Depth / map smoothing (median, derivative, speckle open, voxel DBSCAN) |
| `motion_planner` | A* step size, frontier dilation, goal radii |
| `eqa`, `eqa_vl`, `graph_eqa_*` | EQA and GraphEQA VLM settings |
| `use_instance_memory`, `use_scene_graph` | Rerun instance boxes and scene graph |

The canonical source of truth is [`configs/emet/default.yaml`](../configs/emet/default.yaml) (nested `mapping:`) with per-robot overlays under `robots.<id>.mapping`. [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) is a thin `extends:` alias only.

---

## `map_boundary` — grid-edge barrier

Dynamem’s [`SparseVoxelMapDynamem.get_2d_map()`](../src/emet/mapping/voxel/voxel_dynamem.py) can mark a band of cells along the **fixed grid border** as obstacles so the planner does not drive the robot off the allocated map. That band also appeared as a red frame in the Rerun **`map_topdown`** view (`world/map_snapshot/topdown`).

**Default (current):** barrier **off** — no artificial edge obstacles.

```yaml
map_boundary:
  obstacle_barrier_cells: 0
  history_penalty_cells: 0
```

| Key | Default | Meaning |
|-----|---------|---------|
| `obstacle_barrier_cells` | `0` | Width (in grid cells) of obstacle strips on all four map edges. Affects navigation and top-down visualization. |
| `history_penalty_cells` | `0` | Width of grid-edge band where exploration **history** is maxed (discourages lingering at the border). Used when `get_2d_map(return_history_id=True)`. |

**Restore legacy behavior** (hardcoded pre-2026 values were 30 / 35 cells):

```yaml
map_boundary:
  obstacle_barrier_cells: 30
  history_penalty_cells: 35
```

At `grid_resolution: 0.1` m/cell, 30 cells ≈ 3 m from each edge. Values are clamped to `min(cells, height//2, width//2)` so small maps do not break.

**Note:** `map_boundary` is separate from **`pad_obstacles`**, which morphologically dilates real obstacles detected from depth—not the map grid rim.

---

## Top-down map in Rerun and Discord

The 2D snapshot pipeline lives in [`src/emet/visualization/map_snapshot.py`](../src/emet/visualization/map_snapshot.py).

- **Rerun** blueprint view `map_topdown` (Dynamem / Dynagraph controllers) logs to `world/map_snapshot/topdown`.
- **`send_map_snapshot`** agent tool posts the same style of image to Discord when configured.

Both paths use **`share_topdown_map_rgb`**: render obstacles vs explored, **crop to the explored region** (plus margin and a small neighborhood around the robot), then downsample (`max_side` default 640). Unexplored dark padding outside the crop is not shown.

Live updates come from [`RerunVisualizer.update_voxel_map`](../src/emet/visualization/rerun.py) on each Dynamem map refresh.

---

## Commonly tuned keys

### Voxel map and obstacles

```yaml
voxel_size: 0.1
obs_min_height: 0.2
obs_max_height: 1.5
obs_min_density: 5
pad_obstacles: 2          # dilation radius around detected obstacles (grid cells)
min_pad_obstacles: 1
local_radius: 0.5         # disk marked explored around the robot
```

### Depth / voxel post-filters (DA3 hardware, opt-in)

Optional cleanup for **DA3-inferred** depth on Innate Mars (and any stack with ``depth_source: da3`` / ``auto`` fallback to DA3). **Defaults are off** (``0``) in [`configs/emet/default.yaml`](../configs/emet/default.yaml) — aggressive values can erode thin real structure (chair legs, door frames) or make walls look worse.

| Layer | Key | Default | Applies when |
|-------|-----|---------|--------------|
| Pre-unprojection | `depth_speckle_open_kernel` | `0` | DA3/LingBot **inferred** depth only (not raw ZMQ sensor depth) |
| Pre-unprojection | `depth_speckle_open_iterations` | `1` | Same as speckle kernel; ignored when kernel is ``0`` |
| Post-fusion PCD | `voxel_pcd_dbscan_min_samples` | `0` | **Any** depth source, each mapping frame (``0`` = off; eps ≈ ``4 × voxel_size``) |

**Where:** under ``robots.innate_mars.mapping.filters`` (deep-merged into ``mapping.filters`` at runtime). Workstation stream/dynamem/dynagraph read these via the unified config.

**Runtime:** Speckle open runs in [`DynamemController.update()`](../src/emet/controller/controller_dynamem.py) only when depth came from DA3/LingBot inference (``_depth_map_from_da3_infer``; skipped for raw sensor depth and ``depth_source: auto`` when usable sensor depth is present). Voxel PCD DBSCAN runs in [`SparseVoxelMap.add_observation()`](../src/emet/mapping/voxel/voxel_dynamem.py) whenever ``voxel_pcd_dbscan_min_samples > 0``, regardless of depth source.

**Enable example** (tune per site — try one knob at a time):

```yaml
# configs/emet/default.yaml (or --set overrides)
robots:
  innate_mars:
    mapping:
      filters:
        depth_speckle_open_kernel: 3
        depth_speckle_open_iterations: 1
        voxel_pcd_dbscan_min_samples: 8
```

CLI without editing files:

```bash
emet stream --connection herman --backend dynagraph \
  --set mapping.filters.depth_speckle_open_kernel=3 \
  --set mapping.filters.voxel_pcd_dbscan_min_samples=8
```

**Onboard Jetson DA3** (when depth is computed on the robot): same speckle helper in [`onboard_da3.py`](../src/innate_mars_bridge/innate_mars_bridge/onboard_da3.py); env vars in [environment_variables.md](environment_variables.md). Redeploy bridge after changing onboard defaults.

**Symptom guide:** Floating mid-air blobs in Rerun ``world/point_cloud`` → try speckle open and/or DBSCAN. **Curved / bowed flat walls** with normal RGB → usually DA3 metric quality, lighting, or intrinsics — not fixed by these filters; see [innate_mars.md](robots/innate_mars.md) and ``emet debug-da3-depth``.

### Depth source

```yaml
depth_source: sensor      # sim default: ZMQ depth
# depth_source: da3       # Depth Anything 3 from RGB (see dynav_innate_mars.yaml)
# depth_source: auto      # sensor if present, else DA3
```

### Motion planner / exploration frontier

```yaml
motion_planner:
  frontier:
    dilate_frontier_size: 2
    dilate_obstacle_size: 0
```

### Instance memory and Rerun

```yaml
use_instance_memory: True
use_scene_graph: True
detection:
  confidence_threshold: 0.02
```

---

## Other planner configs

Pick-and-place / instance-memory stacks that do **not** use Dynamem’s voxel class may load separate YAML files, for example:

- `default_planner.yaml`, `sim_planner.yaml`, `a_star_planner.yaml` under `src/emet/config/`

Those files use the same `pad_obstacles` naming but **do not** include `map_boundary` (grid-edge barrier is Dynamem-specific).

---

## See also

- [Dynamem](dynamem.md) — commands, calibration, troubleshooting
- [Dynagraph](dynagraph.md) — `dynagraph_*` keys in the same YAML
- [Graph EQA](graph_eqa.md)
- [Discord bot](discord_bot.md) — `send_map_snapshot` from the agent
