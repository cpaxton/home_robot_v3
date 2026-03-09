# Mapping Module Refactor and Unification

This document describes the **mapping** package layout and how it fits with **memory** and shared UI.

## Current layout (after refactor)

```
emet.mapping
├── __init__.py       # Public API: grid, instance, scene_graph, voxel
├── grid/             # 2D grid params and utilities (GridParams)
├── instance/         # Object instances: Instance, InstanceView, InstanceMemory
├── scene_graph/      # Scene graph over instances (SceneGraph)
└── voxel/            # Voxel maps: base + Dynamem (SparseVoxelMap, navigation space)
```

**Memory models** (semantic / EQA) live under **`emet.memory`** for consistency:

- **`emet.memory.dynamem`** – re-exports DynaMem (voxel) types from `emet.mapping.voxel`.
- **`emet.memory.graph_eqa`** – graph-based EQA memory (`GraphEQAMemory`).

So: **mapping** = spatial representation (grid, voxel, instances, scene graph). **memory** = semantic/EQA backends (DynaMem voxel, GraphEQA graph).

## Shared concepts

- **Grid**: 2D occupancy / exploration grid (resolution, origin, size). Used by voxel and planners.
- **Instance**: A detected object with point cloud, bounds, and views (cropped image, pose, score). Used by voxel (when `use_instance_memory=True`), scene graph, and controllers.
- **Scene graph**: Nodes = instances; edges = spatial relations (near, on). Built from instances; used by operations and Rerun.
- **Voxel map**: 2D/3D voxelized point cloud + optional instance memory. Base class in `voxel/voxel.py`; Dynamem variant in `voxel/voxel_dynamem.py`.

## Instance module

`emet.mapping.instance` provides:

- **Instance**: `global_id`, `id`, `point_cloud`, `point_cloud_rgb`, `bounds`, `category_id`, `get_best_view()`, `show_best_view()`.
- **InstanceView**: `bounds`, `cropped_image`, `mask`, `score`, `get_pose()`, `get_image()`.
- **InstanceMemory**: per-env dict of instances; `reset()`, `process_instances_for_env(...)`, `associate_instances_to_memory()`, `global_box_compression_and_nms()`, `get_instances_by_class()`, `pop_global_instance()`.

When **`use_instance_memory=False`** (default for Dynamem), the voxel map sets `self.instances = None` and never uses instance processing. When **`use_instance_memory=True`**, the voxel map creates `InstanceMemory` and calls `process_instances_for_env` / `associate_instances_to_memory` on each observation. The implementation in this repo is **minimal**: it satisfies the interface so imports and code paths run; `process_instances_for_env` is a no-op so no instances are created unless you plug in a full implementation (e.g. from home_robot or custom).

## Common UI / visualization

- **Rerun**: `emet.visualization.rerun` logs voxel map, scene graph, and instances. All mapping-backed views go through this.
- **Config**: Grid and voxel params come from `emet.config` (e.g. `dynav_config.yaml`, `sim_planner.yaml`). Instance memory and scene graph options live in the same configs under `use_scene_graph`, `scene_graph`, `use_instance_memory`, `instance_memory_kwargs`.

Unifying "common UI" means: one Rerun blueprint for mapping (voxel + instances + scene graph), and one place for config keys that affect mapping and memory.

## Imports

Prefer importing from the top-level package:

```python
from emet.mapping import GridParams
from emet.mapping import Instance, InstanceView, InstanceMemory
from emet.mapping import SceneGraph
from emet.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from emet.mapping.voxel import SparseVoxelMapDynamem, SparseVoxelMapNavigationSpaceDynamem
```

Memory models:

```python
from emet.memory import GraphEQAMemory
from emet.memory import SparseVoxelMapDynamem, SparseVoxelMapNavigationSpaceDynamem  # lazy
```
