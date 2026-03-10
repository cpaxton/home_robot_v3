# Mapping

Spatial representation for the robot: grid, instances, scene graph, and voxel maps.

- **grid** – 2D grid params (resolution, origin, size). Used by voxel and planners.
- **instance** – Object instances (point cloud, bounds, views). Used by voxel when `use_instance_memory=True`, and by scene graph / controllers.
- **scene_graph** – Nodes = instances; edges = spatial relations (near, on). Built from instances.
- **voxel** – 2D/3D voxel maps (base + Dynamem variant). Optionally use instance memory.

There are **three memory models**: **sparse voxel map** (this package, base map), **DynaMem** (voxel + VL + EQA, re-exported in `emet.memory.dynamem`), and **Graph EQA** (`emet.memory.graph_eqa`). See [docs/plans/MAPPING_REFACTOR.md](../../../docs/plans/MAPPING_REFACTOR.md) for layout and imports.
