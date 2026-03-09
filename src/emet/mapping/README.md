# Mapping

Spatial representation for the robot: grid, instances, scene graph, and voxel maps.

- **grid** – 2D grid params (resolution, origin, size). Used by voxel and planners.
- **instance** – Object instances (point cloud, bounds, views). Used by voxel when `use_instance_memory=True`, and by scene graph / controllers.
- **scene_graph** – Nodes = instances; edges = spatial relations (near, on). Built from instances.
- **voxel** – 2D/3D voxel maps (base + Dynamem variant). Optionally use instance memory.

Semantic / EQA **memory** models live under `emet.memory` (DynaMem re-export, GraphEQA). See [docs/plans/MAPPING_REFACTOR.md](../../../docs/plans/MAPPING_REFACTOR.md) for layout and imports.
