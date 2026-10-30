# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded 3D free/occupied/unknown evidence from registered sensor depth.

Free carving assumes each calibrated pinhole pixel covers its pixel footprint,
with the supplied depth-error margin. This is a stationary-camera geometric
snapshot, not sensor-time synchronization or a complete object reconstruction.
No semantic crop removes robot/target/obstacle points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emet.motion.placement_geometry import CORNERS, PlacementScene, bounds_array


@dataclass(frozen=True)
class ObservedWorkspace:
    origin: np.ndarray
    resolution: float
    cells: np.ndarray  # 0 unknown, 1 free, 2 occupied

    def __post_init__(self):
        origin = np.asarray(self.origin, dtype=float)
        cells = np.asarray(self.cells)
        if (origin.shape != (3,) or not np.isfinite(origin).all()
                or not np.isfinite(self.resolution) or self.resolution <= 0
                or cells.ndim != 3 or min(cells.shape) <= 0 or not np.isin(cells, [0, 1, 2]).all()):
            raise ValueError('invalid_observed_workspace')
        object.__setattr__(self, 'origin', np.frombuffer(origin.tobytes(), dtype=origin.dtype))
        object.__setattr__(self, 'cells', np.frombuffer(cells.astype(np.uint8).tobytes(), dtype=np.uint8).reshape(cells.shape))

    @classmethod
    def from_depth(cls, depth, intrinsic, world_from_camera, *, workspace, resolution,
                   depth_margin_m=.02, min_depth_m=.1, max_depth_m=5., max_cells=250_000):
        """Carve a cell only when its *entire* projection is in front of valid depth.

        Every pixel in the projected corner bounding rectangle must return a
        surface beyond the farthest cell corner plus the error margin. Missing
        pixels, clipping, camera-plane crossings and occlusion remain unknown.
        Workspace bounds must lie on this grid; no partial edge cells are added.
        Inputs use the same optical XYZ convention as ``Camera.depth_to_xyz``.
        """
        depth, intrinsic, pose = np.asarray(depth, float), np.asarray(intrinsic, float), np.asarray(world_from_camera, float)
        bounds = bounds_array(workspace)
        if (depth.ndim != 2 or not depth.size or intrinsic.shape != (3, 3) or pose.shape != (4, 4)
                or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1]) or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0)
                or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1.)
                or not np.isfinite(resolution) or resolution <= 0
                or not np.isfinite(depth_margin_m) or depth_margin_m < 0
                or not np.isfinite([min_depth_m, max_depth_m]).all() or not 0 < min_depth_m < max_depth_m):
            raise ValueError('invalid_observed_depth_geometry')
        # Out-of-range values are unknown, including a finite overflow/sentinel.
        depth = np.where(np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m), depth, 0.)
        count = (bounds[1] - bounds[0]) / resolution
        if np.any(count < 1) or not np.allclose(count, np.round(count), atol=1e-7, rtol=0):
            raise ValueError('workspace_must_align_to_grid')
        if np.prod(count) > max_cells:
            raise ValueError('observed_workspace_budget_exceeded')
        shape = np.round(count).astype(int)
        ids = np.indices(shape).reshape(3, -1).T
        centers = bounds[0] + (ids + .5) * resolution
        corners = centers[:, None, :] + CORNERS[None] * resolution / 2
        camera = (corners - pose[:3, 3]) @ pose[:3, :3]
        in_front = np.all(camera[..., 2] > 0, axis=1)
        projected = camera @ intrinsic.T
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = projected[..., :2] / projected[..., 2:]
        cells = np.zeros(tuple(shape), dtype=np.uint8)
        height, width = depth.shape
        for index in np.flatnonzero(in_front):
            lo, hi = uv[index].min(0), uv[index].max(0)
            if np.any(lo < -.5) or hi[0] >= width - .5 or hi[1] >= height - .5:
                continue
            low, high = np.floor(lo + .5).astype(int), np.floor(hi + .5).astype(int)
            patch = depth[low[1]:high[1] + 1, low[0]:high[0] + 1]
            if np.isfinite(patch).all() and np.all(patch > camera[index, :, 2].max() + depth_margin_m):
                cells[tuple(ids[index])] = 1
        # Endpoint cells are occupied; a margin-neighbourhood remains unknown
        # because free carving already stops before each return by that margin.
        y, x = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > 0)
        z = depth[valid]
        camera_points = np.column_stack(((x[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0],
                                         (y[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
        world = camera_points @ pose[:3, :3].T + pose[:3, 3]
        inside = np.all((world >= bounds[0]) & (world < bounds[1]), axis=1)
        endpoints = np.floor((world[inside] - bounds[0]) / resolution).astype(int)
        cells[tuple(endpoints.T)] = 2
        return cls(bounds[0], resolution, cells)

    @classmethod
    def fuse_static(cls, views):
        """Combine aligned views of a static scene; occupied evidence dominates.

        This does not renew capture times or certify that a scene stayed static.
        Callers must discard the scan on scene changes, not fuse moving objects
        into a supposedly fresh execution snapshot.
        """
        views = tuple(views)
        if not views:
            raise ValueError('missing_observed_views')
        first = views[0]
        if any(v.cells.shape != first.cells.shape or v.resolution != first.resolution
               or not np.array_equal(v.origin, first.origin) for v in views):
            raise ValueError('observed_grid_alignment_mismatch')
        cells = np.zeros_like(first.cells)
        for view in views:
            cells[(view.cells == 1) & (cells != 2)] = 1
            cells[view.cells == 2] = 2
        return cls(first.origin, first.resolution, cells)

    def partition(self, points_world, labels):
        """Attribute observed occupied cells without erasing unknown space.

        Provide all measured endpoint labels from this snapshot. Unattributed
        occupied cells remain background. Shared target/obstacle cells retain
        both owners; excluding a target never removes their obstacle evidence.
        """
        points, labels = np.asarray(points_world, float), np.asarray(labels)
        if (points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),)
                or not np.isfinite(points).all() or labels.dtype.kind not in {'U', 'S'}
                or np.any(labels == '')):
            raise ValueError('invalid_occupancy_provenance')
        labels = labels.astype(str)
        upper = self.origin + np.array(self.cells.shape)*self.resolution
        inside = np.all((points >= self.origin) & (points < upper), axis=1)
        points, labels = points[inside], labels[inside]
        indices = np.floor((points-self.origin)/self.resolution).astype(int)
        ids = np.ravel_multi_index(indices.T, self.cells.shape)
        occupied = self.cells.ravel() == 2
        owners = {str(label): np.unique(ids[(labels == label) & occupied[ids]]) for label in np.unique(labels)}
        attributed = np.zeros(self.cells.size, bool)
        for values in owners.values():
            attributed[values] = True
        missing = np.flatnonzero(occupied & ~attributed)
        owners['background'] = np.union1d(owners.get('background', []), missing).astype(int)
        return ObservedOccupancyPartition(self, tuple((name, tuple(map(int, values))) for name, values in sorted(owners.items())))

    def query_region(self, bounds):
        """Slices for all cells touching a closed query, or None outside the grid."""
        lo, hi = bounds_array(bounds)
        upper = self.origin + np.array(self.cells.shape) * self.resolution
        if np.any(lo < self.origin) or np.any(hi > upper):
            return None
        # Include both neighbours at an exact internal boundary. This errs toward
        # rejecting paths touching an unknown/occupied cell rather than skipping it.
        low = np.ceil((lo - self.origin) / self.resolution).astype(int) - 1
        high = np.floor((hi - self.origin) / self.resolution).astype(int)
        low, high = np.maximum(low, 0), np.minimum(high, np.array(self.cells.shape) - 1)
        if np.any(low > high):
            return None  # Floating-point boundary arithmetic may yield no cells.
        return tuple(slice(a, b+1) for a, b in zip(low, high, strict=True))

    def known_free(self, bounds):
        """Require every cell intersecting the closed query volume to be free."""
        region = self.query_region(bounds)
        return region is not None and bool(np.all(self.cells[region] == 1))

    def placement_scene(self):
        ids = np.argwhere(self.cells == 2)
        centers = self.origin + (ids + .5) * self.resolution
        return PlacementScene.from_voxels(centers, resolution=self.resolution,
            workspace=[self.origin, self.origin + np.array(self.cells.shape) * self.resolution],
            known_free=self.known_free)


@dataclass(frozen=True)
class ObservedOccupancyPartition:
    workspace: ObservedWorkspace
    owners: tuple[tuple[str, tuple[int, ...]], ...]

    def excluding(self, object_refs):
        """Remove only exclusive endpoints; the vacated cell becomes UNKNOWN.

        This is not a contact exception or proof of free space inside a shape.
        New measured rays must establish free space after an object moves.
        """
        excluded = frozenset(object_refs)
        if 'background' in excluded:
            raise ValueError('background_occupancy_cannot_be_excluded')
        keep = np.zeros(self.workspace.cells.size, bool)
        for name, ids in self.owners:
            if name not in excluded:
                keep[list(ids)] = True
        cells = self.workspace.cells.copy().reshape(-1)
        cells[(cells == 2) & ~keep] = 0
        return ObservedWorkspace(self.workspace.origin, self.workspace.resolution, cells.reshape(self.workspace.cells.shape))
