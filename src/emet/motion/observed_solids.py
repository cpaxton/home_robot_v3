# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Solid geometry for observed-space queries; bounding boxes are broad phase only.

A convex material declaration is an external calibration assumption, not a
conversion of a hollow CAD surface into solid material. Mesh ownership is never
inferred merely because MuJoCo exposes an enclosing collision AABB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from emet.controller.manipulation.geometry_models import PrimitiveModel, immutable
from emet.motion.placement_geometry import box_corners


@lru_cache(maxsize=512)
def _validated_convex_geometry(shape, vertex_bytes):
    vertices = np.frombuffer(vertex_bytes, dtype=float).reshape(shape)
    if not np.isfinite(vertices).all() or np.linalg.matrix_rank(vertices - vertices.mean(0)) < 3:
        raise ValueError('convex_material_calibration_required')
    return vertices, immutable(ConvexHull(vertices).equations)


@dataclass(frozen=True)
class WorldSolid:
    kind: str
    dimensions: tuple[float, ...]
    world_from_local: np.ndarray
    offset_m: float = 0.0
    vertices: np.ndarray | None = field(default=None, repr=False, compare=False)
    material_calibration_id: str | None = None
    _planes: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        pose = np.asarray(self.world_from_local, float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1.)
                or not np.isfinite(self.offset_m)):
            raise ValueError('invalid_observed_solid')
        dimensions = tuple(map(float, self.dimensions))
        if self.kind == 'convex':
            vertices = np.asarray(self.vertices, float)
            if (not self.material_calibration_id or dimensions or vertices.ndim != 2
                    or vertices.shape[1:] != (3,) or len(vertices) < 4):
                raise ValueError('convex_material_calibration_required')
            vertices, planes = _validated_convex_geometry(vertices.shape, vertices.tobytes())
            object.__setattr__(self, 'vertices', vertices)
            object.__setattr__(self, '_planes', planes)
        elif self.kind == 'capsule':
            if len(dimensions) != 2 or not np.isfinite(dimensions).all() or min(dimensions) <= 0:
                raise ValueError('invalid_observed_solid')
        else:
            PrimitiveModel(self.kind, dimensions)
        object.__setattr__(self, 'world_from_local', immutable(pose))
        object.__setattr__(self, 'dimensions', dimensions)

    def with_offset(self, offset_m):
        """Reuse immutable, validated geometry when only clearance changes."""
        offset = float(offset_m)
        if not np.isfinite(offset):
            raise ValueError('invalid_observed_solid')
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result.__dict__.pop('bounds', None)
        object.__setattr__(result, 'offset_m', offset)
        return result

    @classmethod
    def from_geom(cls, model, data, gid, *, offset_m=0., convex_material_calibration_id=None):
        """Copy one geom pose; meshes require an explicit convex-solid prior."""
        kind = int(model.geom_type[gid])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.geom_xmat[gid].reshape(3, 3), data.geom_xpos[gid]
        shapes = {int(mujoco.mjtGeom.mjGEOM_SPHERE): ('sphere', 1),
                  int(mujoco.mjtGeom.mjGEOM_CAPSULE): ('capsule', 2),
                  int(mujoco.mjtGeom.mjGEOM_CYLINDER): ('cylinder', 2),
                  int(mujoco.mjtGeom.mjGEOM_BOX): ('box', 3)}
        if kind in shapes:
            name, count = shapes[kind]
            return cls(name, tuple(model.geom_size[gid, :count]), pose, offset_m)
        if kind == mujoco.mjtGeom.mjGEOM_MESH and convex_material_calibration_id:
            mesh = int(model.geom_dataid[gid])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            return cls('convex', (), pose, offset_m, model.mesh_vert[start:start+count],
                       convex_material_calibration_id)
        raise ValueError('robot_material_geometry_unavailable')

    def lower_distance(self, points):
        """A 1-Lipschitz lower bound on distance to the solid, negative inside.

        Primitive distances are exact. Convex plane distance is a lower bound
        outside the hull; that can only prevent a disjointness proof, never skip
        queried material. Offsets erode ownership and dilate query envelopes.
        """
        local = (np.asarray(points) - self.world_from_local[:3, 3]) @ self.world_from_local[:3, :3]
        if self.kind == 'convex':
            distance = np.max(local @ self._planes[:, :3].T + self._planes[:, 3], axis=-1)
        elif self.kind == 'capsule':
            local = local.copy()
            local[:, 2] -= np.clip(local[:, 2], -self.dimensions[1], self.dimensions[1])
            distance = np.linalg.norm(local, axis=-1) - self.dimensions[0]
        else:
            distance = PrimitiveModel(self.kind, self.dimensions).distance(local)
        return distance - self.offset_m

    @cached_property
    def bounds(self):
        if self.kind == 'convex':
            vertices = self.vertices @ self.world_from_local[:3, :3].T + self.world_from_local[:3, 3]
            bounds = np.stack((vertices.min(0), vertices.max(0)))
        elif self.kind == 'capsule':
            radius, half_length = self.dimensions
            half = np.abs(self.world_from_local[:3, 2])*half_length + radius
            bounds = self.world_from_local[:3, 3] + np.stack((-half, half))
        else:
            bounds = PrimitiveModel(self.kind, self.dimensions).world_bounds(
                self.world_from_local[:3, :3], self.world_from_local[:3, 3])
        # Negative offsets need no tight bound; keeping the original is safe.
        return immutable(bounds + np.array([[-1.], [1.]])*max(0., self.offset_m))

    def disjoint_box(self, bounds):
        lo, hi = bounds
        if np.any(self.bounds[1] < lo) or np.any(self.bounds[0] > hi):
            return True
        center, radius = (lo+hi)/2, np.linalg.norm((hi-lo)/2)
        return bool(self.lower_distance(center[None])[0] > radius)

    def contains_box(self, bounds):
        # Negative convex offsets are exact erosions. For positive convex
        # offsets, plane tests alone are not sufficient at outside corners.
        if self.kind == 'convex' and self.offset_m > 0:
            raise ValueError('dilated_convex_cannot_certify_material')
        return bool(np.all(self.lower_distance(box_corners(bounds)) <= 0))

    def contains_solid(self, other):
        identical = (self.kind == other.kind and self.dimensions == other.dimensions
                     and np.array_equal(self.world_from_local, other.world_from_local)
                     and (self.kind != 'convex' or np.array_equal(self.vertices, other.vertices)))
        if identical and self.offset_m >= other.offset_m:
            return True
        return self.contains_box(other.bounds)
