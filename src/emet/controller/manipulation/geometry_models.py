# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit geometry priors and measured multi-view surface models; no scene RPCs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from emet.motion.placement_geometry import box_corners


def immutable(value):
    value = np.asarray(value, float)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class PrimitiveModel:
    kind: str
    dimensions: tuple[float, ...]

    def __post_init__(self):
        counts = {"sphere": 1, "cylinder": 2, "box": 3}
        dimensions = tuple(float(v) for v in self.dimensions)
        if (
            self.kind not in counts
            or len(dimensions) != counts[self.kind]
            or not np.isfinite(dimensions).all()
            or min(dimensions) <= 0
        ):
            raise ValueError("invalid_geometry_model")
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def construction(self):
        return "known_model_prior"

    @property
    def model_id(self):
        return hashlib.sha256(json.dumps((self.kind, self.dimensions)).encode()).hexdigest()

    @property
    def symmetry(self):
        return {"sphere": "all_rotations", "cylinder": "axial_unoriented", "box": "finite_box"}[self.kind]

    @property
    def half_extents(self):
        if self.kind == "sphere":
            return np.repeat(self.dimensions[0], 3)
        if self.kind == "cylinder":
            return np.array([self.dimensions[0], self.dimensions[0], self.dimensions[1]])
        return np.array(self.dimensions)

    @property
    def vertices_reference(self):
        return box_corners([-self.half_extents, self.half_extents])

    def distance(self, points):
        """Signed distance to an exact declared solid; dimensions are half-sizes."""
        if self.kind == "sphere":
            return np.linalg.norm(points, axis=1) - self.dimensions[0]
        if self.kind == "cylinder":
            d = np.column_stack(
                (np.linalg.norm(points[:, :2], axis=1) - self.dimensions[0], np.abs(points[:, 2]) - self.dimensions[1])
            )
        else:
            d = np.abs(points) - np.array(self.dimensions)
        return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0)

    def validation_distance(self, points):
        return np.abs(self.distance(points))

    def world_bounds(self, rotation, center):
        if self.kind == "sphere":
            half = self.half_extents
        elif self.kind == "cylinder":
            axis = rotation[:, 2]
            half = self.dimensions[1] * np.abs(axis) + self.dimensions[0] * np.sqrt(np.maximum(0, 1 - axis * axis))
        else:
            half = np.abs(rotation) @ self.half_extents
        return np.array([center - half, center + half])


class MeasuredSurfaceModel:
    """Reference-frame measured surfaces with a separately certified envelope.

    The envelope must come from the enclosure constructor, never the visible
    points' box. Reference axes are a coordinate convention, not measured yaw.
    Registration does not assert that every future surface was already observed.
    """

    construction = "observed_multiview_surface"
    symmetry = "undeclared"

    def __init__(self, points_reference, *, world_from_reference, enclosure):
        points = np.asarray(points_reference, float)
        pose = np.asarray(world_from_reference, float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 25
            or not np.isfinite(points).all()
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
        ):
            raise ValueError("invalid_measured_surface_model")
        corners = (enclosure.vertices_world - pose[:3, 3]) @ pose[:3, :3]
        if np.any(points.min(0) < corners.min(0)) or np.any(points.max(0) > corners.max(0)):
            raise ValueError("surface_outside_certified_enclosure")
        self.points_reference = immutable(points)
        self.vertices_reference = immutable(corners)
        self.half_extents = immutable(np.max(np.abs(corners), axis=0))
        self.observation_ids = enclosure.observation_ids
        self.received_monotonic = enclosure.received_monotonic
        self._tree = cKDTree(self.points_reference, copy_data=True)
        neighbours = self._tree.query(points, k=min(16, len(points)))[1]
        patches = points[neighbours]
        centered = patches - patches.mean(axis=1, keepdims=True)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        self.normals = immutable(vectors[:, -1])
        self.model_id = hashlib.sha256(points.tobytes() + corners.tobytes()).hexdigest()

    def distance(self, points):
        indices = self._tree.query(points)[1]
        return np.sum((points - self.points_reference[indices]) * self.normals[indices], axis=1)

    def validation_distance(self, points):
        return self._tree.query(points)[0]

    def world_bounds(self, rotation, center):
        vertices = self.vertices_reference @ rotation.T + center
        return np.stack((vertices.min(0), vertices.max(0)))
