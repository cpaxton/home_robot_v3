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


def rotations_equivalent(model, a, b):
    """Exact declared shape symmetry, for redundant SEARCH seeds only.

    No undeclared measured-surface symmetry is inferred. This does not collapse
    distinct centers, unobservable translations or physical object hypotheses.
    """
    if model.symmetry == "all_rotations":
        return True
    if model.symmetry == "axial_unoriented":
        return np.allclose(np.outer(a[:, 2], a[:, 2]), np.outer(b[:, 2], b[:, 2]), atol=1e-10, rtol=0)
    if model.symmetry == "axial_oriented":
        return np.allclose(a[:, 2], b[:, 2], atol=1e-10, rtol=0)
    if model.symmetry == "finite_box":
        change = np.abs(a.T @ b)
        return np.allclose(change, np.rint(change), atol=1e-10, rtol=0) and np.allclose(
            change @ model.half_extents, model.half_extents, atol=1e-10, rtol=0
        )
    return np.allclose(a, b, atol=1e-10, rtol=0)


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


class KnownMeshSurface:
    """An independent visual surface for registration and a separate outer envelope."""

    construction = "known_model_prior"
    symmetry = "undeclared"

    def __init__(self, points_reference, normals, vertices_reference):
        points = np.asarray(points_reference, dtype=float)
        normals = np.asarray(normals, dtype=float)
        vertices = np.asarray(vertices_reference, dtype=float)
        if (
            points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 25
            or normals.shape != points.shape or vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or len(vertices) < 4 or not all(np.isfinite(a).all() for a in (points, normals, vertices))
            or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("invalid_known_mesh_prior")
        from scipy.spatial import ConvexHull

        hull = ConvexHull(vertices)
        for start in range(0, len(points), 256):
            if np.any(points[start : start + 256] @ hull.equations[:, :3].T + hull.equations[:, 3] > 1e-7):
                raise ValueError("visual_surface_outside_payload_envelope")
        self.points_reference = immutable(points)
        self.normals = immutable(normals)
        self.vertices_reference = immutable(vertices)
        self.half_extents = immutable(np.max(np.abs(vertices), axis=0))
        self.model_id = hashlib.sha256(points.tobytes() + normals.tobytes() + vertices.tobytes()).hexdigest()
        self._tree = cKDTree(self.points_reference)

    def distance(self, points):
        indices = self._tree.query(points)[1]
        return np.sum((points - self.points_reference[indices]) * self.normals[indices], axis=1)

    def validation_distance(self, points):
        return self._tree.query(points)[0]

    def world_bounds(self, rotation, center):
        vertices = self.vertices_reference @ rotation.T + center
        return np.array([vertices.min(0), vertices.max(0)])


class AxialMeshSurface:
    """Registration in radius-height space for an object with a declared axial bound."""

    construction = "known_model_prior"
    symmetry = "axial_oriented"

    def __init__(self, surface):
        self.vertices_reference, self.half_extents = surface.vertices_reference, surface.half_extents
        self.model_id = hashlib.sha256((surface.model_id + ":radial-registration").encode()).hexdigest()
        points = surface.points_reference
        radius = np.linalg.norm(points[:, :2], axis=1)
        radial = np.zeros(len(points))
        np.divide(np.sum(surface.normals[:, :2] * points[:, :2], axis=1), radius, out=radial, where=radius > 1e-8)
        normals = np.column_stack((radial, surface.normals[:, 2]))
        length = np.linalg.norm(normals, axis=1)
        valid = length > 0.5
        self.samples = immutable(np.column_stack((radius, points[:, 2]))[valid])
        self.normals = immutable(normals[valid] / length[valid, None])
        if len(self.samples) < 25:
            raise ValueError("invalid_axial_surface_prior")
        self.tree = cKDTree(self.samples)

    def distance(self, points):
        xz = np.column_stack((np.linalg.norm(points[:, :2], axis=1), points[:, 2]))
        indices = self.tree.query(xz)[1]
        return np.sum((xz - self.samples[indices]) * self.normals[indices], axis=1)

    def validation_distance(self, points):
        return self.tree.query(np.column_stack((np.linalg.norm(points[:, :2], axis=1), points[:, 2])))[0]

    def world_bounds(self, rotation, center):
        vertices = self.vertices_reference @ rotation.T + center
        return np.array([vertices.min(0), vertices.max(0)])


class MeasuredSurfaceModel:
    """Reference-frame measured surfaces with a separately certified envelope.

    When supplied, the complete envelope comes from the enclosure constructor.
    Without it, bounds describe observed samples only and construction is explicitly
    partial; downstream complete-payload admission must not accept those bounds. Reference axes are a coordinate convention, not measured yaw.
    Registration does not assert that every future surface was already observed.
    """

    construction = "observed_multiview_surface"
    symmetry = "undeclared"

    def __init__(self, points_reference, *, world_from_reference, enclosure=None,
                 observation_ids=(), received_monotonic=None):
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
        self.complete_enclosure = enclosure is not None
        if enclosure is None:
            if (not observation_ids or not all(isinstance(v, str) and v for v in observation_ids)
                    or received_monotonic is None or not np.isfinite(received_monotonic)):
                raise ValueError("partial_surface_capture_required")
            # This box encloses observed samples only, never the complete object.
            corners = box_corners([points.min(0), points.max(0)])
            self.construction = "observed_partial_surface"
            self.observation_ids = tuple(observation_ids)
            self.received_monotonic = float(received_monotonic)
        else:
            corners = (enclosure.vertices_world - pose[:3, 3]) @ pose[:3, :3]
            self.observation_ids = enclosure.observation_ids
            self.received_monotonic = enclosure.received_monotonic
        if np.any(points.min(0) < corners.min(0)) or np.any(points.max(0) > corners.max(0)):
            raise ValueError("surface_outside_certified_enclosure")
        self.points_reference = immutable(points)
        self.vertices_reference = immutable(corners)
        self.half_extents = immutable(np.max(np.abs(corners), axis=0))
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
