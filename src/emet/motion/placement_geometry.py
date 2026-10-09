# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Conservative 3D geometry snapshots for placement; no simulator state mutation.

Observed cells represent occupied space, not a claim that unobserved space is free.
A workspace bounds the query domain. Applications can additionally supply a
known-free-volume predicate to reject occluded/unknown swept volumes.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from emet.motion.mujoco_collision import MujocoSceneCollisionChecker, body_subtree

CORNERS = np.array(list(product((-1, 1), repeat=3)), dtype=float)


def bounds_array(value):
    bounds = np.array(value, dtype=float, copy=True)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError("Expected finite ordered (2, 3) bounds")
    return bounds


def box_corners(bounds):
    lo, hi = bounds_array(bounds)
    return (lo + hi) / 2 + CORNERS * (hi - lo) / 2


class PlacementScene:
    """World-frame obstacle boxes, from GT bounds or occupied 3D voxels.

    The caller must segment the held object out of scene occupancy. No geometric
    crop or receptacle exemption is applied: those would also erase obstacles.
    """

    def __init__(self, boxes, *, source: str, workspace=None, known_free: Callable | None = None):
        boxes = np.array(boxes, dtype=float, copy=True)
        if boxes.size == 0:
            boxes = np.empty((0, 2, 3))
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 3):
            raise ValueError("Expected N x 2 x 3 boxes")
        if not np.isfinite(boxes).all() or np.any(boxes[:, 0] > boxes[:, 1]):
            raise ValueError("Invalid obstacle bounds")
        if source not in {"observed_voxels", "ground_truth"}:
            raise ValueError("Unsupported geometry source")
        self.boxes = boxes
        self.boxes.setflags(write=False)
        self.source = source
        self.workspace = None if workspace is None else bounds_array(workspace)
        self.known_free = known_free
        centers = boxes.mean(axis=1)
        self._tree = cKDTree(centers) if len(boxes) else None
        self._max_radius = float(np.linalg.norm((boxes[:, 1] - boxes[:, 0]) / 2, axis=1).max()) if len(boxes) else 0

    @classmethod
    def from_voxels(cls, centers, *, resolution: float, workspace, known_free=None):
        """Cell centers in world metres; resolution is full cell width (not 2D map resolution)."""
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("Expected finite N x 3 voxel centers")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Positive finite voxel resolution required")
        half = resolution / 2
        return cls(np.stack((centers - half, centers + half), axis=1), source="observed_voxels",
                   workspace=workspace, known_free=known_free)

    @classmethod
    def from_pointcloud(cls, voxel_map, *, workspace, known_free=None):
        """Adapt SparseVoxelMap centroids conservatively (a centroid is not a cell center).

        Supply a scene-only map with the held object and robot segmented out.
        Uses the actual 3D voxel resolution; never projects to a 2D obstacle map.
        """
        points = voxel_map.get_pointcloud()[0]
        if points is None:
            raise ValueError("Missing observed geometry")
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        # A centroid can lie anywhere in its cell. One cell width on each side
        # conservatively contains that cell without guessing grid alignment.
        return cls.from_voxels(points, resolution=2 * float(voxel_map.voxel_resolution),
                               workspace=workspace, known_free=known_free)

    @classmethod
    def from_placements(cls, placements, *, held_object: str):
        if held_object not in placements:
            raise ValueError("Missing held object")
        boxes = []
        for name, entry in placements.items():
            if name != held_object:
                if "bounds" not in entry:
                    raise ValueError(f"Missing scene bounds: {name}")
                boxes.append(bounds_array(entry["bounds"]))
        return cls(boxes, source="ground_truth")

    def collides(self, bounds) -> bool:
        lo, hi = bounds_array(bounds)
        if self.workspace is not None and (np.any(lo < self.workspace[0]) or np.any(hi > self.workspace[1])):
            return True
        if self._tree is not None:
            ids = self._tree.query_ball_point((lo + hi) / 2, np.linalg.norm((hi - lo) / 2) + self._max_radius)
            boxes = self.boxes[ids]
            if np.any(np.all(boxes[:, 1] >= lo, axis=1) & np.all(boxes[:, 0] <= hi, axis=1)):
                return True
        return self.known_free is not None and not bool(self.known_free(np.stack((lo, hi))))


@dataclass(frozen=True)
class HeldObject:
    """Conservative object vertices in the EE frame, with an explicit attachment transform."""
    vertices_ee: np.ndarray

    def __post_init__(self):
        vertices = np.array(self.vertices_ee, dtype=float, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("Finite held-object volume required")
        if np.any(np.ptp(vertices, axis=0) <= 0):
            raise ValueError("Held-object volume must have positive extent")
        vertices.setflags(write=False)
        object.__setattr__(self, "vertices_ee", vertices)

    @classmethod
    def from_world_bounds(cls, bounds, *, ee_position, ee_rotation):
        rotation = np.asarray(ee_rotation, dtype=float).reshape(3, 3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
            raise ValueError("Invalid attachment rotation")
        return cls((box_corners(bounds) - np.asarray(ee_position)) @ rotation)

    def world_vertices(self, position, rotation):
        return self.vertices_ee @ np.asarray(rotation).reshape(3, 3).T + position


class PlacementCollisionChecker:
    """Robot geom volumes + transformed payload versus the scene, plus self collision.

    Robot geom world AABBs conservatively enclose the complete declared volumes,
    including meshes. Payload/robot intersections are rejected except explicitly
    named grasp-contact bodies. Broad boxes may reject geometrically feasible
    poses; they never shrink occupied volumes to link origins.
    """

    def __init__(self, model, *, robot_body: str, ee_body: str, scene: PlacementScene,
                 payload: HeldObject, contact_bodies: Sequence[str] = (), margin_m: float = .005):
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError("Nonnegative finite margin required")
        self.scene, self.payload, self.ee_body, self.margin = scene, payload, ee_body, margin_m
        self.robot_ids = body_subtree(model, robot_body)
        self.allowed_ids = set()
        for name in contact_bodies:
            self.allowed_ids |= body_subtree(model, name)
        self.geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in self.robot_ids
                         and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if not self.geom_ids:
            raise ValueError("Robot collision geometry missing")
        self.self_collision = MujocoSceneCollisionChecker(model, robot_body=robot_body)
        self.last_reason = None
        self.released_bounds = None

    def configuration_collides(self, model, data):
        self.last_reason = None
        if self.self_collision.configuration_collides(model, data):
            self.last_reason = "robot_self_collision"
            return True
        ee = data.body(self.ee_body)
        vertices = self.payload.world_vertices(ee.xpos, ee.xmat)
        payload_bounds = np.stack((vertices.min(axis=0) - self.margin, vertices.max(axis=0) + self.margin))
        if self.released_bounds is not None:
            payload_bounds = self.released_bounds
        if self.scene.collides(payload_bounds):
            self.last_reason = "payload_scene_collision"
            return True
        for gid in self.geom_ids:
            center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
            rotation = data.geom_xmat[gid].reshape(3, 3)
            world_center = rotation @ center + data.geom_xpos[gid]
            world_half = np.abs(rotation) @ half
            bounds = np.stack((world_center - world_half - self.margin, world_center + world_half + self.margin))
            if self.scene.collides(bounds):
                self.last_reason = "robot_scene_collision"
                return True
            if int(model.geom_bodyid[gid]) not in self.allowed_ids:
                if np.all(bounds[1] >= payload_bounds[0]) and np.all(bounds[0] <= payload_bounds[1]):
                    self.last_reason = "payload_robot_collision"
                    return True
        return False
