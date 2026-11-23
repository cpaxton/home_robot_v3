# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Capture-bound grasp proposals from partial measured geometry, without CAD."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _immutable(value):
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


@dataclass(frozen=True)
class ObservedGeometry:
    observation_id: str
    camera_id: str
    acquired_monotonic: float
    points_world: np.ndarray
    position_error_m: float

    def __post_init__(self):
        p = np.asarray(self.points_world, dtype=float)
        if (not self.observation_id or not self.camera_id or p.ndim != 2 or p.shape[1] != 3
                or len(p) < 25 or not np.isfinite(p).all()
                or not np.isfinite(self.acquired_monotonic)
                or not np.isfinite(self.position_error_m) or self.position_error_m < 0):
            raise ValueError('invalid_observed_geometry')
        object.__setattr__(self, 'points_world', _immutable(p))


@dataclass(frozen=True)
class GraspProposal:
    observation_id: str
    world_from_hand: np.ndarray
    opening_m: float
    provenance: str = 'partial_cloud_top_down'

    def __post_init__(self):
        t = np.asarray(self.world_from_hand, dtype=float)
        if (not self.observation_id or t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1.)
                or not np.isfinite(self.opening_m) or self.opening_m <= 0):
            raise ValueError('invalid_grasp_proposal')
        object.__setattr__(self, 'world_from_hand', _immutable(t))


class GraspProvider(Protocol):
    def propose(self, geometry: ObservedGeometry) -> tuple[GraspProposal, ...]: ...


@dataclass(frozen=True)
class TopDownCloudProvider:
    """Proposals only: observed thickness is not proof of hidden extent/contact.

    Frame convention: hand X closes fingers, hand Z points down. The robot port
    must apply its calibrated hand-to-tool transform before motion validation.
    """
    maximum_opening_m: float
    finger_clearance_m: float = .005

    def __post_init__(self):
        if (not np.isfinite(self.maximum_opening_m) or self.maximum_opening_m <= 0
                or not np.isfinite(self.finger_clearance_m) or self.finger_clearance_m < 0):
            raise ValueError('invalid_gripper_geometry')

    def propose(self, geometry):
        points = geometry.points_world
        center = np.median(points, axis=0)
        # A principal direction chooses a proposal family, not object orientation.
        _, vectors = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
        axis = vectors[:, 0]
        angle = np.arctan2(axis[1], axis[0])
        result = []
        for yaw in (angle, angle + np.pi / 2):
            x = np.array([np.cos(yaw), np.sin(yaw), 0.])
            z = np.array([0., 0., -1.])
            projected = points @ x
            width = float(np.ptp(projected)) + 2 * (geometry.position_error_m + self.finger_clearance_m)
            if not 0 < width <= self.maximum_opening_m:
                continue
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
            pose[:3, 3] = center
            result.append(GraspProposal(geometry.observation_id, pose, width))
        return tuple(result)


@dataclass(frozen=True)
class GroundingHint:
    """Image-space semantic proposal; never a metric pose or accepted mask."""
    observation_id: str
    camera_id: str
    box_xyxy: tuple[float, float, float, float]
    label: str

    def __post_init__(self):
        b = np.asarray(self.box_xyxy, float)
        if (not self.observation_id or not self.camera_id or not self.label
                or b.shape != (4,) or not np.isfinite(b).all()
                or np.any(b < 0) or np.any(b > 1000) or np.any(b[:2] >= b[2:])):
            raise ValueError('invalid_grounding_hint')
        object.__setattr__(self, 'box_xyxy', tuple(b))


def geometry_from_mask(hint, *, observation_id, camera_id, acquired_monotonic,
                       mask, world_points, position_error_m):
    """Bind a refined mask to its original aligned depth, retaining all returns.

    Semantic selection/segmentation must happen upstream. No CAD completion or
    hidden-surface claim is made. The source must preserve acquisition time.
    """
    if (hint.observation_id, hint.camera_id) != (observation_id, camera_id):
        raise ValueError('capture_identity_mismatch')
    points = np.asarray(world_points, float)
    selected = np.asarray(mask)
    if points.ndim != 3 or points.shape[2] != 3 or selected.shape != points.shape[:2] or selected.dtype != bool:
        raise ValueError('unaligned_geometry_mask')
    selected = selected & np.isfinite(points).all(axis=-1)
    return ObservedGeometry(observation_id, camera_id, acquired_monotonic,
                            points[selected], position_error_m)


def registration_surface(geometry, *, box_padding_m=None):
    """Create a CAD-free registration reference, not a complete object envelope."""
    from emet.controller.manipulation.geometry_models import MeasuredSurfaceModel

    reference = np.eye(4)
    reference[:3, 3] = np.median(geometry.points_world, axis=0)
    model = MeasuredSurfaceModel(
        geometry.points_world-reference[:3, 3], world_from_reference=reference,
        observation_ids=(geometry.observation_id,), received_monotonic=geometry.acquired_monotonic,
    )
    if box_padding_m is not None:
        from emet.motion.placement_geometry import box_corners

        if not np.isfinite(box_padding_m) or not 0 <= box_padding_m <= .1:
            raise ValueError('invalid_payload_box_padding')
        local = geometry.points_world-reference[:3, 3]
        padding = box_padding_m + geometry.position_error_m
        model.vertices_reference = _immutable(box_corners([local.min(0)-padding, local.max(0)+padding]))
        model.half_extents = _immutable(np.max(np.abs(model.vertices_reference), axis=0))
        model.construction = 'approximate_observed_box'
        model.box_padding_m = box_padding_m
        # Keep the registration surfaces measured; only the payload box expands.
        import hashlib
        model.model_id = hashlib.sha256(model.points_reference.tobytes()+model.vertices_reference.tobytes()).hexdigest()
    return model, _immutable(reference)
