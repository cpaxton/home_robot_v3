# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit geometric pose equivalence; never a unique material orientation."""

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.geometry_models import PrimitiveModel, rotations_equivalent
from emet.controller.manipulation.native_inputs import OBSERVED_POSE_SOURCES, ObjectPoseEvidence, ObservationUnavailable


@lru_cache(maxsize=64)
def _box_group(dimensions):
    model = PrimitiveModel("box", dimensions)
    return tuple(r for r in Rotation.create_group("O").as_matrix() if rotations_equivalent(model, np.eye(3), r))


def shape_rotation_distance(model, a, b):
    if model.symmetry == "all_rotations":
        return 0.0
    if model.symmetry == "axial_unoriented":
        return float(np.arccos(np.clip(abs(a[:, 2] @ b[:, 2]), 0.0, 1.0)))
    group = _box_group(model.dimensions) if model.symmetry == "finite_box" else (np.eye(3),)
    return min(float(np.arccos(np.clip((np.sum((a @ r) * b) - 1) / 2, -1.0, 1.0))) for r in group)


@dataclass(frozen=True)
class ShapePoseEvidence:
    """One representative of a declared solid's pose equivalence class.

    Bounds must be independently supplied/calibrated, never fit residuals. Local
    origin is the declared primitive center. Generic unique-pose readers reject
    this record; only explicitly shape-aware geometry consumers may use it.
    """

    world_from_object: np.ndarray
    received_monotonic: float
    source: str
    observation_id: str
    model: PrimitiveModel
    position_error_m: float
    rotation_error_rad: float
    calibration_id: str

    def __post_init__(self):
        if (
            not isinstance(self.model, PrimitiveModel)
            or self.source not in OBSERVED_POSE_SOURCES
            or not isinstance(self.calibration_id, str)
            or not self.calibration_id
            or not np.isfinite([self.position_error_m, self.rotation_error_rad]).all()
            or not 0 < self.position_error_m <= 0.01
            or not 0 < self.rotation_error_rad <= 0.05
        ):
            raise ValueError("invalid_shape_pose_evidence")
        validated = ObjectPoseEvidence(
            self.world_from_object, self.received_monotonic, self.source, self.observation_id
        )
        object.__setattr__(self, "world_from_object", validated.world_from_object)

    def require(self, inputs, *, now=None, max_age_s=2.0):
        raise ObservationUnavailable("object_pose_ambiguous")

    def require_shape(self, *, now=None, max_age_s=2.0):
        now = time.monotonic() if now is None else now
        if not 0 <= now - self.received_monotonic <= max_age_s:
            raise ObservationUnavailable("object_observation_stale")
        return self.world_from_object.copy()


def shape_identity(pose):
    return (
        (pose.model.model_id, pose.calibration_id, pose.position_error_m, pose.rotation_error_rad)
        if isinstance(pose, ShapePoseEvidence)
        else None
    )


def evidence_rotation_distance(pose, a, b):
    if isinstance(pose, ShapePoseEvidence):
        return shape_rotation_distance(pose.model, a, b)
    return float(np.arccos(np.clip((np.sum(a * b) - 1) / 2, -1.0, 1.0)))
