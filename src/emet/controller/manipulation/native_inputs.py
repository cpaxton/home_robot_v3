# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit input provenance for native manipulation; unknown evidence never passes."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

OBSERVED_POSE_SOURCES = frozenset({'rgbd_tracking', 'stereo_tracking'})


class ObservationUnavailable(RuntimeError):
    """A native operation must stop or inspect rather than invent missing evidence."""


@dataclass(frozen=True)
class ObjectPoseEvidence:
    world_from_object: np.ndarray
    received_monotonic: float
    source: str
    observation_id: str

    def __post_init__(self):
        pose = np.array(self.world_from_object, dtype=float, copy=True)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)
                or not np.isfinite(self.received_monotonic)
                or not self.observation_id):
            raise ValueError('invalid_object_pose_evidence')
        if self.source not in OBSERVED_POSE_SOURCES | {'simulator_ground_truth'}:
            raise ValueError('unsupported_object_pose_source')
        # An immutable bytes backing also prevents setflags(write=True).
        pose = np.frombuffer(pose.tobytes(), dtype=pose.dtype).reshape(4, 4)
        object.__setattr__(self, 'world_from_object', pose)

    def require(self, inputs: str, *, now: float | None = None, max_age_s: float = 2.) -> np.ndarray:
        if inputs not in {'observed', 'privileged'}:
            raise ValueError('invalid_tamp_inputs')
        if inputs == 'observed' and self.source not in OBSERVED_POSE_SOURCES:
            raise ObservationUnavailable('privileged_input_forbidden')
        age = (time.monotonic() if now is None else now) - self.received_monotonic
        if not np.isfinite(age) or not 0 <= age <= max_age_s:
            raise ObservationUnavailable('object_observation_stale')
        return self.world_from_object.copy()


class ObjectPoseProvider(Protocol):
    input_mode: str

    def observe_object(self, object_ref: str) -> ObjectPoseEvidence | None:
        """Return independently observed evidence, or None for unknown/occluded."""
        ...


def read_object_pose(provider: ObjectPoseProvider, object_ref: str, *, inputs: str) -> np.ndarray:
    if provider.input_mode != inputs:
        raise ObservationUnavailable('input_provider_mismatch')
    evidence = provider.observe_object(object_ref)
    if evidence is None:
        raise ObservationUnavailable('object_state_unknown')
    return evidence.require(inputs)
