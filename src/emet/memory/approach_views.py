# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded camera evidence during motion, independent of mapping and instances.

Robot adapters feed already acquired observations. This buffer does no inference,
navigation, or map insertion; it cannot supply manipulation object geometry.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


def _snapshot(value):
    if value is None:
        return None
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ApproachView:
    view_id: str
    timestamp_ns: int
    clock_domain: str
    rgb: np.ndarray
    depth: np.ndarray | None
    camera_K: np.ndarray | None
    camera_pose: np.ndarray


class ApproachViewHistory:
    """Keep at most 32 owned snapshots, sampled by measured camera motion.

    Fixed initial sampling: 25 cm translation OR 15 degrees optical-axis change.
    These are acquisition settings, not answer-label-derived confidence scores.
    Instantiate once per query/world; never carry a buffer across map resets.
    """

    def __init__(self, capacity=32):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.views = deque(maxlen=capacity)
        self._sequence = 0

    def add(self, observation) -> bool:
        if observation is None:
            return False
        timing = (getattr(observation, "image_timing", None) or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if not isinstance(stamp, (int, float)) or not np.isfinite(stamp) or stamp <= 0 or not clock:
            return False
        pose = _snapshot(getattr(observation, "camera_pose", None))
        rgb = getattr(observation, "rgb", None)
        if pose is None or pose.shape != (4, 4) or not np.isfinite(pose).all() or rgb is None:
            return False
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
            return False
        if self.views:
            previous = self.views[-1]
            if clock != previous.clock_domain or stamp <= previous.timestamp_ns:
                return False
            distance = np.linalg.norm(pose[:3, 3] - previous.camera_pose[:3, 3])
            cosine = np.clip(np.dot(pose[:3, 2], previous.camera_pose[:3, 2]), -1, 1)
            if distance < 0.25 and np.arccos(cosine) < np.deg2rad(15) - 1e-6:
                return False
        rgb = _snapshot(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            return False
        self._sequence += 1
        self.views.append(
            ApproachView(
                f"approach:{self._sequence}",
                int(stamp),
                str(clock),
                rgb,
                _snapshot(getattr(observation, "depth", None)),
                _snapshot(getattr(observation, "camera_K", None)),
                pose,
            )
        )
        return True
