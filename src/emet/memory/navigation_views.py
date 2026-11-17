# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Bounded historical RGB evidence sampled independently of map ingestion."""

from collections import Counter, deque
from threading import RLock

import numpy as np

from emet.memory.view_selection import ViewFrame


class NavigationViewBuffer:
    """Retain distinct sensor viewpoints, never depth, labels or object geometry.

    Sampling runs on already acquired frames. It does not render, infer, update
    the map, rank task relevance, or change control. Selection happens later.
    """

    def __init__(self, scope_id, *, max_frames=32, min_translation_m=0.5, min_rotation_deg=20.0, max_side=640):
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("A map/episode scope is required")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 128:
            raise ValueError("max_frames must be between 1 and 128")
        if (
            not np.isfinite([min_translation_m, min_rotation_deg]).all()
            or min(min_translation_m, min_rotation_deg) <= 0
        ):
            raise ValueError("Positive finite viewpoint sampling distances required")
        if isinstance(max_side, bool) or not isinstance(max_side, int) or not 2 <= max_side <= 1280:
            raise ValueError("max_side must be between 2 and 1280 pixels")
        self.scope_id = scope_id
        self.min_translation_m = min_translation_m
        self.min_rotation_deg = min_rotation_deg
        self.max_side = max_side
        self._frames = deque(maxlen=max_frames)
        self._lock = RLock()
        self._next_id = 0
        self.skipped = Counter()

    def frames(self):
        with self._lock:
            return list(self._frames)

    def summary(self):
        with self._lock:
            return {
                "scope_id": self.scope_id,
                "retained": len(self._frames),
                "skipped": dict(self.skipped),
                "sampling": {
                    "max_frames": self._frames.maxlen,
                    "min_translation_m": self.min_translation_m,
                    "min_rotation_deg": self.min_rotation_deg,
                    "max_side": self.max_side,
                },
            }

    def consider(self, *, rgb, camera_pose, image_timing):
        timing = image_timing.get("head_cam_left") if isinstance(image_timing, dict) else None
        timing = timing if isinstance(timing, dict) else {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (int, float))
            or not np.isfinite(stamp)
            or stamp <= 0
            or not isinstance(clock, str)
            or not clock
        ):
            with self._lock:
                self.skipped["acquisition_unavailable"] += 1
            return False
        try:
            pose = camera_pose.detach().cpu().numpy() if hasattr(camera_pose, "detach") else np.asarray(camera_pose)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("invalid camera pose")
            with self._lock:
                scope = f"{self.scope_id}:{clock}"
                previous = self._frames[-1] if self._frames else None
                if previous is not None and previous.scope_id == scope:
                    if stamp / 1e9 <= previous.timestamp_s:
                        self.skipped["nonadvancing_acquisition"] += 1
                        return False
                    distance = np.linalg.norm(pose[:3, 3] - previous.camera_pose[:3, 3])
                    angle = np.degrees(np.arccos(np.clip(np.dot(pose[:3, 2], previous.camera_pose[:3, 2]), -1, 1)))
                    if distance < self.min_translation_m and angle < self.min_rotation_deg:
                        self.skipped["same_viewpoint"] += 1
                        return False
                from emet.llms.vl_image import downsample_rgb_hwc

                pixels = downsample_rgb_hwc(np.asarray(rgb), max_side=self.max_side, max_pixels=self.max_side**2)
                frame = ViewFrame(
                    f"navigation:{self.scope_id}:{self._next_id}", scope, stamp / 1e9, pixels, camera_pose=pose
                )
                self._frames.append(frame)
                self._next_id += 1
                return True
        except (TypeError, ValueError):
            with self._lock:
                self.skipped["invalid_sensor_frame"] += 1
            return False


def bind_navigation_views(agent):
    """Bind one bounded buffer to a real map epoch; explicit config permits ablation."""
    robot = getattr(agent, "robot", None)
    epoch = getattr(getattr(agent, "voxel_map", None), "observation_scope_id", None)
    if robot is None or not isinstance(epoch, str) or not epoch:
        return None
    parameters = getattr(agent, "parameters", {})
    settings = dict((parameters.get("eqa", {}) or {}).get("navigation_views") or {})
    if not settings.pop("enabled", True):
        robot.navigation_view_buffer = None
        return None
    buffer = getattr(robot, "navigation_view_buffer", None)
    if not isinstance(buffer, NavigationViewBuffer) or buffer.scope_id != epoch:
        buffer = robot.navigation_view_buffer = NavigationViewBuffer(epoch, **settings)
    return buffer


def retain_navigation_observation(robot, observation):
    """Shared producer boundary for native observations and decoded ZMQ dictionaries."""
    buffer = getattr(robot, "navigation_view_buffer", None)
    if not isinstance(buffer, NavigationViewBuffer):
        return
    get = observation.get if isinstance(observation, dict) else lambda key: getattr(observation, key, None)
    buffer.consider(rgb=get("rgb"), camera_pose=get("camera_pose"), image_timing=get("image_timing"))
