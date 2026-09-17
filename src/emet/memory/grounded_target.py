# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Ephemeral, observation-backed geometry passed to manipulation adapters."""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class GroundedTarget:
    candidate_id: int
    instance_id: int
    observation_revision: int
    points: np.ndarray
    geometry_source: str = "detector_mask"

    def __post_init__(self):
        points = np.array(self.points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10 or not np.isfinite(points).all():
            raise ValueError("Grounded target requires finite object-specific world points")
        points.flags.writeable = False
        object.__setattr__(self, "points", points)

    @property
    def xyz(self) -> np.ndarray:
        return np.median(self.points, axis=0)

    def project_box(self, camera_K, camera_pose, image_shape):
        """Project observed world bounds as an unverified segmentation prompt.

        This is neither current visibility nor complete object geometry. A
        fresh mask still needs semantic verification and world association.
        Reject bounds crossing the camera plane rather than inventing a box.
        """
        intrinsic = np.asarray(camera_K, dtype=float)
        pose = np.asarray(camera_pose, dtype=float)
        if (
            intrinsic.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(pose).all()
            or intrinsic[0, 0] <= 0
            or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1])
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("Target projection requires calibrated camera intrinsics and a rigid world pose")
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("Target projection requires a nonempty image")
        corners = np.asarray(list(product(*zip(self.points.min(axis=0), self.points.max(axis=0), strict=True))))
        camera = (corners - pose[:3, 3]) @ pose[:3, :3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError("Grounded bounds cross or lie behind the camera plane")
        pixels = camera @ intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        lo = np.clip(pixels.min(axis=0), [0, 0], [width, height])
        hi = np.clip(pixels.max(axis=0), [0, 0], [width, height])
        if not np.isfinite(pixels).all() or np.any(hi <= lo):
            raise ValueError("Grounded bounds have no visible image extent")
        return np.concatenate([lo, hi])

    def select_surface(self, world_xyz, *, margin_m: float = 0.05):
        """Track one connected observed surface, independent of detector classes."""
        from scipy.ndimage import label

        if world_xyz is None:
            raise ValueError("Target tracking requires world-aligned depth")
        xyz = np.asarray(world_xyz)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("Target tracking requires organized world geometry")
        inside = np.isfinite(xyz).all(axis=-1)
        inside &= (xyz >= self.points.min(axis=0) - margin_m).all(axis=-1)
        inside &= (xyz <= self.points.max(axis=0) + margin_m).all(axis=-1)
        components, n = label(inside)
        matches = [components == i for i in range(1, n + 1) if (components == i).sum() >= 25]
        if len(matches) != 1:
            raise ValueError("Grounded surface absent or ambiguous in current frame")
        return matches[0]

    def select_mask(self, instances, class_mask, world_xyz, *, margin_m: float = 0.05):
        """Require one currently visible instance within the grounded world bounds.

        This is a conservative local geometry gate, not semantic verification or
        long-term re-identification. Never fall back to the largest/central mask.
        """
        if world_xyz is None or instances is None:
            raise ValueError("Target tracking requires world-aligned depth and instance masks")
        xyz = np.asarray(world_xyz)
        instances, class_mask = np.asarray(instances), np.asarray(class_mask, dtype=bool)
        if xyz.shape != (*instances.shape, 3) or class_mask.shape != instances.shape:
            raise ValueError("Target tracking requires aligned masks and world geometry")
        lo = self.points.min(axis=0) - margin_m
        hi = self.points.max(axis=0) + margin_m
        finite = np.isfinite(xyz).all(axis=-1)
        inside = finite & (xyz >= lo).all(axis=-1) & (xyz <= hi).all(axis=-1)
        matches = []
        for iid in np.unique(instances):
            if iid < 0:
                continue
            mask = (instances == iid) & class_mask & finite
            support = int((mask & inside).sum())
            if support >= 10 and support >= 0.8 * int(mask.sum()):
                matches.append(mask & inside)
        if len(matches) != 1:
            raise ValueError("Grounded target absent or ambiguous in current frame")
        return matches[0]
