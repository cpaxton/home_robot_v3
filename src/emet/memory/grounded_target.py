# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Ephemeral, observation-backed geometry passed to manipulation adapters."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundedTarget:
    candidate_id: int
    instance_id: int
    observation_revision: int
    points: np.ndarray

    def __post_init__(self):
        points = np.array(self.points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10 or not np.isfinite(points).all():
            raise ValueError("Grounded target requires finite object-specific world points")
        points.flags.writeable = False
        object.__setattr__(self, "points", points)

    @property
    def xyz(self) -> np.ndarray:
        return np.median(self.points, axis=0)

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
