# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Historical environment evidence, separate from current robot/target returns.

Excluding a self endpoint never establishes free space. Only previously measured
free evidence can survive that occlusion, retaining its original per-cell time.
A bounded static-scene assumption may extend its validity, never its timestamp.
"""

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.observed_workspace import ObservedOccupancyPartition, ObservedWorkspace


def _frozen(values):
    values = np.asarray(values, float)
    return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)


@dataclass(frozen=True)
class StaticBackgroundAssumption:
    scene_epoch: str
    started_monotonic: float
    expires_monotonic: float
    assumption_id: str

    def __post_init__(self):
        if (
            not self.scene_epoch
            or not self.assumption_id
            or not np.isfinite([self.started_monotonic, self.expires_monotonic]).all()
            or not 0 < self.expires_monotonic - self.started_monotonic <= 300
        ):
            raise ValueError("invalid_static_background_assumption")


@dataclass(frozen=True)
class ObservedBackground:
    workspace: ObservedWorkspace
    observed_monotonic: np.ndarray
    scene_epoch: str
    capture_ids: frozenset[str]
    latest_capture: CaptureInterval

    def __post_init__(self):
        stamps = np.asarray(self.observed_monotonic, float)
        if (
            not self.scene_epoch
            or not self.capture_ids
            or stamps.shape != self.workspace.cells.shape
            or not np.isfinite(stamps[self.workspace.cells != 0]).all()
            or np.any(stamps[self.workspace.cells != 0] > self.latest_capture.latest)
        ):
            raise ValueError("invalid_background_evidence")
        object.__setattr__(self, "observed_monotonic", _frozen(stamps))

    @classmethod
    def from_capture(
        cls, partition: ObservedOccupancyPartition, *, excluded_refs, scene_epoch, capture: CaptureInterval, now=None
    ):
        capture.require(now=time.monotonic() if now is None else now)
        workspace = partition.excluding(excluded_refs)
        stamps = np.full(workspace.cells.shape, np.nan)
        stamps[workspace.cells != 0] = capture.earliest
        return cls(workspace, stamps, scene_epoch, frozenset((capture.observation_id,)), capture)

    def update(self, partition, *, excluded_refs, scene_epoch, capture: CaptureInterval, now=None):
        """Fuse only an explicitly unchanged background; contradictions reject.

        Current self-only endpoints become unknown before fusion. Mixed owners
        retain environment occupancy, so self masking cannot erase an obstacle.
        The caller must discard this map and invalidate plans on scene changes.
        """
        capture.require(now=time.monotonic() if now is None else now, after=self.latest_capture.latest)
        if (
            scene_epoch != self.scene_epoch
            or capture.observation_id in self.capture_ids
            or capture.calibration_id != self.latest_capture.calibration_id
            or len(self.capture_ids) >= 4096
        ):
            raise ObservationUnavailable("background_epoch_or_capture_invalid")
        current = partition.excluding(excluded_refs)
        old = self.workspace
        if (
            current.resolution != old.resolution
            or current.cells.shape != old.cells.shape
            or not np.array_equal(current.origin, old.origin)
        ):
            raise ObservationUnavailable("background_grid_changed")
        contradiction = ((old.cells == 1) & (current.cells == 2)) | ((old.cells == 2) & (current.cells == 1))
        if np.any(contradiction):
            raise ObservationUnavailable("background_scene_changed")
        cells, stamps = old.cells.copy(), self.observed_monotonic.copy()
        observed = current.cells != 0
        cells[observed], stamps[observed] = current.cells[observed], capture.earliest
        return ObservedBackground(
            ObservedWorkspace(old.origin, old.resolution, cells),
            stamps,
            self.scene_epoch,
            self.capture_ids | {capture.observation_id},
            capture,
        )

    @property
    def geometry_revision(self):
        digest = hashlib.sha256(
            self.scene_epoch.encode()
            + self.workspace.origin.tobytes()
            + np.float64(self.workspace.resolution).tobytes()
            + self.workspace.cells.tobytes()
        )
        return "observed-background:" + digest.hexdigest()

    def known_free(self, bounds, *, now, static_assumption=None, max_age_s=2.0):
        if not np.isfinite(max_age_s) or not 0 < max_age_s <= 2:
            raise ValueError("background_age_requires_static_assumption")
        region = self.workspace.query_region(bounds)
        if region is None or not np.all(self.workspace.cells[region] == 1):
            return False
        stamps = self.observed_monotonic[region]
        if not np.isfinite(now) or np.any(stamps > now):
            return False
        if static_assumption is None:
            return bool(np.all(now - stamps <= max_age_s))
        return bool(
            static_assumption.scene_epoch == self.scene_epoch
            and static_assumption.started_monotonic <= now <= static_assumption.expires_monotonic
            and np.all(stamps >= static_assumption.started_monotonic)
        )

    def placement_scene(self, *, static_assumption=None, clock=time.monotonic):
        scene = self.workspace.placement_scene()
        scene.known_free = lambda bounds: self.known_free(bounds, now=clock(), static_assumption=static_assumption)
        return scene
