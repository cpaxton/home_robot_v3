# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Whole-object bounds only when observed free space encloses a connected object.

All unknown AND occupied cells are possible object volume. Background endpoints
are not treated as solid negative space. A component reaching the scan boundary
is unbounded and needs another view; visible-point extents are never substituted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.placement_geometry import box_corners


@dataclass(frozen=True)
class ObservedObjectBound:
    bounds_world: np.ndarray
    observation_ids: tuple[str, ...]
    received_monotonic: float
    scene_revision: str
    topology: str = "connected_rigid"

    def __post_init__(self):
        bounds = np.asarray(self.bounds_world, float)
        object.__setattr__(self, "bounds_world", np.frombuffer(bounds.tobytes(), dtype=bounds.dtype).reshape(2, 3))

    @property
    def vertices_world(self):
        return box_corners(self.bounds_world)

    def require(self, *, now, max_age_s=2.0):
        if not 0 <= now - self.received_monotonic <= max_age_s:
            raise ObservationUnavailable("object_observation_stale")
        return self.bounds_world.copy()


def enclose_observed_object(partition, object_ref, *, observation_ids, received_monotonic, scene_revision, topology):
    """Enclose all possible connected extent under the grid's free-space model.

    Caller must supply a coherent static scan, its ORIGINAL oldest evidence time,
    and the connected-rigid-object assumption. This does not measure object pose
    or identify material points. Conservative bounds can contain background and
    reject otherwise feasible grasps; background labels do not carve the volume.
    """
    if (
        topology != "connected_rigid"
        or not observation_ids
        or not scene_revision
        or any(not isinstance(value, str) or not value for value in observation_ids)
        or not np.isfinite(received_monotonic)
    ):
        raise ValueError("invalid_object_enclosure_evidence")
    owners = dict(partition.owners)
    seeds = np.array(owners.get(object_ref, ()), dtype=int)
    if not len(seeds):
        raise ObservationUnavailable("object_geometry_unknown")
    grid = partition.workspace
    components, _ = label(grid.cells != 1, structure=np.ones((3, 3, 3), bool))
    identities = np.unique(components.ravel()[seeds])
    if len(identities) != 1 or identities[0] == 0:
        raise ObservationUnavailable("object_geometry_disconnected")
    possible = components == identities[0]
    if any(np.take(possible, edge, axis=axis).any() for axis in range(3) for edge in (0, -1)):
        raise ObservationUnavailable("object_geometry_unbounded")
    indices = np.argwhere(possible)
    lo = grid.origin + indices.min(0) * grid.resolution
    hi = grid.origin + (indices.max(0) + 1) * grid.resolution
    return ObservedObjectBound(np.stack((lo, hi)), tuple(observation_ids), received_monotonic, scene_revision)
