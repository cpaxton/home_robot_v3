# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Conservative local floor support; missing returns are unknown, not flat floor."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FloorEvidence:
    observed: np.ndarray
    inferred: np.ndarray
    drops: np.ndarray
    raised: np.ndarray

    def __post_init__(self):
        for layer in (self.observed, self.inferred, self.drops, self.raised):
            layer.setflags(write=False)


def floor_evidence(points, shape, resolution, origin, *, reference_z_m, tolerance_m=0.03, hazard_height_m=0.08):
    """Fill only a single missing cell with four measured, level floor neighbors.

    Reference height must be calibrated in the point cloud's gravity-aligned
    frame. A lower surface is drop evidence; no depth return alone is not.
    This does not fit slopes, certify overhanging surfaces, or detect every cliff.
    """
    raw_values = [resolution, reference_z_m, tolerance_m, hazard_height_m]
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("Floor calibration must be numeric, not boolean")
    values = np.asarray(raw_values, dtype=float)
    resolution, reference_z_m, tolerance_m, hazard_height_m = values
    if not np.isfinite(values).all() or resolution <= 0 or not 0 < tolerance_m < hazard_height_m:
        raise ValueError("Finite calibrated floor reference and 0 < tolerance < hazard height required")
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    origin = np.asarray(origin, dtype=float).reshape(-1)
    if origin.shape not in {(2,), (3,)} or not np.isfinite(origin).all():
        raise ValueError("Finite XY or XYZ grid origin required")
    origin = origin[:2]
    # Match the map's +0.5 cell offset for in-grid points. Floor intentionally
    # rejects a negative fractional edge coordinate instead of truncating it
    # into cell zero and manufacturing known-floor support outside the grid.
    cells = np.floor(xyz[:, :2] / resolution + origin + 0.5).astype(int)
    valid = ((cells >= 0) & (cells < np.asarray(shape))).all(axis=1)
    cells, dz = cells[valid], xyz[valid, 2] - reference_z_m
    seen, observed, drops, raised, conflicting = (np.zeros(shape, dtype=bool) for _ in range(5))
    for mask, selected in (
        (seen, np.ones(len(cells), dtype=bool)),
        (observed, abs(dz) <= tolerance_m),
        (drops, dz <= -hazard_height_m),
        (raised, dz >= hazard_height_m),
        (conflicting, abs(dz) > tolerance_m),
    ):
        indices = cells[selected]
        mask[indices[:, 0], indices[:, 1]] = True
    observed &= ~conflicting
    inferred = np.zeros(shape, dtype=bool)
    inferred[1:-1, 1:-1] = (
        ~seen[1:-1, 1:-1] & observed[:-2, 1:-1] & observed[2:, 1:-1] & observed[1:-1, :-2] & observed[1:-1, 2:]
    )
    return FloorEvidence(observed, inferred, drops, raised)
