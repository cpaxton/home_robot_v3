# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke-only grasp planting helpers (multi-option TAMP demos / unit tests)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def plant_mixed_grasp_poses(
    reachable_xyz: np.ndarray | Sequence[float],
    *,
    n_infeasible: int = 2,
    infeasible_offset: np.ndarray | Sequence[float] | None = None,
) -> list[Any]:
    """Build grasp candidates with infeasible poses first, then a reachable COM-style grasp.

    Used by the multi-option TAMP smoke so naive ``chosen=0`` would pick a decoy.
    """
    from emet.perception.grasps.oracle import GraspPose

    good = np.asarray(reachable_xyz, dtype=np.float64).reshape(3)
    if infeasible_offset is None:
        # Far + high: outside typical table workspace for rby1 left arm.
        offset = np.array([2.5, 2.5, 1.5], dtype=np.float64)
    else:
        offset = np.asarray(infeasible_offset, dtype=np.float64).reshape(3)
    poses: list[Any] = []
    for i in range(max(1, int(n_infeasible))):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = good + offset * (1.0 + 0.15 * i)
        poses.append(GraspPose(T_world=T, score=0.1, asset_id=f"decoy_{i}"))
    T_ok = np.eye(4, dtype=np.float64)
    T_ok[:3, 3] = good
    poses.append(GraspPose(T_world=T_ok, score=1.0, asset_id="reachable_com"))
    return poses
