# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Procedural scenario generators for whole-body RL training breadth.

The synthetic generator is scene-agnostic: it synthesizes arbitrary obstacle blocks,
rasterizes them into a local occupancy crop, and picks a random free-space target so
the policy learns "reach around arbitrary obstacles" rather than "reach in one kitchen".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.controller.manipulation.case_library import ScenarioRecord


def random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion (xyzw), via a uniform random rotation."""
    u = rng.uniform(0.0, 1.0, size=3)
    q = np.array(
        [
            np.sqrt(1.0 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1.0 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
        ],
        dtype=np.float64,
    )
    return q


def synthetic_obstacle_target(
    rng: np.random.Generator,
    *,
    grid_shape: tuple[int, int, int] = (32, 32, 32),
    workspace: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.6, -0.6, 0.0),
        (0.6, 0.6, 1.0),
    ),
    n_obstacles: tuple[int, int] = (1, 4),
    obstacle_size: tuple[float, float] = (0.05, 0.25),
    n_joints: int = 10,
) -> ScenarioRecord:
    """Generate one reconstructible scenario: random obstacles → occupancy + free target.

    Obstacles are axis-aligned boxes rasterized into ``grid_shape`` binary occupancy over
    ``workspace`` (meters). The target is a random free-space EE pose (pos + unit quat).
    """
    lo, hi = np.asarray(workspace[0], dtype=np.float64), np.asarray(workspace[1], dtype=np.float64)
    shape = tuple(int(s) for s in grid_shape)
    axes = [np.linspace(lo[i], hi[i], shape[i]) for i in range(3)]
    cell = np.array([(hi[i] - lo[i]) / max(1, shape[i] - 1) for i in range(3)], dtype=np.float64)
    occ = np.zeros(shape, dtype=np.uint8)

    n = int(rng.integers(int(n_obstacles[0]), int(n_obstacles[1]) + 1))
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n):
        size = np.asarray(rng.uniform(obstacle_size[0], obstacle_size[1], size=3), dtype=np.float64)
        center = np.asarray(
            [rng.uniform(lo[i] + size[i] / 2, hi[i] - size[i] / 2) for i in range(3)], dtype=np.float64
        )
        bmin, bmax = center - size / 2, center + size / 2
        boxes.append((bmin, bmax))
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    p = np.array([axes[0][x], axes[1][y], axes[2][z]], dtype=np.float64)
                    if np.all(p >= bmin - cell / 2) and np.all(p <= bmax + cell / 2):
                        occ[x, y, z] = 1

    # Random free-space target position (rejection sample away from obstacles).
    target_pos = None
    for _ in range(200):
        cand = np.asarray([rng.uniform(lo[i], hi[i]) for i in range(3)], dtype=np.float64)
        if not any(np.all(cand >= bmin) and np.all(cand <= bmax) for bmin, bmax in boxes):
            target_pos = cand
            break
    if target_pos is None:
        target_pos = np.asarray([0.0, 0.0, (lo[2] + hi[2]) / 2], dtype=np.float64)

    goal: dict[str, Any] = {
        "kind": "ee_target",
        "ee_target": [*target_pos.tolist(), *random_unit_quaternion(rng).tolist()],
    }
    robot_state: dict[str, Any] = {
        "q": [0.0] * n_joints,
        "base_xyt": [0.0, 0.0, 0.0],
    }
    meta: dict[str, Any] = {"source": "synthetic", "failure_mode": ""}
    return ScenarioRecord(voxel=occ, goal=goal, robot_state=robot_state, meta=meta)
