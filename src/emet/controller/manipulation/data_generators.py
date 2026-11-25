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
from emet.controller.manipulation.wbc_contract import EmbodimentProfile


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


def _free_target(
    rng: np.random.Generator,
    lo: np.ndarray,
    hi: np.ndarray,
    boxes: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    for _ in range(200):
        cand = np.asarray([rng.uniform(lo[i], hi[i]) for i in range(3)], dtype=np.float64)
        if not any(np.all(cand >= bmin) and np.all(cand <= bmax) for bmin, bmax in boxes):
            return cand
    return np.asarray([0.0, 0.0, (lo[2] + hi[2]) / 2], dtype=np.float64)


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
    profile: EmbodimentProfile | None = None,
    target_arms: tuple[str, ...] | None = None,
) -> ScenarioRecord:
    """Generate one reconstructible scenario: random obstacles → occupancy + targets.

    Obstacles are axis-aligned boxes rasterized into ``grid_shape`` binary occupancy over
    ``workspace`` (meters). With ``profile``, every selected arm gets an independent
    free-space EE target and ``robot_state`` uses the schema-v2 nested multi-arm layout.
    Without a profile, the legacy single-target / flat-``q`` record is preserved.

    ``target_arms`` selects a subset of the profile's arms; the others stay in the
    observation/action tuple with no target (a one-arm task on a dual-arm robot).
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
        center = np.asarray([rng.uniform(lo[i] + size[i] / 2, hi[i] - size[i] / 2) for i in range(3)], dtype=np.float64)
        bmin, bmax = center - size / 2, center + size / 2
        boxes.append((bmin, bmax))
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    p = np.array([axes[0][x], axes[1][y], axes[2][z]], dtype=np.float64)
                    if np.all(p >= bmin - cell / 2) and np.all(p <= bmax + cell / 2):
                        occ[x, y, z] = 1

    if profile is None:
        target_pos = _free_target(rng, lo, hi, boxes)
        goal: dict[str, Any] = {
            "kind": "ee_target",
            "ee_target": [*target_pos.tolist(), *random_unit_quaternion(rng).tolist()],
        }
        robot_state: dict[str, Any] = {
            "q": [0.0] * n_joints,
            "base_xyt": [0.0, 0.0, 0.0],
        }
        meta: dict[str, Any] = {"source": "synthetic", "failure_mode": ""}
        return ScenarioRecord(initial_voxel=occ, goal=goal, robot_state=robot_state, meta=meta)

    selected = profile.arm_names if target_arms is None else tuple(target_arms)
    unknown = set(selected) - set(profile.arm_names)
    if unknown:
        raise ValueError(f"unknown target arms: {sorted(unknown)}")
    if not selected:
        raise ValueError("target_arms must select at least one arm")

    targets: dict[str, list[float]] = {}
    for name in selected:
        target_pos = _free_target(rng, lo, hi, boxes)
        targets[name] = [*target_pos.tolist(), *random_unit_quaternion(rng).tolist()]

    arms_state: dict[str, Any] = {}
    for arm in profile.arms:
        group = profile.joint_group(arm.joint_group)
        arms_state[arm.name] = {"q": [0.0] * len(group.joints)}
        if arm.gripper is not None:
            arms_state[arm.name]["gripper"] = [0.0] * arm.gripper.dof

    robot_state = {
        "base_xyt": [0.0, 0.0, 0.0],
        "base_twist": [0.0] * profile.base.action_dim,
        "arms": arms_state,
    }
    if profile.torso is not None:
        robot_state["torso"] = {"q": [0.0] * len(profile.torso.joints)}
    goal = {"kind": "ee_targets", "arms": targets, "object": None, "receptacle": None}
    meta = {
        "source": "synthetic",
        "failure_mode": "",
        "embodiment": profile.name,
        "schema_version": 2,
    }
    return ScenarioRecord(initial_voxel=occ, goal=goal, robot_state=robot_state, meta=meta)
