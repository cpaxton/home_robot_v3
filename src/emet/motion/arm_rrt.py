# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Joint-space RRT-Connect for kinematic arm motion (reuse of ``emet.motion.algo``).

Plans collision-aware paths between two arm/torso configurations. Collision uses the
agent voxel / 2D obstacle map when provided — not CuRobo / not MJCF mesh geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from emet.motion.algo import get_planner
from emet.motion.algo.shortcut import Shortcut
from emet.motion.base import ConfigurationSpace
from emet.motion.mujoco_arm_ik import interpolate_arm_waypoints, joint_qpos_addrs
from emet.utils.logger import Logger

logger = Logger(__name__)


@dataclass(frozen=True)
class ArmRrtPlanResult:
    success: bool
    waypoints: list[np.ndarray]
    planner: str
    reason: str | None = None
    detail: str | None = None


def joint_limits_from_model(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mins, maxs) for named 1-DoF joints (unlimited joints → ±π)."""
    mins: list[float] = []
    maxs: list[float] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"joint not found: {name!r}")
        if model.jnt_limited[jid]:
            mins.append(float(model.jnt_range[jid][0]))
            maxs.append(float(model.jnt_range[jid][1]))
        else:
            mins.append(-np.pi)
            maxs.append(np.pi)
    return np.asarray(mins, dtype=np.float64), np.asarray(maxs, dtype=np.float64)


def make_arm_configuration_space(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    *,
    step_size: float = 0.15,
    rng=None,
) -> ConfigurationSpace:
    mins, maxs = joint_limits_from_model(model, joint_names)
    return ConfigurationSpace(len(joint_names), mins, maxs, step_size=float(step_size), rng=rng)


def make_arm_validate_fn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    collision: Any | None = None,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> Callable[[np.ndarray], bool]:
    """Return a validate(q) that checks joint bounds + optional link collision."""
    qadr = joint_qpos_addrs(model, joint_names)
    if mins is None or maxs is None:
        mins, maxs = joint_limits_from_model(model, joint_names)
    lo = np.asarray(mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(maxs, dtype=np.float64).reshape(-1)

    def validate(q: np.ndarray) -> bool:
        qq = np.asarray(q, dtype=np.float64).reshape(-1)
        if qq.shape[0] != len(qadr) or not np.isfinite(qq).all():
            return False
        if np.any(qq < lo - 1e-6) or np.any(qq > hi + 1e-6):
            return False
        for a, v in zip(qadr, qq, strict=True):
            data.qpos[a] = float(v)
        if collision is None:
            return True
        return not collision.configuration_collides(model, data)

    return validate


def arm_config_violation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    q: np.ndarray,
    collision: Any | None = None,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> str | None:
    """Return a detailed reason *q* is an invalid configuration, or ``None``.

    Distinguishes the two ``invalid_start`` predicates the generic reason hides:
    an out-of-range joint (``joint_bounds:<name>``) versus a colliding link
    (``collision``). Writes *q* into ``data`` only when a collision check runs,
    matching :func:`make_arm_validate_fn`.
    """
    qadr = joint_qpos_addrs(model, joint_names)
    qq = np.asarray(q, dtype=np.float64).reshape(-1)
    if qq.shape[0] != len(qadr) or not np.isfinite(qq).all():
        return "nonfinite_configuration"
    if mins is None or maxs is None:
        mins, maxs = joint_limits_from_model(model, joint_names)
    lo = np.asarray(mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(maxs, dtype=np.float64).reshape(-1)
    violated = np.flatnonzero((qq < lo - 1e-6) | (qq > hi + 1e-6))
    if violated.size:
        i = int(violated[0])
        return f"joint_bounds:{joint_names[i]}={qq[i]:.4f} in [{lo[i]:.4f},{hi[i]:.4f}]"
    for a, v in zip(qadr, qq, strict=True):
        data.qpos[a] = float(v)
    if collision is not None and collision.configuration_collides(model, data):
        return "collision"
    return None


def plan_arm_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_names: Sequence[str],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    collision: Any | None = None,
    planner: str = "rrt_connect",
    max_iter: int = 400,
    step_size: float = 0.15,
    goal_tolerance: float = 0.05,
    shortcut: bool = True,
    shortcut_iter: int = 50,
    linear_fallback: bool = True,
    linear_steps: int = 15,
    verbose: bool = False,
    rng=None,
) -> ArmRrtPlanResult:
    """Plan a joint-space path from ``q_start`` to ``q_goal``.

    Default planner is RRT-Connect (same stack as base nav). On failure, optionally falls
    back to linear interpolation (still rejected if any waypoint collides when a checker
    is present).
    """
    q0 = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if q0.shape[0] != len(joint_names) or q1.shape[0] != len(joint_names):
        return ArmRrtPlanResult(False, [], planner, "dof_mismatch")
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        return ArmRrtPlanResult(False, [], planner, "nonfinite_configuration")

    space = make_arm_configuration_space(model, joint_names, step_size=step_size, rng=rng)
    validate = make_arm_validate_fn(model, data, joint_names, collision, mins=space.mins, maxs=space.maxs)

    start_reason = arm_config_violation(model, data, joint_names, q0, collision, mins=space.mins, maxs=space.maxs)
    if start_reason is not None:
        return ArmRrtPlanResult(False, [], planner, "invalid_start", detail=start_reason)
    goal_reason = arm_config_violation(model, data, joint_names, q1, collision, mins=space.mins, maxs=space.maxs)
    if goal_reason is not None:
        return ArmRrtPlanResult(False, [], planner, "invalid_goal", detail=goal_reason)

    # Even a short move can cross an obstacle. Validate it before accepting.
    if float(np.linalg.norm(q1 - q0)) < float(goal_tolerance):
        path = [q0.copy(), *list(space.extend(q0, q1))]
        if not all(validate(q) for q in path):
            return ArmRrtPlanResult(False, [], planner, "invalid_short_path")
        return ArmRrtPlanResult(True, path, planner, None)

    algo = str(planner or "rrt_connect").strip().lower()
    if algo in ("rrt", "rrt_connect"):
        pl = get_planner(
            algo,
            space,
            validate,
            max_iter=int(max_iter),
            goal_tolerance=float(goal_tolerance),
            rng=rng,
        )
        if shortcut:
            pl = Shortcut(pl, shortcut_iter=int(shortcut_iter), rng=rng)
        res = pl.plan(q0, q1, verbose=verbose)
        if res.success and res.trajectory:
            wps = [np.asarray(n.state, dtype=np.float64).copy() for n in res.trajectory]
            # Include the exact requested endpoint; planners may stop within tolerance.
            if not np.array_equal(wps[-1], q1):
                wps.append(q1.copy())
            if all(validate(q) for a, b in zip(wps, wps[1:], strict=False) for q in space.extend(a, b)):
                return ArmRrtPlanResult(True, wps, algo, None)
        reason = getattr(res, "reason", None) or "rrt_failed"
        logger.debug(f"arm_rrt: {algo} failed ({reason}); linear_fallback={linear_fallback}")
        if not linear_fallback:
            return ArmRrtPlanResult(False, [], algo, reason)
    elif algo != "linear":
        return ArmRrtPlanResult(False, [], algo, "unknown_planner")

    # Linear (explicit request or RRT fallback)
    path = interpolate_arm_waypoints(q0, q1, n_steps=int(linear_steps))
    for index, (a, b) in enumerate(zip(path, path[1:], strict=False)):
        if not all(validate(q) for q in space.extend(a, b)):
            return ArmRrtPlanResult(False, [], "linear", f"linear_collision_at_{index}")
    return ArmRrtPlanResult(True, path, "linear", None)


def resolve_agent_manip_planner(*, config_mode: str | None = None) -> str:
    """Resolve ``rrt_connect`` | ``rrt`` | ``linear`` from env then config."""
    from emet.simulation.env_flags import env_manip_planner

    env_p = env_manip_planner()
    if env_p:
        return env_p
    mode = str(config_mode or "rrt_connect").strip().lower()
    if mode in ("rrt_connect", "rrt", "linear"):
        return mode
    return "rrt_connect"
