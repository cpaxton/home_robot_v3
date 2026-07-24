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

import mujoco
import numpy as np

from emet.motion.algo import get_planner
from emet.motion.algo.shortcut import Shortcut
from emet.motion.base import ConfigurationSpace
from emet.motion.mujoco_arm_ik import interpolate_arm_waypoints, joint_qpos_addrs
from emet.motion.voxel_arm_collision import VoxelMapArmCollisionChecker
from emet.utils.logger import Logger

logger = Logger(__name__)


@dataclass(frozen=True)
class ArmRrtPlanResult:
    success: bool
    waypoints: list[np.ndarray]
    planner: str
    reason: str | None = None


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
) -> ConfigurationSpace:
    mins, maxs = joint_limits_from_model(model, joint_names)
    return ConfigurationSpace(len(joint_names), mins, maxs, step_size=float(step_size))


def make_arm_validate_fn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    collision: VoxelMapArmCollisionChecker | None,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> Callable[[np.ndarray], bool]:
    """Return a validate(q) that checks joint bounds + optional voxel link collision."""
    qadr = joint_qpos_addrs(model, joint_names)
    if mins is None or maxs is None:
        mins, maxs = joint_limits_from_model(model, joint_names)
    lo = np.asarray(mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(maxs, dtype=np.float64).reshape(-1)

    def validate(q: np.ndarray) -> bool:
        qq = np.asarray(q, dtype=np.float64).reshape(-1)
        if qq.shape[0] != len(qadr):
            return False
        if np.any(qq < lo - 1e-6) or np.any(qq > hi + 1e-6):
            return False
        for a, v in zip(qadr, qq, strict=True):
            data.qpos[a] = float(v)
        if collision is None:
            return True
        return not collision.configuration_collides(model, data)

    return validate


def plan_arm_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_names: Sequence[str],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    collision: VoxelMapArmCollisionChecker | None = None,
    planner: str = "rrt_connect",
    max_iter: int = 400,
    step_size: float = 0.15,
    goal_tolerance: float = 0.05,
    shortcut: bool = True,
    shortcut_iter: int = 50,
    linear_fallback: bool = True,
    linear_steps: int = 15,
    verbose: bool = False,
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

    space = make_arm_configuration_space(model, joint_names, step_size=step_size)
    # Live sim can report slightly out-of-range q; clamp so RRT can start.
    q0 = np.clip(q0, space.mins, space.maxs)
    q1 = np.clip(q1, space.mins, space.maxs)

    # Trivial: already at goal
    if float(np.linalg.norm(q1 - q0)) < float(goal_tolerance):
        return ArmRrtPlanResult(True, [q0.copy(), q1.copy()], planner, None)

    validate = make_arm_validate_fn(model, data, joint_names, collision, mins=space.mins, maxs=space.maxs)

    if not validate(q0):
        return ArmRrtPlanResult(False, [], planner, "invalid_start")
    if not validate(q1):
        return ArmRrtPlanResult(False, [], planner, "invalid_goal")

    algo = str(planner or "rrt_connect").strip().lower()
    if algo in ("rrt", "rrt_connect"):
        pl = get_planner(
            algo,
            space,
            validate,
            max_iter=int(max_iter),
            goal_tolerance=float(goal_tolerance),
        )
        if shortcut:
            pl = Shortcut(pl, shortcut_iter=int(shortcut_iter))
        res = pl.plan(q0, q1, verbose=verbose)
        if res.success and res.trajectory:
            wps = [np.asarray(n.state, dtype=np.float64).copy() for n in res.trajectory]
            return ArmRrtPlanResult(True, wps, algo, None)
        reason = getattr(res, "reason", None) or "rrt_failed"
        logger.debug(f"arm_rrt: {algo} failed ({reason}); linear_fallback={linear_fallback}")
        if not linear_fallback:
            return ArmRrtPlanResult(False, [], algo, reason)
    elif algo != "linear":
        return ArmRrtPlanResult(False, [], algo, "unknown_planner")

    # Linear (explicit request or RRT fallback)
    path = interpolate_arm_waypoints(q0, q1, n_steps=int(linear_steps))
    if collision is not None:
        hit = collision.trajectory_collides(model, data, joint_names=joint_names, arm_waypoints=path)
        if hit is not None:
            return ArmRrtPlanResult(False, [], "linear", f"linear_collision_at_{hit}")
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
