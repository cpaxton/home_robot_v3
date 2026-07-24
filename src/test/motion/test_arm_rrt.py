# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for joint-space arm RRT-Connect (voxel collision + linear fallback)."""

from __future__ import annotations

import numpy as np

from emet.motion.algo import get_planner
from emet.motion.arm_rrt import plan_arm_joint_path, resolve_agent_manip_planner
from emet.motion.base import ConfigurationSpace
from emet.motion.voxel_arm_collision import link_samples_collide_2d


def test_resolve_agent_manip_planner_default():
    assert resolve_agent_manip_planner(config_mode=None) == "rrt_connect"
    assert resolve_agent_manip_planner(config_mode="linear") == "linear"
    assert resolve_agent_manip_planner(config_mode="bogus") == "rrt_connect"


def test_rrt_connect_2d_avoids_obstacle():
    """Classic 2D RRT-Connect: start/goal free, wall blocks straight line."""
    obs = np.zeros((20, 20), dtype=bool)
    # Vertical wall in the middle (cells x=9..10, y=0..19) — leave a gap at bottom
    obs[9:11, 2:] = True
    origin = np.array([0.0, 0.0])
    res = 0.1

    def validate(q: np.ndarray) -> bool:
        x, y = float(q[0]), float(q[1])
        if not (0.0 <= x <= 1.9 and 0.0 <= y <= 1.9):
            return False
        return not link_samples_collide_2d(obs, grid_origin=origin, resolution=res, sample_xy=[(x, y)], inflate_cells=0)

    space = ConfigurationSpace(2, np.array([0.0, 0.0]), np.array([1.9, 1.9]), step_size=0.15)
    start = np.array([0.2, 1.0])
    goal = np.array([1.7, 1.0])
    assert validate(start) and validate(goal)
    # Straight line through wall must be invalid
    mid = 0.5 * (start + goal)
    assert not validate(mid)

    np.random.seed(0)
    planner = get_planner("rrt_connect", space, validate, max_iter=800, goal_tolerance=0.12)
    result = planner.plan(start, goal, verbose=False)
    assert result.success, "RRT-Connect should find a path around the wall"
    assert result.trajectory is not None and len(result.trajectory) >= 2
    for node in result.trajectory:
        assert validate(node.state)


def test_plan_arm_joint_path_linear_explicit():
    """With planner=linear and no collision, returns interpolated path."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Use only first 2 arm joints for a tiny plan (faster / simpler)
    joints = RBY1_LEFT_ARM_JOINTS[:2]
    from emet.motion.mujoco_arm_ik import joint_qpos_addrs

    qadr = joint_qpos_addrs(model, joints)
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    q1 = q0 + np.array([0.2, -0.15])
    # Clamp into limits
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        q1[i] = float(np.clip(q1[i], lo, hi))

    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q1,
        collision=None,
        planner="linear",
        linear_steps=5,
        linear_fallback=False,
    )
    assert plan.success
    assert plan.planner == "linear"
    assert len(plan.waypoints) == 6


def test_plan_arm_rrt_connect_no_collision():
    """RRT-Connect in joint space with no obstacle map should reach a nearby goal."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, joint_qpos_addrs
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joints = RBY1_LEFT_ARM_JOINTS[:3]
    qadr = joint_qpos_addrs(model, joints)
    for i, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[qadr[i]] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    q0 = np.array([float(data.qpos[a]) for a in qadr], dtype=np.float64)
    q1 = q0.copy()
    q1[0] = float(
        np.clip(
            q0[0] + 0.35,
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])][0],
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])][1],
        )
    )

    np.random.seed(1)
    plan = plan_arm_joint_path(
        model,
        data,
        joint_names=joints,
        q_start=q0,
        q_goal=q1,
        collision=None,
        planner="rrt_connect",
        max_iter=200,
        linear_fallback=True,
        shortcut=False,
    )
    assert plan.success, f"expected success, got {plan}"
    assert plan.planner in ("rrt_connect", "linear")
    assert len(plan.waypoints) >= 2


def test_voxel_checker_rejects_colliding_config():
    obs = np.zeros((10, 10), dtype=bool)
    obs[5, 5] = True
    assert link_samples_collide_2d(obs, grid_origin=np.array([0.0, 0.0]), resolution=0.1, sample_xy=[(0.55, 0.55)])
    assert not link_samples_collide_2d(obs, grid_origin=np.array([0.0, 0.0]), resolution=0.1, sample_xy=[(0.1, 0.1)])
