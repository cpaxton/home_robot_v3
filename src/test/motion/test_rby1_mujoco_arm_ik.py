# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline MuJoCo position-IK smoke for Galaxea R1 / rby1 (no CuRobo, no ZMQ)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    RBY1_LEFT_EE_BODY,
    RBY1_RIGHT_ARM_JOINTS,
    RBY1_RIGHT_EE_BODY,
    solve_position_ik,
)
from emet.robots.rby1 import Rby1Backend


def _rby1_mjcf() -> Path:
    mjcf = Path(Rby1Backend().get_spec().mjcf_path)
    assert mjcf.is_file(), f"missing rby1 MJCF: {mjcf}"
    return mjcf


def _seed_arm_midrange(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> None:
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0
        qadr = int(model.jnt_qposadr[jid])
        if model.jnt_limited[jid]:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            data.qpos[qadr] = 0.5 * (lo + hi)
        else:
            data.qpos[qadr] = 0.3
    mujoco.mj_forward(model, data)


def test_rby1_left_arm_position_ik_reaches_offset():
    model = mujoco.MjModel.from_xml_path(str(_rby1_mjcf()))
    data = mujoco.MjData(model)
    _seed_arm_midrange(model, data, RBY1_LEFT_ARM_JOINTS)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    assert ee_id >= 0
    start = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    target = start + np.array([0.0, 0.0, 0.08])

    result = solve_position_ik(
        model,
        data,
        ee_body=RBY1_LEFT_EE_BODY,
        joint_names=RBY1_LEFT_ARM_JOINTS,
        target_pos=target,
        tol_m=0.015,
    )
    assert result.success, f"left arm IK failed: err={result.pos_error_m:.4f} m after {result.iterations} iters"
    assert result.pos_error_m < 0.015


def test_rby1_right_arm_position_ik_reaches_offset():
    model = mujoco.MjModel.from_xml_path(str(_rby1_mjcf()))
    data = mujoco.MjData(model)
    _seed_arm_midrange(model, data, RBY1_RIGHT_ARM_JOINTS)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_RIGHT_EE_BODY)
    start = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    target = start + np.array([0.05, 0.0, 0.05])

    result = solve_position_ik(
        model,
        data,
        ee_body=RBY1_RIGHT_EE_BODY,
        joint_names=RBY1_RIGHT_ARM_JOINTS,
        target_pos=target,
        tol_m=0.02,
    )
    assert result.success, f"right arm IK failed: err={result.pos_error_m:.4f} m after {result.iterations} iters"


def test_rby1_table_grasp_ik_from_approach_pose():
    """default_table_rby1 approach: pregrasp then grasp must succeed (multi-seed)."""
    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed

    model = mujoco.MjModel.from_xml_path(str(_rby1_mjcf()))
    data = mujoco.MjData(model)
    joints = tuple(f"torso_joint{i}" for i in range(1, 5)) + RBY1_LEFT_ARM_JOINTS
    obj = np.array([0.08, -0.55, 0.59858736])
    base = np.array([0.08, -0.27, -np.pi / 2])
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
    qadr = int(model.jnt_qposadr[jid])
    half = 0.5 * base[2]
    data.qpos[qadr : qadr + 7] = [base[0], base[1], 0.0, np.cos(half), 0.0, 0.0, np.sin(half)]
    home = {
        "torso_joint1": 0.0,
        "torso_joint2": 0.0,
        "torso_joint3": 0.0,
        "torso_joint4": 0.0,
        "left_arm_joint1": 0.0,
        "left_arm_joint2": 0.5,
        "left_arm_joint3": -0.5,
        "left_arm_joint4": 0.0,
        "left_arm_joint5": 0.0,
        "left_arm_joint6": 0.0,
    }
    for name, v in home.items():
        jj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jj])] = v
    mujoco.mj_forward(model, data)

    pre = obj + np.array([0.0, 0.0, 0.15])
    grasp = obj + np.array([0.0, 0.0, 0.02])
    r1 = solve_position_ik_multiseed(
        model,
        data,
        ee_body=RBY1_LEFT_EE_BODY,
        joint_names=joints,
        target_pos=pre,
        tol_m=0.035,
        max_iters=150,
    )
    assert r1.success, f"pregrasp failed err={r1.pos_error_m:.4f}"
    # Perturb like lagging PD feedback, then grasp with last-good seed retry.
    rng = np.random.default_rng(1)
    qadr_list = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in joints]
    good = np.array([float(data.qpos[a]) for a in qadr_list], dtype=np.float64)
    for a in qadr_list:
        data.qpos[a] += float(rng.normal(0, 0.2))
    mujoco.mj_forward(model, data)
    r2 = solve_position_ik_multiseed(
        model,
        data,
        ee_body=RBY1_LEFT_EE_BODY,
        joint_names=joints,
        target_pos=grasp,
        seeds=[good],
        try_midrange=True,
        tol_m=0.035,
        max_iters=150,
    )
    assert r2.success, f"grasp after PD noise failed err={r2.pos_error_m:.4f}"
