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
