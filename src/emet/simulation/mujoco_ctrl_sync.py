# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""MuJoCo ``ctrl`` ↔ joint ``qpos`` sync for merged robots (Galaxea / registry backends)."""

from __future__ import annotations

import mujoco

from emet.robots.base import RobotSpec


def sync_actuator_ctrl_from_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_spec: RobotSpec,
) -> None:
    """Set ``data.ctrl`` so actuators command the current ``qpos`` (wheel *velocity* actuators → 0).

    If ``ctrl`` stays at 0 while ``qpos`` is non-zero, the first ``mj_step`` applies large PD errors and
    the robot can bounce or collapse. Re-call after each ``mj_step`` while settling contacts.
    """
    n = min(len(robot_spec.actuator_names), len(robot_spec.joint_names))
    for i in range(n):
        jname = robot_spec.joint_names[i]
        aname = robot_spec.actuator_names[i]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if jid < 0 or aid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if aname.startswith("wheel"):
            data.ctrl[aid] = 0.0
        else:
            data.ctrl[aid] = float(data.qpos[qadr])


def stabilize_physics_inplace(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_spec: RobotSpec,
    *,
    n_steps: int = 24,
) -> None:
    """Zero velocities, run *n_steps* ``mj_step``, re-syncing ``ctrl`` after each step, then zero again."""
    data.qvel.fill(0.0)
    sync_actuator_ctrl_from_joint_positions(model, data, robot_spec)
    mujoco.mj_forward(model, data)
    for _ in range(int(n_steps)):
        mujoco.mj_step(model, data)
        sync_actuator_ctrl_from_joint_positions(model, data, robot_spec)
    data.qvel.fill(0.0)
    sync_actuator_ctrl_from_joint_positions(model, data, robot_spec)
    mujoco.mj_forward(model, data)
