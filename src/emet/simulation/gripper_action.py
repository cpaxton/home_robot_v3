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

"""Map ZMQ gripper actions to MuJoCo actuators (dual-arm robots)."""

from __future__ import annotations

from typing import Any

import mujoco

from emet.robots.base import RobotSpec
from emet.simulation.head_look_action import _set_ctrl_clipped


def _side_from_gripper_key(key: str) -> str | None:
    k = key.lower()
    if k in ("gripper_left", "left_gripper", "jaw_l"):
        return "left"
    if k in ("gripper_right", "right_gripper", "jaw_r"):
        return "right"
    return None


def apply_gripper_action_robosuite(
    spec: RobotSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: dict[str, Any],
) -> list[str]:
    """Apply ``gripper`` / ``gripper_left`` / ``gripper_right`` to MJCF position actuators.

    Returns actuator names that were updated.
    """
    if spec.name != "xlerobot":
        return []

    from emet.robots.xlerobot import XLEROBOT_GRIPPER_ACTUATORS, jaw_angle_from_normalized

    updated: list[str] = []
    targets: dict[str, float] = {}

    for key, val in action.items():
        side = _side_from_gripper_key(str(key))
        if side is None:
            continue
        targets[side] = jaw_angle_from_normalized(float(val))

    if "gripper" in action and "left" not in targets and "right" not in targets:
        # Stretch-style single key defaults to left jaw on XLeRobot.
        targets["left"] = jaw_angle_from_normalized(float(action["gripper"]))

    for side, angle in targets.items():
        aname = XLEROBOT_GRIPPER_ACTUATORS[side]
        if _set_ctrl_clipped(model, data, aname, angle):
            updated.append(aname)

    return updated


def read_xlerobot_gripper_qpos(
    spec: RobotSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, float]:
    """Return jaw joint angles (radians) keyed by ``left`` / ``right``."""
    from emet.robots.xlerobot import XLEROBOT_GRIPPER_JOINTS

    out: dict[str, float] = {}
    if spec.name != "xlerobot":
        return out
    for side, jname in XLEROBOT_GRIPPER_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        out[side] = float(data.qpos[qadr])
    return out
