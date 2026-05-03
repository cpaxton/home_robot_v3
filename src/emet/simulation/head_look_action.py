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
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""Map Stretch-style ``head_to`` (pan, tilt) to non-Stretch MuJoCo actuators (spec-driven)."""

from __future__ import annotations

import mujoco
import numpy as np

from emet.robots.base import RobotSpec
from emet.utils.logger import Logger

logger = Logger(__name__)


def _set_ctrl_clipped(model: mujoco.MjModel, data: mujoco.MjData, actuator_name: str, value: float) -> bool:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        return False
    lo, hi = model.actuator_ctrlrange[aid]
    v = float(np.clip(value, lo, hi))
    data.ctrl[aid] = v
    return True


def apply_head_to_robosuite(
    spec: RobotSpec, model: mujoco.MjModel, data: mujoco.MjData, pan: float, tilt: float
) -> int:
    """Apply look pan/tilt to ``data.ctrl`` for the loaded MJCF. Returns number of actuators set.

    Stretch is handled by ``MujocoZmqServerStretch`` (``head_pan`` / ``head_tilt`` in sim).
    This covers ``RobosuiteZmqServer`` (Galaxea R1 / rby1 merged MJCF):
    - Actuators named ``head_pan`` / ``head_tilt`` if present.
    - ``rby1`` / ``galaxea_r1``: map to ``torso2`` / ``torso3`` with reduced gain.
    """
    n = 0
    anames = spec.actuator_names
    if not anames:
        return 0

    if _set_ctrl_clipped(model, data, "head_pan", pan):
        n += 1
    if _set_ctrl_clipped(model, data, "head_tilt", tilt):
        n += 1
    if n > 0:
        return n

    if spec.name in ("rby1", "galaxea_r1"):
        n += int(_set_ctrl_clipped(model, data, "torso2", 0.25 * float(np.clip(pan, -1.2, 1.2))))
        n += int(_set_ctrl_clipped(model, data, "torso3", 0.2 * float(np.clip(tilt, -1.2, 0.3))))
        if n == 0:
            logger.debug("head_to: no torso2/torso3 actuators for spec %r; look request ignored", spec.name)
        return n

    logger.debug("head_to: no head_pan/head_tilt and no rby1/galaxea mapping for spec %r; ignored", spec.name)
    return 0
