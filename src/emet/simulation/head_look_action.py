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


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> bool:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return False
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr] = float(value)
    vadr = int(model.jnt_dofadr[jid])
    if vadr >= 0:
        data.qvel[vadr] = 0.0
    return True


def _sync_stretch_robocasa_actuators_from_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Align position ``ctrl`` with ``qpos`` when ``RobotSpec.actuator_names`` is empty (Robocasa Stretch).

    Without this, ``head_to`` / ``posture`` only set ``qpos`` while actuators still command 0 and the
    head drifts back within a few ``mj_step`` calls.
    """
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if aname.startswith("wheel") or "vel" in aname:
            data.ctrl[aid] = 0.0
            continue
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(data.qpos[qadr], lo, hi))


def _set_stretch_head_qpos(model: mujoco.MjModel, data: mujoco.MjData, pan: float, tilt: float) -> int:
    """Robocasa Stretch MJCF uses ``joint_head_*`` hinges (often no ``head_pan`` actuators)."""
    n = 0
    for jname, val in (("joint_head_pan", pan), ("head_pan", pan)):
        if _set_joint_qpos(model, data, jname, val):
            n += 1
            break
    for jname, val in (("joint_head_tilt", tilt), ("head_tilt", tilt)):
        if _set_joint_qpos(model, data, jname, val):
            n += 1
            break
    return n


def apply_stretch_posture_to_robosuite(
    spec: RobotSpec, model: mujoco.MjModel, data: mujoco.MjData, posture: str
) -> None:
    """Snap Robocasa Stretch arm/head to navigation or manipulation posture (``posture`` ZMQ action)."""
    import emet.motion.constants as motion_constants
    from emet.motion.kinematics import HelloStretchIdx
    from emet.robots.stretch.joint_layout import robocasa_mjcf_joint_positions_from_hello_stretch

    if posture not in ("navigation", "manipulation"):
        return
    hello_q = (
        motion_constants.STRETCH_NAVIGATION_Q
        if posture == "navigation"
        else motion_constants.STRETCH_PREGRASP_Q
    )
    mjcf_q = robocasa_mjcf_joint_positions_from_hello_stretch(hello_q)
    if mjcf_q is None:
        return
    for jname, val in zip(spec.joint_names, mjcf_q, strict=True):
        _set_joint_qpos(model, data, jname, val)
    _set_stretch_head_qpos(
        model,
        data,
        float(hello_q[HelloStretchIdx.HEAD_PAN]),
        float(hello_q[HelloStretchIdx.HEAD_TILT]),
    )
    _sync_stretch_robocasa_actuators_from_qpos(model, data)
    mujoco.mj_forward(model, data)


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
    This covers ``RobosuiteZmqServer`` (Galaxea R1 / rby1 / innate_mars merged MJCF):
    - Actuators named ``head_pan`` / ``head_tilt`` if present.
    - ``rby1`` / ``galaxea_r1``: map to ``torso2`` / ``torso3`` with reduced gain.
    - ``innate_mars``: ``joint_head`` position actuator driven by ``tilt`` only (Stretch-style nod).
    """
    if spec.name == "stretch":
        n = _set_stretch_head_qpos(model, data, pan, tilt)
        _set_ctrl_clipped(model, data, "head_pan", pan)
        _set_ctrl_clipped(model, data, "head_tilt", tilt)
        _sync_stretch_robocasa_actuators_from_qpos(model, data)
        mujoco.mj_forward(model, data)
        if n > 0:
            return n
        # Fall through to actuator names if present on other Stretch MJCF variants.

    n = 0
    anames = spec.actuator_names
    if not anames and spec.name != "stretch":
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

    if spec.name == "innate_mars":
        # joint_head sim hinge +base X: stereo lk ~ -world Y, so Stretch tilt pitches nod (URDF hinge differs).
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_head")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head")
        if aid < 0 or jid < 0:
            logger.debug(
                "head_to: innate_mars has no joint_head position actuator (MJCF mismatch?); tilt=%r ignored",
                tilt,
            )
            return 0
        lo, hi = model.actuator_ctrlrange[aid]
        # Stretch look_front uses ~−π/4; innate hinge only allows ~[−10°, 30°].
        stretch_tilt = float(tilt)
        if stretch_tilt <= -0.35:
            target = float(lo) + 0.02
        elif stretch_tilt >= 0.1:
            target = float(hi) * 0.35
        else:
            target = float(np.interp(stretch_tilt, [-0.35, 0.0], [float(lo), 0.08]))
        target = float(np.clip(target, lo, hi))
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        current = float(data.qpos[qadr])
        step = float(np.clip(target - current, -0.06, 0.06))
        new_q = current + step
        data.qpos[qadr] = new_q
        data.qvel[vadr] = 0.0
        data.ctrl[aid] = new_q
        return 1

    logger.debug("head_to: no head_pan/head_tilt and no rby1/galaxea mapping for spec %r; ignored", spec.name)
    return 0
