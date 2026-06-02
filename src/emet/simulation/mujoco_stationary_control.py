# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""MuJoCo stationary joint control (MolmoSpaces-style) for **any** EMET sim that steps ``mj_step`` on ``MjModel``.

This is **not** robosuite-specific: the same ``MjModel`` / ``MjData`` / ``RobotSpec`` contract applies whether
the scene comes from robosuite XML, a MolmoSpaces merge, or a plain merged MJCF. Stretch’s
:class:`emet.simulation.stretch_mujoco.mujoco_server.MujocoServer` applies the same stationary **fill** at the
start of each control callback (before ``StatusCommand`` updates), via
:meth:`emet.robots.stretch.StretchBackend.create_mujoco_stationary_control`.

MolmoSpaces analogue (no ``molmo_spaces`` import in emet):

- ``molmo_spaces/tasks/task.py`` — inner loop: ``robot.compute_control()`` then ``env.step``.
- ``molmo_spaces/robots/rby1.py`` — ``compute_control`` writes ``data.ctrl``.
- ``molmo_spaces/controllers/joint_pos.py`` — stationary joint targets + wheel velocity semantics.

We build a **full-length** ``ctrl`` vector (``model.nu``) from each actuator’s **joint transmission** and
current ``MjData.qpos`` (hinge/slide), with velocity / ``wheel*`` actuators at ``0``, then **overlay** the
spec-sized hold buffer (ZMQ ``joint`` / internal targets) onto named actuators in :class:`emet.robots.base.RobotSpec`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

from emet.robots.base import RobotSpec


def _is_joint_velocity_actuator(model: mujoco.MjModel, aid: int) -> bool:
    """True for MuJoCo ``<velocity`` joint actuators (``gear[0] == 3``) or legacy ``wheel*`` names."""
    if abs(float(model.actuator_gear[int(aid), 0]) - 3.0) < 1e-6:
        return True
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
    return str(nm).startswith("wheel")


def compute_stationary_ctrl_vector(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return ``ctrl`` values that command **current** hinge/slide joint angles and zero wheel speeds."""
    u = np.zeros(int(model.nu), dtype=np.float64)
    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    for a in range(int(model.nu)):
        if int(model.actuator_trntype[a]) != trn_joint:
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        if _is_joint_velocity_actuator(model, a):
            u[a] = 0.0
            continue
        jt = int(model.jnt_type[jid])
        if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            qadr = int(model.jnt_qposadr[jid])
            u[a] = float(data.qpos[qadr])
    return u


@runtime_checkable
class MujocoStationaryControl(Protocol):
    """Per-robot MuJoCo interface: refresh ``ctrl`` + optional ``RobotSpec`` hold buffer before ``mj_step``."""

    def write_ctrl_with_spec_hold(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: RobotSpec,
        hold: np.ndarray | None,
    ) -> None:
        """Fill ``data.ctrl`` from stationary physics, then overlay *hold* on spec actuators (if given)."""
        ...

    def sync_ctrl_and_spec_hold(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: RobotSpec,
        hold: np.ndarray,
    ) -> None:
        """Write stationary ``ctrl`` for the whole model and refresh *hold* from the spec actuator slice."""
        ...


class DefaultMujocoStationaryControl:
    """Default implementation: transmission-based ``ctrl`` + :class:`RobotSpec` hold overlay."""

    def write_ctrl_with_spec_hold(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: RobotSpec,
        hold: np.ndarray | None,
    ) -> None:
        full = compute_stationary_ctrl_vector(model, data)
        np.copyto(data.ctrl, full)
        if hold is None:
            return
        n = min(len(spec.actuator_names), int(hold.shape[0]))
        for i in range(n):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.actuator_names[i])
            if aid >= 0:
                data.ctrl[aid] = float(hold[i])

    def sync_ctrl_and_spec_hold(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: RobotSpec,
        hold: np.ndarray,
    ) -> None:
        self.write_ctrl_with_spec_hold(model, data, spec, None)
        n = min(len(spec.actuator_names), len(spec.joint_names), int(hold.shape[0]))
        for i in range(n):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.actuator_names[i])
            if aid >= 0:
                hold[i] = float(data.ctrl[aid])
                continue
            jname = spec.joint_names[i]
            aname = spec.actuator_names[i]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qadr = int(model.jnt_qposadr[jid])
            if aname.startswith("wheel"):
                val = 0.0
            else:
                val = float(data.qpos[qadr])
            hold[i] = val


_DEFAULT = DefaultMujocoStationaryControl()


def write_ctrl_stationary_with_spec_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotSpec,
    hold: np.ndarray | None,
) -> None:
    """Set ``data.ctrl`` from stationary physics, then overlay *hold* on spec actuators."""
    _DEFAULT.write_ctrl_with_spec_hold(model, data, spec, hold)


def sync_stationary_ctrl_and_spec_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotSpec,
    hold: np.ndarray,
) -> None:
    """Write stationary ``ctrl`` for the whole model and refresh *hold* from the spec actuator slice."""
    _DEFAULT.sync_ctrl_and_spec_hold(model, data, spec, hold)
