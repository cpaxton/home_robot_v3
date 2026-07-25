# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Per-robot arm / torso profiles for kinematic MuJoCo pick-place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    RBY1_LEFT_EE_BODY,
    RBY1_RIGHT_ARM_JOINTS,
    RBY1_RIGHT_EE_BODY,
)


@dataclass(frozen=True)
class ArmManipProfile:
    """Joint / EE / actuator wiring for :class:`KinematicPickPlaceExecutor`."""

    robot_ids: tuple[str, ...]
    ee_body: str
    joint_names: tuple[str, ...]
    link_bodies: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_cmd: tuple[float, ...]
    base_freejoint_name: str = "base_freejoint"
    arm: str = "left"
    home_arm_q: tuple[float, ...] = field(default_factory=tuple)

    @staticmethod
    def for_robot(robot_id: str, *, arm: str = "left") -> ArmManipProfile:
        key = str(robot_id).lower().strip()
        arm_l = str(arm).lower().strip()
        for profile in _all_profiles():
            if key in profile.robot_ids and profile.arm == arm_l:
                return profile
        raise KeyError(
            f"no ArmManipProfile for robot={robot_id!r} arm={arm!r}; known={[p.robot_ids for p in _all_profiles()]}"
        )


_PROFILES: list[ArmManipProfile] | None = None


def _galaxea_profile(*, arm: str) -> ArmManipProfile:
    from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES
    from emet.robots.rby1 import Rby1Backend

    mjcf = Path(Rby1Backend().get_spec().mjcf_path)
    if not mjcf.is_file():
        raise FileNotFoundError(f"missing Galaxea MJCF: {mjcf}")
    if arm == "right":
        joints = tuple(f"torso_joint{i}" for i in range(1, 5)) + RBY1_RIGHT_ARM_JOINTS
        links = tuple(f"right_arm_link{i}" for i in range(3, 7))
        ee = RBY1_RIGHT_EE_BODY
    else:
        joints = tuple(f"torso_joint{i}" for i in range(1, 5)) + RBY1_LEFT_ARM_JOINTS
        links = tuple(f"left_arm_link{i}" for i in range(3, 7))
        ee = RBY1_LEFT_EE_BODY
    home_arm = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0)
    # Matches galaxea_r1.xml key name="home" ctrl (26 actuators).
    home_ctrl = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.5,
        -0.5,
        0,
        0,
        0,
        0.04,
        0.04,
        0,
        0.5,
        -0.5,
        0,
        0,
        0,
        0.04,
        0.04,
    )
    return ArmManipProfile(
        robot_ids=("rby1", "galaxea_r1"),
        ee_body=ee,
        joint_names=joints,
        link_bodies=links,
        actuator_names=tuple(R1_ACTUATOR_NAMES),
        home_cmd=home_ctrl,
        arm=arm,
        home_arm_q=home_arm,
    )


def _all_profiles() -> list[ArmManipProfile]:
    global _PROFILES
    if _PROFILES is None:
        _PROFILES = [_galaxea_profile(arm="left"), _galaxea_profile(arm="right")]
    return _PROFILES


def resolve_manip_mode_for_robot(robot: object, *, manip_mode: str = "auto") -> str:
    """Return ``kinematic`` or ``teleport`` from session capabilities."""
    mode = str(manip_mode or "auto").lower().strip()
    sess: dict = {}
    if hasattr(robot, "get_emet_session"):
        try:
            sess = robot.get_emet_session() or {}
        except Exception:
            sess = {}
    caps = sess.get("capabilities") or {}
    if mode == "auto":
        if caps.get("kinematic_manip"):
            return "kinematic"
        if caps.get("sim_set_body_pose"):
            return "teleport"
        raise RuntimeError("auto manip: server advertises neither kinematic_manip nor sim_set_body_pose")
    if mode == "kinematic":
        if not caps.get("kinematic_manip"):
            raise RuntimeError("manip_mode=kinematic but server lacks kinematic_manip")
        return "kinematic"
    if mode == "teleport":
        if not caps.get("sim_set_body_pose"):
            raise RuntimeError("manip_mode=teleport but server lacks sim_set_body_pose")
        return "teleport"
    raise ValueError(f"unknown manip_mode={manip_mode!r}")


def home_arm_q_array(profile: ArmManipProfile) -> np.ndarray:
    q = profile.home_arm_q
    if len(q) != len(profile.joint_names):
        raise ValueError(f"home_arm_q len {len(q)} != joints {len(profile.joint_names)}")
    return np.asarray(q, dtype=np.float64)


def has_arm_manip_profile(robot_id: str, *, arm: str = "left") -> bool:
    """True when :meth:`ArmManipProfile.for_robot` would succeed."""
    try:
        ArmManipProfile.for_robot(robot_id, arm=arm)
        return True
    except KeyError:
        return False


def robot_id_from_client(robot: object) -> str:
    """Best-effort robot id from GenericZmqClient / Stretch client.

    Raises ``ValueError`` when the id cannot be resolved (do not guess ``rby1``).
    """
    spec = getattr(robot, "_spec", None)
    if spec is not None and getattr(spec, "name", None):
        return str(spec.name)
    for attr in ("robot_id", "robot_name", "_robot_id"):
        v = getattr(robot, attr, None)
        if v:
            return str(v)
    get_sess = getattr(robot, "get_emet_session", None)
    if callable(get_sess):
        try:
            sess = get_sess() or {}
        except Exception:
            sess = {}
        if isinstance(sess, dict):
            for key in ("emet_robot_id", "robot_id"):
                v = sess.get(key)
                if v:
                    return str(v)
            caps_rid = (sess.get("capabilities") or {}).get("emet_robot_id")
            if caps_rid:
                return str(caps_rid)
    raise ValueError("cannot resolve robot id from client (missing _spec / session emet_robot_id)")
