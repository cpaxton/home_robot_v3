# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Nori A3 — affordable bimanual mobile manipulator (Nori Robotics).

19-DoF dual-arm mobile manipulator on a 45x45 cm differential base with a three-stage
telescoping lift (floor-to-counter reach) and two 7+1 DoF arms. Vendored URDF-derived
MJCF under ``src/emet/assets/robot/nori_a3/`` (CC BY-NC-SA 4.0, from
``github.com/Nori-Robotics/nori_description``; kinematics verified against hardware).

Sim runs through :class:`~emet.simulation.robosuite_server.RobosuiteZmqServer` on the
merged MJCF (base = freejoint, nav teleports; differential wheels are fixed visuals).
Real hardware is driven via the `nori-sdk` operator client (WebRTC jog streams).
"""

from __future__ import annotations

from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path

# MJCF joint names (freejoint base; lift; per-arm 7 DoF; per-gripper driven + idler).
# The passive mimic joints (lift middle stage, gripper idler fingers) are constrained
# via <equality><joint> gearings (MuJoCo has no mjEQ_MIMIC), so they exist but have no
# actuators — they do not float independently under physics.
NORI_JOINT_NAMES = [
    "base_freejoint",
    "lift_extension_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_bicep_yaw_joint",
    "left_elbow_pitch_joint",
    "left_forearm_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_gripper_joint",
    "left_gripper_idler_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_bicep_yaw_joint",
    "right_elbow_pitch_joint",
    "right_forearm_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_gripper_joint",
    "right_gripper_idler_joint",
    "lift_middle_joint",
]

# Actuators (named == joint; 1 lift + 14 arm + 2 driven grippers; mimics unactuated).
NORI_ACTUATOR_NAMES = [
    "lift_extension_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_bicep_yaw_joint",
    "left_elbow_pitch_joint",
    "left_forearm_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_bicep_yaw_joint",
    "right_elbow_pitch_joint",
    "right_forearm_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "left_gripper_joint",
    "right_gripper_joint",
]

NORI_CAMERA_NAMES = ["head_camera", "neck_camera", "left_gripper_camera", "right_gripper_camera"]

# Per-arm 7-DoF chains (mirror about the sagittal plane for the right arm).
_ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "bicep_yaw_joint",
    "elbow_pitch_joint",
    "forearm_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)


def _nori_arm_chain(side: str) -> ArmChain:
    # The telescoping lift is part of the arm chain so IK can lower the column to reach
    # floor objects (floor-to-counter reach); otherwise the arm alone can't reach z≈0.
    joints = ("lift_extension_joint",) + tuple(f"{side}_{s}" for s in _ARM_JOINT_SUFFIXES)
    acts = ("lift_extension_joint",) + tuple(f"{side}_{s}" for s in _ARM_JOINT_SUFFIXES)
    link_bodies = (
        "lift_top_link",
        f"{side}_shoulder_pitch_link",
        f"{side}_shoulder_roll_link",
        f"{side}_bicep_yaw_link",
        f"{side}_elbow_pitch_link",
        f"{side}_forearm_yaw_link",
        f"{side}_wrist_pitch_link",
        f"{side}_wrist_roll_link",
    )
    return ArmChain(
        joint_names=joints,
        ee_body=f"{side}_wrist_roll_link",
        actuator_names=acts,
        link_bodies=link_bodies,
        gripper_bodies=(f"{side}_gripper_link", f"{side}_gripper_idler_link"),
    )


def _nori_mjcf_path() -> str:
    p = get_robot_mjcf_path("nori_a3")
    if p is None or not p.is_file():
        raise RuntimeError(
            "Nori A3 MJCF not found (src/emet/assets/robot/nori_a3/nori_a3.xml). "
            "Regenerate with the vendored nori_description URDF if missing."
        )
    return str(p.resolve())


class NoriBackend(RobotBackend):
    """Nori A3 bimanual mobile manipulator."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="nori",
            dof=len(NORI_JOINT_NAMES),
            joint_names=list(NORI_JOINT_NAMES),
            camera_names=list(NORI_CAMERA_NAMES),
            urdf_path=None,
            mjcf_path=_nori_mjcf_path(),
            actuator_names=list(NORI_ACTUATOR_NAMES),
            base_link_name="base_link",
            footprint=Footprint(width=0.45, length=0.45, width_offset=0.0, length_offset=0.0),
            arm_chain=_nori_arm_chain("left"),
            arm_chains={"left": _nori_arm_chain("left"), "right": _nori_arm_chain("right")},
            advertise_kinematic_manip=True,
            optional_uv_extras=(),
            # Freejoint base + nav teleport (differential wheels are visual-only in sim).
            planar_base_joint_names=None,
            # Slightly wider explored stamp vs default (compact base, short column camera).
            dynav_parameter_overrides={"local_radius": 0.85, "max_depth": 3.2},
        )

    def create_client(self, robot_ip: str, **kwargs):
        """Sim: GenericZmqClient against RobosuiteZmqServer (merged MJCF).

        Real hardware: the `nori-sdk` operator client (WebRTC jog) is the supported
        control path — not the emet ZMQ joint-streaming contract — so real-robot
        driving is a follow-up (see docs/robots/nori.md).
        """
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        from emet.robots.spec_robot_model import SpecRobotModel

        return SpecRobotModel(self.get_spec())


__all__ = [
    "NORI_ACTUATOR_NAMES",
    "NORI_CAMERA_NAMES",
    "NORI_JOINT_NAMES",
    "NoriBackend",
    "_nori_arm_chain",
]