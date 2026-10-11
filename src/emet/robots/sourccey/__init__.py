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

"""Sourccey Mk.V: official full assembly adapted to Emet planar simulation.

See assets/robot/sourccey/upstream/manifest.json for pinned source revisions.
The ZMQ client speaks Emet's simulation protocol, not the vendor hardware protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend


def _sourccey_mjcf_path() -> str:
    """Same file as :func:`get_robot_mjcf_path` (importlib resource path)."""
    p = get_robot_mjcf_path("sourccey")
    if p is None or not p.is_file():
        raise RuntimeError(
            "Sourccey MJCF not found. Use a full emet install with package data, or a checkout where "
            "src/emet/assets/robot/sourccey/sourccey.xml exists."
        )
    return str(p.resolve())


def _sourccey_urdf_path() -> str | None:
    """Unmodified official full-robot Mk.V URDF."""
    mjcf = Path(_sourccey_mjcf_path())
    urdf = mjcf.parent / "upstream/models/source/SourcceyURDF/SourcceyMkV.urdf"
    return str(urdf.resolve()) if urdf.is_file() else None


# Planar base + lift + dual 5-DOF arms + grippers. Order matches ``sourccey.xml`` joints.
SOURCCEY_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "lift",
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

SOURCCEY_ACTUATOR_NAMES = [
    "base_x_act",
    "base_y_act",
    "base_yaw_act",
    "lift_act",
    "left_shoulder_pan_act",
    "left_shoulder_lift_act",
    "left_elbow_flex_act",
    "left_wrist_flex_act",
    "left_wrist_roll_act",
    "left_gripper_act",
    "right_shoulder_pan_act",
    "right_shoulder_lift_act",
    "right_elbow_flex_act",
    "right_wrist_flex_act",
    "right_wrist_roll_act",
    "right_gripper_act",
]

SOURCCEY_CAMERA_NAMES = ["front_left", "front_right", "bottom", "wrist_left", "wrist_right"]

# Gripper joint -> actuator name (single revolute gear per side).
SOURCCEY_GRIPPER_JOINTS = {"left": "left_gripper", "right": "right_gripper"}
SOURCCEY_GRIPPER_ACTUATORS = {"left": "left_gripper_act", "right": "right_gripper_act"}

# Raised-arm home used by robosuite_load_utils / spawns.
SOURCCEY_HOME_KEYFRAME = "sourccey_home"

# Per-arm IK chain (shoulder_pan … wrist_roll). The gripper is a separate actuator
# so position IK cannot chew the fingers; ``_set_gripper`` drives ``{side}_gripper_act``.
_ARM_IK_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def _sourccey_arm_chain(side: str) -> ArmChain:
    joints = tuple(f"{side}_{s}" for s in _ARM_IK_SUFFIXES)
    acts = tuple(f"{side}_{s}_act" for s in (*_ARM_IK_SUFFIXES, "gripper"))
    suffix = "1" if side == "left" else "2"
    home = (-0.785, 2.01, -1.570796, 0.691, 0.0) if side == "left" else (-0.801, -2.01, -1.570796, -0.723, 0.0)
    return ArmChain(
        joint_names=joints,
        # Wrist-roll origin is independent of finger opening, as in upstream IK.
        ee_body=f"Gripper_Base_v1_{suffix}",
        actuator_names=acts,
        link_bodies=tuple(
            f"{name}_v1_{suffix}"
            for name in ("Shoulder", "Bicep_Left", "Forearm", "Wrist", "Gripper_Base", "Gripper_Finger")
        ),
        gripper_bodies=(f"Gripper_Finger_v1_{suffix}",),
        home_arm_q=home,
        gripper_open=1.0471975511965976,
        gripper_closed=-0.08726646259971647,
    )


class SourcceyBackend(RobotBackend):
    """Sourccey: planar base + vertical lift + dual 5-DOF arms + grippers.

    Sim runs through :class:`~emet.simulation.robosuite_server.RobosuiteZmqServer` on the
    vendored MJCF; kinematic pick/place (``capabilities.kinematic_manip``) is advertised.
    ``create_client`` returns an Emet ``GenericZmqClient`` for simulation. The
    vendor hardware host uses a different protocol and uncalibrated native units;
    it requires a separate, calibrated adapter before autonomous Emet use.
    """

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="sourccey",
            dof=len(SOURCCEY_JOINT_NAMES),
            joint_names=list(SOURCCEY_JOINT_NAMES),
            camera_names=list(SOURCCEY_CAMERA_NAMES),
            urdf_path=_sourccey_urdf_path(),
            mjcf_path=_sourccey_mjcf_path(),
            actuator_names=list(SOURCCEY_ACTUATOR_NAMES),
            base_link_name="base_root",
            footprint=Footprint(width=0.50, length=0.50, width_offset=0.0, length_offset=0.0),
            planar_base_joint_names=("base_x", "base_y", "base_yaw"),
            arm_chain=_sourccey_arm_chain("left"),
            arm_chains={"left": _sourccey_arm_chain("left"), "right": _sourccey_arm_chain("right")},
            advertise_kinematic_manip=True,
            tamp_approach="front",
            # arm meshes are visual-only in the MJCF; inflate clip erosion + require EE XY inside floor.
            planar_spawn_xy_extra_margin_m=0.25,
            planar_spawn_clip_guard_body_names=(
                "Gripper_Base_v1_1",
                "Gripper_Base_v1_2",
                "Wrist_v1_1",
                "Wrist_v1_2",
            ),
            planar_spawn_clip_guard_pad_m=0.25,
            planar_spawn_robocasa_first_clearance_m=0.068,
            # MuJoCo Renderer already returns top-down pixels. Front cameras
            # have +Y up in the asset; an extra flip inverts their image horizon.
            robosuite_rgb_depth_ops=(),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def get_emote_backend(self) -> EmoteBackend:
        from emet.robots.sourccey.emote_backend import SourcceyEmoteBackend

        return SourcceyEmoteBackend()

    def create_model(self, **kwargs):
        from emet.robots.spec_robot_model import SpecRobotModel

        return SpecRobotModel(self.get_spec())


__all__ = [
    "SOURCCEY_ACTUATOR_NAMES",
    "SOURCCEY_CAMERA_NAMES",
    "SOURCCEY_GRIPPER_ACTUATORS",
    "SOURCCEY_GRIPPER_JOINTS",
    "SOURCCEY_HOME_KEYFRAME",
    "SOURCCEY_JOINT_NAMES",
    "SourcceyBackend",
    "_sourccey_arm_chain",
]
