# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Pick / place wrappers over DynamemManipulationWrapper."""

from __future__ import annotations

from emet.controller.dynamem.constants import (
    INIT_ARM_POS,
    INIT_HEAD_PAN,
    INIT_HEAD_TILT,
    INIT_LIFT_POS,
    INIT_WRIST_PITCH,
    INIT_WRIST_ROLL,
    INIT_WRIST_YAW,
)
from emet.controller.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    move_to_point,
    pickup,
    process_image_for_placing,
)
from emet.perception.detection.yoloe import YoloEPerception
from emet.utils.logger import Logger

logger = Logger(__name__)


def place(
    self,
    text,
    local=True,
    init_tilt=INIT_HEAD_TILT,
    base_node="camera_depth_optical_frame",
):
    """
    An API for running placing. By calling this API, human will ask the robot to place whatever it holds
    onto objects specified by text queries A
    - hello_robot: a wrapper for home-robot StretchClient controller
    - socoket: we use this to communicate with workstation to get estimated gripper pose
    - text: queries specifying target object
    - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
    - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
    """
    self.robot.switch_to_manipulation_mode()
    self.robot.look_at_ee()
    self.manip_wrapper.move_to_position(head_pan=INIT_HEAD_PAN, head_tilt=init_tilt)

    if not local:
        rotation, translation = capture_and_process_image(
            mode="place",
            obj=text,
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
        )
    else:
        if self.owl_sam_detector is None:
            # We can opt to use OWLv2 + SAMv2 for accurate object detection, but the placing receptacles are usually very easy to detect,
            # so we don't see the point of installing SAMv2 and using it
            # from emet.perception.detection.owl import OWLSAMProcessor

            # self.owl_sam_detector = OWLSAMProcessor(confidence_threshold=0.1)

            # A misnomer, this is actually YOLOE while its named as self.owl_sam_detector
            self.owl_sam_detector = YoloEPerception(confidence_threshold=0.05, size="l", device=self.device)
        rotation, translation = process_image_for_placing(
            obj=text,
            hello_robot=self.manip_wrapper,
            detection_model=self.owl_sam_detector,
            save_dir=self.log,
        )
    logger.debug("Place: rotation=%s translation=%s", rotation, translation)

    if rotation is None:
        return False

    # lift arm to the top before the robot extends the arm, prepare the pre-placing gripper pose
    self.manip_wrapper.move_to_position(lift_pos=1.05)
    self.manip_wrapper.move_to_position(wrist_yaw=0, wrist_pitch=0)

    # Placing the object
    move_to_point(self.manip_wrapper, translation, base_node, self.transform_node, move_mode=0)
    self.manip_wrapper.move_to_position(gripper_pos=1, blocking=True)

    # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
    self.manip_wrapper.move_to_position(lift_pos=min(self.manip_wrapper.robot.get_six_joints()[1] + 0.3, 1.1))
    self.manip_wrapper.move_to_position(wrist_roll=2.5, blocking=True)
    self.manip_wrapper.move_to_position(wrist_roll=-2.5, blocking=True)

    # Wait for some time and shrink the arm back
    self.manip_wrapper.move_to_position(gripper_pos=1, lift_pos=1.05, arm_pos=0)
    self.manip_wrapper.move_to_position(wrist_pitch=-1.57)

    # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
    self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])
    return True


def get_voxel_map(self):
    """Return the voxel map"""
    return self.voxel_map


def manipulate(
    self,
    text,
    init_tilt=INIT_HEAD_TILT,
    base_node="camera_depth_optical_frame",
    skip_confirmation: bool = False,
):
    """
    An API for running manipulation. By calling this API, human will ask the robot to pick up objects
    specified by text queries A
    - hello_robot: a wrapper for home-robot StretchClient controller
    - socoket: we use this to communicate with workstation to get estimated gripper pose
    - text: queries specifying target object
    - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
    - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
    """

    self.robot.switch_to_manipulation_mode()
    self.robot.look_at_ee()

    gripper_pos = 1

    self.manip_wrapper.move_to_position(
        arm_pos=INIT_ARM_POS,
        head_pan=INIT_HEAD_PAN,
        head_tilt=init_tilt,
        gripper_pos=gripper_pos,
        lift_pos=INIT_LIFT_POS,
        wrist_pitch=INIT_WRIST_PITCH,
        wrist_roll=INIT_WRIST_ROLL,
        wrist_yaw=INIT_WRIST_YAW,
    )

    rotation, translation, depth, width = capture_and_process_image(
        mode="pick",
        obj=text,
        socket=self.manip_socket,
        hello_robot=self.manip_wrapper,
    )

    if rotation is None:
        return False

    if width < 0.05 and self.re == 3:
        gripper_width = 0.45
    elif width < 0.075 and self.re == 3:
        gripper_width = 0.6
    else:
        gripper_width = 1

    confirmed = skip_confirmation or input("Do you want to do this manipulation? Y or N ") != "N"
    if confirmed:
        pickup(
            self.manip_wrapper,
            rotation,
            translation,
            base_node,
            self.transform_node,
            gripper_depth=depth,
            gripper_width=gripper_width,
        )

    # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
    self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])

    return bool(confirmed)
