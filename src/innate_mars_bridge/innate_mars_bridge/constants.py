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

"""Topic and frame names for Innate Mars (maurice) robot.

Pinned against innate-os ``main`` (2026-06): ``maurice_cam``, ``maurice_arm``, ``maurice_nav``.
Run ``uv run python scripts/audit_innate_os_topics.py`` on a live ROS graph to verify.
"""

# Upstream reference (https://github.com/innate-inc/innate-os)
INNATE_OS_GIT_REF = "main"
INNATE_OS_REPO = "https://github.com/innate-inc/innate-os"

# Arm joint state (JointState: joint1..joint6)
ARM_STATE_TOPIC = "/mars/arm/state"

# Odometry and TF
ODOM_TOPIC = "/odom"
BASE_FOOTPRINT_FRAME = "base_footprint"
ODOM_FRAME = "odom"
MAP_FRAME = "map"

# Head (stereo) camera - left and right
HEAD_LEFT_IMAGE_TOPIC = "/mars/main_camera/left/image_raw"
HEAD_LEFT_CAMERA_INFO_TOPIC = "/mars/main_camera/left/camera_info"
HEAD_RIGHT_IMAGE_TOPIC = "/mars/main_camera/right/image_raw"
HEAD_RIGHT_CAMERA_INFO_TOPIC = "/mars/main_camera/right/camera_info"

# End-effector / arm camera
EE_IMAGE_TOPIC = "/mars/arm/image_raw"

# Optional: head position (if published)
HEAD_POSITION_TOPIC = "/mars/head/current_position"

# Default camera frame IDs (from maurice_cam)
HEAD_LEFT_FRAME_ID = "camera_optical_frame"
HEAD_RIGHT_FRAME_ID = "right_camera_optical_frame"
EE_CAMERA_FRAME_ID = "arm_camera_optical_frame"

# Nav2 (maurice_nav) — standard Nav2 action; goal topic fallback per innate-os / PAL docs
NAVIGATE_TO_POSE_ACTION = "navigate_to_pose"
GOAL_POSE_TOPIC = "/goal_pose"

# Expected on a running Mars stack (audit script checks these)
EXPECTED_TOPICS = (
    ARM_STATE_TOPIC,
    ODOM_TOPIC,
    HEAD_LEFT_IMAGE_TOPIC,
    HEAD_LEFT_CAMERA_INFO_TOPIC,
    HEAD_RIGHT_IMAGE_TOPIC,
    HEAD_RIGHT_CAMERA_INFO_TOPIC,
    EE_IMAGE_TOPIC,
)

EXPECTED_TF_FRAMES = (
    BASE_FOOTPRINT_FRAME,
    ODOM_FRAME,
    MAP_FRAME,
    HEAD_LEFT_FRAME_ID,
    HEAD_RIGHT_FRAME_ID,
    EE_CAMERA_FRAME_ID,
    "ee_link",
    "base_link",
)
