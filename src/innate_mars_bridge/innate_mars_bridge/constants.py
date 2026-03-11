# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Topic and frame names for Innate Mars (maurice) robot."""

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
