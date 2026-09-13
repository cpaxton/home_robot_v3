# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}

# Conservative manipulation setpoint speeds (m/s for arm/lift, rad/s for
# wrist). Both wrist folding and a full-extension position step can eject a
# payload. These are simulator settings, not calibrated real-robot limits.
joint_position_rates = {"arm": 0.1, "lift": 0.15, "wrist_yaw": 0.8, "wrist_pitch": 0.8, "wrist_roll": 0.8}

# The base is also a manipulation joint. Its 2 cm navigation deadband used to
# discard small visual-servo corrections entirely. Keep ordinary navigation
# policies unchanged and target 5 mm when executing an arm's base component.
manipulation_base_xy_tolerance = 0.005


base_motion = {"timeout": 15, "default_x_vel": 0.3, "default_r_vel": 1.0}

# TODO: Add params to tune joints response motion profiles
