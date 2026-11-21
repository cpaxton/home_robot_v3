# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from .utils import default_robot_xml_path, default_scene_xml_path, models_path

__all__ = ["StretchMujocoSimulator", "default_robot_xml_path", "default_scene_xml_path", "models_path"]


def __getattr__(name):
    if name == "StretchMujocoSimulator":
        from .stretch_mujoco_simulator import StretchMujocoSimulator

        return StretchMujocoSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
