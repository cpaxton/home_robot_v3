# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

from emet.robots.footprint import Footprint


class RobotModel(abc.ABC):
    """placeholder"""

    def __init__(
        self,
        name="robot",
        urdf_path: str | None = None,
        visualize=False,
        assets_path=None,
    ):
        self.name = name
        self.urdf_path = urdf_path
        self.visualize = visualize
        self.assets_apth = assets_path

    @abc.abstractmethod
    def get_dof(self) -> int:
        """return degrees of freedom of the robot"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_config(self, q):
        """put the robot in the right position for bullet planning"""
        raise NotImplementedError

    def get_config(self):
        """turn current state into a vector"""
        raise NotImplementedError

    def set_head_config(self, q):
        """just for the head"""
        raise NotImplementedError

    def set_camera_to_head(self, camera, q=None):
        """take a bullet camera and put it on the robot's head"""
        if q is not None:
            self.set_head_config(q)
        raise NotImplementedError

    @abc.abstractmethod
    def get_footprint(self) -> Footprint:
        """return a footprint mask that we can check 2d collisions against"""
        raise NotImplementedError
