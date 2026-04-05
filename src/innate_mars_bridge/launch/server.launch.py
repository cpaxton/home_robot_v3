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

"""Launch the Innate Mars ZMQ bridge server (pose, proprioception, head + EE cameras)."""

import launch
from launch_ros.actions import Node


def generate_launch_description():
    start_server = Node(
        package="innate_mars_bridge",
        executable="server",
        name="innate_mars_zmq_server",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    return launch.LaunchDescription([start_server])
