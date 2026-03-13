#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


import click

from emet.controller.controller_instance_memory import RobotAgent
from emet.controller.task.emote import EmoteTask
from emet.controller.zmq_client import StretchZmqClient
from emet.core import get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--local", is_flag=True, help="Set if we are executing on the robot and not on a remote computer")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "default_planner.yaml",
    port_offset: int = 0,
):
    # Create robot client
    parameters = get_parameters(parameter_file)
    robot = StretchZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
        port_offset=port_offset,
    )

    robot.move_to_nav_posture()

    # create robot agent
    demo = RobotAgent(robot, parameters=parameters)

    # create emote task
    emote_task = EmoteTask(demo)
    task = emote_task.get_task("nod")
    task.run()

    task = emote_task.get_task("shake_head")
    task.run()

    task = emote_task.get_task("wave")
    task.run()

    task = emote_task.get_task("avert_gaze")
    task.run()


if __name__ == "__main__":
    main()
