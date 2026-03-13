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
#
# Entry point for GraphEQA: graph-based memory model for Embodied Question Answering.
# Re-implementation inspired by GraphEQA (https://arxiv.org/abs/2412.14480).

import click

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
from emet.controller.zmq_client import StretchZmqClient
from emet.core.parameters import get_parameters


@click.command()
@click.option(
    "--robot_ip",
    "--robot-ip",
    default="127.0.0.1",
    type=str,
    help="Robot IP address (leave empty for saved default)",
)
@click.option(
    "--not_rotate_in_place",
    "-N",
    is_flag=True,
    help="Whether the robot rotates in place at the beginning",
)
@click.option(
    "--discord",
    "-D",
    is_flag=True,
    help="Whether to launch Discord bot",
)
@click.option(
    "--save_rerun",
    "--SR",
    is_flag=True,
    help="Whether to save Rerun rrd",
)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
def main(
    robot_ip: str,
    discord: bool = False,
    not_rotate_in_place: bool = False,
    save_rerun: bool = False,
    port_offset: int = 0,
    **kwargs,
) -> None:
    """Run GraphEQA: EQA using graph-based semantic memory (see docs/graph_eqa.md)."""
    click.echo("GraphEQA: connecting to robot and starting graph-based EQA.")
    robot = StretchZmqClient(robot_ip=robot_ip, port_offset=port_offset)

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    print("- Start GraphEQA agent (graph memory + voxel map for navigation)")
    agent = GraphEQAController(robot, parameters, save_rerun=save_rerun)
    agent.start()

    if discord:
        from emet.llms.discord_bot import EmetDiscordBot

        bot = EmetDiscordBot(agent, task="graph_eqa")
        if not not_rotate_in_place:
            bot.executor.rotate_in_place()

        @bot.client.command(name="summon", help="Summon the bot to a channel.")
        async def summon(ctx):
            print("Summoning the bot.")
            print(" -> Channel name:", ctx.channel.name)
            print(" -> Channel ID:", ctx.channel.id)
            bot.allowed_channels.visit(ctx.channel)
            await ctx.send("Hello! I am here to help you (GraphEQA).")

        obs = robot.get_observation()
        bot.push_task_to_all_channels(content=obs.rgb)
        bot.run()
    else:
        executor = EQAExecuter(agent)
        if not not_rotate_in_place:
            executor.rotate_in_place()

        while True:
            question = input("Question (Press enter to quit): ").strip()
            if not question:
                break
            robot.move_to_nav_posture()
            robot.switch_to_navigation_mode()
            robot.say("Answering the question " + question)
            executor(question)


if __name__ == "__main__":
    main()
