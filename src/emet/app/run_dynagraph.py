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

"""CLI entry for Dynagraph (DynaMem + GraphEQA graph lifecycle). See docs/dynagraph.md."""

from __future__ import annotations

import click

from emet.app.robot_cli import create_robot_client_from_cli
from emet.core.parameters import get_parameters
from emet.memory.headless_export import export_graph_eqa_dir


@click.command()
@click.option(
    "--robot_ip",
    "--robot-ip",
    default="127.0.0.1",
    type=str,
    help="Robot IP address (leave empty for saved default)",
)
@click.option(
    "--robot",
    "robot_backend",
    default="stretch",
    type=str,
    help="Robot backend (stretch, rby1, galaxea_r1, etc.). Must match emet serve mujoco --robot.",
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
@click.option(
    "--input-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Load graph memory from a saved directory before running",
)
@click.option(
    "--export",
    "export_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Headless: after spin, save graph here and exit",
)
@click.option(
    "--dump-memory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Save graph memory to this directory when the session ends",
)
@click.option(
    "--cpu-only",
    is_flag=True,
    help="CPU-only: skip loading Qwen3.5 multimodal for scene labels; use voxel fallback",
)
@click.option(
    "--no-sensor-perception",
    is_flag=True,
    help="Do not use VLM scene labels; use voxel image_descriptions only",
)
@click.option(
    "--no-instance-graph",
    is_flag=True,
    help="Disable YoloE instance masks for graph labels",
)
@click.option(
    "--merge-xy-m",
    type=float,
    default=None,
    help="Override dynagraph_merge_xy_m (0 disables merge; default from dynagraph block in yaml if set)",
)
@click.option(
    "--staleness-horizon",
    type=int,
    default=None,
    help="Override dynagraph_staleness_horizon (0 disables pruning)",
)
def main(
    robot_ip: str,
    robot_backend: str = "stretch",
    discord: bool = False,
    not_rotate_in_place: bool = False,
    save_rerun: bool = False,
    port_offset: int = 0,
    input_path: str | None = None,
    export_dir: str | None = None,
    dump_memory: str | None = None,
    cpu_only: bool = False,
    no_sensor_perception: bool = False,
    no_instance_graph: bool = False,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    **kwargs,
) -> None:
    """Run Dynagraph: voxel + graph EQA with optional merge and staleness (see docs/dynagraph.md)."""
    click.echo("Dynagraph: graph memory with DynaMem-style voxel navigation.")
    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=True,
    )

    print("- Load parameters")
    parameters: dict = get_parameters("dynav_config.yaml")
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if merge_xy_m is not None:
        parameters["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        parameters["dynagraph_staleness_horizon"] = int(staleness_horizon)

    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    print("- Start Dynagraph agent")
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.controller.task.dynamem import EQAExecuter

    agent = DynagraphController(
        robot,
        parameters,
        save_rerun=save_rerun,
        graph_memory_input_path=input_path,
        use_sensor_perception=not no_sensor_perception,
        cpu_only=cpu_only,
        use_instance_graph=not no_instance_graph,
    )
    agent.start()

    def _save_dump() -> None:
        if not dump_memory:
            return
        text = export_graph_eqa_dir(
            agent.graph_memory,
            getattr(agent, "voxel_map", None),
            dump_memory,
            title="Scene graph (Dynagraph, saved)",
        )
        print(f"Saved graph memory to {dump_memory}")
        print(text)

    try:
        if export_dir and discord:
            raise click.UsageError("Use either --export or --discord, not both.")

        if export_dir:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()
            text = export_graph_eqa_dir(
                agent.graph_memory,
                getattr(agent, "voxel_map", None),
                export_dir,
                title="Scene graph (Dynagraph export)",
            )
            print(text)
            print(f"Exported graph memory to {export_dir}")
            return

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
                await ctx.send("Hello! I am here to help you (Dynagraph).")

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
                discord_text, _imgs = executor(question)
                if not discord_text.strip():
                    print("(Empty EQA reply — check graph memory / observations.)")
    finally:
        _save_dump()


if __name__ == "__main__":
    main()
