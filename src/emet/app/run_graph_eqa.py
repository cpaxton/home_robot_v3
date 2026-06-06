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

from emet.app.robot_cli import create_robot_client_from_cli
from emet.app.run_interactive import run_graph_eqa_loop
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
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
    help="Load graph memory from a saved directory (common format) before running",
)
@click.option(
    "--export",
    "export_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help=(
        "Headless: after spin, save graph + scene_graph_report.txt here, print graph to stdout, "
        "and exit (no question loop). Use for machines without a TTY."
    ),
)
@click.option(
    "--dump-memory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Save graph memory to this directory when the session ends (empty line to quit)",
)
@click.option(
    "--cpu-only",
    is_flag=True,
    help="CPU-only: skip loading Qwen3.5 multimodal for scene labels; use voxel fallback",
)
@click.option(
    "--no-sensor-perception",
    is_flag=True,
    help="Do not use VLM scene labels; use voxel image_descriptions only (legacy)",
)
@click.option(
    "--no-instance-graph",
    is_flag=True,
    help="Disable YoloE instance masks for graph labels; use voxel VLM list_objects + legacy labeling",
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
    **kwargs,
) -> None:
    """Run GraphEQA: EQA using graph-based semantic memory (see docs/graph_eqa.md)."""
    click.echo("GraphEQA: connecting to robot and starting graph-based EQA.")
    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun_server=True,
    )
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    ev = parameters.get("eqa_vl", {}) or {}
    ms = ev.get("model_size")
    qn = ev.get("quantization", "int4")
    if ms is None or str(ms).lower() == "null":
        print(
            "- EQA VL: one Qwen3.5 load sized by VRAM tiers (see eqa_vl/vram_mib_tier_* in dynav_config.yaml),",
            f"quantization={qn}",
        )
    else:
        print(f"- EQA VL: single shared Qwen3.5-{ms} ({qn}) for labels + EQA")

    print("- Start GraphEQA agent (graph memory + voxel map for navigation)")
    agent = GraphEQAController(
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
            title="Scene graph (saved)",
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
                title="Scene graph (export)",
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
                await ctx.send("Hello! I am here to help you (GraphEQA).")

            obs = robot.get_observation()
            bot.push_task_to_all_channels(content=obs.rgb)
            bot.run()
        else:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()

            run_graph_eqa_loop(agent, executor, robot, app_name="GraphEQA")
    finally:
        _save_dump()
        if dump_memory:
            from emet.memory.utils import print_memory_view_help_on_quit

            print_memory_view_help_on_quit(dump_memory)


if __name__ == "__main__":
    main()
