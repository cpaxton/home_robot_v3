# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for Dynagraph (DynaMem + GraphEQA graph lifecycle). See docs/dynagraph.md."""

from __future__ import annotations

import os

import click
import numpy as np

from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import get_parameters
from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    build_ground_truth_graph_from_session,
    ground_truth_alignment_report,
    read_sim_object_placements,
)
from emet.memory.headless_export import export_graph_eqa_dir


def _export_ground_truth_graph(
    robot,
    parameters: dict,
    export_dir: str,
    *,
    input_path: str | None,
) -> None:
    """Headless GT export without loading Dynamem/CLIP (sim smoke tests)."""
    mem = GraphEQAMemory(parameters=parameters, defer_llm_clients=True)
    if input_path:
        from emet.memory.backend import get_memory_backend

        backend = get_memory_backend("graph_eqa", graph_memory=mem, voxel_map=None)
        backend.load(input_path)
    obs = robot.get_observation()
    n_gt, placements = build_ground_truth_graph_from_session(
        mem,
        np.asarray(obs.rgb, dtype=np.uint8),
        robot.get_emet_session(),
    )
    if n_gt == 0:
        raise click.ClickException(
            "Ground-truth mode: emet_session has no sim_object_placements. "
            "Start emet serve mujoco (default, --use-robocasa, or --molmospaces-scene …) first."
        )
    click.echo(ground_truth_alignment_report(mem, placements))
    text = export_graph_eqa_dir(mem, None, export_dir, title="Scene graph (Dynagraph GT export)")
    print(text)
    print(f"Exported graph memory to {export_dir}")


def _print_dynagraph_rerun_help(*, enabled: bool, headless: bool) -> None:
    """Dynagraph-specific Rerun hints (web URL is printed from RerunVisualizer after rr.serve)."""
    if not enabled:
        click.echo("Rerun visualization is disabled (--no-rerun).")
        return
    if headless:
        click.echo("Rerun headless: no auto-open browser (use the URL printed when the viewer started).")
    click.echo("Dynagraph: use blueprint columns «Dynagraph 3D» and «Dynagraph graph» (3D nodes + tree).")


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
@click.option(
    "--headless",
    is_flag=True,
    help="No auto-open browser for Rerun; open http://<host>:9090 manually.",
)
@click.option(
    "--no-rerun",
    is_flag=True,
    help="Disable Rerun visualization entirely",
)
@click.option(
    "--rerun-native",
    is_flag=True,
    help="Use the native Rerun desktop viewer instead of the browser (needs DISPLAY).",
)
@click.option(
    "--rerun-show-panels",
    is_flag=True,
    help="Show Rerun blueprint/selection panel (useful for debugging)",
)
@click.option(
    "--rerun-debug",
    is_flag=True,
    help="Print Rerun logging status (obs/servo received, step count)",
)
@click.option(
    "--rerun-bind",
    is_flag=True,
    help="Bind Rerun to 0.0.0.0 for remote viewing (Tailscale, etc.).",
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
@click.option(
    "--ground-truth",
    is_flag=True,
    help=(
        "Sim only: build the scene graph from emet_session sim_object_placements "
        "(Robocasa wizard, default table, or MolmoSpaces MJCF scan) instead of VLM perception. "
        "Use with --export for headless GT smoke tests."
    ),
)
@click.option(
    "--compare-to-gt",
    is_flag=True,
    help=(
        "Sim only: after building the graph from sensors (full Dynagraph --export path), "
        "print alignment vs emet_session sim_object_placements."
    ),
)
def main(
    robot_ip: str,
    robot_backend: str = "stretch",
    discord: bool = False,
    not_rotate_in_place: bool = False,
    save_rerun: bool = False,
    headless: bool = False,
    no_rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    rerun_bind: bool = False,
    port_offset: int = 0,
    input_path: str | None = None,
    export_dir: str | None = None,
    dump_memory: str | None = None,
    cpu_only: bool = False,
    no_sensor_perception: bool = False,
    no_instance_graph: bool = False,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    ground_truth: bool = False,
    compare_to_gt: bool = False,
    **kwargs,
) -> None:
    """Run Dynagraph: voxel + graph EQA with optional merge and staleness (see docs/dynagraph.md)."""
    click.echo("Dynagraph: graph memory with DynaMem-style voxel navigation.")
    if ground_truth:
        click.echo("Ground-truth mode: graph nodes from sim_object_placements (no VLM / instance graph).")
        cpu_only = True
        no_sensor_perception = True
        no_instance_graph = True

    if rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"
    if rerun_native and headless:
        raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")

    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=not no_rerun,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
    )
    _print_dynagraph_rerun_help(enabled=not no_rerun, headless=headless)

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if merge_xy_m is not None:
        parameters["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        parameters["dynagraph_staleness_horizon"] = int(staleness_horizon)

    if ground_truth and export_dir:
        if discord:
            raise click.UsageError("Use either --export or --discord, not both.")
        if not not_rotate_in_place:
            click.echo(
                "Note: --ground-truth --export uses the lightweight GT path (no rotate_in_place). "
                "Omit --export for the full Dynagraph stack with rotation."
            )
        try:
            _export_ground_truth_graph(robot, parameters, export_dir, input_path=input_path)
        finally:
            robot.stop()
        return

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

    print("- Start Dynagraph agent")
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
            if compare_to_gt:
                placements = read_sim_object_placements(robot.get_emet_session())
                if placements:
                    click.echo(ground_truth_alignment_report(agent.graph_memory, placements))
                else:
                    click.echo("Note: --compare-to-gt skipped (no sim_object_placements in emet_session).")
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
            if ground_truth:
                obs = robot.get_observation()
                n_gt, placements = build_ground_truth_graph_from_session(
                    agent.graph_memory,
                    np.asarray(obs.rgb, dtype=np.uint8),
                    robot.get_emet_session(),
                )
                if n_gt == 0:
                    raise click.ClickException("Ground-truth mode: emet_session has no sim_object_placements.")
                click.echo(ground_truth_alignment_report(agent.graph_memory, placements))

            click.echo(
                "Interactive mode: type a **question** to run graph EQA, "
                "**explore** (or **e**) to extend the map without calling the EQA model, "
                "or Enter to quit."
            )
            while True:
                question = input("Dynagraph [question | explore | Enter=quit]: ").strip()
                if not question:
                    break
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                low = question.lower()
                if low in ("explore", "e", "map", "nav"):
                    click.echo("- Exploring (frontier navigation, no EQA call)…")
                    finished, _pt = agent.execute_action("")
                    if finished is None:
                        click.echo("Explore step failed (no plan / blocked). Map may still grow on the next update.")
                    elif finished:
                        click.echo("Explore step finished at a manipulation-ready pose.")
                    else:
                        click.echo("Explore step advanced; ask a question or explore again.")
                    continue
                robot.say("Answering the question " + question)
                discord_text, _imgs = executor(question)
                if not discord_text.strip():
                    print("(Empty EQA reply — check graph memory / observations.)")
    finally:
        _save_dump()


if __name__ == "__main__":
    main()
