# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for Dynagraph (DynaMem + GraphEQA graph lifecycle). See docs/dynagraph.md."""

from __future__ import annotations

import os

import click
import numpy as np

from emet.app.robot_cli import create_robot_client_from_cli
from emet.app.run_interactive import run_graph_eqa_loop
from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import get_parameters
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    ground_truth_alignment_report,
    gt_pose_sanity_report,
    read_sim_object_placements,
)
from emet.memory.headless_export import export_dynagraph_episode, export_graph_eqa_dir


def _ensure_ground_truth_ready(agent: DynagraphController, *, context: str) -> None:
    """Populate GT graph + Rerun immediately; fail fast when session has no placements."""
    session = agent.robot.get_emet_session()
    if session is None:
        raise click.ClickException(
            f"Ground-truth mode ({context}): no emet_session from the ZMQ server. "
            "Start emet serve mujoco (default, --scene robocasa, or --scene ithor …) with the "
            "same --port-offset as this client, then retry."
        )
    n_bodies = agent.refresh_ground_truth()
    if n_bodies == 0:
        runtime = session.get("runtime_kind", "?")
        raise click.ClickException(
            f"Ground-truth mode ({context}): emet_session has no sim_object_placements "
            f"(runtime_kind={runtime!r}). Restart the sim server from this branch with the same "
            "--port-offset — servers started before the ground-truth feature do not publish placements."
        )
    n_nodes = len(agent.graph_memory.get_nodes()) if agent.graph_memory is not None else 0
    n_boxes = sum(
        1
        for n in (agent.graph_memory.get_nodes() if agent.graph_memory else [])
        if getattr(n, "extent_half", None) is not None
    )
    click.echo(f"Ground truth: {n_bodies} sim bodies → {n_nodes} graph nodes ({n_boxes} with 3D bounds).")
    click.echo(
        "Rerun: «Graph (ground truth)» column — nodes at world/dynagraph/nodes, boxes at world/dynagraph/bboxes."
    )
    placements = read_sim_object_placements(agent.robot.get_emet_session())
    if agent.graph_memory is not None and placements:
        click.echo(ground_truth_alignment_report(agent.graph_memory, placements))
    session = agent.robot.get_emet_session()
    try:
        from emet.utils.geometry import nav_xyt_to_world_xyt

        obs = agent.robot.get_observation()
        gps = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
        comp = np.asarray(obs.compass, dtype=np.float64).ravel()
        local = np.array([float(gps[0]), float(gps[1]), float(comp[0]) if comp.size else 0.0])
        robot_world = nav_xyt_to_world_xyt(local, session)
    except Exception:
        robot_world = None
    click.echo(gt_pose_sanity_report(placements, robot_world_xyt=robot_world, session=session))


def _print_dynagraph_rerun_help(
    *,
    enabled: bool,
    headless: bool,
    ground_truth: bool = False,
    compare_to_gt: bool = False,
) -> None:
    """Dynagraph-specific Rerun hints (web URL is printed from RerunVisualizer after rr.serve)."""
    if not enabled:
        click.echo("Rerun visualization is disabled (--no-rerun).")
        return
    if headless:
        click.echo("Rerun headless: no auto-open browser (use the URL printed when the viewer started).")
    if ground_truth:
        click.echo(
            "Ground-truth mode: use «Graph (ground truth)» for labeled nodes and 3D boxes "
            "(world/dynagraph/nodes, world/dynagraph/bboxes)."
        )
        return
    click.echo("Dynagraph: use «Dynagraph 3D» and «Dynagraph graph» for sensor-built nodes.")
    if compare_to_gt:
        click.echo("Compare mode: green sim reference under «Sim GT (reference)» (world/dynagraph/ground_truth/).")


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
    """Run Dynagraph: graph EQA with DynaMem-style voxel navigation (see docs/dynagraph.md)."""
    click.echo("Dynagraph: connecting to robot and starting graph-based EQA (with merge/staleness).")
    if ground_truth and compare_to_gt:
        raise click.UsageError(
            "--ground-truth and --compare-to-gt are mutually exclusive. "
            "Use --ground-truth to build the graph from sim GT, or --compare-to-gt to evaluate sensor perception."
        )
    if compare_to_gt and not export_dir:
        click.echo(
            "Note: --compare-to-gt prints a full alignment report on --export. "
            "Interactive runs still show the GT Rerun layer; a summary prints when you quit."
        )

    if ground_truth:
        click.echo(
            "Ground-truth mode: graph nodes from sim_object_placements; "
            "voxel map, rotate/explore, and instance detections still run (detections attach to GT nodes)."
        )
        no_sensor_perception = True

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
    _print_dynagraph_rerun_help(
        enabled=not no_rerun,
        headless=headless,
        ground_truth=ground_truth,
        compare_to_gt=compare_to_gt,
    )

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if merge_xy_m is not None:
        parameters["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        parameters["dynagraph_staleness_horizon"] = int(staleness_horizon)

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

    print("- Start Dynagraph agent (graph memory + voxel map for navigation)")
    agent: DynagraphController | None = None
    agent = DynagraphController(
        robot,
        parameters,
        save_rerun=save_rerun,
        graph_memory_input_path=input_path,
        use_sensor_perception=not no_sensor_perception,
        cpu_only=cpu_only,
        use_instance_graph=not no_instance_graph,
        ground_truth_mode=ground_truth,
        visualize_ground_truth=compare_to_gt,
    )
    agent.start()

    if ground_truth:
        _ensure_ground_truth_ready(agent, context="export" if export_dir else "interactive")

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
            placements = read_sim_object_placements(robot.get_emet_session())
            gt_report: str | None = None
            if ground_truth and placements:
                gt_report = ground_truth_alignment_report(agent.graph_memory, placements)
                click.echo(gt_report)
            elif compare_to_gt:
                if placements:
                    gt_report = ground_truth_alignment_report(
                        agent.graph_memory,
                        placements,
                        perception_nodes_only=True,
                    )
                    click.echo(gt_report)
                else:
                    click.echo("Note: --compare-to-gt skipped (no sim_object_placements in emet_session).")
            text = export_dynagraph_episode(
                agent.graph_memory,
                getattr(agent, "voxel_map", None),
                export_dir,
                title="Scene graph (Dynagraph GT export)" if ground_truth else "Scene graph (Dynagraph export)",
                ground_truth_mode=ground_truth,
                sim_object_placements=placements,
                gt_alignment_report_text=gt_report,
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

            run_graph_eqa_loop(agent, executor, robot, app_name="Dynagraph")
    finally:
        _save_dump()
        if dump_memory:
            from emet.memory.utils import print_memory_view_help_on_quit

            print_memory_view_help_on_quit(dump_memory)
        if compare_to_gt and not export_dir and agent is not None:
            placements = read_sim_object_placements(robot.get_emet_session())
            if placements and agent.graph_memory is not None:
                click.echo(
                    ground_truth_alignment_report(
                        agent.graph_memory,
                        placements,
                        perception_nodes_only=True,
                    )
                )


if __name__ == "__main__":
    main()
