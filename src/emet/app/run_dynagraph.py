# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for Dynagraph (DynaMem + GraphEQA graph lifecycle). See docs/dynagraph.md."""

from __future__ import annotations

import os
import time
from typing import Any

import click

from emet.app.dynagraph_explore import dynagraph_explore_until_terminated
from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import get_parameters
from emet.memory.graph_eqa import format_scene_graph_pretty
from emet.memory.headless_export import export_graph_eqa_dir
from emet.robots import apply_robot_dynav_parameter_overrides, resolve_dynav_config_yaml
from emet.utils.logger import Logger

logger = Logger(__name__)


def _print_dynagraph_rerun_help(*, enabled: bool, headless: bool) -> None:
    """Dynagraph-specific Rerun hints (web URL is printed from RerunVisualizer after rr.serve)."""
    if not enabled:
        click.echo("Rerun visualization is disabled (--no-rerun).")
        return
    if headless:
        click.echo("Rerun headless: no auto-open browser (use the URL printed when the viewer started).")
    click.echo("Dynagraph: 3D world view + graph node list; full tree text is export/stdout only (not live Rerun).")


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
    "--rerun",
    is_flag=True,
    help="Rerun is already on by default; accepted for compatibility with `emet run agent --rerun` (no-op).",
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
    "--dynav-config",
    "--dynav_config",
    type=str,
    default="dynav_config.yaml",
    help="Graph/voxel YAML: basename under emet/config/, cwd path, or absolute. Resolved with robot preset "
    "(same as dynamem); use dynav_innate_mars.yaml for Innate Mars + DA3 when needed.",
)
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
    help="save graph backend + scene_graph_report.txt here, print summary, exit (unless combined with discord)",
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
    "--perfect-depth",
    "--perfect_depth",
    is_flag=True,
    help="Prefer observation sensor depth over DA3 when available (sim calibration / Robocasa)",
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
    "--explore-loop",
    is_flag=True,
    help="Frontier exploration batch (run_exploration) before interactive/export continues (heuristic termination).",
)
@click.option(
    "--explore-max-iters",
    type=int,
    default=64,
    show_default=True,
    help="Max frontier excursions when --explore-loop is set",
)
@click.option(
    "--explore-max-failures",
    type=int,
    default=3,
    show_default=True,
    help="Stop explore-loop after this many consecutive failed run_exploration calls",
)
@click.option(
    "--explore-timeout-s",
    type=float,
    default=None,
    help="Wall-clock timeout (seconds) for explore-loop; omit for no timeout",
)
@click.option(
    "--question",
    type=str,
    default=None,
    help="Single NL GraphEQA question (batch); exits after answering unless combined with discord",
)
@click.option(
    "--print-graph",
    is_flag=True,
    help="Pretty-print GraphEQAMemory at session exit (finally); often used with interactive runs",
)
@click.option(
    "--dump-sim-ground-truth",
    type=str,
    default=None,
    help="MuJoCo sim (`emet serve mujoco`): simulator writes body pose snapshot on the simulator host at PATH (.json ⇒ JSON)",
)
@click.option(
    "--dump-sim-gt-include-robot",
    is_flag=True,
    help="With --dump-sim-ground-truth: include kinematic subtree of robot base_link (verbose)",
)
@click.option(
    "--calibration-export",
    default=None,
    type=str,
    help="Append raw instance detections per step to JSONL for emet tune-graph-fusion",
)
def main(
    robot_ip: str,
    robot_backend: str = "stretch",
    discord: bool = False,
    not_rotate_in_place: bool = False,
    save_rerun: bool = False,
    headless: bool = False,
    rerun: bool = False,
    no_rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    rerun_bind: bool = False,
    port_offset: int = 0,
    dynav_config: str = "dynav_config.yaml",
    input_path: str | None = None,
    export_dir: str | None = None,
    dump_memory: str | None = None,
    cpu_only: bool = False,
    perfect_depth: bool = False,
    no_sensor_perception: bool = False,
    no_instance_graph: bool = False,
    merge_xy_m: float | None = None,
    staleness_horizon: int | None = None,
    explore_loop: bool = False,
    explore_max_iters: int = 64,
    explore_max_failures: int = 3,
    explore_timeout_s: float | None = None,
    question: str | None = None,
    print_graph: bool = False,
    dump_sim_ground_truth: str | None = None,
    dump_sim_gt_include_robot: bool = False,
    calibration_export: str | None = None,
) -> None:
    """Run Dynagraph: voxel + graph EQA with optional merge and staleness (see docs/dynagraph.md)."""
    click.echo("Dynagraph: graph memory with DynaMem-style voxel navigation.")

    if rerun and no_rerun:
        raise click.UsageError("Cannot use both --rerun and --no-rerun.")
    if rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"
    if rerun_native and headless:
        raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")
    if discord and explore_loop:
        raise click.UsageError("--explore-loop is not supported together with --discord.")
    if discord and question:
        raise click.UsageError("Use Discord commands for questions instead of --question with --discord.")

    dynav_resolved = resolve_dynav_config_yaml(robot_backend, dynav_config)
    if dynav_resolved != dynav_config:
        logger.info(
            f"Dynagraph: resolved dynav {dynav_resolved!r} (CLI default was {dynav_config!r}, robot preset)"
        )

    if explore_loop and explore_max_iters < 1:
        raise click.UsageError("--explore-max-iters must be >= 1 when --explore-loop is set.")
    if explore_max_failures < 1:
        raise click.UsageError("--explore-max-failures must be >= 1.")

    logger.info(f"Dynagraph startup: dynav={dynav_resolved} robot={robot_backend}")

    click.echo("- Load parameters")
    parameters = get_parameters(dynav_resolved)
    robot_key = robot_backend.lower().replace("-", "_")
    apply_robot_dynav_parameter_overrides(robot_backend, parameters)
    if perfect_depth:
        parameters["debug_perfect_sensor_depth"] = True
        logger.info("debug: perfect sensor depth (DA3 skipped when observation depth is present)")
    elif robot_ip.strip() in ("127.0.0.1", "localhost", "::1") and str(
        parameters.get("depth_source", "")
    ).lower() == "da3":
        parameters["depth_source"] = "auto"
        logger.info(
            "Dynagraph: local sim (robot_ip=%s); depth_source da3 -> auto (prefer ZMQ sensor depth)",
            robot_ip,
        )
    if robot_key == "stretch" and os.environ.get("EMET_STRETCH_ROBOSUITE_ZMQ", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        parameters["local_radius"] = max(float(parameters.get("local_radius", 0.5)), 1.4)
        parameters["max_depth"] = max(float(parameters.get("max_depth", 2.5)), 3.8)
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if parameters.get("graph_object_fusion") is None:
        from dataclasses import asdict

        from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config

        fc = load_graph_object_fusion_config()
        parameters["graph_object_fusion"] = asdict(fc)
    if merge_xy_m is not None:
        parameters["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        parameters["dynagraph_staleness_horizon"] = int(staleness_horizon)

    depth_mode = str(parameters.get("depth_source", "sensor")).lower()
    allow_missing_depth = depth_mode in ("da3", "auto") or robot_key in (
        "innate_mars",
        "galaxea_r1",
        "rby1",
        "stretch",
    )

    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun_server=not no_rerun,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow_missing_depth,
        start_immediately=False,
    )
    if not robot.start():
        raise click.ClickException(
            "Could not connect to the robot/sim ZMQ server. "
            "Start `emet serve mujoco --use-robocasa --robot stretch` first, then re-run dynagraph. "
            "Large scenes may need: export EMET_ZMQ_STARTUP_TIMEOUT=120"
        )
    if hasattr(robot, "wait_for_obs"):
        robot.wait_for_obs(timeout=30.0)
    _print_dynagraph_rerun_help(enabled=not no_rerun, headless=headless)

    robot.move_to_nav_posture()
    if robot_key == "stretch":
        robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    ev = parameters.get("eqa_vl", {}) or {}
    ms = ev.get("model_size")
    qn = ev.get("quantization", "int4")
    if ms is None or str(ms).lower() == "null":
        print(
            "- EQA VL: one Qwen3.5 load sized by VRAM tiers (see eqa_vl/vram_mib_tier_* in dynav YAML),",
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
    if calibration_export:
        from emet.memory.graph_eqa.calibration_export import CalibrationFrameWriter

        agent._calibration_writer = CalibrationFrameWriter(calibration_export)
        click.echo(f"- Calibration export: {calibration_export!r}")

    def _maybe_explore(reason: str) -> None:
        if not explore_loop:
            return
        reason_lab, ok, nit = dynagraph_explore_until_terminated(
            agent,
            max_iterations=int(explore_max_iters),
            max_consecutive_failures=int(explore_max_failures),
            timeout_s=float(explore_timeout_s) if explore_timeout_s is not None else None,
            log_fn=lambda m: logger.info(m),
        )
        click.echo(f"- Explore-loop [{reason}] done: reason={reason_lab} successes={ok} iterations_executed={nit}")

    def _export_session_fields() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        get_sess = getattr(robot, "get_emet_session", None)
        if get_sess is None:
            return None, None
        sess = get_sess()
        if not sess:
            return None, None
        env = sess.get("environment")
        spawn = sess.get("spawn_floor_map")
        return (dict(env) if isinstance(env, dict) else None, dict(spawn) if isinstance(spawn, dict) else None)

    def _save_dump() -> None:
        if not dump_memory:
            return
        env, spawn = _export_session_fields()
        text = export_graph_eqa_dir(
            agent.graph_memory,
            getattr(agent, "voxel_map", None),
            dump_memory,
            title="Scene graph (Dynagraph, saved)",
            robot=robot_backend,
            environment=env,
            spawn_floor_map=spawn,
        )
        print(f"Saved graph memory to {dump_memory}")
        print(text)

    def _snapshot_graph_stdout(title: str) -> None:
        click.echo(format_scene_graph_pretty(agent.graph_memory, title=title))

    def _maybe_request_sim_truth_snapshot() -> None:
        req = dump_sim_ground_truth
        if not req:
            return
        path_on_sim_host = req.strip()
        if not path_on_sim_host:
            return
        as_json = path_on_sim_host.lower().endswith(".json")
        try:
            robot.request_sim_mujoco_ground_truth_snapshot(
                path_on_sim_host,
                exclude_robot=not dump_sim_gt_include_robot,
                as_json=as_json,
            )
            time.sleep(0.08)
            click.echo(
                f"- Requested MuJoCo ground-truth snapshot **on simulator host**: {path_on_sim_host!r} "
                f"(exclude_robot={not dump_sim_gt_include_robot}, json={as_json})"
            )
        except Exception as e:
            logger.warning(f"MuJoCo ground-truth snapshot request failed: {e}")

    try:
        if export_dir and discord:
            raise click.UsageError("Use either --export or --discord, not both.")

        if export_dir:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()
            _maybe_explore("export-path")
            if question:
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                robot.say("Answering the question " + question)
                try:
                    discord_text, _imgs = executor(question)
                    if not discord_text.strip():
                        click.echo("(Empty EQA reply — check graph memory / observations.)")
                    else:
                        click.echo(discord_text)
                except Exception as e:
                    logger.warning(f"EQA question failed (export will continue): {e}")
                    click.echo(f"EQA question failed: {e}")
            env, spawn = _export_session_fields()
            text = export_graph_eqa_dir(
                agent.graph_memory,
                getattr(agent, "voxel_map", None),
                export_dir,
                title="Scene graph (Dynagraph export)",
                robot=robot_backend,
                environment=env,
                spawn_floor_map=spawn,
            )
            click.echo(text)
            click.echo(f"Exported graph memory to {export_dir}")
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

        elif question:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()
            _maybe_explore("question-only")
            robot.move_to_nav_posture()
            robot.switch_to_navigation_mode()
            robot.say("Answering the question " + question)
            discord_text, _imgs = executor(question)
            if not discord_text.strip():
                click.echo("(Empty EQA reply — check graph memory / observations.)")
            else:
                click.echo(discord_text)
        else:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()
            _maybe_explore("interactive-prefix")

            click.echo(
                "Interactive mode: type a **question** to run graph EQA, "
                "**explore** (or **e**) to extend the map without calling the EQA model, "
                "or Enter to quit."
            )
            while True:
                qline = input("Dynagraph [question | explore | Enter=quit]: ").strip()
                if not qline:
                    break
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                low = qline.lower()
                if low in ("explore", "e", "map", "nav"):
                    click.echo("- Exploring (frontier navigation, no EQA call)…")
                    finished, _pt = agent.execute_action("")
                    if finished is None:
                        click.echo(
                            "Explore step failed (no plan / blocked). Map may still grow on the next update.",
                        )
                    elif finished:
                        click.echo("Explore step finished at a manipulation-ready pose.")
                    else:
                        click.echo("Explore step advanced; ask a question or explore again.")
                    continue
                robot.say("Answering the question " + qline)
                discord_text, _imgs = executor(qline)
                if not discord_text.strip():
                    print("(Empty EQA reply — check graph memory / observations.)")
    finally:
        _maybe_request_sim_truth_snapshot()
        _save_dump()
        if print_graph:
            _snapshot_graph_stdout("Dynagraph graph (snapshot on exit)")


if __name__ == "__main__":
    main()
