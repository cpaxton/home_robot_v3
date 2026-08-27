# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared Click CLI for Dynagraph and LazyGraph. See docs/dynagraph.md and docs/lazy_graph.md."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import click

from emet.app.config_cli import emet_config_options, load_runtime_from_cli
from emet.app.dynagraph_explore import dynagraph_explore_until_terminated
from emet.app.graph_nav_gt import ensure_ground_truth_ready, print_graph_nav_rerun_help
from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.task.dynamem import EQAExecuter
from emet.memory.graph_eqa import format_scene_graph_pretty
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    ground_truth_alignment_report,
    read_sim_object_placements,
)
from emet.memory.headless_export import export_dynagraph_episode, export_graph_eqa_dir
from emet.utils.logger import Logger

logger = Logger(__name__)

_GRAPH_NAV: dict[str, Any] = {
    "controller_cls": None,
    "product": "Dynagraph",
}


class _GraphNavScope:
    """Applies ``configure_graph_nav``; restores the previous selection on ``with`` exit."""

    def __init__(self, controller_cls: type, *, product: str) -> None:
        self._prev_cls = _GRAPH_NAV.get("controller_cls")
        self._prev_product = _GRAPH_NAV.get("product")
        _GRAPH_NAV["controller_cls"] = controller_cls
        _GRAPH_NAV["product"] = str(product)

    def __enter__(self) -> _GraphNavScope:
        return self

    def __exit__(self, *exc: object) -> None:
        _GRAPH_NAV["controller_cls"] = self._prev_cls
        _GRAPH_NAV["product"] = self._prev_product


def configure_graph_nav(controller_cls: type, *, product: str = "Dynagraph") -> _GraphNavScope:
    """Select controller + product name for the shared Click ``main``.

    Call as a statement for process-lifetime config (``run_dynagraph`` /
    ``run_lazy_graph``). Use ``with configure_graph_nav(...)`` to restore the
    previous selection on exit.
    """
    return _GraphNavScope(controller_cls, product=product)


def _product() -> str:
    return str(_GRAPH_NAV.get("product") or "Dynagraph")


def _controller_cls() -> type:
    cls = _GRAPH_NAV.get("controller_cls")
    if cls is None:
        from emet.controller.controller_dynagraph import DynagraphController

        return DynagraphController
    return cls


@click.command()
@emet_config_options()
@click.pass_context
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
    default=None,
    type=str,
    help="Robot backend (optional: config, connection profile, or ZMQ discovery).",
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
    "--start-sim",
    "start_sim",
    is_flag=True,
    default=False,
    help="Spawn MuJoCo ZMQ server subprocess before connecting (default table / Robocasa / MolmoSpaces from config)",
)
@click.option(
    "--start-habitat",
    "start_habitat",
    is_flag=True,
    default=False,
    help="Spawn ``emet-habitat serve`` subprocess (requires .venv-habitat)",
)
@click.option(
    "--habitat-question-id",
    type=int,
    default=None,
    help="With --start-habitat: HM-EQA question id (scene + init pose from CSV)",
)
@click.option(
    "--habitat-scene-id",
    default=None,
    help="With --start-habitat: HM3D scene id when questions.csv is unavailable",
)
@click.option(
    "--habitat-floor",
    default=0,
    type=int,
    help="With --start-habitat: floor index for init pose CSV lookup",
)
@click.option(
    "--sim-show-subprocess-output",
    is_flag=True,
    default=False,
    help="With --start-sim or --start-habitat: inherit this terminal for sim stdout/stderr",
)
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
        "Save graph backend + scene_graph_report.txt here, print summary, and exit "
        "(unless combined with --discord or --question). Use for headless / no-TTY runs."
    ),
)
@click.option(
    "--dump-memory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Save graph memory to this directory when the session ends (empty line to quit)",
)
@click.option(
    "--export-voxel-pickle/--no-export-voxel-pickle",
    default=True,
    show_default=True,
    help=(
        "With --export/--dump-memory: write voxel_map.pkl (full DynaMem voxel state) so a "
        "later run can resume the map with --input-path (lifelong checkpoints). On by default; "
        "use --no-export-voxel-pickle to skip if disk size is a concern (~tens of MB per scan)."
    ),
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
    "--question-file",
    "question_file",
    default=None,
    type=click.Path(exists=True),
    help="YAML question bank; run all questions (optionally filtered by --question-env)",
)
@click.option(
    "--question-env",
    default=None,
    help="Filter --question-file to one environment tag (e.g. robocasa_seed0)",
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
@click.option(
    "--calibration-steps",
    type=int,
    default=0,
    show_default=True,
    help=(
        "With --calibration-export: run this many agent.update() cycles after rotate-in-place "
        "(instance detections only; skips explore-loop unless also set)"
    ),
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
@click.option(
    "--graph-fusion-config",
    default=None,
    type=click.Path(exists=True),
    help="Override graph_object_fusion block from standalone YAML (A/B experiments)",
)
@click.option(
    "--benchmark-harness",
    type=click.Choice(
        [
            "interactive",
            "habitat_eqa",
            "habitat_ovmm_find",
            "ovmm_find_phase",
            "sqa3d",
            "dynamic_explore",
        ],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Apply configs/benchmarks/dynagraph.yaml harness profile + EQA flags "
        "(memory_summary, mcq_debias, explore_when_uncovered, siglip_grounding, "
        "merged_memory) before starting the agent. Use with --benchmark-method for paper rows."
    ),
)
@click.option(
    "--benchmark-method",
    type=click.Choice(["dynagraph", "static_graph", "graph_eqa"], case_sensitive=False),
    default=None,
    help=(
        "Harness method row (default dynagraph when --benchmark-harness is set; "
        "graph_eqa is a legacy alias for static_graph)"
    ),
)
def main(
    ctx: click.Context,
    robot_ip: str,
    robot_backend: str | None = None,
    emet_config: str = "",
    config_sets: tuple[str, ...] = (),
    connection: str | None = None,
    agent_config: str | None = None,
    dynav_config: str | None = None,
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
    start_sim: bool = False,
    start_habitat: bool = False,
    habitat_question_id: int | None = None,
    habitat_scene_id: str | None = None,
    habitat_floor: int = 0,
    sim_show_subprocess_output: bool = False,
    input_path: str | None = None,
    export_dir: str | None = None,
    dump_memory: str | None = None,
    export_voxel_pickle: bool = True,
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
    question_file: str | None = None,
    question_env: str | None = None,
    print_graph: bool = False,
    dump_sim_ground_truth: str | None = None,
    dump_sim_gt_include_robot: bool = False,
    calibration_export: str | None = None,
    calibration_steps: int = 0,
    ground_truth: bool = False,
    compare_to_gt: bool = False,
    graph_fusion_config: str | None = None,
    benchmark_harness: str | None = None,
    benchmark_method: str | None = None,
) -> None:
    """Run graph-nav (Dynagraph / LazyGraph): voxel navigation + graph EQA."""
    product = _product()
    click.echo(f"{product}: connecting to robot and starting graph-based EQA (with merge/staleness).")
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
    if question and question_file:
        raise click.UsageError("Use either --question or --question-file, not both.")
    if question_file and not export_dir and not question:
        click.echo("Note: --question-file without --export runs questions then exits (no eqa_results.json).")
    if start_sim and start_habitat:
        raise click.UsageError("Use either --start-sim or --start-habitat, not both.")
    if start_habitat and not habitat_question_id and not habitat_scene_id:
        raise click.UsageError(
            "--start-habitat requires --habitat-question-id or --habitat-scene-id "
            "(e.g. --habitat-scene-id Y8Y6ukxGMvn)."
        )

    runtime = load_runtime_from_cli(
        ctx,
        emet_config=emet_config,
        config_sets=config_sets,
        agent_config=agent_config,
        dynav_config=dynav_config,
        robot=robot_backend,
        robot_ip=robot_ip,
        connection=connection,
        port_offset=port_offset,
        zmq_discover=not (start_sim or start_habitat),
    )
    robot_backend = runtime.robot_id
    robot_ip = runtime.host
    parameters = runtime.parameters
    robot_key = robot_backend
    allow_missing_depth = runtime.allow_missing_depth
    if runtime.robot_source == "zmq":
        click.echo(f"Using robot from ZMQ server: {robot_backend!r} (pass --robot to override).")
    logger.info(
        f"{product} startup: config={runtime.config_path} robot={robot_backend} (source={runtime.robot_source})"
    )

    if explore_loop and explore_max_iters < 1:
        raise click.UsageError("--explore-max-iters must be >= 1 when --explore-loop is set.")
    if explore_max_failures < 1:
        raise click.UsageError("--explore-max-failures must be >= 1.")
    if calibration_steps < 0:
        raise click.UsageError("--calibration-steps must be >= 0.")
    if calibration_export and calibration_steps < 1 and not explore_loop and not export_dir:
        raise click.UsageError(
            "Use --calibration-steps with --calibration-export (or --export / --explore-loop) "
            "so instance detections are recorded."
        )

    click.echo("- Load parameters")
    if perfect_depth:
        parameters["debug_perfect_sensor_depth"] = True
        logger.info("debug: perfect sensor depth (DA3 skipped when observation depth is present)")
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if benchmark_harness:
        from emet.eval.benchmark_dynagraph import apply_dynagraph_harness

        method = str(benchmark_method or "dynagraph").strip().lower()
        apply_dynagraph_harness(parameters, str(benchmark_harness).strip().lower(), method)
        logger.info(f"{product}: applied benchmark harness={benchmark_harness!r} method={method!r}")
    elif benchmark_method:
        raise click.UsageError("--benchmark-method requires --benchmark-harness")
    if graph_fusion_config:
        from dataclasses import asdict

        from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config

        fc = load_graph_object_fusion_config(graph_fusion_config)
        parameters["graph_object_fusion"] = asdict(fc)
        logger.info(f"{product}: graph_object_fusion from {graph_fusion_config}")
    elif parameters.get("graph_object_fusion") is None:
        from dataclasses import asdict

        from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config

        fc = load_graph_object_fusion_config()
        parameters["graph_object_fusion"] = asdict(fc)
    if merge_xy_m is not None:
        parameters["dynagraph_merge_xy_m"] = float(merge_xy_m)
    if staleness_horizon is not None:
        parameters["dynagraph_staleness_horizon"] = int(staleness_horizon)

    if start_sim:
        from dataclasses import replace

        from emet.config.sim_launch_config import (
            SimLaunchMolmospaces,
            resolve_serve_robot,
            resolve_sim_launch_for_agent,
        )
        from emet.simulation.sim_subprocess import spawn_mujoco_server_subprocess

        sim_cfg = resolve_sim_launch_for_agent(
            agent_config_path=runtime.config_path,
            sim_config_cli=None,
            port_offset_cli=port_offset,
            default_mujoco_table_if_missing=True,
            default_robot=robot_backend,
            default_headless=headless or not no_rerun,
        )
        sim_cfg = replace(sim_cfg, headless=True)
        if isinstance(sim_cfg, SimLaunchMolmospaces):
            sim_robot = resolve_serve_robot(sim_cfg.robot, is_molmospaces=True)
            sim_cfg = replace(sim_cfg, robot=sim_robot)
            robot_backend = sim_robot
        click.echo(f"{product}: starting MuJoCo sim subprocess (--start-sim)…", err=True)
        spawn_mujoco_server_subprocess(sim_cfg, silence_sim_output=not sim_show_subprocess_output)
        click.echo(f"Sim is up; connecting {product.lower()}.", err=True)
    elif start_habitat:
        from emet.habitat.habitat_subprocess import spawn_habitat_server_subprocess

        click.echo(f"{product}: starting Habitat sim subprocess (--start-habitat)…", err=True)
        spawn_habitat_server_subprocess(
            question_id=habitat_question_id,
            scene_id=habitat_scene_id,
            floor=habitat_floor,
            port_offset=port_offset,
            silence_sim_output=not sim_show_subprocess_output,
        )
        robot_backend = "stretch"
        click.echo(f"Habitat ZMQ server is up; connecting {product.lower()} as stretch.", err=True)

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
            f"Start `emet serve mujoco --use-robocasa --robot stretch` first, then re-run {product.lower()}. "
            "Large scenes may need: export EMET_ZMQ_STARTUP_TIMEOUT=120"
        )
    if hasattr(robot, "wait_for_obs"):
        robot.wait_for_obs(timeout=30.0)
    print_graph_nav_rerun_help(
        product=product,
        enabled=not no_rerun,
        headless=headless,
        ground_truth=ground_truth,
        compare_to_gt=compare_to_gt,
    )

    robot.move_to_nav_posture()
    if robot_key == "stretch":
        robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    eqa_cfg = parameters.get("eqa", {}) or {}
    ev = parameters.get("eqa_vl", {}) or {}
    vl_family = str(eqa_cfg.get("vl_family", "") or "").strip() or "qwen3_vl"
    vl_hf = str(eqa_cfg.get("vl_hf_model_id", "") or "").strip()
    vl_q = eqa_cfg.get("vl_quantization", ev.get("quantization", "int4"))
    if vl_hf:
        print(f"- EQA VL: shared {vl_family} ({vl_hf}, quant={vl_q}) for labels + EQA")
    else:
        ms = ev.get("model_size")
        print(f"- EQA VL: shared {vl_family} (size={ms!r}, quant={vl_q}) for labels + EQA")

    print(f"- Start {product} agent (graph memory + voxel map for navigation)")
    controller_cls = _controller_cls()
    agent: Any | None = None
    agent = controller_cls(
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
    if explore_loop:
        # Shorter head sweeps during scripted frontier batches (Molmo/MuJoCo smoke).
        agent._fast_explore_lookaround = True
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

    def _run_eqa_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa
        from emet.memory.graph_eqa.question_bank import write_eqa_results
        from emet.utils.port_utils import release_zmq_ports

        rows: list[dict[str, Any]] = []
        from emet.memory.graph_eqa import agentic_verify_enabled

        agentic = agentic_verify_enabled(agent)
        # Answer-only: drop MuJoCo/EGL before Qwen load. Agentic verify keeps sim up for nav.
        if not agentic:
            try:
                freed = release_zmq_ports(int(port_offset or 0))
                if freed:
                    click.echo(
                        f"- Released sim ZMQ ports before EQA: {freed}",
                        err=True,
                    )
                else:
                    click.echo("- No sim ZMQ listeners to release before EQA", err=True)
            except Exception as e:
                logger.warning(f"release_zmq_ports before EQA failed (non-fatal): {e}")
        # Free perception / voxel SigLIP before Qwen3-VL load (Phase-1 dynamic explore OOM fix).
        # Agentic verify keeps SigLIP until submit_answer; warm only (no release yet).
        if agentic:
            click.echo("- Warming SigLIP for agentic verify (sim stays up)", err=True)
            from emet.eval.dynagraph_vram import warm_siglip_confirmed_memory

            warm_siglip_confirmed_memory(agent)
        else:
            click.echo("- Preparing VRAM for EQA (release non-EQA GPU caches)", err=True)
            prepare_dynagraph_vram_for_eqa(agent)
        gm = getattr(agent, "graph_memory", None)
        if gm is not None and getattr(gm, "memory_summary_enabled", False):
            refresh = getattr(gm, "refresh_siglip_confirmed_memory", None)
            if callable(refresh):
                refresh()
        # Question bank: answer-only by default; agentic_verify navigates+verifies in-loop.
        agent._fast_explore_lookaround = True
        n_q = sum(1 for q in questions if str(q.get("question", "")).strip())
        mode = "agentic verify" if agentic else "answer-only: max_planning_steps=1, allow_navigation=False"
        click.echo(f"- EQA question bank: {n_q} question(s) ({mode})", err=True)
        qi = 0
        for qspec in questions:
            qtext = str(qspec.get("question", "")).strip()
            if not qtext:
                continue
            qi += 1
            click.echo(f"- EQA question {qi}/{n_q} start: {qtext}", err=True)
            t_q0 = time.monotonic()
            try:
                if agentic:
                    from emet.memory.graph_eqa import run_agentic_eqa

                    discord_text, _imgs = run_agentic_eqa(
                        agent,
                        qtext,
                        trace_path=(Path(export_dir) / "agentic_trace.jsonl") if export_dir else None,
                        trace_meta={"gt_body_key": str(qspec.get("gt_body_key") or "")},
                    )
                else:
                    try:
                        discord_text, _imgs = agent.run_eqa(
                            qtext,
                            max_planning_steps=1,
                            allow_navigation=False,
                        )
                    except TypeError:
                        # Older Dynamem-only agents lack allow_navigation.
                        discord_text, _imgs = agent.run_eqa(qtext, max_planning_steps=1)
            except Exception as e:
                logger.warning(f"EQA question failed: {e}")
                discord_text = f"EQA question failed: {e}"
            q_wall = time.monotonic() - t_q0
            answer = ""
            m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", discord_text or "")
            if m:
                answer = m.group(1).strip()
            elif discord_text:
                answer = discord_text.strip()
            row = {
                **qspec,
                "question": qtext,
                "discord_text": discord_text,
                "answer": answer,
                "eqa_wall_s": q_wall,
            }
            rows.append(row)
            click.echo(
                f"- EQA question {qi}/{n_q} done wall_s={q_wall:.1f} answer={answer[:120]!r}",
                err=True,
            )
            if discord_text.strip():
                click.echo(discord_text, err=True)
            else:
                click.echo("(Empty EQA reply — check graph memory / observations.)", err=True)
        if export_dir and rows:
            out = write_eqa_results(Path(export_dir) / "eqa_results.json", rows)
            click.echo(f"Wrote EQA results -> {out}", err=True)
        return rows

    def _run_calibration_capture(steps: int, *, rotate: bool) -> None:
        if steps < 1:
            return
        executor = EQAExecuter(agent)
        if rotate and not not_rotate_in_place:
            executor.rotate_in_place()
        click.echo(f"- Calibration capture: {steps} update steps (instance detections -> JSONL)")
        for i in range(int(steps)):
            agent.update()
            if (i + 1) % 10 == 0 or i + 1 == steps:
                click.echo(f"  calibration step {i + 1}/{steps}")

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

    if ground_truth:
        ensure_ground_truth_ready(agent, context="export" if export_dir else "interactive")

    def _save_dump() -> None:
        if not dump_memory:
            return
        env, spawn = _export_session_fields()
        text = export_graph_eqa_dir(
            agent.graph_memory,
            getattr(agent, "voxel_map", None),
            dump_memory,
            title=f"Scene graph ({product}, saved)",
            robot=robot_backend,
            environment=env,
            spawn_floor_map=spawn,
            final_step=int(getattr(agent, "obs_count", 0)),
            save_voxel_pickle=export_voxel_pickle,
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
            if calibration_export and calibration_steps > 0:
                _run_calibration_capture(calibration_steps, rotate=True)
            elif not not_rotate_in_place:
                executor = EQAExecuter(agent)
                executor.rotate_in_place()
            if explore_loop:
                _maybe_explore("export-path")
            if question_file:
                from emet.memory.graph_eqa.question_bank import load_question_bank

                bank = load_question_bank(question_file, env_filter=question_env)
                if not bank:
                    raise click.ClickException(f"No questions loaded from {question_file}")
                _run_eqa_questions(bank)
            elif question:
                _run_eqa_questions([{"question": question}])
            env, spawn = _export_session_fields()
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
                title=f"Scene graph ({product} GT export)" if ground_truth else f"Scene graph ({product} export)",
                robot=robot_backend,
                environment=env,
                spawn_floor_map=spawn,
                ground_truth_mode=ground_truth,
                sim_object_placements=placements,
                gt_alignment_report_text=gt_report,
                final_step=int(getattr(agent, "obs_count", 0)),
                save_voxel_pickle=export_voxel_pickle,
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
                await ctx.send(f"Hello! I am here to help you ({product}).")

            obs = robot.get_observation()
            bot.push_task_to_all_channels(content=obs.rgb)
            bot.run()

        elif question or question_file:
            executor = EQAExecuter(agent)
            if not not_rotate_in_place:
                executor.rotate_in_place()
            _maybe_explore("question-only")
            if question_file:
                from emet.memory.graph_eqa.question_bank import load_question_bank

                bank = load_question_bank(question_file, env_filter=question_env)
                _run_eqa_questions(bank)
            else:
                _run_eqa_questions([{"question": question}])
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
                qline = input(f"{product} [question | explore | Enter=quit]: ").strip()
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
            _snapshot_graph_stdout(f"{product} graph (snapshot on exit)")
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
        # Stop client spin threads so the process can exit cleanly in batch harnesses.
        if agent is not None:
            agent_stop = getattr(agent, "stop", None)
            if callable(agent_stop):
                agent_stop()
        robot_stop = getattr(robot, "stop", None)
        if callable(robot_stop):
            robot_stop()


if __name__ == "__main__":
    main()
