# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent chatbot: local LLM for tool routing + optional DynaMem captions/EQA.
# Run with: emet run agent
# Default: ``qwen35-4B`` (fast text tool-router). Shared VL: ``--llm qwen3-vl-eqa``.
# Use ``--eqa --share-memory-vllm`` with a VL --llm so the voxel map reuses the same load.

import os

# PyTorch: must be set before the first CUDA allocation in this process (subprocess from ``emet run agent``).
# User override: ``export PYTORCH_ALLOC_CONF=...`` before launch replaces this default.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import tempfile
import timeit
from collections.abc import Callable

import click
from click.core import ParameterSource
from termcolor import colored

from emet.agent.loop import DEFAULT_AGENT_LLM, run_agent_with_robot
from emet.agent.model_debug import print_offline_model_line
from emet.agent.prompt import DEFAULT_AGENT_NAME
from emet.app.config_cli import (
    emet_config_options,
    load_finalized_config_from_cli,
    load_runtime_from_cli,
    resolve_agent_cli_options,
    resolve_effective_config_path,
)
from emet.audio import AudioRecorder
from emet.audio.speech_to_text import WhisperSpeechToText
from emet.core import get_parameters
from emet.llms import (
    get_llm_choices,
    get_llm_client,
    get_prompt_builder,
    get_prompt_choices,
    validate_llm_client_type,
)
from emet.llms.remote_ops import DEFAULT_LLM_PORT, DEFAULT_VL_PORT, apply_llm_host
from emet.utils.logger import Logger

log = Logger(__name__)


@click.command()
@click.option(
    "--llm",
    default=DEFAULT_AGENT_LLM,
    help=f"LLM to use (default: {DEFAULT_AGENT_LLM}). Registry key, 'openai', or "
    "openai@http://host:port/v1[#model]. "
    f"Examples: {', '.join(sorted(get_llm_choices())[:6])}…",
)
@click.option(
    "--host",
    "llm_host",
    default=None,
    help="LAN OpenAI host for text+VL (sets EMET_* + forces openai@HOST). Example: --host caliban",
)
@click.option(
    "--llm-port",
    default=DEFAULT_LLM_PORT,
    show_default=True,
    type=int,
    help="Text OpenAI port with --host.",
)
@click.option(
    "--vl-port",
    default=None,
    type=int,
    help=f"VL port with --host (default {DEFAULT_VL_PORT}; dual-2b: 8001).",
)
@click.option(
    "--prompt",
    default="simple",
    help="Prompt builder for the assistant.",
    type=click.Choice(get_prompt_choices()),
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cuda", "cpu", "mps"]),
    help="Device for inference. Use 'cpu' to test without GPU (slow).",
)
@click.option("--voice", is_flag=True, help="Use voice input (Whisper).")
@click.option(
    "--max-tokens", default=256, type=int, help="Max new tokens per reply (default: 256; keep low for tool JSON)."
)
@click.option(
    "--robot-ip",
    "--robot_ip",
    default="127.0.0.1",
    show_default=True,
    help="Simulator or robot IP (default: 127.0.0.1). Ignored with --offline.",
)
@click.option(
    "--offline",
    "offline",
    is_flag=True,
    help="Local LLM chat only (uses --prompt); do not connect to ZMQ / sim. No tools or Discord bridge.",
)
@click.option(
    "--input-path",
    "--input_path",
    type=click.Path(),
    default=None,
    help="Memory directory to load when using --robot-ip (graph + voxel_map.pkl for dynagraph).",
)
@click.option(
    "--refine-start/--no-refine-start",
    default=False,
    show_default=True,
    help=(
        "With --input-path: after an optional live frame, estimate a small SE(2) fudge "
        "aligning the saved map to the live cloud. On failure, keep the assumed pose."
    ),
)
@click.option(
    "--confirm-nav/--no-confirm-nav",
    "confirm_nav",
    default=None,
    help=(
        "Before executing a motion plan: show the path on the 2D map (Rerun + Discord image) "
        "and wait for y/n (terminal or Discord). Recommended on the real robot. "
        "Default off; set EMET_CONFIRM_NAV=1 to enable without the flag. "
        "Scripted ``-c`` / ``--command`` runs auto-accept (no prompt)."
    ),
)
@click.option(
    "--discord/--no-discord",
    "discord",
    default=True,
    help=(
        "Start Discord bot when DISCORD_TOKEN is set (default: on; ignored with --offline). "
        "Non-interactive ``--command`` / ``-c`` runs disable Discord automatically; pass ``--no-discord`` "
        "to acknowledge and hide the warning."
    ),
)
@click.option(
    "--no-llm",
    "--no_llm",
    "no_llm",
    is_flag=True,
    help="Disable LLM in agent mode (--robot-ip). By default the agent always uses an LLM (e.g. Qwen Coder) to parse natural language.",
)
@click.option(
    "--debug",
    "--debug-llm",
    "debug_llm",
    is_flag=True,
    help="Print full prompt, user input, raw LLM response, and parsed response.",
)
@click.option(
    "--debug-tools",
    "debug_tools",
    is_flag=True,
    help=(
        "Print raw tool_calls JSON, full tool return strings, executor tuples, combined [tool results] lines, "
        "the exact [Tool results] block sent back to the LLM, VL camera stats, and send_image array stats. "
        "Same as setting EMET_AGENT_TOOL_DEBUG=1."
    ),
)
@click.option(
    "--debug-models",
    "debug_models",
    is_flag=True,
    help=(
        "Print which models/clients are in use (chat LLM, describe_scene detector, DynaMem VLM, etc.). "
        "Same as setting EMET_AGENT_MODEL_DEBUG=1. Also enables VRAM snapshots (nvidia-smi + torch CUDA)."
    ),
)
@click.option(
    "--debug-vram",
    "debug_vram",
    is_flag=True,
    help=(
        "Print nvidia-smi and torch CUDA memory at major load milestones (SigLIP, agent LLM, EQA bind/materialize, "
        "shared Qwen3.5-VL). Same as EMET_VRAM_DEBUG=1. Combine with --debug-models for full model + VRAM report."
    ),
)
@click.option(
    "--debug-camera",
    "debug_camera",
    is_flag=True,
    help=(
        "Print head-camera frame stats for describe_scene, send_image, and Discord encode (black-PNG diagnosis). "
        "Same as EMET_AGENT_CAMERA_DEBUG=1. Discord assumes RGB buffers (matching compression.from_jpg); legacy "
        "OpenCV BGR pipelines can set EMET_DISCORD_IMAGES_BGR=1."
    ),
)
@click.option(
    "--thinking-status/--no-thinking-status",
    "thinking_status",
    default=True,
    help=(
        "Emit *Thinking…* / *Running tools…* status while waiting on the LLM or tools (default: on). "
        "Same as EMET_AGENT_THINKING_STATUS=1/0."
    ),
)
@click.option(
    "--cache-vl-prefix/--no-cache-vl-prefix",
    "cache_vl_prefix",
    default=None,
    help=(
        "Cache system-prompt KV for Qwen3-VL agent turns (default: eqa.vl_cache_system_prefix in config). "
        "Same as EMET_VL_CACHE_SYSTEM_PREFIX=1/0."
    ),
)
@click.option(
    "--name",
    "agent_name",
    default=DEFAULT_AGENT_NAME,
    help=f"Agent / persona name in the system prompt and greetings (default: {DEFAULT_AGENT_NAME!r}).",
)
@click.option(
    "-c",
    "--command",
    "commands",
    multiple=True,
    help=(
        "Run one or more commands non-interactively, then exit (embodied mode only). "
        'Same flag as -c; use quotes for multi-word phrases, e.g. --command "find red cylinder". '
        "With --no-llm: E/M/Q/P/FIND or find …; with an LLM: natural language per turn. "
        "Discord is disabled for this mode (see ``--discord`` / ``--no-discord``)."
    ),
)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
@click.option(
    "--headless",
    is_flag=True,
    help=(
        "Simulation / Rerun: no auto-open browser for Rerun; use http://<this-host>:9090?url=ws://... manually. "
        "Does not select the native Rerun app (use --rerun-native for that)."
    ),
)
@click.option(
    "--rerun",
    is_flag=True,
    help="Enable Rerun live visualization (default: web viewer at :9090; use --rerun-native for desktop app).",
)
@click.option(
    "--rerun-native",
    is_flag=True,
    help="With --rerun: use the native Rerun desktop viewer instead of the browser (needs DISPLAY).",
)
@click.option(
    "--rerun-show-panels",
    is_flag=True,
    help="Rerun: show blueprint/selection panel (debug).",
)
@click.option(
    "--rerun-debug",
    is_flag=True,
    help="Print periodic Rerun / ZMQ stream status for generic robots.",
)
@click.option(
    "--rerun-bind",
    is_flag=True,
    help="Bind Rerun to 0.0.0.0 (sets RERUN_BIND_ALL=1; same as emet run dynamem --rerun-bind).",
)
@click.option(
    "--robot",
    metavar="NAME",
    default=None,
    help=(
        "Robot backend (optional: config, connection profile, or ZMQ discovery). "
        "Overrides top-level ``robot`` in --config when set."
    ),
)
@click.option(
    "--vl-include-camera",
    "vl_include_camera",
    is_flag=True,
    help=(
        "Pass latest robot RGB into the chat VL model on each user turn. "
        "Off by default — vision questions should use describe_scene / send_image (much faster)."
    ),
)
@click.option(
    "--no-vl-camera",
    "no_vl_camera",
    is_flag=True,
    help="Explicitly disable camera→chat VL (default already off; kept for scripts).",
)
@click.option(
    "--eqa",
    "dynamem_eqa",
    is_flag=True,
    help=(
        "Enable EQA/caption VLM on the voxel map (reuses agent VL when --share-memory-vllm; "
        "else loads Qwen3-VL-8B int4 from dynav_config.yaml eqa:). "
        "Heavier GPU/RAM and slower startup; default is off (query_memory falls back to localize_text)."
    ),
)
@click.option(
    "--memory-backend",
    "memory_backend",
    type=click.Choice(
        ["dynagraph", "lazy_graph", "static_graph", "graph_eqa", "dynamem", "open_vocab"],
        case_sensitive=False,
    ),
    default="dynagraph",
    show_default=True,
    help=(
        "Object-graph plug-in on the voxel map (mutually exclusive): "
        "dynagraph (default; streaming GraphEQAMemory + merge/staleness), "
        "lazy_graph (DynaMem find + Qwen graph commit on nav arrival only; no YoloE graph labels), "
        "static_graph (zero-merge GraphEQA-inspired baseline; graph_eqa is a legacy alias), "
        "open_vocab (SAM3/OWL scene graph only), or dynamem (voxels only)."
    ),
)
@click.option(
    "--share-memory-vllm/--no-share-memory-vllm",
    "share_memory_vllm",
    default=True,
    help=(
        "With --eqa and an LLM: defer loading DynaMem's local caption VLM and bind the agent VL client when possible "
        "(saves VRAM vs loading two VL models). Use --no-share-memory-vllm to always load the EQA VLM from config."
    ),
)
@click.option(
    "--start-sim",
    "start_sim",
    is_flag=True,
    help=(
        "Start ``emet.simulation.mujoco_server`` as a subprocess before connecting. "
        "Uses ``sim_config`` / ``sim:`` in the agent YAML, ``--sim-config``, or (if none are set) "
        "the packaged default-table MuJoCo scene with the same ``--robot`` / YAML robot and ``--headless``."
    ),
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
    "--eqa-eval",
    "eqa_eval",
    is_flag=True,
    default=False,
    help=(
        "Run one HM-EQA episode via the shared Habitat episode function (same as emet-habitat; "
        "no chat tool-router). Requires --habitat-question-id. Optional --extra-instruction."
    ),
)
@click.option(
    "--extra-instruction",
    default=None,
    type=str,
    help="With --eqa-eval: text appended to the EQA question (identical compose path as emet-habitat).",
)
@click.option(
    "--eqa-eval-output",
    default=None,
    type=str,
    help="With --eqa-eval: append episode JSONL here (default under ~/.cache/habitat_eqa/results/).",
)
@click.option(
    "--eqa-eval-mock-llm",
    is_flag=True,
    default=False,
    help="With --eqa-eval: mock EQA VLM (wiring smoke; same as emet-habitat --mock-llm).",
)
@click.option(
    "--sim-config",
    "sim_config",
    default=None,
    type=str,
    metavar="PATH",
    help="YAML sim launch profile (overrides sim_config / sim: in --agent-config). See configs/sim/*.yaml.",
)
@click.option(
    "--scene",
    "sim_scene",
    default=None,
    metavar="NAME|PATH",
    help="With --start-sim: robocasa, MolmoSpaces name (ithor), path to MJCF, or omit for default table.",
)
@click.option(
    "--split",
    "sim_split",
    default=None,
    type=click.Choice(["train", "val", "test"]),
    help="With --start-sim and MolmoSpaces --scene: data split (default train or from sim YAML).",
)
@click.option(
    "--index",
    "sim_index",
    default=None,
    type=int,
    help="With --start-sim and MolmoSpaces --scene: scene index (default 0 or from sim YAML).",
)
@click.option(
    "--install-scene-if-missing",
    "sim_install_scene_if_missing",
    is_flag=True,
    help="With --start-sim and MolmoSpaces --scene: download scene assets if missing (non-interactive).",
)
@click.option(
    "--robocasa-task",
    "sim_robocasa_task",
    default=None,
    type=str,
    metavar="NAME",
    help="With --start-sim and --scene robocasa: task name (default from sim YAML or PickPlaceCounterToCabinet).",
)
@click.option(
    "--sim-seed",
    "sim_seed",
    default=None,
    type=int,
    help="With --start-sim: MuJoCo server --seed (overrides sim YAML).",
)
@click.option(
    "--sim-steps",
    "sim_steps",
    default=None,
    type=int,
    help="With --start-sim: stop server after N MuJoCo steps (debug).",
)
@click.option(
    "--sim-no-cameras",
    "sim_no_cameras",
    is_flag=True,
    help="With --start-sim: pass --no-cameras to mujoco_server (e.g. WSL EGL camera hang).",
)
@click.option(
    "--sim-use-glx",
    "sim_use_glx",
    is_flag=True,
    help="With --start-sim: pass --use-glx (Xvfb / GLX instead of EGL).",
)
@click.option(
    "--sim-show-viewer-ui",
    "sim_show_viewer_ui",
    is_flag=True,
    help="With --start-sim: pass --show-viewer-ui (only when not headless).",
)
@click.option(
    "--sim-debug-molmospaces-spawn",
    "sim_debug_molmospaces_spawn",
    is_flag=True,
    help="With --start-sim: pass --debug-molmospaces-spawn.",
)
@click.option(
    "--sim-show-subprocess-output",
    "sim_show_subprocess_output",
    is_flag=True,
    help=(
        "With --start-sim or --start-habitat: inherit this terminal for sim stdout/stderr (verbose). "
        "Default is to discard sim logs so scripted runs stay readable."
    ),
)
@click.pass_context
@emet_config_options()
def main(
    ctx: click.Context,
    llm: str,
    prompt: str,
    device: str,
    voice: bool,
    max_tokens: int,
    robot_ip: str,
    offline: bool,
    input_path: str | None,
    refine_start: bool,
    confirm_nav: bool | None,
    discord: bool,
    no_llm: bool,
    debug_llm: bool,
    debug_tools: bool,
    debug_models: bool,
    debug_vram: bool,
    debug_camera: bool,
    thinking_status: bool,
    cache_vl_prefix: bool | None,
    agent_name: str,
    commands: tuple[str, ...],
    port_offset: int = 0,
    headless: bool = False,
    rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    rerun_bind: bool = False,
    robot: str | None = None,
    emet_config: str = "",
    config_sets: tuple[str, ...] = (),
    connection: str | None = None,
    agent_config: str | None = None,
    dynav_config: str | None = None,
    vl_include_camera: bool = False,
    no_vl_camera: bool = False,
    dynamem_eqa: bool = False,
    memory_backend: str = "dynagraph",
    share_memory_vllm: bool = True,
    start_sim: bool = False,
    start_habitat: bool = False,
    habitat_question_id: int | None = None,
    habitat_scene_id: str | None = None,
    habitat_floor: int = 0,
    eqa_eval: bool = False,
    extra_instruction: str | None = None,
    eqa_eval_output: str | None = None,
    eqa_eval_mock_llm: bool = False,
    sim_config: str | None = None,
    sim_scene: str | None = None,
    sim_split: str | None = None,
    sim_index: int | None = None,
    sim_install_scene_if_missing: bool = False,
    sim_robocasa_task: str | None = None,
    sim_seed: int | None = None,
    sim_steps: int | None = None,
    sim_no_cameras: bool = False,
    sim_use_glx: bool = False,
    sim_show_viewer_ui: bool = False,
    sim_debug_molmospaces_spawn: bool = False,
    sim_show_subprocess_output: bool = False,
    llm_host: str | None = None,
    llm_port: int = DEFAULT_LLM_PORT,
    vl_port: int | None = None,
) -> None:
    """Run the agent chatbot (default: fast ``qwen35-4B`` text tool-router).

    Default: connect to 127.0.0.1 (start ``emet serve mujoco`` first). Use --offline for local chat only.
    The --prompt option applies to --offline only; embodied mode uses the agent tool prompt (JSON tool_calls).

    Examples:
      emet run agent --offline
      emet run agent --device cpu --offline
      emet run agent --llm qwen35-9B --offline   # larger text-only chat model
      emet run agent --start-sim -c "describe the scene"   # default: qwen35-4B tool router
      emet serve habitat --habitat-scene-id Y8Y6ukxGMvn   # terminal 1; then emet run agent -c "..."
      emet run agent --start-habitat --habitat-question-id 17 -c "describe the scene"
      emet run agent --llm qwen3-vl-eqa --eqa --debug-vram   # one Qwen3-VL for chat + voxel captions/EQA
      emet run agent --robot rby1   # ZMQ @ 127.0.0.1; Discord if DISCORD_TOKEN set
      # MolmoSpaces: ``emet serve mujoco --scene ithor ...`` (often DISPLAY=:1 instead of --headless); same --port-offset as serve:
      emet run agent --robot rby1 --config configs/agent_rby1_discord.yaml
      emet run agent --config configs/agent_rby1_discord.yaml   # uses robot: from YAML
      emet run agent --robot stretch --config configs/agent_stretch_discord.yaml
      emet run agent --robot innate_mars --config configs/agent_innate_mars.yaml
      emet run agent --connection herman --host caliban   # remote text+VL on Orin :8000
      emet run agent --input-path logs/memory_xxx --no-discord
      emet run agent --no-llm   # letter commands (E/M/Q/P)
      emet run agent --no-llm --command 'find red cylinder'
      emet run agent --no-llm -c 'FIND blue cube'
      emet run agent --robot stretch --start-sim --command "describe the scene"
      emet run agent --robot rby1 --start-sim --scene ithor --headless -c "describe the scene"
    """
    cmd_list = list(commands) if commands else None
    if eqa_eval:
        if habitat_question_id is None:
            raise click.UsageError("--eqa-eval requires --habitat-question-id.")
        if start_sim:
            raise click.UsageError("--eqa-eval uses the Habitat episode runner; do not pass --start-sim.")
        if offline:
            raise click.UsageError("Cannot combine --eqa-eval with --offline.")
        import json
        from pathlib import Path

        from emet.eval.habitat_eqa_agent import run_hmeqa_via_shared_episode
        from emet.eval.memory_backends import DYNAGRAPH, normalize_hmeqa_method

        try:
            method = normalize_hmeqa_method(memory_backend or DYNAGRAPH)
        except ValueError:
            method = DYNAGRAPH
        log.info(
            f"EQA eval mode: shared HM-EQA episode (question_id={habitat_question_id}, "
            f"method={method}) — no chat tool-router."
        )
        out = Path(eqa_eval_output) if eqa_eval_output else None
        payload = run_hmeqa_via_shared_episode(
            question_id=int(habitat_question_id),
            method=method,
            mock_llm=bool(eqa_eval_mock_llm),
            extra_instruction=extra_instruction,
            device=str(device or "cuda"),
            output=out,
        )
        print(json.dumps(payload, indent=2, default=str))
        return

    host_specs = apply_llm_host(llm_host, port=llm_port, vl_port=vl_port)

    config_path = resolve_effective_config_path(
        ctx,
        emet_config=emet_config,
        agent_config=agent_config,
        dynav_config=dynav_config,
        connection=connection,
    )
    robot_from_cli = robot is not None and str(robot).strip() != ""
    runtime = None

    if offline:
        resolved_robot = "stretch"
    else:
        runtime = load_runtime_from_cli(
            ctx,
            emet_config=emet_config,
            config_sets=config_sets,
            agent_config=agent_config,
            dynav_config=dynav_config,
            robot=robot,
            robot_ip=robot_ip,
            connection=connection,
            port_offset=port_offset,
            zmq_discover=not (start_sim or start_habitat),
            force_localhost=bool(start_sim or start_habitat)
            and ctx.get_parameter_source("robot_ip") == ParameterSource.DEFAULT,
        )
        resolved_robot = runtime.robot_id
        robot_ip = runtime.host
        if runtime.robot_source == "zmq":
            log.info("Using robot from ZMQ server: %r (pass --robot to override).", resolved_robot)
    robot = resolved_robot
    if robot and robot.startswith("-"):
        raise click.UsageError(
            "`--robot` must be followed by a backend name (e.g. `stretch`, `rby1`, `innate_mars`). "
            "You left it empty or the next token was parsed as the value (often another flag)."
        )

    if offline and (start_sim or start_habitat):
        raise click.UsageError("Cannot combine --offline with --start-sim or --start-habitat.")
    if start_sim and start_habitat:
        raise click.UsageError("Use either --start-sim or --start-habitat, not both.")
    if start_habitat and not habitat_question_id and not habitat_scene_id:
        raise click.UsageError(
            "--start-habitat requires --habitat-question-id or --habitat-scene-id "
            "(e.g. --habitat-scene-id Y8Y6ukxGMvn)."
        )

    if offline:
        agent_config_resolved = load_finalized_config_from_cli(
            ctx,
            emet_config=emet_config,
            config_sets=config_sets,
            agent_config=agent_config,
            dynav_config=dynav_config,
            connection=connection,
            robot_id=resolved_robot,
        )
    else:
        assert runtime is not None
        agent_config_resolved = runtime.config

    agent_opts = resolve_agent_cli_options(
        ctx,
        agent_config_resolved.agent_section(),
        llm=llm,
        prompt=prompt,
        device=device,
        max_tokens=max_tokens,
        discord=discord,
        dynamem_eqa=dynamem_eqa,
        share_memory_vllm=share_memory_vllm,
        memory_backend=memory_backend,
        agent_name=agent_name,
    )
    # --host / EMET_LLM_HOST wins over YAML agent.llm (and forces openai@…).
    llm = host_specs[0] if host_specs is not None else agent_opts.llm
    if not no_llm:
        llm = validate_llm_client_type(llm)
    prompt = agent_opts.prompt
    device = agent_opts.device
    max_tokens = agent_opts.max_tokens
    discord = agent_opts.discord
    dynamem_eqa = agent_opts.eqa
    share_memory_vllm = agent_opts.share_memory_vllm
    memory_backend = str(agent_opts.memory_backend or "dynagraph").strip().lower()
    agent_name = agent_opts.name
    if host_specs is not None:
        log.info("Remote LLM host: text=%s vl=%s", host_specs[0], host_specs[1])

    sim_cli_used = any(
        [
            sim_scene is not None and str(sim_scene).strip() != "",
            sim_robocasa_task is not None and str(sim_robocasa_task).strip() != "",
            sim_split is not None,
            sim_index is not None,
            sim_install_scene_if_missing,
            sim_seed is not None,
            sim_steps is not None,
            sim_no_cameras,
            sim_use_glx,
            sim_show_viewer_ui,
            sim_debug_molmospaces_spawn,
            sim_show_subprocess_output,
        ]
    )
    if sim_cli_used and not start_sim:
        raise click.UsageError(
            "MuJoCo sim-only flags (--scene, --split, --sim-seed, etc.) require --start-sim (not --start-habitat)."
        )

    if not offline and cmd_list:
        discord_src = ctx.get_parameter_source("discord")
        explicit_no_discord = discord_src == ParameterSource.COMMANDLINE and not discord
        if discord and not explicit_no_discord:
            log.warning(
                "`--command` / `-c` runs are non-interactive; Discord is disabled for this run. "
                "Pass `--no-discord` to acknowledge and hide this warning in scripts."
            )
        discord = False

    if debug_models:
        os.environ["EMET_AGENT_MODEL_DEBUG"] = "1"
    if debug_vram:
        os.environ["EMET_VRAM_DEBUG"] = "1"
    if debug_camera:
        os.environ["EMET_AGENT_CAMERA_DEBUG"] = "1"
    if not thinking_status:
        os.environ["EMET_AGENT_THINKING_STATUS"] = "0"
    if cache_vl_prefix is False:
        os.environ["EMET_VL_CACHE_SYSTEM_PREFIX"] = "0"
    elif cache_vl_prefix is True:
        os.environ["EMET_VL_CACHE_SYSTEM_PREFIX"] = "1"
    if debug_tools:
        os.environ["EMET_AGENT_TOOL_DEBUG"] = "1"

    # Embodied mode: default IP 127.0.0.1 unless --offline
    robot_effective: str | None = None
    if not offline:
        robot_effective = str(robot_ip or "").strip() or "127.0.0.1"

    # Vision on the *chat* VL is opt-in. Default off: tool path uses describe_scene (detector)
    # instead of prefilling Qwen3-VL with a full camera frame (~tens of seconds saved per turn).
    vl_include_effective = bool(vl_include_camera) and (not no_vl_camera)

    if robot_effective:
        if rerun_bind:
            os.environ["RERUN_BIND_ALL"] = "1"
        if rerun_native and headless:
            raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")
        sim_shutdown: Callable[[], None] | None = None
        if start_sim:
            from dataclasses import replace

            from emet.config.sim_launch_config import (
                SimLaunchMolmospaces,
                apply_sim_launch_cli_overrides,
                resolve_serve_robot,
                resolve_sim_launch_for_agent,
            )
            from emet.simulation.sim_subprocess import (
                shutdown_mujoco_server_subprocess,
                spawn_mujoco_server_subprocess,
            )

            sim_shutdown = shutdown_mujoco_server_subprocess

            try:
                sim_cfg = resolve_sim_launch_for_agent(
                    agent_config_path=config_path,
                    sim_config_cli=sim_config,
                    port_offset_cli=port_offset,
                    default_mujoco_table_if_missing=True,
                    default_robot=robot,
                    default_headless=headless,
                )
            except ValueError as e:
                raise click.UsageError(str(e)) from e
            if sim_cli_used:
                try:
                    sim_cfg = apply_sim_launch_cli_overrides(
                        sim_cfg,
                        scene=sim_scene,
                        split=sim_split,
                        index=sim_index,
                        install_scene_if_missing=True if sim_install_scene_if_missing else None,
                        robocasa_task=sim_robocasa_task,
                        headless=None,
                        show_viewer_ui=True if sim_show_viewer_ui else None,
                        no_cameras=True if sim_no_cameras else None,
                        use_glx=True if sim_use_glx else None,
                        seed=sim_seed,
                        steps=sim_steps,
                        debug_molmospaces_spawn=True if sim_debug_molmospaces_spawn else None,
                        robot=None,
                    )
                except ValueError as e:
                    raise click.UsageError(str(e)) from e
            # Stretch (passive viewer) exits when there is no DISPLAY or when stdout is not a TTY for
            # the viewer subprocess; scripted --command runs do not need a window. Force headless so
            # the sim stays up for ZMQ (4402) while the agent loads.
            had_headless = bool(sim_cfg.headless)
            need_sim_headless = (
                headless
                or not str(os.environ.get("DISPLAY", "")).strip()
                or (bool(cmd_list) and not sim_show_viewer_ui)
            )
            if need_sim_headless:
                sim_cfg.headless = True
            if sim_cfg.headless and not had_headless:
                log.info(
                    "MuJoCo server is running headless (no DISPLAY, --command/-c, or --headless) "
                    "so the sim keeps publishing observations."
                )
            if isinstance(sim_cfg, SimLaunchMolmospaces):
                from emet.simulation.molmospaces_config import normalize_molmospaces_robot_key

                sim_robot = resolve_serve_robot(sim_cfg.robot, is_molmospaces=True)
                sim_cfg = replace(sim_cfg, robot=sim_robot)
                agent_norm = normalize_molmospaces_robot_key(robot)
                sim_norm = normalize_molmospaces_robot_key(sim_robot)
                if robot_from_cli and agent_norm != sim_norm:
                    raise click.UsageError(
                        f"Agent --robot {robot!r} does not match MolmoSpaces sim robot {sim_robot!r}. "
                        "Use the same id for both (or omit --robot on the agent to follow the sim)."
                    )
                robot = sim_robot
            log.info("Starting MuJoCo sim subprocess (--start-sim)…")
            spawn_mujoco_server_subprocess(
                sim_cfg,
                silence_sim_output=not sim_show_subprocess_output,
            )
            log.info("Sim is up; connecting agent.")
        elif start_habitat:
            from emet.habitat.habitat_subprocess import (
                shutdown_habitat_server_subprocess,
                spawn_habitat_server_subprocess,
            )

            sim_shutdown = shutdown_habitat_server_subprocess
            log.info("Starting Habitat sim subprocess (--start-habitat)…")
            spawn_habitat_server_subprocess(
                question_id=habitat_question_id,
                scene_id=habitat_scene_id,
                floor=habitat_floor,
                port_offset=port_offset,
                silence_sim_output=not sim_show_subprocess_output,
            )
            robot = "stretch"
            log.info("Habitat ZMQ server is up; connecting agent as stretch.")
        log.info(f"Robot backend: {robot} (from --robot or config `{config_path}`; must match the ZMQ server).")
        try:
            run_agent_with_robot(
                robot_ip=robot_effective,
                robot=robot,
                input_path=input_path,
                refine_start=refine_start,
                confirm_nav=confirm_nav,
                discord=discord,
                use_llm=not no_llm,
                llm=llm,
                skip_confirmations=True,
                debug_llm=debug_llm,
                tool_debug=debug_tools,
                agent_name=agent_name,
                commands=cmd_list,
                port_offset=port_offset,
                agent_config=config_path,
                device=device,
                max_tokens=max_tokens,
                vl_include_camera=vl_include_effective,
                eqa=dynamem_eqa,
                share_memory_vllm=share_memory_vllm,
                memory_backend=memory_backend,
                headless=headless,
                rerun=rerun,
                rerun_native=rerun_native,
                rerun_show_panels=rerun_show_panels,
                rerun_debug=rerun_debug,
                shutdown_sim_subprocess=sim_shutdown,
                parameters=runtime.parameters if runtime is not None else None,
                allow_missing_depth=runtime.allow_missing_depth if runtime is not None else None,
                embodied_overlay=runtime.config.embodied_agent() if runtime is not None else None,
                thinking_status=thinking_status,
                agent_section=agent_config_resolved.agent_section(),
            )
        finally:
            if sim_shutdown is not None:
                sim_shutdown()
        return

    prompt_builder = get_prompt_builder(prompt)
    dynav_params = get_parameters(
        config_path,
        overrides=list(config_sets) if config_sets else None,
        robot=resolved_robot if offline else None,
    )
    client = get_llm_client(llm, prompt_builder, device=device, parameters=dynav_params)
    if hasattr(client, "max_tokens"):
        client.max_tokens = max_tokens
    print_offline_model_line(llm, client, device, max_tokens)

    if voice:
        audio_recorder = AudioRecorder()
        whisper = WhisperSpeechToText()
    else:
        audio_recorder = None
        whisper = None

    log.info("Offline LLM chat (--offline). Type a message and press Enter. Empty line to quit.")
    log.info("LLM: %s  device: %s", llm, device)
    if debug_llm:
        log.info("Debug: full prompt, raw and parsed response will be printed.")
    print("-" * 60)

    while True:
        if voice:
            input(colored("Press Enter to speak (or Ctrl+C to quit)...", "yellow"))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            try:
                audio_recorder.record(path, duration=10, silence_limit=2)
                user_text = whisper.transcribe_file(path)
            finally:
                if os.path.exists(path):
                    os.unlink(path)
            print(colored("You (voice):", "green"), user_text)
        else:
            user_text = input(colored("You: ", "green")).strip()

        if not user_text:
            print("Bye.")
            break

        if debug_llm:
            system_prompt = getattr(client, "system_prompt", None) or getattr(client, "_prompt", "")
            print(colored("[DEBUG] System prompt:", "yellow"))
            print("-" * 40)
            print(system_prompt[:2000] + ("..." if len(system_prompt) > 2000 else ""))
            print("-" * 40)
            print(colored("[DEBUG] User input:", "yellow"), repr(user_text))
            print("-" * 40)

        t0 = timeit.default_timer()
        try:
            reply = client(user_text, verbose=debug_llm)
        except TypeError:
            reply = client(user_text)
        t1 = timeit.default_timer()

        if hasattr(prompt_builder, "parse_response") and callable(prompt_builder.parse_response):
            parsed = prompt_builder.parse_response(reply)
            if debug_llm:
                print(colored("[DEBUG] Raw LLM response:", "yellow"), repr(reply))
                print(colored("[DEBUG] Parsed response:", "blue"), parsed)
            reply = parsed
        if debug_llm:
            print("Time taken:", t1 - t0)
        print(colored("Agent:", "blue"), reply)
        print(colored(f"({t1 - t0:.2f}s)", "yellow"))
        print("-" * 60)


if __name__ == "__main__":
    main()
