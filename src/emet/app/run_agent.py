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
# Agent chatbot: lightweight LLM (default Qwen 3.5) for local testing.
# Run with: emet run agent
# Default: connect to sim/robot at 127.0.0.1 (embodied agent). Use --offline for local LLM chat only.

import os

# PyTorch: must be set before the first CUDA allocation in this process (subprocess from ``emet run agent``).
# User override: ``export PYTORCH_ALLOC_CONF=...`` before launch replaces this default.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import tempfile
import timeit

import click
from termcolor import colored

from emet.agent.loop import run_agent_with_robot
from emet.agent.model_debug import print_offline_model_line
from emet.agent.prompt import DEFAULT_AGENT_NAME
from emet.audio import AudioRecorder
from emet.audio.speech_to_text import WhisperSpeechToText
from emet.core import get_parameters
from emet.llms import get_llm_choices, get_llm_client, get_prompt_builder, get_prompt_choices
from emet.utils.config import read_top_level_robot_from_yaml

# Default: Qwen 3.5 9B (with expandable CUDA segments enabled above to reduce fragmentation).
# Use ``--llm qwen35-4B`` if you still hit OOM on a single consumer GPU.
DEFAULT_AGENT_LLM = "qwen35-9B"


@click.command()
@click.option(
    "--llm",
    default=DEFAULT_AGENT_LLM,
    help=f"LLM to use (default: {DEFAULT_AGENT_LLM}). Case-insensitive. "
    f"Use qwen3-vl-eqa with --eqa to load one Qwen3-VL from dynav ``eqa:`` for chat + shared DynaMem captions.",
    type=click.Choice(get_llm_choices(), case_sensitive=False),
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
@click.option("--max-tokens", default=1024, type=int, help="Max new tokens per reply.")
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
    type=click.Path(),
    default=None,
    help="Memory directory to load when using --robot-ip.",
)
@click.option(
    "--discord/--no-discord",
    "discord",
    default=True,
    help="Start Discord bot when DISCORD_TOKEN is set (default: on; ignored with --offline). Use --no-discord to skip.",
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
        "With --no-llm: E/M/Q/P/FIND or find …; with an LLM: natural language per turn."
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
        "Robot backend (stretch, rby1, galaxea_r1, innate_mars). Overrides top-level ``robot`` in --agent-config when set; "
        "if omitted, that YAML key is used (default ``stretch`` when the key is absent). Must match "
        "emet serve mujoco --robot after any CLI remaps. MolmoSpaces (--molmospaces-scene) uses rby1 on the "
        "server even when serve is started with default stretch—set ``robot: rby1`` in YAML or pass --robot rby1. "
        "Always put the name immediately after --robot (e.g. --robot stretch); if you omit the name, the next "
        "flag may be parsed as the value."
    ),
)
@click.option(
    "--agent-config",
    "agent_config",
    default="dynav_config.yaml",
    help="DynaMem / scene YAML: basename under emet/config, or path to a YAML file (cwd or absolute).",
)
@click.option(
    "--vl-include-camera",
    "vl_include_camera",
    is_flag=True,
    help="Pass latest robot RGB to VL models each user turn (on by default for *VL* models).",
)
@click.option(
    "--no-vl-camera",
    "no_vl_camera",
    is_flag=True,
    help="Do not pass robot RGB to VL models (saves VRAM / faster).",
)
@click.option(
    "--eqa",
    "dynamem_eqa",
    is_flag=True,
    help=(
        "Enable DynaMem EQA on the voxel map (Qwen2.5-VL for captions + answers by default; see dynav_config.yaml eqa:). "
        "Heavier GPU/RAM and slower startup; default is off (query_memory falls back to localize_text)."
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
    "--sim-config",
    "sim_config",
    default=None,
    type=str,
    metavar="PATH",
    help="YAML sim launch profile (overrides sim_config / sim: in --agent-config). See configs/sim/*.yaml.",
)
@click.option(
    "--use-robocasa",
    "sim_use_robocasa",
    is_flag=True,
    help="With --start-sim: Robocasa kitchen (same as emet serve mujoco --use-robocasa). Incompatible with --molmospaces-scene / --scene-path.",
)
@click.option(
    "--robocasa-task",
    "sim_robocasa_task",
    default=None,
    type=str,
    metavar="NAME",
    help="With --start-sim and Robocasa: task name (default from sim YAML or PickPlaceCounterToCabinet).",
)
@click.option(
    "--scene-path",
    "sim_scene_path",
    default=None,
    type=click.Path(),
    metavar="PATH",
    help="With --start-sim: load this MJCF (default MuJoCo path). Incompatible with --molmospaces-scene.",
)
@click.option(
    "--molmospaces-scene",
    "sim_molmospaces_scene",
    default=None,
    metavar="NAME",
    help="With --start-sim: MolmoSpaces scene (e.g. ithor). Same as emet serve mujoco --molmospaces-scene.",
)
@click.option(
    "--molmospaces-split",
    "sim_molmospaces_split",
    default=None,
    type=click.Choice(["train", "val", "test"]),
    help="With --start-sim and MolmoSpaces: data split (default train or from sim YAML).",
)
@click.option(
    "--molmospaces-index",
    "sim_molmospaces_index",
    default=None,
    type=int,
    help="With --start-sim and MolmoSpaces: scene index (default 0 or from sim YAML).",
)
@click.option(
    "--molmospaces-install",
    "sim_molmospaces_install",
    is_flag=True,
    help="With --start-sim and MolmoSpaces: download scene assets if missing (non-interactive).",
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
def main(
    llm: str,
    prompt: str,
    device: str,
    voice: bool,
    max_tokens: int,
    robot_ip: str,
    offline: bool,
    input_path: str | None,
    discord: bool,
    no_llm: bool,
    debug_llm: bool,
    debug_tools: bool,
    debug_models: bool,
    debug_vram: bool,
    debug_camera: bool,
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
    agent_config: str = "dynav_config.yaml",
    vl_include_camera: bool = False,
    no_vl_camera: bool = False,
    dynamem_eqa: bool = False,
    share_memory_vllm: bool = True,
    start_sim: bool = False,
    sim_config: str | None = None,
    sim_use_robocasa: bool = False,
    sim_robocasa_task: str | None = None,
    sim_scene_path: str | None = None,
    sim_molmospaces_scene: str | None = None,
    sim_molmospaces_split: str | None = None,
    sim_molmospaces_index: int | None = None,
    sim_molmospaces_install: bool = False,
    sim_seed: int | None = None,
    sim_steps: int | None = None,
    sim_no_cameras: bool = False,
    sim_use_glx: bool = False,
    sim_show_viewer_ui: bool = False,
    sim_debug_molmospaces_spawn: bool = False,
) -> None:
    """Run the agent as a chatbot (lightweight Qwen Coder by default for local testing).

    Default: connect to 127.0.0.1 (start ``emet serve mujoco`` first). Use --offline for local chat only.
    The --prompt option applies to --offline only; embodied mode uses the agent tool prompt (JSON tool_calls).

    Examples:
      emet run agent --offline
      emet run agent --device cpu --offline
      emet run agent --llm qwen35-9B --offline
      emet run agent --llm gemma4-e4b --device cuda --offline   # Gemma 4 (HF any-to-any)
      emet run agent --robot rby1   # ZMQ @ 127.0.0.1; Discord if DISCORD_TOKEN set
      # MolmoSpaces: ``emet serve mujoco --molmospaces-scene ithor ...`` (often DISPLAY=:1 instead of --headless); same --port-offset as serve:
      emet run agent --robot rby1 --agent-config configs/agent_rby1_discord.yaml
      emet run agent --agent-config configs/agent_rby1_discord.yaml   # uses robot: from YAML
      emet run agent --robot stretch --agent-config configs/agent_stretch_discord.yaml
      emet run agent --robot innate_mars --agent-config configs/agent_innate_mars.yaml
      emet run agent --input-path logs/memory_xxx --no-discord
      emet run agent --no-llm   # letter commands (E/M/Q/P)
      emet run agent --no-llm --command 'find red cylinder'
      emet run agent --no-llm -c 'FIND blue cube'
      emet run agent --llm qwen3-vl-eqa --eqa --debug-vram   # one Qwen3-VL + VRAM milestones
      emet run agent --robot stretch --start-sim --no-discord --command "describe the scene"
      emet run agent --robot rby1 --start-sim --molmospaces-scene ithor --headless --no-discord -c "describe the scene"
    """
    cmd_list = list(commands) if commands else None

    if robot is None or str(robot).strip() == "":
        r_yaml = read_top_level_robot_from_yaml(agent_config)
        robot = r_yaml if r_yaml is not None else str(get_parameters(agent_config).get("robot", "stretch")).strip()
    else:
        robot = str(robot).strip()
    if not robot or robot.startswith("-"):
        raise click.UsageError(
            "`--robot` must be followed by a backend name (e.g. `stretch`, `rby1`, `innate_mars`). "
            "You left it empty or the next token was parsed as the value (often another flag); "
            "use e.g. `emet run agent --robot stretch --agent-config configs/agent_stretch_discord.yaml --rerun`."
        )

    if offline and start_sim:
        raise click.UsageError("Cannot combine --offline with --start-sim.")

    sim_cli_used = any(
        [
            sim_use_robocasa,
            sim_robocasa_task is not None and str(sim_robocasa_task).strip() != "",
            sim_scene_path is not None and str(sim_scene_path).strip() != "",
            sim_molmospaces_scene is not None and str(sim_molmospaces_scene).strip() != "",
            sim_molmospaces_split is not None,
            sim_molmospaces_index is not None,
            sim_molmospaces_install,
            sim_seed is not None,
            sim_steps is not None,
            sim_no_cameras,
            sim_use_glx,
            sim_show_viewer_ui,
            sim_debug_molmospaces_spawn,
        ]
    )
    if sim_cli_used and not start_sim:
        raise click.UsageError(
            "Sim-only flags (--use-robocasa, --molmospaces-scene, --sim-seed, etc.) require --start-sim."
        )

    if debug_models:
        os.environ["EMET_AGENT_MODEL_DEBUG"] = "1"
    if debug_vram:
        os.environ["EMET_VRAM_DEBUG"] = "1"
    if debug_camera:
        os.environ["EMET_AGENT_CAMERA_DEBUG"] = "1"

    # Embodied mode: default IP 127.0.0.1 unless --offline
    robot_effective: str | None = None
    if not offline:
        robot_effective = str(robot_ip or "").strip() or "127.0.0.1"

    # Vision LLMs: include camera RGB on new user turns (default on for *VL*; use --no-vl-camera to disable)
    llm_l = llm.lower()
    is_vl_name = "-vl-" in llm_l or "vl-" in llm_l
    vl_include_effective = (not no_vl_camera) and (vl_include_camera or is_vl_name)

    if robot_effective:
        if rerun_bind:
            os.environ["RERUN_BIND_ALL"] = "1"
        if rerun_native and headless:
            raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")
        sim_spawned_ok = False
        if start_sim:
            from emet.config.sim_launch_config import apply_sim_launch_cli_overrides, resolve_sim_launch_for_agent
            from emet.simulation.sim_subprocess import (
                shutdown_mujoco_server_subprocess,
                spawn_mujoco_server_subprocess,
            )

            try:
                sim_cfg = resolve_sim_launch_for_agent(
                    agent_config_path=agent_config,
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
                        use_robocasa=sim_use_robocasa,
                        robocasa_task=sim_robocasa_task,
                        scene_path=str(sim_scene_path) if sim_scene_path else None,
                        molmospaces_scene=sim_molmospaces_scene,
                        molmospaces_split=sim_molmospaces_split,
                        molmospaces_index=sim_molmospaces_index,
                        molmospaces_install=True if sim_molmospaces_install else None,
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
            if headless:
                sim_cfg.headless = True
            if robot and robot.lower() not in ("stretch", "hello_stretch", "hellostretch", ""):
                sim_cfg.robot = robot
            print(colored("Starting MuJoCo sim subprocess (--start-sim)…", "cyan"))
            spawn_mujoco_server_subprocess(sim_cfg)
            sim_spawned_ok = True
            print(colored("Sim is up; connecting agent.", "green"))
        print(
            colored(
                f"Robot backend: {robot} (from --robot or YAML `{agent_config}`; must match `emet serve mujoco --robot`).",
                "cyan",
            )
        )
        try:
            run_agent_with_robot(
                robot_ip=robot_effective,
                robot=robot,
                input_path=input_path,
                discord=discord,
                use_llm=not no_llm,
                llm=llm,
                skip_confirmations=True,
                debug_llm=debug_llm,
                tool_debug=debug_tools,
                agent_name=agent_name,
                commands=cmd_list,
                port_offset=port_offset,
                agent_config=agent_config,
                device=device,
                max_tokens=max_tokens,
                vl_include_camera=vl_include_effective,
                eqa=dynamem_eqa,
                share_memory_vllm=share_memory_vllm,
                headless=headless,
                rerun=rerun,
                rerun_native=rerun_native,
                rerun_show_panels=rerun_show_panels,
                rerun_debug=rerun_debug,
            )
        finally:
            if start_sim and sim_spawned_ok:
                shutdown_mujoco_server_subprocess()
        return

    prompt_builder = get_prompt_builder(prompt)
    dynav_params = get_parameters(agent_config)
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

    print(
        colored(
            "Offline LLM chat (--offline). Type a message and press Enter. Empty line to quit.",
            "green",
        )
    )
    print(colored(f"LLM: {llm}  device: {device}", "yellow"))
    if debug_llm:
        print(colored("Debug: full prompt, raw and parsed response will be printed.", "yellow"))
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
