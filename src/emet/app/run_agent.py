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

import logging
import os
import tempfile
import timeit

# Suppress HuggingFace/transformers/httpx INFO spam when loading models (e.g. SigLIP)
for _name in ("httpx", "httpcore", "huggingface_hub", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import click
from termcolor import colored

from emet.agent import run_agent_with_robot
from emet.audio import AudioRecorder
from emet.audio.speech_to_text import WhisperSpeechToText
from emet.llms import get_llm_choices, get_llm_client, get_prompt_builder, get_prompt_choices

# Default: Qwen 3.5 9B (good quality on 24GB GPU; use qwen35-4B if VRAM is tight)
DEFAULT_AGENT_LLM = "qwen35-9B"


@click.command()
@click.option(
    "--llm",
    default=DEFAULT_AGENT_LLM,
    help=f"LLM to use (default: {DEFAULT_AGENT_LLM}). Case-insensitive (e.g. qwen35-vl-9b = qwen35-vl-9B).",
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
    "--name",
    "agent_name",
    default="Emet",
    help="Agent name used in the system prompt (default: Emet).",
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
    "--robot",
    default="stretch",
    help="Robot backend (stretch, rby1, galaxea_r1). Must match emet serve mujoco --robot.",
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
    agent_name: str,
    commands: tuple[str, ...],
    port_offset: int = 0,
    robot: str = "stretch",
    agent_config: str = "dynav_config.yaml",
    vl_include_camera: bool = False,
    no_vl_camera: bool = False,
) -> None:
    """Run the agent as a chatbot (lightweight Qwen Coder by default for local testing).

    Default: connect to 127.0.0.1 (start ``emet serve mujoco`` first). Use --offline for local chat only.
    The --prompt option applies to --offline only; embodied mode uses the agent tool prompt (JSON tool_calls).

    Examples:
      emet run agent --offline
      emet run agent --device cpu --offline
      emet run agent --llm qwen35-9B --offline
      emet run agent --robot rby1   # ZMQ @ 127.0.0.1; Discord if DISCORD_TOKEN set
      emet run agent --input-path logs/memory_xxx --no-discord
      emet run agent --no-llm   # letter commands (E/M/Q/P)
      emet run agent --no-llm --command 'find red cylinder'
      emet run agent --no-llm -c 'FIND blue cube'
      emet run agent -c 'find the red cylinder' -c 'what objects do you see?'
    """
    cmd_list = list(commands) if commands else None

    # Embodied mode: default IP 127.0.0.1 unless --offline
    robot_effective: str | None = None
    if not offline:
        robot_effective = str(robot_ip or "").strip() or "127.0.0.1"

    # Vision LLMs: include camera RGB on new user turns (default on for *VL*; use --no-vl-camera to disable)
    llm_l = llm.lower()
    is_vl_name = "-vl-" in llm_l or "vl-" in llm_l
    vl_include_effective = (not no_vl_camera) and (vl_include_camera or is_vl_name)

    if robot_effective:
        run_agent_with_robot(
            robot_ip=robot_effective,
            robot=robot,
            input_path=input_path,
            discord=discord,
            use_llm=not no_llm,
            llm=llm,
            skip_confirmations=True,
            debug_llm=debug_llm,
            agent_name=agent_name,
            commands=cmd_list,
            port_offset=port_offset,
            agent_config=agent_config,
            device=device,
            max_tokens=max_tokens,
            vl_include_camera=vl_include_effective,
        )
        return

    prompt_builder = get_prompt_builder(prompt)
    client = get_llm_client(llm, prompt_builder, device=device)
    if hasattr(client, "max_tokens"):
        client.max_tokens = max_tokens

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
