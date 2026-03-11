# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent chatbot: lightweight LLM (default Qwen 3.5) for local testing.
# Run with: emet run agent
# With --robot-ip: start robot with logging, optional --input-path and --discord; tool loop.

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


# Default: Qwen 3.5 4B (good balance of quality and speed for agent)
DEFAULT_AGENT_LLM = "qwen35-4B"


@click.command()
@click.option(
    "--llm",
    default=DEFAULT_AGENT_LLM,
    help=f"LLM to use (default: {DEFAULT_AGENT_LLM}). Use a small Coder model for local testing.",
    type=click.Choice(get_llm_choices()),
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
    default="",
    help="Robot IP. If set, start robot with logging and run agent loop (explore, pick/place, query memory, optional Discord).",
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Memory directory to load when using --robot-ip.",
)
@click.option(
    "--discord",
    is_flag=True,
    help="Start Discord bot when using --robot-ip (DISCORD_TOKEN in env; install deps: uv sync -e discord).",
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
def main(
    llm: str,
    prompt: str,
    device: str,
    voice: bool,
    max_tokens: int,
    robot_ip: str,
    input_path: str | None,
    discord: bool,
    no_llm: bool,
    debug_llm: bool,
) -> None:
    """Run the agent as a chatbot (lightweight Qwen Coder by default for local testing).

    With --robot-ip: start robot with LLM enabled by default to parse natural language; optional memory load and Discord.

    Examples:
      emet run agent
      emet run agent --device cpu
      emet run agent --llm qwen35-9B
      emet run agent --robot-ip 127.0.0.1 --input-path logs/memory_xxx --discord
      emet run agent --robot-ip 127.0.0.1 --no-llm   # letter commands only (E/M/Q/P)
    """
    if robot_ip:
        run_agent_with_robot(
            robot_ip=robot_ip,
            input_path=input_path,
            discord=discord,
            use_llm=not no_llm,
            llm=llm,
            skip_confirmations=True,
            debug_llm=debug_llm,
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

    print(colored("Agent chatbot (Qwen Coder). Type a message and press Enter. Empty line to quit.", "green"))
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
