# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Interactive / one-shot chat against local or remote OpenAI-compatible LLMs/VLMs.

Examples::

    # Local gemma (legacy Stretch prompt)
    emet run chat --llm gemma --prompt simple

    # Remote multi-turn REPL (history retained; conversational prompt)
    emet run chat --host caliban
    emet run chat --host caliban --once "Reply with exactly: pong"

    # Vision-language
    emet run chat --host caliban --vl --image /path/to.jpg --once "What do you see?"
"""

from __future__ import annotations

import os
import tempfile
import time
import timeit
from typing import Any

import click
import numpy as np
from PIL import Image
from termcolor import colored

from emet.audio import AudioRecorder
from emet.audio.speech_to_text import WhisperSpeechToText
from emet.controller.zmq_client import StretchZmqClient
from emet.llms import get_llm_choices, get_llm_client, get_prompt_builder, get_prompt_choices
from emet.llms.remote_ops import DEFAULT_LLM_PORT, DEFAULT_VL_PORT, apply_llm_host, resolve_llm_host


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _build_text_client(llm: str, prompt: Any):
    return get_llm_client(llm, prompt)


def _build_vl_client(vl_endpoint: str, *, max_tokens: int = 512):
    from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec

    base, model = parse_openai_endpoint_spec(vl_endpoint)
    return OpenaiVLLMClient(
        prompt=None,
        model=model or "emet-vl",
        base_url=base,
        max_tokens=max_tokens,
    )


@click.command()
@click.option(
    "--llm",
    default="gemma",
    show_default=True,
    help=(f"LLM key or openai@http://host:port/v1[#model]. Common: {', '.join(sorted(get_llm_choices())[:8])}…"),
)
@click.option(
    "--prompt",
    default=None,
    help="Prompt builder (default: chat with --host, else simple). "
    f"Choices: {', '.join(sorted(get_prompt_choices()))}.",
)
@click.option("--name", "persona_name", default="Assistant", show_default=True, help="Persona for --prompt chat.")
@click.option("--max_audio_duration", default=10.0, help="The maximum duration of the audio recording")
@click.option("--silence_limit", default=2.0, help="The amount of silence before stopping the recording")
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--voice", default=False, help="Enable voice chat", is_flag=True)
@click.option("--talk", default=False, help="Robot will speak its responses out load", is_flag=True)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
@click.option(
    "--host",
    "llm_host",
    default=None,
    help="LAN OpenAI host (sets EMET_* endpoints). Example: --host caliban",
)
@click.option(
    "--port",
    "llm_port",
    default=DEFAULT_LLM_PORT,
    show_default=True,
    type=int,
    help="OpenAI serve port used with --host (text; also VL unless --vl-port).",
)
@click.option(
    "--vl-port",
    default=None,
    type=int,
    help=f"VL port with --host (default {DEFAULT_VL_PORT}; use 8001 for dual-2b).",
)
@click.option(
    "--vl/--no-vl",
    "use_vl",
    default=False,
    show_default=True,
    help="Use OpenaiVLLMClient (multimodal). Requires --image or paste paths each turn.",
)
@click.option(
    "--vl-endpoint",
    default=None,
    help="Remote VL openai@URL (default: EMET_VL_ENDPOINT or --host / EMET_LLM_HOST).",
)
@click.option(
    "--image",
    "images",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Image path(s) for VL turns (repeatable).",
)
@click.option(
    "--once",
    default=None,
    help="Send one message and exit (non-interactive smoke).",
)
@click.option("--max-tokens", default=512, show_default=True, type=int, help="Generation length.")
def main(
    llm="gemma",
    max_audio_duration: float = 10.0,
    silence_limit: float = 2.0,
    voice=False,
    prompt: str | None = None,
    persona_name: str = "Assistant",
    robot_ip="",
    talk=False,
    port_offset: int = 0,
    llm_host: str | None = None,
    llm_port: int = DEFAULT_LLM_PORT,
    vl_port: int | None = None,
    use_vl: bool = False,
    vl_endpoint: str | None = None,
    images: tuple[str, ...] = (),
    once: str | None = None,
    max_tokens: int = 512,
):
    specs = apply_llm_host(llm_host, port=llm_port, vl_port=vl_port)
    host = resolve_llm_host(llm_host)
    if specs is not None:
        llm = specs[0]
    prompt_key = prompt
    if prompt_key is None:
        prompt_key = "chat" if host else "simple"

    if use_vl:
        ep = (vl_endpoint or os.environ.get("EMET_VL_ENDPOINT") or "").strip()
        if not ep:
            if specs is not None:
                ep = specs[1]
            else:
                raise click.UsageError("--vl needs --vl-endpoint, EMET_VL_ENDPOINT, or --host / EMET_LLM_HOST")
        client: Any = _build_vl_client(ep, max_tokens=max_tokens)
        prompt_builder = None
        click.echo(colored(f"VL client: {ep}", "cyan"))
    else:
        if prompt_key == "chat":
            prompt_builder = get_prompt_builder("chat", name=persona_name)
        else:
            prompt_builder = get_prompt_builder(prompt_key)
        client = _build_text_client(llm, prompt_builder)
        click.echo(colored(f"Text LLM: {llm}  prompt={prompt_key}", "cyan"))

    if talk:
        robot = StretchZmqClient(robot_ip, port_offset=port_offset)
    else:
        robot = None

    if voice:
        if use_vl:
            raise click.UsageError("--voice is not supported with --vl")
        audio_recorder = AudioRecorder()
        whisper = WhisperSpeechToText()
    else:
        audio_recorder = None
        whisper = None

    image_arrays = [_load_rgb(p) for p in images]

    def _one_turn(input_text: str, *, reset_context: bool = False) -> None:
        t0 = timeit.default_timer()
        if use_vl:
            parts: list[Any] = []
            if input_text.strip():
                parts.append(input_text)
            parts.extend(image_arrays)
            if not parts:
                raise click.UsageError("VL turn needs --once/--image text or image content")
            assistant_response = client.generate_multimodal(
                parts if len(parts) > 1 else parts[0],
                max_new_tokens=max_tokens,
                reset_context=reset_context or once is not None,
            )
            response: Any = assistant_response
        else:
            assistant_response = client(input_text, reset_context=reset_context)
            response = prompt_builder.parse_response(assistant_response) if prompt_builder else assistant_response
        t1 = timeit.default_timer()

        print(colored("Response:", "blue"), response)
        if robot is not None:
            if isinstance(response, str):
                robot.say(response)
            elif isinstance(response, list):
                for r in response:
                    if r[0] == "say":
                        robot.say_sync(r[1])
                        time.sleep(5.0)

        print("-" * 80)
        print("Time taken:", t1 - t0)
        print("-" * 80)

    if once is not None:
        _one_turn(once, reset_context=True)
        return

    if voice:
        print("Talk to me! If you don't say anything, I will give up.")
    else:
        print(colored("Multi-turn chat (history on). Empty line to exit.", "yellow"))
    for _i in range(50):
        if voice:
            input(colored("Press enter to speak or ctrl+c to exit.", "yellow"))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_filename = temp_audio_file.name
                audio_recorder.record(temp_filename, duration=max_audio_duration, silence_limit=silence_limit)
                input_text = whisper.transcribe_file(temp_filename)
                os.remove(temp_filename)
                print(colored("I heard:", "green"), input_text)
        else:
            input_text = input(colored("You: ", "green"))

        if len(input_text) == 0:
            break
        _one_turn(input_text)


if __name__ == "__main__":
    main()
