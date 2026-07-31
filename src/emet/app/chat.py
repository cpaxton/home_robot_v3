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

    # Local gemma (legacy)
    emet run chat --llm gemma

    # Caliban text router
    emet run chat --caliban --once "Reply with exactly: pong"

    # Local / LAN VL serve (``emet serve llm --vl --port 8001``)
    emet run chat --vl --image /path/to.jpg --once "What do you see?"
    emet run chat --caliban --vl --vl-endpoint openai@http://127.0.0.1:8001/v1 \\
      --image shot.jpg --once "Describe briefly"
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
from emet.llms.remote_ops import DEFAULT_TEXT_BASE, DEFAULT_VL_BASE


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
    help=(
        "LLM key or openai@http://host:port/v1[#model]. "
        f"Common: {', '.join(sorted(get_llm_choices())[:8])}…"
    ),
)
@click.option("--prompt", default="simple", help="The prompt to use", type=click.Choice(get_prompt_choices()))
@click.option("--max_audio_duration", default=10.0, help="The maximum duration of the audio recording")
@click.option("--silence_limit", default=2.0, help="The amount of silence before stopping the recording")
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--voice", default=False, help="Enable voice chat", is_flag=True)
@click.option("--talk", default=False, help="Robot will speak its responses out load", is_flag=True)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
@click.option(
    "--caliban",
    is_flag=True,
    help=f"Preset text LLM to openai@{DEFAULT_TEXT_BASE} (Herman / LAN router).",
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
    help=(
        "Remote VL openai@URL (default: EMET_VL_ENDPOINT or "
        f"openai@{DEFAULT_VL_BASE})."
    ),
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
    prompt="simple",
    robot_ip="",
    talk=False,
    port_offset: int = 0,
    caliban: bool = False,
    use_vl: bool = False,
    vl_endpoint: str | None = None,
    images: tuple[str, ...] = (),
    once: str | None = None,
    max_tokens: int = 512,
):
    if caliban:
        llm = f"openai@{DEFAULT_TEXT_BASE}"
    if use_vl:
        ep = (vl_endpoint or os.environ.get("EMET_VL_ENDPOINT") or "").strip()
        if not ep:
            ep = f"openai@{DEFAULT_VL_BASE}"
        client: Any = _build_vl_client(ep, max_tokens=max_tokens)
        prompt_builder = None
        click.echo(colored(f"VL client: {ep}", "cyan"))
    else:
        prompt_builder = get_prompt_builder(prompt)
        client = _build_text_client(llm, prompt_builder)
        click.echo(colored(f"Text LLM: {llm}", "cyan"))

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

    def _one_turn(input_text: str) -> None:
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
            )
            response: Any = assistant_response
        else:
            assistant_response = client(input_text)
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
        _one_turn(once)
        return

    if voice:
        print("Talk to me, Stretch! If you don't say anything, I will give up.")
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
