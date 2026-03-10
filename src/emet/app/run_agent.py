# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent chatbot: lightweight LLM (default Qwen 2.5 Coder) for local testing.
# Run with: emet run agent
# Optional: --llm, --device cpu, --voice. Tool-calling and robot integration can be added later.

import os
import timeit

import click
from termcolor import colored

from emet.llms import get_llm_choices, get_llm_client, get_prompt_builder, get_prompt_choices


# Default: lightweight Qwen 2.5 Coder (designed for tool use), small and easy to run locally
DEFAULT_AGENT_LLM = "qwen25-Coder-1.5B-Instruct-Int4"


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
@click.option("--robot-ip", "--robot_ip", default="", help="Robot IP (for future robot integration; ignored in chat-only mode).")
def main(
    llm: str,
    prompt: str,
    device: str,
    voice: bool,
    max_tokens: int,
    robot_ip: str,
) -> None:
    """Run the agent as a chatbot (lightweight Qwen Coder by default for local testing).

    Examples:
      emet run agent
      emet run agent --device cpu
      emet run agent --llm qwen25-Coder-3B-Instruct-Int4
    """
    prompt_builder = get_prompt_builder(prompt)
    client = get_llm_client(llm, prompt_builder, device=device)
    if hasattr(client, "max_tokens"):
        client.max_tokens = max_tokens

    if voice:
        import tempfile
        from emet.audio import AudioRecorder
        from emet.audio.speech_to_text import WhisperSpeechToText
        audio_recorder = AudioRecorder()
        whisper = WhisperSpeechToText()
    else:
        audio_recorder = None
        whisper = None

    print(colored("Agent chatbot (Qwen Coder). Type a message and press Enter. Empty line to quit.", "green"))
    print(colored(f"LLM: {llm}  device: {device}", "yellow"))
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

        t0 = timeit.default_timer()
        reply = client(user_text)
        t1 = timeit.default_timer()

        if hasattr(prompt_builder, "parse_response") and callable(prompt_builder.parse_response):
            reply = prompt_builder.parse_response(reply)
        print(colored("Agent:", "blue"), reply)
        print(colored(f"({t1 - t0:.2f}s)", "yellow"))
        print("-" * 60)


if __name__ == "__main__":
    main()
