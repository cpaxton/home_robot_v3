# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import os
from io import BytesIO
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from emet.llms.base import AbstractLLMClient, AbstractPromptBuilder


def resolve_openai_base_url(base_url: str | None = None) -> str | None:
    """Return OpenAI-compatible API base URL, or None for the public OpenAI default.

    Precedence: explicit ``base_url`` arg, then ``EMET_OPENAI_BASE_URL``, then ``OPENAI_BASE_URL``.
    """
    if base_url is not None and str(base_url).strip():
        return str(base_url).strip().rstrip("/")
    for key in ("EMET_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.rstrip("/")
    return None


def resolve_openai_api_key(api_key: str | None = None) -> str | None:
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    for key in ("EMET_OPENAI_API_KEY", "OPENAI_API_KEY", "EMET_LLM_SERVE_API_KEY"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    # Local emet serve llm often has no auth; OpenAI SDK still wants a string.
    return "emet-local"


class OpenaiClient(AbstractLLMClient):
    """Client for OpenAI or any OpenAI-compatible HTTP server (e.g. ``emet serve llm``).

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those.
        base_url: API root including ``/v1`` (or set ``EMET_OPENAI_BASE_URL``).
        api_key: Bearer token (or ``OPENAI_API_KEY`` / ``EMET_LLM_SERVE_API_KEY``).

    TODO: add the support for audio input
    """

    model_choices = ["gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest"]

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        prompt_kwargs: dict[str, Any] | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(prompt, prompt_kwargs)
        self.model = model
        self.base_url = resolve_openai_base_url(base_url)
        self.api_key = resolve_openai_api_key(api_key)
        if self.base_url is None and self.model not in self.model_choices:
            print("Your GPT model:", self.model)
            print("Below are some recommended GPT models:")
            for model_choice in self.model_choices:
                print(model_choice)
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._openai = OpenAI(**client_kwargs)

    def _process_input(self, command, verbose=False):
        """
        Transform command sent from the user to the command query OpenAI GPT

        TODO: Add audio support
        """
        if isinstance(command, str):
            user_commands = command
        else:
            user_commands = []  # type:ignore
            for c in command:
                # If this is a dict, then we assume it has already been formtted in the form of {"type": ""}
                if isinstance(c, dict):
                    user_commands.append(c)
                # If this is a strungm then we assume it is a text message from the user
                elif isinstance(c, str):
                    user_commands.append({"type": "text", "text": c})
                # For now, the only remaining option is image
                elif isinstance(c, Image.Image) or isinstance(c, np.ndarray):
                    if isinstance(c, np.ndarray):
                        image = Image.fromarray(c.astype(np.uint8), mode="RGB")
                    else:
                        image = c

                    buffered = BytesIO()
                    (image.save(buffered, format="PNG"),)
                    img_bytes = buffered.getvalue()
                    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                    user_commands.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}",
                            },
                        }
                    )
                else:
                    raise NotImplementedError("We only support text and image for now!")

        if verbose:
            print("input to the model:")
            if isinstance(user_commands, str):
                print(user_commands)
            else:
                for idx, user_command in enumerate(user_commands):
                    if "image_url" in user_command:
                        print(idx, ".", user_command["type"])
                    else:
                        print(idx, ".", user_command["type"], user_command["text"])
        return user_commands

    def __call__(self, command: str | list, verbose: bool = False, tools: list | None = None):
        if verbose:
            print(f"{self.system_prompt=}")
            if self.base_url:
                print(f"base_url={self.base_url}")

        command = self._process_input(command, verbose=verbose)  # type:ignore

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
            **kwargs,
        )
        msg = completion.choices[0].message

        # If the model made tool calls via native API, return structured data
        if tools and msg.tool_calls:
            import json as _json

            result_calls = []
            for tc in msg.tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except (ValueError, TypeError):
                    args = {}
                result_calls.append({"name": tc.function.name, "arguments": args})
            return _json.dumps({"tool_calls": result_calls, "message": msg.content or ""})

        output_text = msg.content
        if verbose:
            print(f"output_text={output_text}")
        return output_text

    def sample(self, command: str | list, n_samples: int, verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")

        command = self._process_input(command, verbose=verbose)  # type:ignore

        completion = self._openai.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
            n=n_samples,
        )
        choices = completion.choices
        if verbose:
            print(f"choices={choices}")
        return choices


if __name__ == "__main__":
    from emet.llms.prompts.ok_robot_prompt import OkRobotPromptBuilder

    prompt = OkRobotPromptBuilder(use_specific_objects=True)
    client = OpenaiClient(prompt, model="gpt-4o")
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)

    choices = client.sample("this room is a mess, could you put away the dirty towel?", n_samples=2, verbose=True)
