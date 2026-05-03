# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# Gemma 4 small variants (E2B, E4B) are published with the Hugging Face
# ``any-to-any`` pipeline (see https://huggingface.co/blog/gemma4).

from __future__ import annotations

import timeit
from typing import Any

import torch
from termcolor import colored
from transformers import pipeline

from emet.llms.base import AbstractLLMClient, AbstractPromptBuilder
from emet.utils.logger import Logger

_logger = Logger(__name__)


def _extract_text_from_any_to_any_output(outputs: Any) -> str:
    """Normalize pipeline return shape across transformers versions."""
    if not outputs:
        return ""
    o0 = outputs[0]
    if not isinstance(o0, dict):
        return str(o0).strip()
    if "generated_text" in o0:
        gen = o0["generated_text"]
        if isinstance(gen, str):
            return gen.strip()
        if isinstance(gen, list) and gen:
            last = gen[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"]).strip()
        if isinstance(gen, dict) and "content" in gen:
            return str(gen["content"]).strip()
    for key in ("text", "content", "message"):
        if key in o0 and isinstance(o0[key], str):
            return o0[key].strip()
    return str(o0).strip()


class Gemma4AnyToAnyClient(AbstractLLMClient):
    """Gemma 4 (E2B / E4B) via Hugging Face ``any-to-any`` for **text** agent turns.

    Multimodal (image / camera) can be added later; the agent loop currently passes
    only text. Native JSON tool-calling in Gemma 4 is available via the processor;
    the embodied agent still relies on the system prompt (same pattern as Qwen)."""

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        prompt_kwargs: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        device: str = "cuda",
        hf_model_id: str = "google/gemma-4-E4B-it",
    ) -> None:
        super().__init__(prompt, prompt_kwargs)
        self.hf_model_id = hf_model_id
        self.max_tokens = int(max_tokens)
        if device not in ("cuda", "mps", "cpu"):
            raise ValueError(f"Invalid device: {device!r} (use cuda, mps, or cpu)")

        device_arg: int | str
        if device == "cpu":
            device_arg = -1
        else:
            device_arg = device

        mk: dict[str, Any] = {}
        if device == "cuda":
            mk["torch_dtype"] = torch.bfloat16
        pkw: dict[str, Any] = {"model": hf_model_id, "device": device_arg}
        if mk:
            pkw["model_kwargs"] = mk
        self._pipe = pipeline("any-to-any", **pkw)

    def __call__(self, command: str, verbose: bool = False, **kwargs: Any) -> str:
        # OpenAIClient-style ``tools=`` from the agent loop — tools are in the system prompt.
        if kwargs and kwargs.keys() - {"image"}:
            _logger.debug("Gemma4AnyToAnyClient ignoring non-image kwargs: %s", list(kwargs.keys()))
        t0 = timeit.default_timer()
        new_message = {"role": "user", "content": command}
        self.add_history(new_message)
        thread = self.get_history()
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for m in thread:
            messages.append(m)

        out = self._pipe(
            messages,
            max_new_tokens=self.max_tokens,
            return_full_text=False,
        )
        text = _extract_text_from_any_to_any_output(out)
        self.add_history({"role": "assistant", "content": text})
        t1 = timeit.default_timer()
        if verbose:
            print(colored("Assistant (gemma-4 any-to-any):", "blue"), text)
            print(colored("Time (s):", "blue"), f"{t1 - t0:.2f}")
        return text
