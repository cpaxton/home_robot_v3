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
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Gemma 3 / 4 multimodal VLM via Hugging Face ``AutoModelForImageTextToText``."""

from __future__ import annotations

import timeit
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from emet.llms.base import AbstractPromptBuilder, AbstractVLLMClient


def _content_to_gemma_messages(
    user_content: str | list[Any],
    *,
    system_prompt: str | None,
    image: Any | None,
) -> list[dict[str, Any]]:
    """Build chat messages for Gemma multimodal (image + text blocks)."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    parts: list[dict[str, Any]] = []
    if image is not None:
        pil = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")
        parts.append({"type": "image", "image": pil})
    if isinstance(user_content, str):
        parts.append({"type": "text", "text": user_content})
    else:
        for c in user_content:
            if isinstance(c, str):
                parts.append({"type": "text", "text": c})
            elif isinstance(c, Image.Image):
                parts.append({"type": "image", "image": c})
            elif isinstance(c, np.ndarray):
                parts.append({"type": "image", "image": Image.fromarray(c.astype(np.uint8), mode="RGB")})
            else:
                raise NotImplementedError("Gemma VLM: only str, PIL, ndarray supported in list content.")
    messages.append({"role": "user", "content": parts})
    return messages


class Gemma4VLLMClient(AbstractVLLMClient):
    """Multimodal Gemma (Gemma 3/4 ``image-text-to-text`` checkpoints)."""

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder | None = None,
        prompt_kwargs: dict[str, Any] | None = None,
        hf_model_id: str = "google/gemma-3-4b-it",
        max_tokens: int = 4096,
        device: str = "cuda",
        torch_dtype: torch.dtype | str = "bfloat16",
    ):
        super().__init__(prompt, prompt_kwargs)
        if device not in ("cuda", "mps", "cpu"):
            raise ValueError(f"Invalid device: {device}")
        self._device = device
        self.max_tokens = max_tokens
        self._resolved_hf_model_id = hf_model_id
        dtype = (
            torch_dtype if isinstance(torch_dtype, torch.dtype) else getattr(torch, str(torch_dtype), torch.bfloat16)
        )
        print(f"Loading Gemma multimodal model: {hf_model_id}")
        load_kw: dict[str, Any] = {"torch_dtype": dtype}
        if device == "cuda":
            load_kw["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(hf_model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_model_id, **load_kw)
        if device == "cpu":
            self.model = self.model.to("cpu")
        elif device == "mps":
            self.model = self.model.to("mps")

    @property
    def canonical_model_key(self) -> str:
        return f"gemma4:{self._resolved_hf_model_id}:{self._device}"

    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image: Any | None = None,
    ) -> str:
        if reset_context:
            self.reset()
        sys_use = system_prompt if system_prompt is not None else (self.system_prompt or None)
        messages = _content_to_gemma_messages(user_content, system_prompt=sys_use, image=image)
        ntok = max_new_tokens if max_new_tokens is not None else self.max_tokens
        t0 = timeit.default_timer()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        dev = next(self.model.parameters()).device
        inputs = inputs.to(dev)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=ntok, do_sample=False)
        trimmed = gen[0][input_len:]
        text_out = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        t1 = timeit.default_timer()
        if verbose:
            print(f"Gemma VLM response (truncated): {text_out[:500]}...")
            print(f"Time taken: {t1 - t0:.2f}s")
        self.add_history({"role": "assistant", "content": text_out})
        return text_out
