# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Qwen3.5 client (natively multimodal, ``Qwen3_5ForConditionalGeneration``).

Qwen3.5 (Feb 2026) is trained from scratch on interleaved image/video/text and
outperforms same-scale Qwen3-VL on visual understanding. Same message contract
as :class:`Qwen3VLClient`; differences handled here:

- thinking mode is ON by default -> disabled via ``enable_thinking=False`` plus a
  defensive ``<think>`` strip (EQA wants terse answers, not reasoning streams);
- post-trained checkpoints have no ``-Instruct`` suffix (``Qwen/Qwen3.5-9B``).

Requires ``transformers`` with qwen3_5 support (>=5.x). Optional ``flash-attn`` +
``fla`` packages enable the fast Gated DeltaNet kernels; without them HF falls back
to slower PyTorch ops (correct, just slower).
"""

from __future__ import annotations

import re
from typing import Any

from transformers import Qwen3_5ForConditionalGeneration

from emet.llms.base import AbstractPromptBuilder
from emet.llms.qwen3_vl_client import Qwen3VLClient

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


class Qwen35Client(Qwen3VLClient):
    """Qwen3.5 multimodal client (image+text -> text, thinking disabled)."""

    _MODEL_CLS = Qwen3_5ForConditionalGeneration
    _FAMILY_KEY = "qwen3_5"
    _TEMPLATE_KWARGS = {"enable_thinking": False}

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder | None = None,
        prompt_kwargs: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        num_beams: int = 1,
        device: str = "cuda",
        quantization: str | None = "int4",
        use_fast_attn: bool = False,
        hf_model_id: str | None = None,
    ):
        super().__init__(
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            hf_model_id=hf_model_id or "Qwen/Qwen3.5-9B",
            max_tokens=max_tokens,
            num_beams=num_beams,
            device=device,
            quantization=quantization,
            use_fast_attn=use_fast_attn,
        )

    def _postprocess_output(self, text: str) -> str:
        """Strip any residual ``<think>...</think>`` block despite enable_thinking=False."""
        return _THINK_BLOCK_RE.sub("", text).lstrip()
