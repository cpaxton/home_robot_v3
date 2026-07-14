# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Pick a Transformers ``attn_implementation`` for local VL / LLM loads."""

from __future__ import annotations

from typing import Literal

AttnImpl = Literal["flash_attention_2", "sdpa", "eager"]


def resolve_attn_implementation(*, prefer_flash: bool = True, device: str = "cuda") -> AttnImpl:
    """Return the best available attention backend.

    Order on CUDA: ``flash_attention_2`` (if installed) → ``sdpa`` (PyTorch built-in) → ``eager``.
    Flash-Attn is optional and often needs a matching CUDA/torch wheel; SDPA is the practical
    default on RTX 40xx and is much faster than eager for long prompts.
    """
    if not str(device).startswith("cuda"):
        return "eager"
    if prefer_flash:
        try:
            from transformers.utils import is_flash_attn_2_available

            if is_flash_attn_2_available():
                return "flash_attention_2"
        except Exception:
            pass
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"
