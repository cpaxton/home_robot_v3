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

"""Factory for DynaMem / EQA local VLLM clients."""

from __future__ import annotations

from typing import Any

from emet.llms.base import AbstractVLLMClient
from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family


def create_dynamem_vllm(
    vl_family: str,
    *,
    hf_model_id: str | None,
    vl_model_size: str,
    max_tokens: int,
    device: str,
    quantization: str | None,
    prompt: str | None = None,
) -> AbstractVLLMClient:
    """Construct a single local VLLM for captions + EQA (shared instance in caller).

    Default HF ids come from :data:`emet.llms.vllm_registry.SUPPORTED_VLLMS` when ``hf_model_id`` is omitted.
    """
    fam = normalize_vl_family(vl_family or "qwen3_vl")
    if fam == "qwen3_vl":
        from emet.llms.qwen3_vl_client import Qwen3VLClient

        mid = hf_model_id or default_hf_model_id("qwen3_vl") or "Qwen/Qwen3-VL-4B-Instruct"
        return Qwen3VLClient(
            prompt=prompt,
            hf_model_id=mid,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
        )
    if fam == "qwen2_5_vl":
        from emet.llms.qwen_client import Qwen25VLClient

        # bitsandbytes int4/int8 breaks Qwen2.5-VL-3B attention (4096 vs 2560 hidden); bf16 fits 24GB.
        if quantization in ("int4", "int8"):
            quantization = None
        mid = hf_model_id or default_hf_model_id("qwen2_5_vl")
        return Qwen25VLClient(
            prompt=prompt,
            model_size=vl_model_size,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
            hf_model_id=mid,
        )
    if fam == "gemma4":
        from emet.llms.gemma4_vllm_client import Gemma4VLLMClient

        mid = hf_model_id or default_hf_model_id("gemma4") or "google/gemma-4-E4B-it"
        return Gemma4VLLMClient(
            prompt=prompt,
            hf_model_id=mid,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
        )
    raise ValueError(
        f"Unknown vl_family {vl_family!r}; use qwen3_vl, qwen2_5_vl, or gemma4 (see dynav_config.yaml eqa:)."
    )


def dynamem_vllm_call(
    client: Any,
    user_content: str | list[Any],
    *,
    system_prompt: str | None,
    max_new_tokens: int,
) -> str:
    """Dispatch to :meth:`generate_multimodal` for VLLM clients; otherwise ``__call__``."""
    if isinstance(client, AbstractVLLMClient):
        return client.generate_multimodal(
            user_content,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            reset_context=True,
        )
    return client(user_content)
