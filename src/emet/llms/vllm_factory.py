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
from emet.llms.prefix_kv_cache import env_vl_cache_system_prefix
from emet.llms.vl_image import eqa_vl_image_kwargs
from emet.llms.vllm_registry import DEFAULT_QWEN3_VL_HF_MODEL_ID, default_hf_model_id, normalize_vl_family


def eqa_prefix_cache_kwargs(eqa_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ``cache_system_prefix`` / ``max_cached_prefixes`` from ``eqa:`` config + env."""
    cfg = eqa_cfg if isinstance(eqa_cfg, dict) else {}
    cache = bool(cfg.get("vl_cache_system_prefix", True))
    env_override = env_vl_cache_system_prefix()
    if env_override is not None:
        cache = env_override
    max_cached = int(cfg.get("vl_max_cached_prefixes", 1) or 1)
    return {"cache_system_prefix": cache, "max_cached_prefixes": max(1, max_cached)}


def eqa_vl_client_kwargs(eqa_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """All VL client knobs from ``eqa:`` (prefix cache + image downsample)."""
    return {**eqa_prefix_cache_kwargs(eqa_cfg), **eqa_vl_image_kwargs(eqa_cfg)}


def create_dynamem_vllm(
    vl_family: str,
    *,
    hf_model_id: str | None,
    vl_model_size: str,
    max_tokens: int,
    device: str,
    quantization: str | None,
    prompt: str | None = None,
    cache_system_prefix: bool = False,
    max_cached_prefixes: int = 1,
    image_max_side: int = 512,
    image_max_pixels: int = 0,
    endpoint: str | None = None,
    model: str | None = None,
) -> AbstractVLLMClient:
    """Construct a VLLM for captions + EQA (shared instance in caller).

    When ``endpoint`` is set (``eqa.vl_endpoint`` / ``EMET_VL_ENDPOINT``), returns
    :class:`~emet.llms.openai_vllm_client.OpenaiVLLMClient` and skips local weights.
    Default HF ids come from :data:`emet.llms.vllm_registry.SUPPORTED_VLLMS` when
    ``hf_model_id`` is omitted for local loads.
    """
    ep = (endpoint or "").strip()
    if ep:
        from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec

        base_url, model_from_spec = parse_openai_endpoint_spec(ep)
        mid = (model or model_from_spec or hf_model_id or "emet-vl").strip()
        return OpenaiVLLMClient(
            prompt=prompt,
            model=mid,
            base_url=base_url,
            max_tokens=max_tokens,
            image_max_side=image_max_side,
            image_max_pixels=image_max_pixels,
            device="remote",
        )

    fam = normalize_vl_family(vl_family or "qwen3_vl")
    if fam == "qwen3_vl":
        from emet.llms.qwen3_vl_client import Qwen3VLClient

        mid = hf_model_id or default_hf_model_id("qwen3_vl") or DEFAULT_QWEN3_VL_HF_MODEL_ID
        return Qwen3VLClient(
            prompt=prompt,
            hf_model_id=mid,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
            cache_system_prefix=cache_system_prefix,
            max_cached_prefixes=max_cached_prefixes,
            image_max_side=image_max_side,
            image_max_pixels=image_max_pixels,
        )
    if fam == "qwen3_5":
        from emet.llms.qwen3_5_client import Qwen35Client

        mid = hf_model_id or default_hf_model_id("qwen3_5") or "Qwen/Qwen3.5-9B"
        return Qwen35Client(
            prompt=prompt,
            hf_model_id=mid,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
            cache_system_prefix=cache_system_prefix,
            max_cached_prefixes=max_cached_prefixes,
            image_max_side=image_max_side,
            image_max_pixels=image_max_pixels,
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
    if fam == "internvl":
        from emet.llms.internvl_client import DEFAULT_INTERNVL_HF_MODEL_ID, InternVLClient

        mid = hf_model_id or default_hf_model_id("internvl") or DEFAULT_INTERNVL_HF_MODEL_ID
        return InternVLClient(
            prompt=prompt,
            hf_model_id=mid,
            max_tokens=max_tokens,
            device=device,
            quantization=quantization,
            cache_system_prefix=cache_system_prefix,
            max_cached_prefixes=max_cached_prefixes,
            image_max_side=image_max_side,
            image_max_pixels=image_max_pixels,
        )
    raise ValueError(
        f"Unknown vl_family {vl_family!r}; use qwen3_vl, qwen3_5, qwen2_5_vl, gemma4, or internvl "
        "(see dynav_config.yaml eqa:)."
    )


def dynamem_vllm_call(
    client: Any,
    user_content: str | list[Any],
    *,
    system_prompt: str | None,
    max_new_tokens: int,
    assistant_prefill: str | None = None,
) -> str:
    """Dispatch to :meth:`generate_multimodal` for VLLM clients; otherwise ``__call__``."""
    if isinstance(client, AbstractVLLMClient):
        return client.generate_multimodal(
            user_content,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            reset_context=True,
            assistant_prefill=assistant_prefill,
        )
    return client(user_content)
