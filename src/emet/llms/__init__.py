# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
"""LLM client registry.

Heavy backends (Qwen / Gemma / transformers) are imported lazily inside
``get_llm_client`` so lightweight paths (``OpenaiClient``, ``emet serve llm`` HTTP
helpers) work on Jetson aarch64 without pulling sklearn/libgomp at import time.
"""

from __future__ import annotations

import os
from typing import Any, Union

from .base import AbstractLLMClient, AbstractPromptBuilder, AbstractVLLMClient, VLInferenceKind
from .chat_wrapper import LLMChatWrapper
from .openai_client import OpenaiClient
from .prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
from .prompts.ok_robot_prompt import OkRobotPromptBuilder
from .prompts.pickup_prompt import PickupPromptBuilder
from .prompts.simple_prompt import SimpleStretchPromptBuilder

__all__ = [
    "Gemma4AnyToAnyClient",
    "Gemma4VLLMClient",
    "GemmaClient",
    "LlamaClient",
    "OpenaiClient",
    "ObjectManipNavPromptBuilder",
    "SimpleStretchPromptBuilder",
    "OkRobotPromptBuilder",
    "PickupPromptBuilder",
    "AbstractLLMClient",
    "AbstractPromptBuilder",
    "AbstractVLLMClient",
    "VLInferenceKind",
    "LLMChatWrapper",
    "Qwen25Client",
    "Qwen25VLClient",
    "QWEN_VL_PRESETS",
    "GEMMA4_PRESETS",
    "GEMMA4_VLM_PRESETS",
    "is_vl_llm_key",
    "get_llm_client",
    "get_llm_choices",
]

# Gemma 4 small (E2B / E4B) on HF ``any-to-any``; keys must be matched before the generic ``"gemma" in client_type`` branch.
GEMMA4_PRESETS: dict[str, str] = {
    "gemma4-e2b": "google/gemma-4-e2b-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}

# Multimodal Gemma 4 (image-text-to-text) for agent + camera / shared EQA VLM.
GEMMA4_VLM_PRESETS: dict[str, str] = {
    "gemma4-vlm-e2b": "google/gemma-4-e2b-it",
    "gemma4-vlm-e4b": "google/gemma-4-E4B-it",
}


def __getattr__(name: str) -> Any:
    """Lazy export of heavy client classes (and Qwen helpers) for ``from emet.llms import …``."""
    if name == "Gemma4AnyToAnyClient":
        from .gemma4_any_client import Gemma4AnyToAnyClient

        return Gemma4AnyToAnyClient
    if name == "Gemma4VLLMClient":
        from .gemma4_vllm_client import Gemma4VLLMClient

        return Gemma4VLLMClient
    if name == "GemmaClient":
        from .gemma_client import GemmaClient

        return GemmaClient
    if name == "LlamaClient":
        from .llama_client import LlamaClient

        return LlamaClient
    if name in ("Qwen25Client", "Qwen25VLClient", "QWEN_VL_PRESETS", "get_qwen_variants", "get_qwen35_variants"):
        from . import qwen_client as qc

        return getattr(qc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _agent_prompt_builder() -> AbstractPromptBuilder:
    from emet.agent.prompt import AgentPromptBuilder

    return AgentPromptBuilder()


class ConversationalChatPromptBuilder(AbstractPromptBuilder):
    """Short multi-turn chat system prompt (LAN dogfood / ``emet run chat``)."""

    def configure(self, name: str = "Assistant", **kwargs) -> str:
        who = (name or "Assistant").strip() or "Assistant"
        return (
            f"You are {who}, a helpful conversational assistant. "
            "Remember earlier turns in this chat. Be concise and friendly. "
            "Do not claim to be named Stretch unless the user asks you to role-play. "
            "You cannot see cameras unless the user provides an image."
        )


prompts = {
    "simple": SimpleStretchPromptBuilder,
    "chat": ConversationalChatPromptBuilder,
    "object_manip_nav": ObjectManipNavPromptBuilder,
    "ok_robot": OkRobotPromptBuilder,
    "pickup": PickupPromptBuilder,
    "agent": _agent_prompt_builder,
}


def get_prompt_builder(prompt_type: str, **kwargs) -> AbstractPromptBuilder:
    """Return a prompt builder of the specified type."""
    if prompt_type not in prompts:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    factory = prompts[prompt_type]
    if prompt_type == "agent":
        return factory()
    return factory(**kwargs)


def get_prompt_choices():
    """Return a list of available prompt builders."""
    return prompts.keys()


def validate_llm_client_type(client_type: str) -> str:
    """Accept registry keys, ``openai``, or ``openai@http://…`` specs."""
    raw = (client_type or "").strip()
    if not raw:
        raise ValueError("llm client type is empty")
    low = raw.lower()
    if low == "openai" or low.startswith("openai@"):
        return raw
    choices = set(get_llm_choices())
    for c in choices:
        if c.lower() == low:
            return c
    raise ValueError(
        f"Unknown LLM {client_type!r}. Use a registry key, 'openai', or "
        f"'openai@http://host:port/v1[#model]'."
    )


def is_vl_llm_key(client_type: str) -> bool:
    """True when ``--llm`` selects a vision-language client (head camera on by default)."""
    from .qwen_client import QWEN_VL_PRESETS

    k = (client_type or "").strip().lower()
    if k in QWEN_VL_PRESETS or k in GEMMA4_VLM_PRESETS:
        return True
    if k in ("qwen3-vl-eqa", "gemma4-vl-eqa"):
        return True
    if k.startswith("qwen35-vlm-"):
        return True
    return "-vl-" in k or k.startswith("vl-")


def get_llm_choices():
    """Return a list of available LLM clients."""
    from .qwen_client import QWEN_VL_PRESETS, get_qwen35_variants, get_qwen_variants

    qwen35_vlm = [f"qwen35-vlm-{s}" for s in ("0.8B", "2B", "4B", "9B", "27B")]
    return sorted(
        set(["gemma", "llama", "openai", "qwen25", "gemma4b", "gemma1b"])
        | set(get_qwen_variants())
        | set(get_qwen35_variants())
        | set(QWEN_VL_PRESETS.keys())
        | set(GEMMA4_PRESETS.keys())
        | set(GEMMA4_VLM_PRESETS.keys())
        | {"qwen3-vl-eqa", "gemma4-vl-eqa"}
        | set(qwen35_vlm)
    )


def process_incoming_qwen_types(qwen_type: str):
    terms = qwen_type.split("-")
    if len(terms) == 1:
        model_size, typing_option, finetuning_option, quantization_option = (
            "3B",
            None,
            "Instruct",
            "int4",
        )
    else:
        if terms[1] not in ["Math", "Coder", "VL", "Deepseek"]:
            typing_option = None
        else:
            typing_option = terms[1]
            terms.remove(terms[1])
        model_size = terms[1]
        if len(terms) < 3:
            if len(terms) == 2 and terms[0].lower() == "qwen35":
                finetuning_option, quantization_option = None, "int4"
            else:
                finetuning_option, quantization_option = "Instruct", None
        elif len(terms) >= 4:
            q = terms[3].lower()
            finetuning_option, quantization_option = terms[2], "int4" if q == "int" else q
        elif "awq" in terms[2].lower():
            finetuning_option, quantization_option = "Instruct", "awq"
        elif "Instruct" in terms[2]:
            finetuning_option, quantization_option = "Instruct", None
        else:
            finetuning_option, quantization_option = None, terms[2].lower()

    return model_size, typing_option, finetuning_option, quantization_option


def get_llm_client(client_type: str, prompt: str | AbstractPromptBuilder, **kwargs) -> AbstractLLMClient:
    """Return an LLM client of the specified type.

    Args:
        client_type: The type of client to create.
        kwargs: Additional keyword arguments to pass to the client constructor.
            ``parameters`` (dynav :class:`~emet.core.parameters.Parameters`) is consumed
            only by ``qwen3-vl-eqa`` and is **not** forwarded to other clients.

    Returns:
        An LLM client.
    """
    kwargs = dict(kwargs)
    parameters = kwargs.pop("parameters", None)

    if client_type.lower() in GEMMA4_PRESETS:
        from .gemma4_any_client import Gemma4AnyToAnyClient

        key = client_type.lower()
        return Gemma4AnyToAnyClient(prompt, hf_model_id=GEMMA4_PRESETS[key], **kwargs)
    if client_type.lower() in GEMMA4_VLM_PRESETS:
        from .gemma4_vllm_client import Gemma4VLLMClient

        key = client_type.lower()
        dev = str(kwargs.get("device", "cuda"))
        mt = int(kwargs.get("max_tokens", 4096))
        return Gemma4VLLMClient(
            prompt,
            hf_model_id=GEMMA4_VLM_PRESETS[key],
            max_tokens=mt,
            device=dev,
        )
    if (client_type or "").strip().lower() == "gemma4-vl-eqa":
        from emet.core.parameters import get_parameters
        from emet.llms.eqa_vl_settings import resolve_vl_endpoint
        from emet.llms.vllm_factory import create_dynamem_vllm, eqa_vl_client_kwargs

        p = parameters if parameters is not None else get_parameters("dynav_config.yaml")
        eqa_cfg = p.get("eqa", {}) or {}
        if not isinstance(eqa_cfg, dict):
            eqa_cfg = {}
        vl_family = str(eqa_cfg.get("vl_family", "gemma4") or "gemma4").strip()
        hf_id = eqa_cfg.get("vl_hf_model_id")
        vl_sz = str(eqa_cfg.get("vl_model_size", "8B") or "8B")
        vl_tok = int(eqa_cfg.get("vl_max_tokens", 512) or 512)
        vl_q = eqa_cfg.get("vl_quantization", "int4")
        dev = str(kwargs.get("device", "cuda"))
        return create_dynamem_vllm(
            vl_family,
            hf_model_id=hf_id,
            vl_model_size=vl_sz,
            max_tokens=vl_tok,
            device=dev,
            quantization=vl_q,
            prompt=prompt,
            endpoint=resolve_vl_endpoint(p),
            **eqa_vl_client_kwargs(eqa_cfg),
        )
    if client_type.lower() in ("gemma", "gemma4b", "gemma1b"):
        from .gemma_client import GemmaClient

        if client_type == "gemma":
            model_size = "1b"
        else:
            model_size = client_type[-2:]
        return GemmaClient(prompt, model_size=model_size, **kwargs)
    if "gemma" in client_type:
        raise ValueError(
            f"Unknown Gemma client type: {client_type!r}. "
            f"Use gemma4-e2b/e4b, gemma4-vlm-e2b/e4b, gemma4-vl-eqa, or legacy gemma/gemma4b/gemma1b (Gemma 3)."
        )
    if client_type == "llama":
        from .llama_client import LlamaClient

        return LlamaClient(prompt, **kwargs)
    if client_type == "openai":
        model = kwargs.pop("model", None) or os.environ.get("EMET_OPENAI_MODEL", "").strip() or "gpt-4o"
        return OpenaiClient(prompt, model=model, **kwargs)
    if client_type.startswith("openai@"):
        rest = client_type[len("openai@") :].strip()
        model = kwargs.pop("model", None) or os.environ.get("EMET_OPENAI_MODEL", "").strip()
        if "#" in rest:
            rest, model_from_url = rest.rsplit("#", 1)
            model = model or model_from_url.strip()
        return OpenaiClient(prompt, model=model or "emet", base_url=rest, **kwargs)

    from .qwen_client import QWEN_VL_PRESETS, Qwen25Client, Qwen25VLClient

    if client_type in QWEN_VL_PRESETS:
        preset = QWEN_VL_PRESETS[client_type]
        return Qwen25VLClient(
            prompt,
            model_size=preset["model_size"],
            hf_model_id=preset.get("hf_model_id"),
            **kwargs,
        )
    if (client_type or "").strip().lower() == "qwen3-vl-eqa":
        from emet.core.parameters import get_parameters
        from emet.llms.eqa_vl_settings import resolve_vl_endpoint
        from emet.llms.vllm_factory import create_dynamem_vllm, eqa_vl_client_kwargs

        p = parameters if parameters is not None else get_parameters("dynav_config.yaml")
        eqa_cfg = p.get("eqa", {}) or {}
        if not isinstance(eqa_cfg, dict):
            eqa_cfg = {}
        vl_family = str(eqa_cfg.get("vl_family", "qwen3_vl") or "qwen3_vl").strip()
        hf_id = eqa_cfg.get("vl_hf_model_id")
        vl_sz = str(eqa_cfg.get("vl_model_size", "8B") or "8B")
        vl_tok = int(eqa_cfg.get("vl_max_tokens", 512) or 512)
        vl_q = eqa_cfg.get("vl_quantization", "int4")
        dev = str(kwargs.get("device", "cuda"))
        return create_dynamem_vllm(
            vl_family,
            hf_model_id=hf_id,
            vl_model_size=vl_sz,
            max_tokens=vl_tok,
            device=dev,
            quantization=vl_q,
            prompt=prompt,
            endpoint=resolve_vl_endpoint(p),
            **eqa_vl_client_kwargs(eqa_cfg),
        )
    if client_type.lower().startswith("qwen35-vlm-"):
        from emet.llms.eqa_qwen import Qwen35SharedVLChatClient

        tail = client_type[len("qwen35-vlm-") :].strip()
        if not tail:
            tail = "4B"
        model_size = tail.upper().replace(" ", "")
        dev = str(kwargs.get("device", "cuda"))
        q = kwargs.get("quantization")
        return Qwen35SharedVLChatClient(
            prompt,
            model_size=model_size,
            device=dev,
            parameters=parameters,
            quantization=q,
            max_tokens=1024,
        )
    if "qwen" in client_type:
        model_size, typing_option, fine_tuning, quantization_option = process_incoming_qwen_types(client_type)
        if str(client_type).lower().startswith("qwen35"):
            from emet.core.parameters import get_parameters
            from emet.llms.qwen3_5_client import Qwen35Client
            from emet.llms.vllm_factory import eqa_vl_image_kwargs

            p = parameters if parameters is not None else get_parameters("dynav_config.yaml")
            eqa_cfg = p.get("eqa", {}) or {}
            if not isinstance(eqa_cfg, dict):
                eqa_cfg = {}
            dev = str(kwargs.get("device", "cuda"))
            mt = int(kwargs.get("max_tokens", 256) or 256)
            return Qwen35Client(
                prompt,
                hf_model_id=f"Qwen/Qwen3.5-{model_size}",
                max_tokens=mt,
                device=dev,
                quantization=quantization_option or "int4",
                cache_system_prefix=False,
                **eqa_vl_image_kwargs(eqa_cfg),
            )
        return Qwen25Client(
            prompt,
            model_size=model_size,
            fine_tuning=fine_tuning,
            model_type=typing_option,
            quantization=quantization_option,
            **kwargs,
        )
    raise ValueError(f"Invalid client type: {client_type}")
