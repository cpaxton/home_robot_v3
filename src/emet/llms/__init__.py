# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from typing import Union

from .base import AbstractLLMClient, AbstractPromptBuilder, AbstractVLLMClient, VLInferenceKind
from .chat_wrapper import LLMChatWrapper
from .gemma4_any_client import Gemma4AnyToAnyClient
from .gemma4_vllm_client import Gemma4VLLMClient
from .gemma_client import GemmaClient
from .llama_client import LlamaClient
from .openai_client import OpenaiClient
from .prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
from .prompts.ok_robot_prompt import OkRobotPromptBuilder
from .prompts.pickup_prompt import PickupPromptBuilder
from .prompts.simple_prompt import SimpleStretchPromptBuilder
from .qwen_client import QWEN_VL_PRESETS, Qwen25Client, Qwen25VLClient, get_qwen35_variants, get_qwen_variants

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
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

llms = {
    "gemma": GemmaClient,
    "llama": LlamaClient,
    "openai": OpenaiClient,
    "qwen25": Qwen25Client,
}


# Add all the various Qwen 2.5 and Qwen 3.5 variants
qwen_variants = get_qwen_variants()
llms.update(dict.fromkeys(qwen_variants, Qwen25Client))
for variant in get_qwen35_variants():
    llms[variant] = Qwen25Client
llms.update(dict.fromkeys(GEMMA4_PRESETS, Gemma4AnyToAnyClient))
llms.update(dict.fromkeys(GEMMA4_VLM_PRESETS, Gemma4VLLMClient))
llms.update(dict.fromkeys(["gemma4b", "gemma1b"], GemmaClient))


def process_incoming_qwen_types(qwen_type: str):
    terms = qwen_type.split("-")
    if len(terms) == 1:
        # default configuration
        model_size, typing_option, finetuning_option, quantization_option = (
            "3B",
            None,
            "Instruct",
            "int4",
        )
    else:
        # model type is None = using LM chat
        if terms[1] not in ["Math", "Coder", "VL", "Deepseek"]:
            typing_option = None
        else:
            typing_option = terms[1]
            terms.remove(terms[1])
        # the next item is model size
        model_size = terms[1]
        # if the quantization is None, meaning no quantization shall be applied
        if len(terms) < 3:
            # qwen35-9B is only two tokens; without this branch we would load full-precision 3.5 (huge VRAM / OOM).
            # Match the one-token default (int4) for local agent use.
            if len(terms) == 2 and terms[0].lower() == "qwen35":
                finetuning_option, quantization_option = None, "int4"
            else:
                finetuning_option, quantization_option = "Instruct", None
        # This means finetune with Instruct and using quantization "Instruct-Int4" or "Instruct-Int" (alias for Int4)
        elif len(terms) >= 4:
            q = terms[3].lower()
            finetuning_option, quantization_option = terms[2], "int4" if q == "int" else q
        # "AWQ"
        elif "awq" in terms[2].lower():
            finetuning_option, quantization_option = "Instruct", "awq"
        # "Int4"
        elif "Instruct" in terms[2]:
            finetuning_option, quantization_option = "Instruct", None
        else:
            finetuning_option, quantization_option = None, terms[2].lower()

    return model_size, typing_option, finetuning_option, quantization_option


def _agent_prompt_builder() -> "AbstractPromptBuilder":
    from emet.agent.prompt import AgentPromptBuilder

    return AgentPromptBuilder()


prompts = {
    "simple": SimpleStretchPromptBuilder,
    "object_manip_nav": ObjectManipNavPromptBuilder,
    "ok_robot": OkRobotPromptBuilder,
    "pickup": PickupPromptBuilder,
    "agent": _agent_prompt_builder,
}


def get_prompt_builder(prompt_type: str) -> AbstractPromptBuilder:
    """Return a prompt builder of the specified type.

    Args:
        prompt_type: The type of prompt builder to create.

    Returns:
        A prompt builder.
    """
    if prompt_type not in prompts:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    return prompts[prompt_type]()


def get_prompt_choices():
    """Return a list of available prompt builders."""
    return prompts.keys()


def is_vl_llm_key(client_type: str) -> bool:
    """True when ``--llm`` selects a vision-language client (head camera on by default)."""
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
    qwen35_vlm = [f"qwen35-vlm-{s}" for s in ("0.8B", "2B", "4B", "9B", "27B")]
    return sorted(
        set(llms.keys())
        | set(QWEN_VL_PRESETS.keys())
        | set(GEMMA4_VLM_PRESETS.keys())
        | {"qwen3-vl-eqa", "gemma4-vl-eqa"}
        | set(qwen35_vlm)
    )


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
        key = client_type.lower()
        return Gemma4AnyToAnyClient(prompt, hf_model_id=GEMMA4_PRESETS[key], **kwargs)
    if client_type.lower() in GEMMA4_VLM_PRESETS:
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
        from emet.llms.vllm_factory import create_dynamem_vllm

        p = parameters if parameters is not None else get_parameters("dynav_config.yaml")
        eqa_cfg = p.get("eqa", {}) or {}
        if not isinstance(eqa_cfg, dict):
            eqa_cfg = {}
        vl_family = str(eqa_cfg.get("vl_family", "gemma4") or "gemma4").strip()
        hf_id = eqa_cfg.get("vl_hf_model_id")
        vl_sz = str(eqa_cfg.get("vl_model_size", "4B") or "4B")
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
        )
    if client_type.lower() in ("gemma", "gemma4b", "gemma1b"):
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
    elif client_type == "llama":
        return LlamaClient(prompt, **kwargs)
    elif client_type == "openai":
        return OpenaiClient(prompt, **kwargs)
    elif client_type in QWEN_VL_PRESETS:
        preset = QWEN_VL_PRESETS[client_type]
        return Qwen25VLClient(
            prompt,
            model_size=preset["model_size"],
            hf_model_id=preset.get("hf_model_id"),
            **kwargs,
        )
    elif (client_type or "").strip().lower() == "qwen3-vl-eqa":
        # One Qwen3-VL load from dynav ``eqa:`` — same class as DynaMem captions so ``bind_shared_vllm_from_agent`` can dedup.
        from emet.core.parameters import get_parameters
        from emet.llms.vllm_factory import create_dynamem_vllm

        p = parameters if parameters is not None else get_parameters("dynav_config.yaml")
        eqa_cfg = p.get("eqa", {}) or {}
        if not isinstance(eqa_cfg, dict):
            eqa_cfg = {}
        vl_family = str(eqa_cfg.get("vl_family", "qwen3_vl") or "qwen3_vl").strip()
        hf_id = eqa_cfg.get("vl_hf_model_id")
        vl_sz = str(eqa_cfg.get("vl_model_size", "4B") or "4B")
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
        )
    elif client_type.lower().startswith("qwen35-vlm-"):
        # Shared Qwen3.5-VLM checkpoint with GraphEQA / EQA VL (avoid loading text-only qwen35-* + multimodal twice).
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
    elif "qwen" in client_type:
        # Parse model size and fine-tuning from client_type
        model_size, typing_option, fine_tuning, quantization_option = process_incoming_qwen_types(client_type)
        version = "3.5" if client_type.startswith("qwen35") else None
        return Qwen25Client(
            prompt,
            model_size=model_size,
            fine_tuning=fine_tuning,
            model_type=typing_option,
            quantization=quantization_option,
            version=version,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid client type: {client_type}")
