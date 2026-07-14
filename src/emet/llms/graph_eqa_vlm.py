# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared local VLM clients for GraphEQA (EQA + label extraction on one checkpoint)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from emet.benchmarks.sqa3d.prompts import SQA3D_EQA_PROMPT
from emet.core.parameters import Parameters
from emet.llms.eqa_vl_settings import apply_eqa_vl_runtime_settings, get_eqa_vl_int, resolve_vl_hf_model_id
from emet.llms.prompts.eqa_prompt import EQA_PROMPT
from emet.llms.prompts.hmeqa_eqa_prompt import HMEQA_EQA_PROMPT
from emet.llms.vllm_factory import create_dynamem_vllm, dynamem_vllm_call, eqa_vl_client_kwargs
from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

_SHARED_VLM: Any | None = None
_SHARED_VLM_KEY: tuple[Any, ...] | None = None


def _eqa_system_prompt(parameters: Parameters | dict | None) -> str:
    eqa = _eqa_cfg_dict(parameters)
    variant = str(eqa.get("prompt_variant", "") or "").strip().lower()
    if variant in ("hmeqa", "mcq"):
        return HMEQA_EQA_PROMPT
    if variant in ("sqa3d", "situated"):
        return SQA3D_EQA_PROMPT
    return EQA_PROMPT


def _eqa_cfg_dict(parameters: Parameters | dict | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        raw = parameters.get("eqa", {}) or {}
    else:
        raw = parameters.get("eqa", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _registry_vl_family(parameters: Parameters | dict | None) -> str | None:
    eqa = _eqa_cfg_dict(parameters)
    backend = str(eqa.get("backend", "qwen_vl") or "qwen_vl").strip()
    if backend != "qwen_vl":
        return None
    fam = normalize_vl_family(str(eqa.get("vl_family", "") or ""))
    if fam in ("qwen3_vl", "qwen3_5", "qwen2_5_vl", "gemma4"):
        return fam
    return None


def _get_shared_vlm(
    parameters: Parameters | dict | None,
    *,
    device: str | None = None,
) -> Any:
    global _SHARED_VLM, _SHARED_VLM_KEY
    eqa = _eqa_cfg_dict(parameters)
    fam = _registry_vl_family(parameters)
    if fam is None:
        raise ValueError("build_graph_eqa_vlm_clients requires eqa.vl_family qwen3_vl|qwen3_5|qwen2_5_vl|gemma4")
    dev = _resolve_device(device)
    hf_id = resolve_vl_hf_model_id(fam, parameters, device=dev) or default_hf_model_id(fam)
    vl_sz = str(eqa.get("vl_model_size", "8B") or "8B")
    vl_tok = int(eqa.get("vl_max_tokens", 512) or 512)
    vl_q = eqa.get("vl_quantization", "int4")
    key = (fam, hf_id, vl_sz, vl_tok, vl_q, dev)
    if _SHARED_VLM is not None and _SHARED_VLM_KEY == key:
        return _SHARED_VLM
    if _SHARED_VLM is not None and _SHARED_VLM_KEY != key:
        release_shared_graph_eqa_vlm()
    _SHARED_VLM = create_dynamem_vllm(
        fam,
        hf_model_id=hf_id,
        vl_model_size=vl_sz,
        max_tokens=vl_tok,
        device=dev,
        quantization=vl_q,
        prompt=None,
        **eqa_vl_client_kwargs(eqa),
    )
    _SHARED_VLM_KEY = key
    return _SHARED_VLM


class GraphEQAVLMClient:
    """Callable matching GraphEQA memory (command list + optional system prompt)."""

    def __init__(self, vl_client: Any, *, system_prompt: str | None, max_tokens: int):
        self._vl = vl_client
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    def __call__(self, command: str | list, **kwargs) -> str:
        max_new = int(kwargs.get("max_new_tokens", self._max_tokens))
        system_prompt = kwargs.get("system_prompt", self._system_prompt)
        return dynamem_vllm_call(
            self._vl,
            command,
            system_prompt=system_prompt,
            max_new_tokens=max_new,
        )


def trim_shared_graph_eqa_vlm_cache() -> None:
    """Release activation memory after an EQA forward without unloading weights."""
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_shared_graph_eqa_vlm() -> None:
    """Drop the process-global GraphEQA VLM so Habitat episodes do not accumulate VRAM."""
    global _SHARED_VLM, _SHARED_VLM_KEY
    if _SHARED_VLM is None:
        return
    try:
        del _SHARED_VLM
    except Exception:
        pass
    _SHARED_VLM = None
    _SHARED_VLM_KEY = None
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_graph_eqa_vlm_clients(
    *,
    parameters: Parameters | dict | None = None,
    device: str | None = None,
    keyword_max_tokens: int | None = None,
    eqa_max_tokens: int | None = None,
) -> tuple[Callable[..., str], Callable[..., str]]:
    """Return (keyword/image-description client, EQA client) sharing one VLM load."""
    apply_eqa_vl_runtime_settings(parameters)
    fam = _registry_vl_family(parameters)
    if fam is not None:
        shared = _get_shared_vlm(parameters, device=device)
        if keyword_max_tokens is None:
            keyword_max_tokens = get_eqa_vl_int(parameters, "graph_keyword_max_tokens", 64)
        if eqa_max_tokens is None:
            eqa_max_tokens = get_eqa_vl_int(parameters, "eqa_max_tokens", 1024)
        keyword_client = GraphEQAVLMClient(shared, system_prompt=None, max_tokens=keyword_max_tokens)
        eqa_client = GraphEQAVLMClient(
            shared,
            system_prompt=_eqa_system_prompt(parameters),
            max_tokens=eqa_max_tokens,
        )
        return keyword_client, eqa_client

    from emet.llms.eqa_qwen import build_shared_eqa_clients

    kw = (
        keyword_max_tokens
        if keyword_max_tokens is not None
        else get_eqa_vl_int(parameters, "graph_keyword_max_tokens", 64)
    )
    return build_shared_eqa_clients(
        parameters=parameters,
        keyword_max_tokens=kw,
        eqa_max_tokens=eqa_max_tokens,
        device=device,
    )
