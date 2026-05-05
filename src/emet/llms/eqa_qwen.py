# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Default EQA backend: **Qwen3.5** multimodal on Hugging Face (``Qwen/Qwen3.5-*``), loaded once per
# process and shared between EQA and short text helpers (keyword extraction, etc.).
#
# Model size and Hub logging follow ``dynav_config.yaml`` → ``eqa_vl:`` (see that file). Env overrides:
# ``EMET_EQA_VL_MODEL_SIZE``, ``EMET_VERBOSE_HF``.

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

import torch

from emet.core.parameters import Parameters
from emet.llms.base import AbstractLLMClient, AbstractPromptBuilder
from emet.llms.eqa_vl_settings import (
    apply_eqa_vl_runtime_settings,
    get_eqa_vl_int,
    get_eqa_vl_str,
    resolve_eqa_vl_model_size,
    sync_resolved_eqa_vl_model_size_from_explicit,
)
from emet.llms.prompts.eqa_prompt import EQA_PROMPT
from emet.llms.qwen_client import Qwen35VLClient
from emet.utils.logger import Logger
from emet.utils.vram_debug import print_vram_snapshot

logger = Logger(__name__)

_shared_qwen35_vl: Qwen35VLClient | None = None
_shared_qwen35_vl_lock = threading.Lock()

# Multimodal checkpoints to try on CUDA OOM (largest → smallest). Matches ``eqa_vl/model_size`` presets.
_EQA_VL_SIZE_FALLBACK_CHAIN = ("27B", "9B", "4B", "2B", "0.8B")


def _eqa_vl_sizes_to_try(preferred: str) -> list[str]:
    """Return ``preferred`` first, then smaller Qwen3.5-VL checkpoints until exhausted."""
    ps = str(preferred).strip()
    chain = _EQA_VL_SIZE_FALLBACK_CHAIN
    if ps in chain:
        return list(chain[chain.index(ps) :])
    # MoE / uncommon Hub ids: try once, then fall back from 9B downward.
    return [ps] + list(chain[chain.index("9B") :])


def _is_vram_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg


def _release_cuda_memory_best_effort() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_shared_qwen35_vl_for_tests() -> None:
    """Clear the process-wide VL client (pair with ``reset_eqa_vl_resolution_for_tests`` in tests)."""
    global _shared_qwen35_vl
    _shared_qwen35_vl = None


def default_eqa_vl_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "Qwen3.5 multimodal EQA requires a GPU (CUDA or Apple MPS). "
        "Set up CUDA/MPS or use an environment with GPU support.",
    )


class Qwen35SharedVLChatClient(AbstractLLMClient):
    """Agent chat that uses :func:`get_shared_qwen35_vl_client` — **same** Qwen3.5-VLM weights as GraphEQA / EQA VL.

    Prefer ``--llm qwen35-vlm-4B`` (or ``9B``) instead of ``qwen35-4B`` / ``qwen35-9B`` when using embodied
    GraphEQA so Hugging Face does not load a second checkpoint (text-only ``AutoModelForCausalLM`` +
    multimodal ``Qwen3_5ForConditionalGeneration``).
    """

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        *,
        model_size: str,
        device: str = "cuda",
        parameters: Parameters | dict | None = None,
        quantization: str | None = None,
        max_tokens: int = 1024,
        prompt_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(prompt, prompt_kwargs)
        apply_eqa_vl_runtime_settings(parameters)
        q = quantization if quantization is not None else get_eqa_vl_str(parameters, "quantization", "int4")
        self._vl = get_shared_qwen35_vl_client(
            model_size=model_size,
            device=device,
            quantization=q,
            parameters=parameters,
        )
        self.max_tokens = max_tokens

    def __call__(self, command: str, verbose: bool = False, image: Any | None = None, **kwargs: Any) -> str:
        del kwargs
        if self.is_first_message():
            self.add_history({"role": "system", "content": self.system_prompt})
        if image is not None:
            import numpy as np
            from PIL import Image

            pil = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")
            user_content: Any = [
                {"type": "image", "image": pil},
                {"type": "text", "text": command},
            ]
        else:
            user_content = command
        self.add_history({"role": "user", "content": user_content})
        messages = self.get_history()
        out = self._vl(
            messages,
            verbose=verbose,
            system_prompt=None,
            max_new_tokens=self.max_tokens,
        )
        self.add_history({"role": "assistant", "content": out})
        self._iterations += 1
        return out


def _warn_shared_vl_size_mismatch(requested: str | None) -> None:
    global _shared_qwen35_vl
    if _shared_qwen35_vl is None or not (requested and str(requested).strip()):
        return
    got = getattr(_shared_qwen35_vl, "model_size", None)
    req = str(requested).strip()
    if got and req and got != req:
        logger.warning(
            "EQA VL shared client is already Qwen3.5-%s; ignoring requested size %r "
            "(one multimodal load per process). Prefer matching ``--llm qwen35-vlm-%s`` with ``eqa_vl/model_size: \"%s\"``.",
            got,
            req,
            got,
            got,
        )


def get_shared_qwen35_vl_client(
    *,
    model_size: str | None = None,
    device: str | None = None,
    quantization: str | None = None,
    parameters: Parameters | dict | None = None,
) -> Qwen35VLClient:
    """
    Return a single process-wide Qwen3.5 multimodal client (same weights for EQA + helpers).

    Loads at most once: later calls return the existing client even if free VRAM dropped (avoids a
    second checkpoint after SigLIP / first VL pass). Size comes from ``resolve_eqa_vl_model_size``
    (or an explicit ``model_size``). On **CUDA/MPS OOM** while building the first client, automatically
    retries with smaller Qwen3.5-VL checkpoints (e.g. 9B → 4B → 2B → 0.8B) before failing.
    """
    global _shared_qwen35_vl
    if _shared_qwen35_vl is not None:
        _warn_shared_vl_size_mismatch(model_size)
        return _shared_qwen35_vl
    with _shared_qwen35_vl_lock:
        if _shared_qwen35_vl is not None:
            _warn_shared_vl_size_mismatch(model_size)
            return _shared_qwen35_vl
        if device is None:
            device = default_eqa_vl_device()
        if quantization is None:
            quantization = get_eqa_vl_str(parameters, "quantization", "int4")
        if model_size is None:
            model_size = resolve_eqa_vl_model_size(parameters, device=device)
        else:
            model_size = sync_resolved_eqa_vl_model_size_from_explicit(model_size)

        requested_start = model_size
        sizes = _eqa_vl_sizes_to_try(model_size)
        for idx, sz in enumerate(sizes):
            try:
                _shared_qwen35_vl = Qwen35VLClient(
                    prompt=None,
                    model_size=sz,
                    max_tokens=4096,
                    num_beams=1,
                    device=device,
                    quantization=quantization,
                )
                sync_resolved_eqa_vl_model_size_from_explicit(sz)
                if sz != requested_start:
                    logger.warning(
                        "EQA VL: loaded Qwen3.5-%s instead of requested %s (OOM while loading a larger checkpoint).",
                        sz,
                        requested_start,
                    )
                print_vram_snapshot(
                    "eqa_qwen_shared_qwen35_vl_first_load",
                    extra=f"Qwen3.5-{sz} multimodal (eqa_vl / graph helpers)",
                )
                return _shared_qwen35_vl
            except Exception as e:
                if _is_vram_oom(e) and idx < len(sizes) - 1:
                    logger.warning(
                        "EQA VL: failed to load Qwen3.5-%s (%s); retrying smaller multimodal checkpoint.",
                        sz,
                        e,
                    )
                    _release_cuda_memory_best_effort()
                    continue
                raise


class Qwen3VLEQAClient:
    """
    Drop-in replacement for ``GeminiClient`` in EQA paths: ``__call__(command)`` where
    ``command`` is a list of text strings and PIL images (same as graph / voxel EQA).
    """

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        *,
        model_size: str | None = None,
        device: str | None = None,
        max_tokens: int = 1024,
        quantization: str | None = None,
        shared_vl: Qwen35VLClient | None = None,
        parameters: Parameters | dict | None = None,
    ):
        if isinstance(prompt, str):
            system_prompt = prompt
        elif isinstance(prompt, AbstractPromptBuilder):
            system_prompt = str(prompt)
        else:
            system_prompt = str(prompt)
        if device is None:
            device = default_eqa_vl_device()
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        if shared_vl is not None:
            self._vl = shared_vl
        else:
            self._vl = get_shared_qwen35_vl_client(
                model_size=model_size,
                device=device,
                quantization=quantization,
                parameters=parameters,
            )

    def __call__(
        self,
        command: str | list,
        model: str | None = None,
        verbose: bool = False,
    ) -> str:
        del model  # API parity with GeminiClient
        return self._vl(
            command,
            verbose=verbose,
            system_prompt=self._system_prompt,
            max_new_tokens=self._max_tokens,
        )

    def sample(
        self,
        command: str | list,
        model: str | None = None,
        n_samples: int = 4,
        verbose: bool = False,
    ) -> list:
        return [self.__call__(command, model=model, verbose=verbose) for _ in range(n_samples)]


def build_shared_eqa_clients(
    *,
    keyword_max_tokens: int | None = None,
    eqa_max_tokens: int | None = None,
    model_size: str | None = None,
    device: str | None = None,
    quantization: str | None = None,
    parameters: Parameters | dict | None = None,
) -> tuple[Callable[..., str], Qwen3VLEQAClient]:
    """
    Pair of callables sharing one Qwen3.5 load: short keyword-style calls vs full EQA.

    Token defaults come from ``eqa_vl/keyword_max_tokens`` / ``eqa_vl/eqa_max_tokens`` in dynav config
    when ``keyword_max_tokens`` / ``eqa_max_tokens`` are left unset.
    """
    apply_eqa_vl_runtime_settings(parameters)
    if device is None:
        device = default_eqa_vl_device()
    if quantization is None:
        quantization = get_eqa_vl_str(parameters, "quantization", "int4")
    if keyword_max_tokens is None:
        keyword_max_tokens = get_eqa_vl_int(parameters, "keyword_max_tokens", 64)
    if eqa_max_tokens is None:
        eqa_max_tokens = get_eqa_vl_int(parameters, "eqa_max_tokens", 1024)
    shared = get_shared_qwen35_vl_client(
        model_size=model_size,
        device=device,
        quantization=quantization,
        parameters=parameters,
    )

    def keyword_client(cmd: str | list) -> str:
        return shared(cmd, system_prompt=None, max_new_tokens=keyword_max_tokens)

    eqa = Qwen3VLEQAClient(
        EQA_PROMPT,
        model_size=model_size,
        device=device,
        shared_vl=shared,
        quantization=quantization,
        max_tokens=eqa_max_tokens,
        parameters=parameters,
    )
    return keyword_client, eqa


# Back-compat alias for code that imported the old name
Qwen25VLEQAClient = Qwen3VLEQAClient


def create_default_eqa_vl_client(
    system_prompt: str | AbstractPromptBuilder,
    *,
    model_size: str | None = None,
    device: str | None = None,
    parameters: Parameters | dict | None = None,
) -> Qwen3VLEQAClient:
    """Construct the default local EQA client (Qwen3.5 multimodal, shared weights)."""
    apply_eqa_vl_runtime_settings(parameters)
    if device is None:
        device = default_eqa_vl_device()
    q = get_eqa_vl_str(parameters, "quantization", "int4")
    shared = get_shared_qwen35_vl_client(
        model_size=model_size, device=device, quantization=q, parameters=parameters
    )
    return Qwen3VLEQAClient(
        system_prompt,
        model_size=model_size,
        device=device,
        shared_vl=shared,
        quantization=q,
        parameters=parameters,
    )
