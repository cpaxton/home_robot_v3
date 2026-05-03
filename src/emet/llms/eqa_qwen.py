# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Default EQA backend: **Qwen3.5** multimodal on Hugging Face (``Qwen/Qwen3.5-*``), loaded once per
# process and shared between EQA and short text helpers (keyword extraction, etc.).
#
# Model size and Hub logging follow ``dynav_config.yaml`` → ``eqa_vl:`` (see that file). Env overrides:
# ``EMET_EQA_VL_MODEL_SIZE``, ``EMET_VERBOSE_HF``.

from __future__ import annotations

import threading
from collections.abc import Callable

import torch

from emet.core.parameters import Parameters
from emet.llms.base import AbstractPromptBuilder
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
    second checkpoint after SigLIP / first VL pass). Size comes from ``resolve_eqa_vl_model_size``.

    Initialization is serialized (lock): concurrent first callers must not each run VRAM tiering
    after the other has allocated GPU memory, which previously could load a second checkpoint (e.g.
    4B then 2B).
    """
    global _shared_qwen35_vl
    if _shared_qwen35_vl is not None:
        return _shared_qwen35_vl
    with _shared_qwen35_vl_lock:
        if _shared_qwen35_vl is not None:
            return _shared_qwen35_vl
        if device is None:
            device = default_eqa_vl_device()
        if quantization is None:
            quantization = get_eqa_vl_str(parameters, "quantization", "int4")
        if model_size is None:
            model_size = resolve_eqa_vl_model_size(parameters, device=device)
        else:
            model_size = sync_resolved_eqa_vl_model_size_from_explicit(model_size)
        _shared_qwen35_vl = Qwen35VLClient(
            prompt=None,
            model_size=model_size,
            max_tokens=4096,
            num_beams=1,
            device=device,
            quantization=quantization,
        )
        print_vram_snapshot(
            "eqa_qwen_shared_qwen35_vl_first_load",
            extra=f"Qwen3.5-{model_size} multimodal (eqa_vl / graph helpers)",
        )
        return _shared_qwen35_vl


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
