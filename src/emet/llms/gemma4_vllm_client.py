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

import gc
import timeit
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from emet.llms.base import AbstractPromptBuilder, AbstractVLLMClient
from emet.llms.repetition_stop import repetition_stopping_criteria
from emet.utils.logger import Logger

logger = Logger(__name__)


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


def _trim_cuda_cache() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


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
        quantization: str | None = "int4",
    ):
        super().__init__(prompt, prompt_kwargs)
        if device not in ("cuda", "mps", "cpu"):
            raise ValueError(f"Invalid device: {device}")
        self._device = device
        self.max_tokens = max_tokens
        self._resolved_hf_model_id = hf_model_id
        self._quantization = quantization
        dtype = (
            torch_dtype if isinstance(torch_dtype, torch.dtype) else getattr(torch, str(torch_dtype), torch.bfloat16)
        )

        model_kwargs: dict[str, Any] = {}
        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            if quantization in ("int8", "int4"):
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                except ImportError as e:
                    raise ImportError(
                        "bitsandbytes required for int4/int8 quantization: pip install bitsandbytes"
                    ) from e
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "int4"),
                    load_in_8bit=(quantization == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = quantization_config
            elif quantization in ("none", "bf16", "bfloat16"):
                model_kwargs["torch_dtype"] = dtype
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")
        else:
            model_kwargs["torch_dtype"] = dtype

        print(f"Loading Gemma multimodal model: {hf_model_id} (quant={quantization or 'none'}, device={device})")
        from emet.llms.hf_local import merge_pretrained_kwargs, resolve_pretrained_source
        from emet.llms.vlm_device import assert_cuda_placement, env_allow_cpu_vlm, summarize_model_devices

        source, local_kw = resolve_pretrained_source(hf_model_id)
        pretrained_kw: dict[str, Any] = merge_pretrained_kwargs(dict(model_kwargs), local_kw)
        if device == "cuda":
            if quantization_config is not None:
                pretrained_kw["device_map"] = {"": 0}
            else:
                pretrained_kw["device_map"] = "auto"
        elif device == "mps":
            pretrained_kw["device_map"] = "mps"

        self.processor = AutoProcessor.from_pretrained(source, **local_kw)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(source, **pretrained_kw)
        except (ValueError, RuntimeError) as e:
            err = str(e).lower()
            recoverable = (
                device == "cuda"
                and quantization_config is not None
                and ("dispatched" in err or "disk" in err or "out of memory" in err or "cuda out of memory" in err)
            )
            if not recoverable:
                raise
            if not env_allow_cpu_vlm():
                raise RuntimeError(
                    f"Gemma VLM GPU load failed ({e}). Refusing silent CPU fallback. "
                    "Free VRAM or set EMET_ALLOW_CPU_VLM=1."
                ) from e
            logger.warning(
                "Gemma VLM int4 GPU load failed (%s); EMET_ALLOW_CPU_VLM=1 — retrying bf16 on CPU (slow).",
                e,
            )
            import warnings

            warnings.warn(f"Gemma VLM falling back to CPU bf16 after int4 GPU failure: {e}", UserWarning, stacklevel=2)
            self._device = "cpu"
            self._quantization = None
            self.model = AutoModelForImageTextToText.from_pretrained(source, torch_dtype=dtype, **local_kw)
            self.model = self.model.to("cpu")
        else:
            if device == "cpu" and quantization_config is None:
                self.model = self.model.to("cpu")
            elif device == "mps" and quantization_config is None:
                self.model = self.model.to("mps")

        print(f"  Gemma VLM devices: {summarize_model_devices(self.model)}", flush=True)
        assert_cuda_placement(
            self.model,
            requested_device=self._device,
            model_label=f"Gemma VLM ({hf_model_id})",
        )

        try:
            from emet.utils.vram_debug import print_vram_snapshot

            print_vram_snapshot(
                "gemma4_vllm_client_init",
                extra=f"{hf_model_id!r} quant={self._quantization!r} device={self._device!r}",
            )
        except Exception:
            pass

    @property
    def canonical_model_key(self) -> str:
        q = self._quantization or "none"
        return f"gemma4:{self._resolved_hf_model_id}:{self._device}:{q}"

    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image: Any | None = None,
        assistant_prefill: str | None = None,
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
        try:
            with torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=ntok,
                    do_sample=False,
                    stopping_criteria=repetition_stopping_criteria(int(input_len)),
                )
            trimmed = gen[0][input_len:]
            text_out = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        finally:
            del inputs
            if "gen" in locals():
                del gen
            _trim_cuda_cache()
        t1 = timeit.default_timer()
        if verbose:
            print(f"Gemma VLM response (truncated): {text_out[:500]}...")
            print(f"Time taken: {t1 - t0:.2f}s")
        self.add_history({"role": "assistant", "content": text_out})
        return text_out
