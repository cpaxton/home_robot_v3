# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Qwen3-VL client (Hugging Face ``Qwen3VLForConditionalGeneration``).

Requires ``transformers`` with Qwen3-VL support (project pins ``transformers>=4.55``).
"""

from __future__ import annotations

import logging
import os
import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from emet.agent.env_flags import env_agent_model_debug
from emet.llms.attn_impl import resolve_attn_implementation
from emet.llms.base import AbstractPromptBuilder, AbstractVLLMClient
from emet.llms.hf_local import merge_pretrained_kwargs, resolve_pretrained_source
from emet.llms.prefix_kv_cache import (
    PrefixKVCache,
    PrefixKVCacheEntry,
    clone_past_key_values,
    system_prompt_cache_key,
)
from emet.llms.repetition_stop import repetition_stopping_criteria
from emet.llms.vl_image import downsample_rgb_hwc
from emet.llms.vlm_device import (
    assert_cuda_placement,
    env_allow_cpu_vlm,
    summarize_model_devices,
)

logger = logging.getLogger(__name__)

# Soft timeout for prefix-KV generate before falling back to full generate (seconds).
_PREFIX_GENERATE_SOFT_TIMEOUT_S = float(os.environ.get("EMET_VL_PREFIX_GENERATE_TIMEOUT_S", "45") or 45)


class Qwen3VLClient(AbstractVLLMClient):
    """Qwen3-VL multimodal client (same message contract as Qwen25VLClient).

    Subclass hooks (see ``Qwen35Client``): ``_MODEL_CLS`` swaps the HF model class,
    ``_FAMILY_KEY`` namespaces ``canonical_model_key``, ``_TEMPLATE_KWARGS`` is forwarded
    to ``apply_chat_template`` (e.g. ``enable_thinking=False``), and
    ``_postprocess_output`` cleans raw decodes.
    """

    _MODEL_CLS: type = Qwen3VLForConditionalGeneration
    _FAMILY_KEY: str = "qwen3_vl"
    _TEMPLATE_KWARGS: dict[str, Any] = {}

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder | None = None,
        prompt_kwargs: dict[str, Any] | None = None,
        model_size: str = "4B",
        fine_tuning: str | None = "Instruct",
        max_tokens: int = 4096,
        num_beams: int = 1,
        device: str = "cuda",
        quantization: str | None = "int4",
        use_fast_attn: bool = False,
        hf_model_id: str | None = None,
        cache_system_prefix: bool = False,
        max_cached_prefixes: int = 1,
        image_max_side: int = 512,
        image_max_pixels: int = 0,
    ):
        super().__init__(prompt, prompt_kwargs)
        if device == "cpu":
            import warnings

            warnings.warn(
                "Qwen3VLClient on CPU: very slow; prefer GPU or a smaller VL model.",
                UserWarning,
                stacklevel=2,
            )
        if hf_model_id is None:
            assert model_size in ["2B", "4B", "8B", "32B"], f"Invalid Qwen3-VL model size: {model_size}"
        assert fine_tuning in [None, "Instruct"], f"Invalid fine-tuning: {fine_tuning}"

        self._device = device
        self.max_tokens = max_tokens
        self.num_beams = num_beams
        self.use_fast_attn = use_fast_attn
        # Always prefer Flash-Attn 2 when installed; otherwise PyTorch SDPA (not eager).
        # ``use_fast_attn=False`` only forces eager when EMET_ATTN_EAGER=1 (debug).
        if (
            not use_fast_attn
            and os.environ.get("EMET_ATTN_EAGER", "").strip().lower() in ("1", "true", "yes", "on")
        ):
            self._attn_implementation = "eager"
        else:
            self._attn_implementation = resolve_attn_implementation(prefer_flash=True, device=device)
        self.cache_system_prefix = bool(cache_system_prefix)
        self._prefix_cache = PrefixKVCache(max_entries=max_cached_prefixes)
        self.image_max_side = int(image_max_side or 0)
        self.image_max_pixels = int(image_max_pixels or 0)

        if hf_model_id is not None:
            model_name = hf_model_id
        elif fine_tuning is None:
            model_name = f"Qwen/Qwen3-VL-{model_size}"
        else:
            model_name = f"Qwen/Qwen3-VL-{model_size}-{fine_tuning}"

        self._resolved_hf_model_id = model_name
        self._quantization = quantization

        source, local_kw = resolve_pretrained_source(model_name)
        from_local = source != model_name or bool(local_kw.get("local_files_only"))
        where = "local HF cache" if from_local else "Hugging Face Hub"
        family = "Qwen3.5" if self._FAMILY_KEY == "qwen3_5" else "Qwen3-VL"
        print(
            f"Loading {family} from {where}: {model_name} "
            f"(quant={quantization!r}, device={device!r}, attn={self._attn_implementation})"
        )

        model_kwargs: dict[str, Any] = {"use_safetensors": True}

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            if quantization in ["int8", "int4"]:
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
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")
        else:
            model_kwargs["dtype"] = "auto"

        t_proc0 = timeit.default_timer()
        self.processor = AutoProcessor.from_pretrained(source, **local_kw)
        _vl_tok = getattr(self.processor, "tokenizer", None)
        if _vl_tok is not None and getattr(_vl_tok, "pad_token_id", None) is None:
            _vl_tok.pad_token = _vl_tok.eos_token
        print(f"  processor: {timeit.default_timer() - t_proc0:.1f}s", flush=True)

        attn_implementation = self._attn_implementation
        pretrained_kw: dict[str, Any] = merge_pretrained_kwargs(
            {
                "attn_implementation": attn_implementation,
                **model_kwargs,
            },
            local_kw,
        )
        # bitsandbytes: omit dtype/torch_dtype (match text Qwen client).
        if quantization_config is not None:
            pretrained_kw.pop("dtype", None)
            pretrained_kw.pop("torch_dtype", None)
        if device == "cuda":
            pretrained_kw["device_map"] = {"": 0}
        elif device == "mps":
            pretrained_kw["device_map"] = "mps"

        t_w0 = timeit.default_timer()
        try:
            self.model = self._MODEL_CLS.from_pretrained(source, **pretrained_kw)
        except (ValueError, RuntimeError) as e:
            err = str(e).lower()
            recoverable = device == "cuda" and (
                "dispatched" in err or "disk" in err or "out of memory" in err or "cuda out of memory" in err
            )
            if not recoverable:
                raise
            if not env_allow_cpu_vlm():
                raise RuntimeError(
                    f"Qwen3-VL GPU load failed ({e}). Refusing silent CPU fallback (multi-minute "
                    "inference looks like a hang). Free VRAM, use a smaller --llm, or set "
                    "EMET_ALLOW_CPU_VLM=1 to allow the slow CPU bf16 path."
                ) from e
            logger.warning(
                "Qwen3-VL GPU load failed (%s); EMET_ALLOW_CPU_VLM=1 — retrying on CPU in bfloat16 "
                "without bitsandbytes (very slow).",
                e,
            )
            import warnings

            warnings.warn(
                "Qwen3-VL fell back to CPU (bf16, no int4). Inference will be very slow.",
                UserWarning,
                stacklevel=2,
            )
            self._device = "cpu"
            self._quantization = None
            fallback_kw: dict[str, Any] = merge_pretrained_kwargs(
                {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn_implementation,
                    "use_safetensors": True,
                },
                local_kw,
            )
            self.model = self._MODEL_CLS.from_pretrained(source, **fallback_kw)
            self.model = self.model.to("cpu")
        else:
            if device == "cpu":
                self.model = self.model.to("cpu")

        quant_label = self._quantization or "none"
        print(
            f"  weights+{quant_label}: {timeit.default_timer() - t_w0:.1f}s "
            f"(devices: {summarize_model_devices(self.model)})",
            flush=True,
        )
        assert_cuda_placement(
            self.model,
            requested_device=self._device,
            model_label=f"Qwen3-VL ({model_name})",
        )

        try:
            from emet.utils.vram_debug import print_vram_snapshot

            print_vram_snapshot(
                "qwen3_vl_client_init",
                extra=f"{model_name!r} quant={self._quantization!r} device={self._device!r}",
            )
        except Exception:
            pass

    @property
    def canonical_model_key(self) -> str:
        q = self._quantization or "none"
        return f"{self._FAMILY_KEY}:{self._resolved_hf_model_id}:{self._device}:{q}"

    def clear_prefix_cache(self) -> None:
        """Drop cached system-prefix KV state (tests / prompt changes)."""
        self._prefix_cache.clear()

    def _downsample_image(self, image: Any) -> np.ndarray:
        return downsample_rgb_hwc(
            image,
            max_side=self.image_max_side,
            max_pixels=self.image_max_pixels,
        )

    def _has_system_in_history(self) -> bool:
        return any(isinstance(m, dict) and m.get("role") == "system" for m in self.conversation_history)

    def _postprocess_output(self, text: str) -> str:
        """Hook for subclasses to clean raw decoded output (default: passthrough)."""
        return text

    def _process_input(self, command: Any) -> Any:
        if isinstance(command, str):
            return command
        user_commands: list[Any] = []
        for c in command:
            if isinstance(c, str):
                user_commands.append({"type": "text", "text": c})
            elif isinstance(c, Image.Image) or isinstance(c, np.ndarray):
                image = self._downsample_image(c)
                user_commands.append({"type": "image", "image": Image.fromarray(image, mode="RGB")})
            else:
                raise NotImplementedError("Only text and image content supported for VL.")
        return user_commands

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _processor_inputs(self, messages: list[Any], *, add_generation_prompt: bool) -> Any:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self._TEMPLATE_KWARGS,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model_device())

    def _encode_system_prefix(self, sys_txt: str) -> tuple[torch.Tensor, Any]:
        prefix_messages = [{"role": "system", "content": sys_txt}]
        prefix_inputs = self._processor_inputs(prefix_messages, add_generation_prompt=False)
        prefix_ids = prefix_inputs["input_ids"]
        with torch.inference_mode():
            forward_kw = {k: v for k, v in prefix_inputs.items() if k in ("input_ids", "attention_mask")}
            outputs = self.model(**forward_kw, use_cache=True)
        return prefix_ids, outputs.past_key_values

    def _ensure_prefix_cached(self, sys_txt: str) -> PrefixKVCacheEntry | None:
        if not self.cache_system_prefix or not sys_txt:
            return None
        key = system_prompt_cache_key(sys_txt)
        entry = self._prefix_cache.get(key)
        if entry is not None:
            return entry
        try:
            prefix_ids, past = self._encode_system_prefix(sys_txt)
            entry = PrefixKVCacheEntry(
                past_key_values=clone_past_key_values(past),
                prefix_token_len=int(prefix_ids.shape[1]),
                prefix_token_ids=prefix_ids.detach().clone(),
            )
            self._prefix_cache.put(key, entry)
            if env_agent_model_debug():
                logger.info(
                    "VL prefix cache: stored system prefix (%d tokens) key=%s…",
                    entry.prefix_token_len,
                    key[:12],
                )
            return entry
        except Exception as e:
            logger.warning("VL prefix cache: prefill failed (%s); falling back to full generate", e)
            return None

    def warm_system_prefix_cache(self, system_prompt: str | None = None) -> int | None:
        """Prefill and store the system-prompt KV cache (call once after load).

        Moves the expensive ~system-token prefill off the first user turn.
        Returns cached prefix token length, or None when caching is disabled/failed.
        """
        sys_txt = system_prompt if system_prompt is not None else self.system_prompt
        if not sys_txt:
            return None
        entry = self._ensure_prefix_cached(str(sys_txt))
        return None if entry is None else int(entry.prefix_token_len)

    def _prefix_ids_align(self, full_ids: torch.Tensor, entry: PrefixKVCacheEntry) -> bool:
        plen = entry.prefix_token_len
        if plen <= 0 or full_ids.shape[1] < plen:
            return False
        cached = entry.prefix_token_ids.to(full_ids.device)
        return torch.equal(full_ids[:, :plen], cached)

    def _messages_have_vision(self, messages: list[Any]) -> bool:
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("image", "video"):
                        return True
                    if isinstance(part, (Image.Image, np.ndarray)):
                        return True
            elif isinstance(content, (Image.Image, np.ndarray)):
                return True
        return False

    def _generate_ids(
        self,
        inputs: Any,
        *,
        max_new_tokens: int,
        past_key_values: Any | None = None,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        input_len = int(inputs.input_ids.shape[1])
        model_inputs = dict(inputs.items())

        if past_key_values is not None and prefix_len > 0 and input_len > prefix_len:
            # Text-only prefix reuse. Pass the FULL prompt ids + mask; transformers 5.x
            # crops with ``input_ids[:, past_length:]`` when mask length matches ids.
            # Pre-slicing to a suffix made that crop empty → reshape [1, 0, -1, head_dim].
            for vision_key in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
            ):
                model_inputs.pop(vision_key, None)
            model_inputs["past_key_values"] = clone_past_key_values(past_key_values)
            ids = model_inputs["input_ids"]
            if (
                "attention_mask" not in model_inputs
                or model_inputs["attention_mask"] is None
                or int(model_inputs["attention_mask"].shape[1]) != input_len
            ):
                model_inputs["attention_mask"] = torch.ones(
                    (ids.shape[0], input_len),
                    dtype=torch.long,
                    device=ids.device,
                )
            model_inputs.pop("position_ids", None)
            prompt_len_for_stop = input_len
        else:
            past_key_values = None
            prefix_len = 0
            prompt_len_for_stop = input_len

        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": self.num_beams,
            "stopping_criteria": repetition_stopping_criteria(prompt_len_for_stop),
        }
        if pad_id is not None:
            gen_kw["pad_token_id"] = pad_id
        return self.model.generate(**model_inputs, **gen_kw)

    def _generate_ids_with_prefix_guard(
        self,
        inputs: Any,
        *,
        max_new_tokens: int,
        past_key_values: Any,
        prefix_len: int,
    ) -> tuple[torch.Tensor, bool]:
        """Try prefix-KV generate; on failure disable cache and fall back to full generate.

        Returns ``(generated_ids, used_prefix_cache)``.
        """
        t0 = timeit.default_timer()
        try:
            generated_ids = self._generate_ids(
                inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
                prefix_len=prefix_len,
            )
            elapsed = timeit.default_timer() - t0
            if elapsed > _PREFIX_GENERATE_SOFT_TIMEOUT_S:
                logger.warning(
                    "VL prefix cache: generate with cache took %.1fs (>%ss). Disabling prefix cache "
                    "for subsequent turns (use --no-cache-vl-prefix to skip at startup).",
                    elapsed,
                    _PREFIX_GENERATE_SOFT_TIMEOUT_S,
                )
                self.cache_system_prefix = False
                self._prefix_cache.clear()
            return generated_ids, True
        except Exception as e:
            logger.warning("VL prefix cache: generate with cache failed (%s); full generate", e)
            self.cache_system_prefix = False
            self._prefix_cache.clear()
            return self._generate_ids(inputs, max_new_tokens=max_new_tokens), False

    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image: Any | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        def _progress(msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        if reset_context:
            self.reset()
        sys_txt = system_prompt if system_prompt is not None else self.system_prompt
        if sys_txt and (reset_context or not self._has_system_in_history()):
            self.add_history({"role": "system", "content": sys_txt})

        if image is not None:
            pil = Image.fromarray(self._downsample_image(image), mode="RGB")
            content: Any = [{"type": "image", "image": pil}, {"type": "text", "text": user_content}]
        else:
            content = self._process_input(user_content)

        self.add_history({"role": "user", "content": content})
        messages = self.get_history()

        if verbose:
            print("Qwen3-VL messages (truncated):", str(messages)[:800])

        t0 = timeit.default_timer()
        _progress("building prompt")
        t_prep0 = timeit.default_timer()
        inputs = self._processor_inputs(messages, add_generation_prompt=True)
        prep_s = timeit.default_timer() - t_prep0
        ntok = max_new_tokens if max_new_tokens is not None else self.max_tokens
        input_len = int(inputs.input_ids.shape[1])
        model_dev = str(self._model_device())

        # Prefix KV reuse is text-only. Multimodal turns (pixel_values) fall back to full generate.
        has_vision = image is not None or self._messages_have_vision(messages)
        if not has_vision:
            pv = None
            if hasattr(inputs, "get"):
                pv = inputs.get("pixel_values")
            elif "pixel_values" in getattr(inputs, "keys", lambda: ())():
                pv = inputs["pixel_values"]
            if pv is not None and torch.is_tensor(pv):
                has_vision = True

        if env_agent_model_debug() or verbose:
            print(
                f"[vl] generate prep={prep_s:.2f}s tokens={input_len} device={model_dev} "
                f"vision={has_vision} max_new={ntok} prefix_cache={self.cache_system_prefix}",
                flush=True,
            )

        if model_dev.startswith("cpu") and self._device == "cuda" and not env_allow_cpu_vlm():
            raise RuntimeError(
                f"Qwen3-VL parameters are on {model_dev} but client device is cuda. "
                "Refusing CPU generate (set EMET_ALLOW_CPU_VLM=1 to override)."
            )

        prefix_entry: PrefixKVCacheEntry | None = None
        used_cache = False
        trim_from_len = int(inputs.input_ids.shape[1])
        _progress(
            f"generate prompt={input_len} max_new={ntok} on {model_dev}"
            + (" prefix_cache" if (self.cache_system_prefix and sys_txt and not has_vision) else "")
        )
        t_gen0 = timeit.default_timer()
        if self.cache_system_prefix and sys_txt and not has_vision:
            prefix_entry = self._ensure_prefix_cached(sys_txt)
            if prefix_entry is not None and self._prefix_ids_align(inputs.input_ids, prefix_entry):
                _progress(
                    f"generate prompt={input_len} (cached_prefix={prefix_entry.prefix_token_len}) "
                    f"max_new={ntok} on {model_dev}"
                )
                generated_ids, used_cache = self._generate_ids_with_prefix_guard(
                    inputs,
                    max_new_tokens=ntok,
                    past_key_values=prefix_entry.past_key_values,
                    prefix_len=prefix_entry.prefix_token_len,
                )
                trim_from_len = int(inputs.input_ids.shape[1])
                if used_cache and env_agent_model_debug():
                    logger.info(
                        "VL prefix cache: hit (%d-token system prefix)",
                        prefix_entry.prefix_token_len,
                    )
            else:
                if prefix_entry is not None and env_agent_model_debug():
                    logger.info("VL prefix cache: token mismatch; full generate")
                generated_ids = self._generate_ids(inputs, max_new_tokens=ntok)
        else:
            if has_vision and self.cache_system_prefix and env_agent_model_debug():
                logger.info("VL prefix cache: skipped (vision inputs present)")
            generated_ids = self._generate_ids(inputs, max_new_tokens=ntok)
        gen_s = timeit.default_timer() - t_gen0

        if verbose and used_cache:
            print("VL prefix cache: used cached system KV")

        _progress("decode")
        generated_ids_trimmed = [out_ids[trim_from_len:] for out_ids in generated_ids]
        output_text = self._postprocess_output(
            self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        )

        self.add_history({"role": "assistant", "content": output_text})
        t1 = timeit.default_timer()

        if verbose or env_agent_model_debug():
            print(
                f"[vl] generate done prep={prep_s:.2f}s generate={gen_s:.2f}s total={t1 - t0:.2f}s "
                f"out_chars={len(output_text)} prefix_hit={used_cache}",
                flush=True,
            )
        if verbose:
            print(f"Assistant response: {output_text[:500]}...")
            print(f"Time taken: {t1 - t0:.2f}s")

        return output_text
