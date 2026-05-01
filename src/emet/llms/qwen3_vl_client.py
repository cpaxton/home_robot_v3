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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Qwen3-VL client (Hugging Face ``Qwen3VLForConditionalGeneration``).

Requires ``transformers`` with Qwen3-VL support (project pins ``transformers>=4.55``).
"""

from __future__ import annotations

import logging
import timeit
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from emet.llms.base import AbstractPromptBuilder, AbstractVLLMClient

logger = logging.getLogger(__name__)


class Qwen3VLClient(AbstractVLLMClient):
    """Qwen3-VL multimodal client (same message contract as Qwen25VLClient)."""

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

        if hf_model_id is not None:
            model_name = hf_model_id
        elif fine_tuning is None:
            model_name = f"Qwen/Qwen3-VL-{model_size}"
        else:
            model_name = f"Qwen/Qwen3-VL-{model_size}-{fine_tuning}"

        self._resolved_hf_model_id = model_name
        self._quantization = quantization

        print(f"Loading Qwen3-VL model: {model_name}")
        model_kwargs: dict[str, Any] = {"dtype": "auto"}

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

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.processor = AutoProcessor.from_pretrained(model_name)
        _vl_tok = getattr(self.processor, "tokenizer", None)
        if _vl_tok is not None and getattr(_vl_tok, "pad_token_id", None) is None:
            _vl_tok.pad_token = _vl_tok.eos_token
        attn_implementation = "flash_attention_2" if self.use_fast_attn else None
        pretrained_kw: dict[str, Any] = {
            "attn_implementation": attn_implementation,
            **model_kwargs,
        }
        # bitsandbytes 4/8-bit: all quantized modules must stay on one accelerator; ``device_map="auto"``
        # can place shards on CPU/disk when VRAM is tight (e.g. after a large agent LLM), which raises
        # ValueError from the HF quantizer. Prefer a single GPU index first, then CPU bf16 fallback.
        if device == "cuda":
            pretrained_kw["device_map"] = {"": 0}
        elif device == "mps":
            pretrained_kw["device_map"] = "mps"

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **pretrained_kw)
        except (ValueError, RuntimeError) as e:
            err = str(e).lower()
            recoverable = device == "cuda" and (
                "dispatched" in err or "disk" in err or "out of memory" in err or "cuda out of memory" in err
            )
            if not recoverable:
                raise
            logger.warning(
                "Qwen3-VL GPU load failed (%s); retrying on CPU in bfloat16 without bitsandbytes quantization "
                "(slow but avoids split CPU/GPU layouts). Consider a smaller agent --llm, --device cpu for the "
                "agent, --no-share-memory-vllm with eqa on CPU in config, or a VL agent model to share weights.",
                e,
            )
            import warnings

            warnings.warn(
                "Qwen3-VL fell back to CPU (bf16, no int4). EQA/captions will be slow.",
                UserWarning,
                stacklevel=2,
            )
            self._device = "cpu"
            self._quantization = None
            fallback_kw: dict[str, Any] = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": attn_implementation,
            }
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **fallback_kw)
            self.model = self.model.to("cpu")
        else:
            if device == "cpu":
                self.model = self.model.to("cpu")

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
        return f"qwen3_vl:{self._resolved_hf_model_id}:{self._device}:{q}"

    def _process_input(self, command: Any) -> Any:
        if isinstance(command, str):
            return command
        user_commands: list[Any] = []
        for c in command:
            if isinstance(c, str):
                user_commands.append({"type": "text", "text": c})
            elif isinstance(c, Image.Image) or isinstance(c, np.ndarray):
                image = Image.fromarray(c.astype(np.uint8), mode="RGB") if isinstance(c, np.ndarray) else c
                user_commands.append({"type": "image", "image": image})
            else:
                raise NotImplementedError("Only text and image content supported for VL.")
        return user_commands

    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image: Any | None = None,
    ) -> str:
        if reset_context:
            self.reset()
        sys_txt = system_prompt if system_prompt is not None else self.system_prompt
        if sys_txt:
            self.add_history({"role": "system", "content": sys_txt})

        if image is not None:
            pil = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")
            content: Any = [{"type": "image", "image": pil}, {"type": "text", "text": user_content}]
        else:
            content = self._process_input(user_content)

        self.add_history({"role": "user", "content": content})
        messages = self.get_history()

        if verbose:
            print("Qwen3-VL messages (truncated):", str(messages)[:800])

        t0 = timeit.default_timer()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        dev = next(self.model.parameters()).device
        inputs = inputs.to(dev)

        pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        ntok = max_new_tokens if max_new_tokens is not None else self.max_tokens
        gen_kw: dict[str, Any] = {
            "max_new_tokens": ntok,
            "num_beams": self.num_beams,
        }
        if pad_id is not None:
            gen_kw["pad_token_id"] = pad_id
        generated_ids = self.model.generate(**inputs, **gen_kw)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        self.add_history({"role": "assistant", "content": output_text})
        t1 = timeit.default_timer()

        if verbose:
            print(f"Assistant response: {output_text[:500]}...")
            print(f"Time taken: {t1 - t0:.2f}s")

        return output_text
