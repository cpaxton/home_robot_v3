# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import timeit
from typing import Any

import numpy as np
import torch
from PIL import Image
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from emet.llms.base import AbstractLLMClient, AbstractPromptBuilder

# Presets for get_llm_client (Qwen2.5-VL / stand-in for Qwen3.5-VL-9B until a dedicated HF id is wired).
QWEN_VL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen25-VL-3B": {"model_size": "3B", "hf_model_id": None},
    "qwen25-VL-7B": {"model_size": "7B", "hf_model_id": None},
    # Maps to 2.5-VL-7B-Instruct (closest widely available); swap hf_model_id when Qwen3.5-VL-9B is published.
    "qwen35-vl-9B": {"model_size": "7B", "hf_model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
}

qwen_typing_options = ["Math", "Coder", "Deepseek", None]
qwen_quantization_options = {
    None: [None, "Int4", "Int8", "Int", "Instruct", "Instruct-Int4", "Instruct-Int8", "Instruct-Int"],
    "Coder": [None, "Int4", "Int8", "Int", "Instruct", "Instruct-Int4", "Instruct-Int8", "Instruct-Int"],
    "Math": [None, "Int4", "Int8", "Int", "Instruct", "Instruct-Int4", "Instruct-Int8", "Instruct-Int"],
    "Deepseek": [None, "Int4", "Int8", "Int"],
}
qwen_sizes = {
    None: ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
    "Coder": ["0.5B", "1.5B", "3B", "7B", "14B", "32B"],
    "Math": ["1.5B", "7B", "72B"],
    "Deepseek": ["1.5B", "7B", "14B", "72B"],
}


def get_qwen_variants():
    qwen_variants = []
    for qwen_typing_option in qwen_typing_options:
        for qwen_quantization_option in qwen_quantization_options[qwen_typing_option]:
            for qwen_size in qwen_sizes[qwen_typing_option]:
                qwen_type = "qwen25"
                if qwen_typing_option is not None:
                    qwen_type += "-" + qwen_typing_option
                qwen_type += "-" + qwen_size
                if qwen_quantization_option is not None:
                    qwen_type += "-" + qwen_quantization_option
                qwen_variants.append(qwen_type)
    return qwen_variants


def get_qwen35_variants():
    """Return Qwen 3.5 model variant names (e.g. qwen35-4B, qwen35-9B)."""
    return [f"qwen35-{s}" for s in qwen35_sizes]


# Qwen 3.5 sizes on HuggingFace (Qwen/Qwen3.5-*)
qwen35_sizes = ["0.8B", "2B", "4B", "9B"]


class Qwen25Client(AbstractLLMClient):
    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        prompt_kwargs: dict[str, Any] | None = None,
        model_size: str = "3B",
        fine_tuning: str | None = "Instruct",
        model_type: str | None = None,
        max_tokens: int = 4096,
        device: str = "cuda",
        quantization: str | None = "int4",
        version: str | None = None,
    ):
        super().__init__(prompt, prompt_kwargs)
        assert device in ["cuda", "mps", "cpu"], f"Invalid device: {device}"
        if device == "cpu":
            import warnings

            warnings.warn(
                "Qwen client on CPU: inference will be slow; use a small model (e.g. 1.5B) and Int4 for local testing.",
                UserWarning,
                stacklevel=2,
            )
        if version == "3.5":
            assert model_size in qwen35_sizes, f"Invalid Qwen 3.5 size: {model_size}, use one of {qwen35_sizes}"
        else:
            assert model_type in qwen_typing_options, f"Invalid model type: {model_type}"
            assert model_size in qwen_sizes[model_type], f"Invalid model size: {model_size}"
            assert fine_tuning in [None, "Instruct"], f"Invalid fine-tuning: {fine_tuning}"

        self._version = version
        self.max_tokens = max_tokens

        if version == "3.5":
            model_name = f"Qwen/Qwen3.5-{model_size}"
        elif model_type == "Deepseek":
            model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{model_size}"
        elif model_type is None:
            if fine_tuning is None:
                model_name = f"Qwen/Qwen2.5-{model_size}"
            else:
                model_name = f"Qwen/Qwen2.5-{model_size}-{fine_tuning}"
        else:
            if fine_tuning is None:
                model_name = f"Qwen/Qwen2.5-{model_type}-{model_size}"
            else:
                model_name = f"Qwen/Qwen2.5-{model_type}-{model_size}-{fine_tuning}"

        print(f"Loading model: {model_name}")
        model_kwargs = {"dtype": "auto"}

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            # "int" is alias for int4 (e.g. --llm qwen25-Coder-3B-Instruct-Int)
            if quantization == "int":
                quantization = "int4"
            # Note: there were supposed to be other options but this is the only one that worked this way
            if quantization == "awq":
                model_kwargs["dtype"] = torch.float16
                model_name += "-AWQ"
            elif quantization in ["int8", "int4"]:
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # When using quantization, omit torch_dtype and use device_map so bitsandbytes loads correctly
        load_kwargs = dict(model_kwargs)
        if quantization_config is not None:
            load_kwargs.pop("dtype", None)
            load_kwargs.pop("torch_dtype", None)
            if device != "cpu":
                load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        # Pipeline device: 0 or "cuda" for GPU, -1 for CPU
        pipe_device = -1 if device == "cpu" else (0 if device == "cuda" else device)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipe_device,
            model_kwargs=model_kwargs,
        )

    def __call__(self, command: str, verbose: bool = False):
        if self.is_first_message():
            system_message = {"role": "system", "content": self.system_prompt}
            self.add_history(system_message)

        # Prepare the messages including the conversation history
        new_message = {"role": "user", "content": command}

        self.add_history(new_message)
        messages = self.get_history()

        template_kwargs: dict = {}
        if self._version == "3.5":
            template_kwargs["enable_thinking"] = False
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )

        t0 = timeit.default_timer()
        gen_config = GenerationConfig(
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        outputs = self.pipe(text, generation_config=gen_config)
        t1 = timeit.default_timer()

        generated = outputs[0]["generated_text"]
        # Extract the last assistant turn. Qwen uses <|im_start|>assistant\n as delimiter.
        if "<|im_start|>assistant" in generated:
            assistant_response = generated.rsplit("<|im_start|>assistant", 1)[-1]
            # Strip the role marker and any trailing end tokens
            assistant_response = assistant_response.lstrip("\n").rstrip()
            if assistant_response.endswith("<|im_end|>"):
                assistant_response = assistant_response[: -len("<|im_end|>")].rstrip()
        else:
            assistant_response = generated.split("assistant")[-1].strip()

        self.add_history({"role": "assistant", "content": assistant_response})
        if verbose:
            print(f"Assistant response: {assistant_response}")
            print(f"Time taken: {t1 - t0:.2f}s")
        return assistant_response

    def sample(self, command: str, n_samples: int = 5, verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")
        plan = []
        for _i in range(n_samples):
            self.reset()
            plan.append(self.__call__(command, verbose))
        if verbose:
            print(f"plan={plan}")
        return plan


from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLClient(AbstractLLMClient):
    """Qwen2.5-VL multimodal chat for agent tool JSON (same text contract as Qwen25Client).

    Supports optional ``image=`` (RGB ndarray) on a user turn for camera-conditioned replies.
    """

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder | None = None,
        prompt_kwargs: dict[str, Any] | None = None,
        model_size: str = "3B",
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
                "Qwen25VLClient on CPU: very slow; prefer GPU or a smaller VL model.",
                UserWarning,
                stacklevel=2,
            )
        assert model_size in ["3B", "7B", "72B"], f"Invalid Qwen VL model size: {model_size}"
        assert fine_tuning in [None, "Instruct"], f"Invalid fine-tuning: {fine_tuning}"

        self._device = device
        self.max_tokens = max_tokens
        self.num_beams = num_beams
        self.use_fast_attn = use_fast_attn

        if hf_model_id is not None:
            model_name = hf_model_id
        elif fine_tuning is None:
            model_name = f"Qwen/Qwen2.5-VL-{model_size}"
        else:
            model_name = f"Qwen/Qwen2.5-VL-{model_size}-{fine_tuning}"

        if model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
            print(
                "Note: qwen35-vl-9B currently loads Qwen/Qwen2.5-VL-7B-Instruct; "
                "set hf_model_id when a Qwen3.5-VL-9B checkpoint is available.",
            )
        print(f"Loading VL model: {model_name}")
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
        if device == "cuda":
            pretrained_kw["device_map"] = "auto"
        elif device == "mps":
            pretrained_kw["device_map"] = "mps"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **pretrained_kw)
        if device == "cpu":
            self.model = self.model.to("cpu")

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

    def __call__(
        self,
        command: str | list[Any],
        image: np.ndarray | None = None,
        verbose: bool = False,
        tools: list[Any] | None = None,
    ) -> str:
        if tools is not None:
            pass  # Agent uses JSON-in-text, not native tool APIs.
        if self.is_first_message():
            self.add_history({"role": "system", "content": self.system_prompt})

        if image is not None:
            pil = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")
            user_content: Any = [{"type": "image", "image": pil}, {"type": "text", "text": command}]
        else:
            user_content = self._process_input(command)

        self.add_history({"role": "user", "content": user_content})
        messages = self.get_history()

        if verbose:
            print("VL messages (truncated):", str(messages)[:800])

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
        gen_kw: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
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


if __name__ == "__main__":
    # from emet.llms.prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
    from emet.llms.prompts.pickup_prompt import PickupPromptBuilder

    prompt = PickupPromptBuilder()
    client = Qwen25Client(prompt, model_size="1.5B", fine_tuning="Instruct")
    for _ in range(50):
        msg = input("Enter a message (empty to quit): ")
        if len(msg) == 0:
            break
        response = client(msg, verbose=True)
        print()
        print("-" * 80)
        print(colored("You said:", "green"), msg)
        print(colored("Response", "blue"), response)
