# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Remote OpenAI-compatible VLM client (JPEG ``image_url`` over HTTP)."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from emet.llms.base import AbstractPromptBuilder, AbstractVLLMClient
from emet.llms.openai_client import resolve_openai_api_key, resolve_openai_base_url
from emet.llms.vl_image import downsample_rgb_hwc


def parse_openai_endpoint_spec(spec: str) -> tuple[str, str | None]:
    """Parse ``openai@http://host:port/v1[#model]`` or a bare ``http://…/v1`` URL.

    Returns ``(base_url_without_trailing_slash, optional_model_id)``.
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("empty OpenAI endpoint spec")
    model: str | None = None
    if raw.lower().startswith("openai@"):
        raw = raw[len("openai@") :].strip()
    if "#" in raw:
        raw, model_part = raw.rsplit("#", 1)
        model = model_part.strip() or None
    base = raw.strip().rstrip("/")
    if not base:
        raise ValueError(f"invalid OpenAI endpoint spec: {spec!r}")
    return base, model


class OpenaiVLLMClient(AbstractVLLMClient):
    """Stateless multimodal client posting JPEG data-URLs to ``/v1/chat/completions``."""

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder | None = None,
        *,
        model: str = "emet-vl",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        image_max_side: int = 512,
        image_max_pixels: int = 0,
        jpeg_quality: int = 85,
        device: str = "remote",
        **_kwargs: Any,
    ) -> None:
        super().__init__(prompt if prompt is not None else "", None)
        self.model = model
        self.base_url = resolve_openai_base_url(base_url)
        if not self.base_url:
            raise ValueError("OpenaiVLLMClient requires base_url or EMET_OPENAI_BASE_URL / EMET_VL_ENDPOINT")
        self.api_key = resolve_openai_api_key(api_key)
        self.max_tokens = int(max_tokens)
        self.image_max_side = int(image_max_side or 0)
        self.image_max_pixels = int(image_max_pixels or 0)
        self.jpeg_quality = int(jpeg_quality)
        self.device = device
        self._openai = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def canonical_model_key(self) -> str:
        return f"openai_vl:{self.base_url}:{self.model}"

    def _rgb_to_jpeg_data_url(self, rgb: np.ndarray | Image.Image) -> str:
        arr = downsample_rgb_hwc(
            rgb,
            max_side=self.image_max_side,
            max_pixels=self.image_max_pixels,
        )
        pil = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=max(1, min(95, self.jpeg_quality)), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _content_blocks(self, user_content: str | list[Any], image: Any | None) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        parts: list[Any]
        if isinstance(user_content, str):
            parts = [user_content] if user_content.strip() else []
        elif isinstance(user_content, list):
            parts = list(user_content)
        else:
            parts = [user_content]
        if image is not None:
            parts = list(parts) + [image]
        for part in parts:
            if part is None:
                continue
            if isinstance(part, str):
                if part.strip():
                    blocks.append({"type": "text", "text": part})
                continue
            if isinstance(part, dict) and part.get("type") in ("text", "image_url"):
                blocks.append(part)
                continue
            if isinstance(part, (Image.Image, np.ndarray)):
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._rgb_to_jpeg_data_url(part)},
                    }
                )
                continue
            raise TypeError(f"Unsupported multimodal part type: {type(part)!r}")
        if not blocks:
            raise ValueError("OpenaiVLLMClient: empty user content")
        return blocks

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
        sys_text = system_prompt if system_prompt is not None else (self.system_prompt or None)
        messages: list[dict[str, Any]] = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": self._content_blocks(user_content, image)})
        prefill = (assistant_prefill or "").strip()
        if prefill:
            # Remote OpenAI servers continue from a trailing assistant message when they
            # support it; servers that render it as a closed turn still emit the prefill
            # as the first field, which is what the local prefill shims also achieve.
            messages.append({"role": "assistant", "content": prefill})
        mt = int(max_new_tokens if max_new_tokens is not None else self.max_tokens)
        if verbose:
            n_img = sum(1 for b in messages[-1]["content"] if isinstance(b, dict) and b.get("type") == "image_url")
            print(f"OpenaiVLLMClient model={self.model} base_url={self.base_url} images={n_img} max_tokens={mt}")
        completion = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=mt,
        )
        text = completion.choices[0].message.content or ""
        if verbose:
            print(f"OpenaiVLLMClient output={text[:200]!r}")
        return text
