# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Health / smoke helpers for remote OpenAI text + VL serve (caliban / LAN)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_TEXT_BASE = "http://caliban:8000/v1"
DEFAULT_VL_BASE = "http://caliban:8001/v1"


def normalize_openai_base(url: str) -> str:
    """Return ``http://host:port/v1`` without a trailing slash."""
    raw = (url or "").strip().rstrip("/")
    if raw.lower().startswith("openai@"):
        raw = raw[len("openai@") :].strip().rstrip("/")
    if "#" in raw:
        raw = raw.split("#", 1)[0].rstrip("/")
    if not raw.endswith("/v1"):
        if raw.endswith("/health"):
            raw = raw[: -len("/health")].rstrip("/")
        if not raw.endswith("/v1"):
            raw = f"{raw}/v1"
    return raw


def health_url_from_base(base_url: str) -> str:
    base = normalize_openai_base(base_url)
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    return f"{root}/health"


@dataclass
class HealthResult:
    ok: bool
    url: str
    payload: dict[str, Any] | None
    error: str | None = None


def fetch_health(base_or_health_url: str, *, timeout_s: float = 10.0) -> HealthResult:
    raw = (base_or_health_url or "").strip()
    url = raw if raw.rstrip("/").endswith("/health") else health_url_from_base(raw)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        if not isinstance(body, dict):
            return HealthResult(ok=False, url=url, payload=None, error="non-object JSON")
        ready = bool(body.get("ready", False))
        return HealthResult(ok=ready, url=url, payload=body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return HealthResult(ok=False, url=url, payload=None, error=f"{type(exc).__name__}: {exc}")


def smoke_chat_completions(
    base_url: str,
    *,
    message: str = "Reply with exactly: pong",
    max_tokens: int = 16,
    timeout_s: float = 60.0,
    model: str = "emet",
) -> str:
    """POST a text-only chat completion; return assistant content."""
    base = normalize_openai_base(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode())
    return str(body["choices"][0]["message"]["content"] or "")


def smoke_vl_completions(
    base_url: str,
    *,
    image_path: str | None = None,
    prompt: str = "Describe this image in one short sentence.",
    max_tokens: int = 64,
    timeout_s: float = 120.0,
    model: str = "emet-vl",
) -> str:
    """POST a multimodal completion (synthetic or file JPEG) via OpenaiVLLMClient."""
    import numpy as np
    from PIL import Image

    from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec

    base, model_from_spec = parse_openai_endpoint_spec(normalize_openai_base(base_url))
    client = OpenaiVLLMClient(
        prompt=None,
        model=model_from_spec or model,
        base_url=base,
        max_tokens=max_tokens,
        image_max_side=256,
    )
    if image_path:
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    else:
        # Deterministic tiny RGB so smoke needs no fixture file.
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rgb[:, :16, 0] = 220
        rgb[:, 16:, 2] = 200
    return client.generate_multimodal([prompt, rgb], max_new_tokens=max_tokens)
