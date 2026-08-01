# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for OpenAI-compatible LLM serve helpers (no model download).

Imports modules by file path so Jetson aarch64 TLS issues in ``emet.llms.__init__``
(transformers → sklearn) do not break collection of these lightweight tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_LLMS = Path(__file__).resolve().parents[2] / "emet" / "llms"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / relative patterns resolve.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Lightweight deps: openai_client → emet.llms.base (package init is still pulled for base).
# Prefer direct file load of openai_client after ensuring base is importable.
from emet.llms.base import AbstractLLMClient  # noqa: E402,F401
from emet.llms.openai_client import resolve_openai_api_key, resolve_openai_base_url  # noqa: E402
from emet.llms.openai_server import (  # noqa: E402
    LLMServeState,
    _make_handler,
    _openai_messages_have_images,
    _openai_messages_to_chat,
    _openai_messages_to_multimodal,
    decode_data_url_image,
    generate_chat,
    resolve_serve_device,
)


def test_resolve_openai_base_url_env(monkeypatch) -> None:
    monkeypatch.delenv("EMET_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert resolve_openai_base_url(None) is None
    monkeypatch.setenv("EMET_OPENAI_BASE_URL", "http://caliban:8000/v1/")
    assert resolve_openai_base_url(None) == "http://caliban:8000/v1"
    assert resolve_openai_base_url("http://other:9/v1") == "http://other:9/v1"


def test_resolve_openai_api_key_default() -> None:
    assert resolve_openai_api_key(None) == "emet-local"
    assert resolve_openai_api_key("secret") == "secret"


def test_resolve_serve_device_cpu(monkeypatch) -> None:
    monkeypatch.setenv("EMET_LLM_SERVE_DEVICE", "cpu")
    assert resolve_serve_device(None) == "cpu"
    assert resolve_serve_device("cuda") == "cuda"


def test_openai_messages_to_chat_multipart() -> None:
    msgs = _openai_messages_to_chat(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
            },
        ]
    )
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"] == "hello\nworld"


class _FakePipeClient:
    def __init__(self) -> None:
        self.max_tokens = 32
        self._version = None

        class _Tok:
            pad_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
                return "PROMPT"

        self.tokenizer = _Tok()

        def _pipe(text, **kwargs):
            return [{"generated_text": text + "<|im_start|>assistant\nok\n<|im_end|>"}]

        self.pipe = _pipe


def test_generate_chat_via_pipe() -> None:
    out = generate_chat(
        _FakePipeClient(),
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
    )
    assert out == "ok"


def test_decode_data_url_and_multimodal_route() -> None:
    import base64
    from io import BytesIO
    from typing import Any

    import numpy as np
    from PIL import Image

    from emet.llms.base import AbstractVLLMClient

    img = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    url = f"data:image/jpeg;base64,{b64}"
    arr = decode_data_url_image(url)
    assert arr.shape == (4, 6, 3)

    msgs = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]
    assert _openai_messages_have_images(msgs) is True
    system, parts = _openai_messages_to_multimodal(msgs)
    assert system == "sys"
    assert parts[0] == "describe"
    assert isinstance(parts[1], np.ndarray)

    class _FakeVL(AbstractVLLMClient):
        def __init__(self) -> None:
            super().__init__("", None)
            self.max_tokens = 16
            self.calls: list[Any] = []

        def generate_multimodal(self, user_content, **kwargs):
            self.calls.append((user_content, kwargs))
            return "caption"

    vl = _FakeVL()
    out = generate_chat(vl, msgs, max_tokens=8)
    assert out == "caption"
    assert len(vl.calls) == 1


def test_generate_chat_images_on_text_client_errors() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                },
            ],
        }
    ]
    with pytest.raises(ValueError, match="text-only"):
        generate_chat(_FakePipeClient(), msgs, max_tokens=8)


def test_http_chat_completions_roundtrip() -> None:
    state = LLMServeState(llm_key="fake", device="cpu", max_tokens=16, api_key=None)
    state.client = _FakePipeClient()
    state.ready = True
    state.model_id = "fake"
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "ok"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        assert health["ready"] is True
    finally:
        httpd.shutdown()


def test_serve_llm_help() -> None:
    from click.testing import CliRunner

    from emet.app.serve_llm import main as serve_llm_main

    r = CliRunner().invoke(serve_llm_main, ["--help"])
    assert r.exit_code == 0
    assert "--llm" in r.output
    assert "--vl" in r.output


def test_vl_serve_load_ignores_remote_endpoint(monkeypatch) -> None:
    """``emet serve llm --vl`` must load local weights, not proxy via EMET_VL_ENDPOINT."""
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("EMET_VL_ENDPOINT", "openai@http://caliban:8001/v1")
    state = LLMServeState(
        llm_key="qwen3-vl-eqa",
        device="cpu",
        max_tokens=32,
        multimodal=True,
    )
    fake = MagicMock(name="local_vlm")
    with patch("emet.llms.vllm_factory.create_dynamem_vllm", return_value=fake) as m:
        client = state._load_local_vlm(device="cpu", max_tokens=32)
    assert client is fake
    assert m.call_args.kwargs.get("endpoint") is None


def test_get_llm_client_openai_at_url() -> None:
    """May skip on aarch64 if transformers/sklearn static TLS is not preloaded."""
    try:
        from emet.llms import get_llm_client
        from emet.llms.openai_client import OpenaiClient
    except ImportError as exc:
        if "TLS" in str(exc) or "libgomp" in str(exc):
            pytest.skip(f"aarch64 TLS: {exc}")
        raise
    client = get_llm_client("openai@http://192.168.1.10:8000/v1#qwen25-14B", prompt="sys")
    assert isinstance(client, OpenaiClient)
    assert client.base_url == "http://192.168.1.10:8000/v1"
    assert client.model == "qwen25-14B"
