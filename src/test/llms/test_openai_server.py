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
    _openai_messages_to_chat,
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
