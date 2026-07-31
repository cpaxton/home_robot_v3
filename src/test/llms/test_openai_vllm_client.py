# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for remote OpenAI VLM client (no network)."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from emet.llms.eqa_vl_settings import resolve_vl_endpoint
from emet.llms.openai_vllm_client import OpenaiVLLMClient, parse_openai_endpoint_spec
from emet.llms.vllm_factory import create_dynamem_vllm


def test_parse_openai_endpoint_spec() -> None:
    base, model = parse_openai_endpoint_spec("openai@http://caliban:8001/v1#emet-vl")
    assert base == "http://caliban:8001/v1"
    assert model == "emet-vl"
    base2, model2 = parse_openai_endpoint_spec("http://host:9/v1/")
    assert base2 == "http://host:9/v1"
    assert model2 is None


def test_resolve_vl_endpoint_env_and_config(monkeypatch) -> None:
    for key in (
        "EMET_VL_ENDPOINT",
        "EMET_OPENAI_BASE_URL",
        "EMET_LLM_HOST",
        "EMET_CALIBAN_HOST",
    ):
        monkeypatch.delenv(key, raising=False)
    assert resolve_vl_endpoint({"eqa": {}}) is None
    assert resolve_vl_endpoint({"eqa": {"vl_endpoint": "openai@http://x:8001/v1"}}) == (
        "openai@http://x:8001/v1"
    )
    monkeypatch.setenv("EMET_VL_ENDPOINT", "openai@http://env:8001/v1")
    assert resolve_vl_endpoint({"eqa": {"vl_endpoint": "openai@http://cfg:8001/v1"}}) == (
        "openai@http://env:8001/v1"
    )


def test_openai_vllm_client_jpeg_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    def _create(**kwargs):
        captured.update(kwargs)
        return _Completion()

    client = OpenaiVLLMClient(
        model="emet-vl",
        base_url="http://caliban:8001/v1",
        max_tokens=32,
        image_max_side=8,
    )
    client._openai = MagicMock()
    client._openai.chat.completions.create = _create

    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    out = client.generate_multimodal(["caption this", rgb], system_prompt="sys", max_new_tokens=16)
    assert out == "ok"
    assert captured["model"] == "emet-vl"
    assert captured["max_tokens"] == 16
    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    user = messages[1]["content"]
    assert user[0] == {"type": "text", "text": "caption this"}
    assert user[1]["type"] == "image_url"
    url = user[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    blob = base64.b64decode(url.split(",", 1)[1])
    assert blob[:2] == b"\xff\xd8"  # JPEG SOI


def test_create_dynamem_vllm_remote_endpoint() -> None:
    client = create_dynamem_vllm(
        "qwen3_vl",
        hf_model_id=None,
        vl_model_size="8B",
        max_tokens=64,
        device="cuda",
        quantization="int4",
        endpoint="openai@http://caliban:8001/v1#remote-vl",
        image_max_side=64,
    )
    assert isinstance(client, OpenaiVLLMClient)
    assert client.base_url == "http://caliban:8001/v1"
    assert client.model == "remote-vl"


def test_agent_innate_mars_vl_endpoint_resolves(monkeypatch) -> None:
    """Without host env, preset has no VL URL; --host / EMET_* fallthrough works."""
    from emet.core.parameters import get_parameters
    from emet.llms.openai_vllm_client import OpenaiVLLMClient
    from emet.llms.remote_ops import apply_llm_host
    from emet.llms.vllm_factory import create_dynamem_vllm

    monkeypatch.delenv("EMET_VL_ENDPOINT", raising=False)
    monkeypatch.delenv("EMET_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMET_LLM_HOST", raising=False)
    monkeypatch.delenv("EMET_CALIBAN_HOST", raising=False)

    p = get_parameters("configs/agent_innate_mars.yaml")
    assert resolve_vl_endpoint(p) is None

    specs = apply_llm_host("caliban")
    assert specs is not None
    ep = resolve_vl_endpoint(p)
    assert ep == "openai@http://caliban:8000/v1"
    eqa = p.get("eqa") or {}
    client = create_dynamem_vllm(
        str(eqa.get("vl_family") or "qwen3_vl"),
        hf_model_id=eqa.get("vl_hf_model_id"),
        vl_model_size=str(eqa.get("vl_model_size") or "8B"),
        max_tokens=int(eqa.get("vl_max_tokens") or 512),
        device="cpu",
        quantization=eqa.get("vl_quantization"),
        endpoint=ep,
        image_max_side=int(eqa.get("vl_image_max_side") or 512),
    )
    assert isinstance(client, OpenaiVLLMClient)
    assert client.base_url == "http://caliban:8000/v1"


def test_serve_llm_help_lists_vl() -> None:
    from click.testing import CliRunner

    from emet.app.serve_llm import main as serve_llm_main

    r = CliRunner().invoke(serve_llm_main, ["--help"])
    assert r.exit_code == 0
    assert "--vl" in r.output
