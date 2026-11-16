# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for remote OpenAI VLM client (no network)."""

from __future__ import annotations

import base64
import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

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
    assert resolve_vl_endpoint({"eqa": {"vl_endpoint": "openai@http://x:8001/v1"}}) == ("openai@http://x:8001/v1")
    monkeypatch.setenv("EMET_VL_ENDPOINT", "openai@http://env:8001/v1")
    assert resolve_vl_endpoint({"eqa": {"vl_endpoint": "openai@http://cfg:8001/v1"}}) == ("openai@http://env:8001/v1")


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


def test_openai_vllm_client_assistant_prefill(monkeypatch) -> None:
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

    out = client.generate_multimodal(
        "caption this",
        system_prompt="sys",
        max_new_tokens=16,
        assistant_prefill="Reasoning:",
    )
    assert out == "ok"
    messages = captured["messages"]
    assert messages[-1] == {"role": "assistant", "content": "Reasoning:"}

    captured.clear()
    client.generate_multimodal("caption this", system_prompt="sys", max_new_tokens=16)
    assert all(m["role"] != "assistant" for m in captured["messages"])


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


def test_call_timing_separates_encoding_from_request_and_preserves_errors():
    from types import SimpleNamespace

    import pytest

    client = OpenaiVLLMClient(model="model", base_url="http://localhost:8000/v1")
    client._openai = MagicMock()
    client._openai.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    assert client.generate_multimodal(["look", np.zeros((8, 8, 3), dtype=np.uint8)]) == "ok"
    metrics = client.last_call_metrics
    assert metrics["completed"] and metrics["image_count"] == 1
    assert metrics["encode_s"] >= 0 and metrics["request_s"] >= 0
    assert abs(metrics["total_s"] - metrics["encode_s"] - metrics["request_s"]) < 1e-6
    client._openai.chat.completions.create.side_effect = TimeoutError("timeout")
    with pytest.raises(TimeoutError):
        client.generate_multimodal("look")
    assert client.last_call_metrics["completed"] is False
    assert client.last_call_metrics["image_count"] == 0


def test_opt_in_continuation_replays_images_but_next_default_call_discards_them():
    client = OpenaiVLLMClient(model="fixture", base_url="http://localhost:9999/v1")
    complete = MagicMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
    )
    client._openai.chat.completions.create = complete
    image = np.zeros((4, 4, 3), np.uint8)
    client.generate_multimodal(["first", image])
    first_blocks = copy.deepcopy(complete.call_args.kwargs["messages"][0]["content"])
    client.generate_multimodal(["followup", image + 127], reset_context=False)
    continued = complete.call_args.kwargs["messages"]
    assert [row["role"] for row in continued] == ["user", "assistant", "user"]
    assert continued[0]["content"] == first_blocks
    assert client.steps == 2
    prior_metrics = client.last_call_metrics.copy()
    assert prior_metrics["image_count"] == 1 and prior_metrics["request_image_count"] == 2
    client.generate_multimodal("independent caption")
    assert complete.call_args.kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "independent caption"}]}
    ]
    assert client.steps == 1 and client.last_call_metrics["request_image_count"] == 0
    assert prior_metrics["request_image_count"] == 2


@pytest.mark.parametrize("raises", [False, True])
def test_explicit_dialogue_encodes_prior_images_and_restores_shared_state(raises):
    client = OpenaiVLLMClient(model="fixture", base_url="http://localhost:9999/v1")
    complete = MagicMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
    )
    client._openai.chat.completions.create = complete
    client.generate_multimodal("unrelated perception")
    history, steps = copy.deepcopy(client.get_history()), client.steps
    hook = MagicMock(wraps=client._conversation_user_content)
    client._conversation_user_content = hook
    image = np.zeros((4, 4, 3), np.uint8)
    prior_content = ["earlier observation", image, image + 127]
    dialogue = [
        {"role": "system", "content": "agent task"},
        {"role": "user", "content": prior_content},
        {"role": "assistant", "content": "inspect next"},
        {"role": "user", "content": ["current observation", image + 255]},
    ]
    if raises:
        complete.side_effect = TimeoutError("remote timeout")
        with pytest.raises(TimeoutError):
            client.generate_messages(dialogue)
    else:
        assert client.generate_messages(dialogue) == "answer"
    sent = complete.call_args.kwargs["messages"]
    assert [row["role"] for row in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["content"] == "inspect next"
    assert [b["type"] for b in sent[1]["content"]] == ["text", "image_url", "image_url"]
    assert sent[3]["content"][1]["type"] == "image_url"
    assert "unrelated perception" not in str(sent)
    assert hook.call_count == 1 and hook.call_args.args[0] is prior_content
    assert client.get_history() == history and client.steps == steps
    assert client.last_call_metrics["image_count"] == 1
    assert client.last_call_metrics["request_image_count"] == 3
    assert client.last_call_metrics["completed"] is not raises
