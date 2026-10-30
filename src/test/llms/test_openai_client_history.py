# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OpenaiClient multi-turn history (no network)."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image

from emet.llms.openai_client import OpenaiClient


@pytest.mark.parametrize("native_tools", [False, True, None])
def test_declared_provider_tool_protocol_is_honored_without_discarding_arguments(monkeypatch, native_tools):
    import httpx
    from openai import OpenAI

    calls = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "provided",
                            "object": "model",
                            "created": 1,
                            "owned_by": "fixture",
                            "capabilities": {"native_tools": native_tools},
                        }
                    ],
                },
            )
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "provided",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"tool_calls":[],"message":"done"}'},
                    }
                ],
            },
        )

    sdk = OpenAI(
        base_url="http://fixture/v1",
        api_key="unused",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr("emet.llms.openai_client.OpenAI", lambda **_: sdk)
    client = OpenaiClient("Tool schemas and text-JSON protocol", model="provided", base_url="http://fixture/v1")
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
    if native_tools is False:
        assert not client.supports_native_tools
        with pytest.raises(ValueError, match="prompt-based tool protocol"):
            client("inspect", tools=tools)
        assert not calls
        assert client.get_history() == []
        client("inspect with the prompt tool schema", tools=None)
        assert "tools" not in calls[0]
    else:
        assert client.supports_native_tools
        client("inspect", tools=tools)
        assert calls[0]["tools"] == tools


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def create(self, *, model: str, messages: list, **kwargs: Any):
        self.calls.append(list(messages))

        class _Msg:
            content = f"reply-{len(self.calls)}"
            tool_calls = None

        class _Choice:
            message = _Msg()

        class _Completion:
            choices = [_Choice()]

        return _Completion()


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions()})()


def test_openai_client_retains_history(monkeypatch) -> None:
    monkeypatch.setattr("emet.llms.openai_client.OpenAI", _FakeOpenAI)
    client = OpenaiClient("You are a test bot.", model="emet", base_url="http://127.0.0.1:9/v1")
    assert client("hello") == "reply-1"
    assert client("what did I say?") == "reply-2"
    comps = client._openai.chat.completions
    assert len(comps.calls) == 2
    assert comps.calls[0][0]["role"] == "system"
    assert comps.calls[0][1] == {"role": "user", "content": "hello"}
    # Second turn includes prior assistant turn.
    roles = [m["role"] for m in comps.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert comps.calls[1][2]["content"] == "reply-1"
    assert comps.calls[1][3]["content"] == "what did I say?"


def test_openai_client_reset_context_clears_history(monkeypatch) -> None:
    monkeypatch.setattr("emet.llms.openai_client.OpenAI", _FakeOpenAI)
    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client("one")
    client("two", reset_context=True)
    comps = client._openai.chat.completions
    assert len(comps.calls[1]) == 2  # system + user only
    assert comps.calls[1][1]["content"] == "two"


def test_chat_camera_keyword_reaches_http_payload_and_survives_followup(monkeypatch):
    """Use the real CHAT call boundary, not a hand-assembled multimodal input."""
    from emet.agent.loop import _call_llm

    monkeypatch.setattr("emet.llms.openai_client.OpenAI", _FakeOpenAI)
    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    rgb = np.full((4, 6, 3), [20, 180, 40], dtype=np.uint8)
    _call_llm(client, "Observe this", [], False, image=rgb, reset_context=True)
    _call_llm(client, "What color was it?", [], False, reset_context=False)
    messages = client._openai.chat.completions.calls[-1]
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Observe this"}
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    decoded = np.asarray(Image.open(BytesIO(base64.b64decode(encoded))))
    np.testing.assert_array_equal(decoded, rgb)
