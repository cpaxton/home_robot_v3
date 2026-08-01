# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OpenaiClient multi-turn history (no network)."""

from __future__ import annotations

from typing import Any

from emet.llms.openai_client import OpenaiClient


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
