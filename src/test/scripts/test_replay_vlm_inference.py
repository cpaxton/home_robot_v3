# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Frozen inference uses the production client without actions or silent retries."""

import json
import runpy
from pathlib import Path

import httpx
import pytest
from PIL import Image


def module():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/replay_vlm_context.py"))


def packet(tmp_path):
    Image.new("RGB", (4, 4), "red").save(tmp_path / "old.png")
    Image.new("RGB", (4, 4), "blue").save(tmp_path / "new.png")
    source = {
        "system_prompt": "Choose the next action; tools are not executed in replay.",
        "conversation_history": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "image_id=image_original_1"},
                    {"type": "image", "path": "old.png"},
                ],
            },
            {"role": "assistant", "content": "inspect_graph"},
            {"role": "user", "content": "TOOL_RESULT movement rejected"},
        ],
        "user_content": [
            {"type": "text", "text": "Current task"},
            {"type": "image", "path": "new.png"},
            {"type": "text", "text": "End marker"},
        ],
        "metadata": {"scorer_only": "MUST_NOT_REACH_MODEL"},
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(source))
    return path


@pytest.mark.parametrize(
    "status,finish,expected", [(200, "stop", "complete"), (200, "length", "incomplete"), (500, None, "failed")]
)
def test_exact_interleaved_context_one_call_and_failures_retained(tmp_path, monkeypatch, status, finish, expected):
    calls = []

    def send(transport, request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            status,
            request=request,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "raw response"}, "finish_reason": finish}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    source = packet(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "result"
    row = module()["infer_recorded"](source, output, endpoint="http://model.invalid/v1#test", max_tokens=32)
    assert row["status"] == expected
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 32
    messages = calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "user"]
    assert messages[1]["content"][0]["text"] == "image_id=image_original_1"
    assert messages[3]["content"][0]["text"] == "TOOL_RESULT movement rejected"
    assert [p["type"] for p in messages[-1]["content"]] == ["text", "image_url", "text"]
    assert messages[-1]["content"][-1]["text"] == "End marker"
    assert "MUST_NOT_REACH_MODEL" not in json.dumps(calls)
    assert row["tools_executed"] == 0
    assert row["usage"]["prompt_tokens"] == 12
    assert source.read_bytes() == original
    assert "test-secret" not in (output / "wire-request.json").read_text()
    assert json.loads((output / "result.json").read_text())["status"] == expected
    with pytest.raises(ValueError, match="fresh output"):
        module()["infer_recorded"](source, output, endpoint="http://model.invalid/v1#test")


def test_recorded_packet_path_escape_fails_before_model_or_output(tmp_path):
    source = packet(tmp_path)
    x = json.loads(source.read_text())
    x["user_content"][1]["path"] = "../outside.png"
    source.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="escapes"):
        module()["infer_recorded"](source, tmp_path / "out", endpoint="http://model.invalid/v1#test")
    assert not (tmp_path / "out").exists()
