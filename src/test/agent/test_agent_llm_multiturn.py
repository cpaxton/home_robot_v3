# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from emet.agent.loop import _call_llm, _format_fast_tool_reply, _should_skip_llm_summarize
from emet.llms.base import AbstractVLLMClient


class _FakeVLClient(AbstractVLLMClient):
    def __init__(self) -> None:
        self.conversation_history: list = []
        self._prompt = "system tools"
        self.max_tokens = 128
        self.calls: list[dict] = []

    @property
    def system_prompt(self) -> str:
        return self._prompt

    def generate_multimodal(self, user_content, **kwargs):
        self.calls.append(kwargs)
        return '{"message":"ok","tool_calls":[]}'

    def reset(self) -> None:
        self.conversation_history.clear()


def test_call_llm_vl_client_passes_reset_context():
    client = _FakeVLClient()
    _call_llm(client, "summarize", None, False, image=None, reset_context=False)
    assert len(client.calls) == 1
    assert client.calls[0]["reset_context"] is False


def test_call_llm_vl_client_defaults_reset_context_true():
    client = _FakeVLClient()
    _call_llm(client, "hello", None, False, image=None)
    assert client.calls[0]["reset_context"] is True


def test_agent_followup_round_uses_reset_context_false():
    """Simulate two-round tool loop kwargs pattern."""
    client = _FakeVLClient()
    _call_llm(client, "what do you see?", None, False, reset_context=True)
    _call_llm(client, "[Tool results]\n...", None, False, reset_context=False)
    assert client.calls[0]["reset_context"] is True
    assert client.calls[1]["reset_context"] is False


def test_fast_tool_reply_skips_summarize_for_describe_scene():
    tool_calls = [{"name": "describe_scene", "arguments": {}}, {"name": "send_image", "arguments": {}}]
    results = [
        "[describe_scene] From my head camera I can make out: sofa, lamp.",
        "[send_image] Image queued for Discord (attached to the reply).",
    ]
    assert _should_skip_llm_summarize(tool_calls, results) is True
    msg = _format_fast_tool_reply(results)
    assert msg is not None
    assert "sofa" in msg
    assert "lamp" in msg


def test_fast_tool_reply_does_not_skip_query_memory():
    tool_calls = [{"name": "query_memory", "arguments": {"question": "where?"}}]
    results = ["[query_memory] Answer: on the table"]
    assert _should_skip_llm_summarize(tool_calls, results) is False
