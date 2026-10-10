# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared inference must not turn a task follow-up into a caption dialogue."""

import pytest

from emet.llms.base import AbstractLLMClient


class Client(AbstractLLMClient):
    def __call__(self, *args, **kwargs):
        return ""


@pytest.mark.parametrize("raises", [False, True])
def test_perception_restores_agent_history_on_success_and_failure(raises):
    client = Client("agent tools")
    original = [{"role": "system", "content": "agent tools"}, {"role": "user", "content": "put cup on table"}]
    client.conversation_history = original.copy()
    client._iterations = 2
    try:
        with client.preserve_conversation():
            client.reset()
            client.add_history({"role": "system", "content": "list object labels only"})
            if raises:
                raise RuntimeError("perception failed")
    except RuntimeError:
        pass
    assert client.get_history() == original
    assert client.steps == 2


def test_nested_perception_context_is_restored_in_order():
    client = Client("agent")
    client.add_history({"role": "user", "content": "task"})
    with client.preserve_conversation():
        client.reset()
        client.add_history({"role": "user", "content": "caption"})
        with client.preserve_conversation():
            client.reset()
        assert client.get_history() == [{"role": "user", "content": "caption"}]
    assert client.get_history() == [{"role": "user", "content": "task"}]
