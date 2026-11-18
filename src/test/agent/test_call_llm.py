# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""CHAT binds supported arguments before inference and never retries internal errors."""

from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.llm_call import call_adapter
from emet.agent.loop import _call_llm
from emet.llms.base import AbstractVLLMClient


def test_plain_text_client_binds_optional_legacy_keywords_without_trial_calls():
    calls = []

    def client(text, verbose=False):
        calls.append((text, verbose))
        return "ok"

    assert _call_llm(client, "hi", None, True, reset_context=True)[0] == "ok"
    assert calls == [("hi", True)]


@pytest.mark.parametrize("keyword", ["image", "tools"])
def test_unsupported_requested_inputs_fail_before_inference(keyword):
    calls = []

    def client(text, verbose=False):
        calls.append(text)
        return "text-only"

    image = np.zeros((2, 2, 3), dtype=np.uint8) if keyword == "image" else None
    tools = [{"type": "function"}] if keyword == "tools" else None
    with pytest.raises(TypeError, match=f"does not support requested {keyword}"):
        _call_llm(client, "hi", tools, False, image=image)
    assert calls == []


def test_image_tools_and_history_reach_the_client_unchanged():
    received = []

    def client(text, *, verbose=False, tools=None, image=None, reset_context=True):
        received.append((text, verbose, tools, image, reset_context))
        return "with_image"

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    tools = [{"type": "function", "function": {"name": "x"}}]
    assert _call_llm(client, "q", tools, False, image=image, reset_context=False)[0] == "with_image"
    assert len(received) == 1
    assert received[0][2] is tools and received[0][3] is image
    assert received[0][4] is False


def test_internal_typeerror_propagates_after_one_model_call():
    calls = []

    def client(text, **kwargs):
        calls.append(kwargs)
        raise TypeError("internal decoder bug")

    with pytest.raises(TypeError, match="internal decoder bug"):
        _call_llm(client, "q", [{"type": "function"}], False, image=np.zeros((2, 2, 3)))
    assert len(calls) == 1
    assert "tools" in calls[0] and "image" in calls[0]


def test_adapter_is_bound_once_per_client(monkeypatch):
    from emet.agent import llm_call

    inspect_signature = Mock(wraps=llm_call.signature)
    monkeypatch.setattr(llm_call, "signature", inspect_signature)

    def client(text):
        return "ok"

    _call_llm(client, "first", None, False)
    _call_llm(client, "second", None, False)
    assert inspect_signature.call_count == 1
    assert call_adapter(client) is call_adapter(client)


class _NoProgressVL(AbstractVLLMClient):
    def __init__(self):
        super().__init__("test")
        self.calls = 0

    def generate_multimodal(self, text, *, image=None, reset_context=True):
        self.calls += 1
        return "ok"


def test_vl_client_without_progress_keyword_runs_once():
    client = _NoProgressVL()
    assert _call_llm(client, "q", None, False, progress_callback=lambda msg: None)[0] == "ok"
    assert client.calls == 1


def test_vl_internal_typeerror_is_not_retried_as_missing_progress():
    class BrokenVL(_NoProgressVL):
        def generate_multimodal(self, text, *, progress_callback=None, **kwargs):
            self.calls += 1
            raise TypeError("decoder failed")

    client = BrokenVL()
    with pytest.raises(TypeError, match="decoder failed"):
        _call_llm(client, "q", None, False, progress_callback=lambda msg: None)
    assert client.calls == 1


def test_bound_method_adapter_never_reuses_a_different_receiver():
    class Receiver:
        def __init__(self, name):
            self.name = name

        def generate(self, text):
            return self.name

    first, second = Receiver("first"), Receiver("second")
    assert _call_llm(first.generate, "hello", None, False)[0] == "first"
    assert _call_llm(second.generate, "hello", None, False)[0] == "second"
    assert "_agent_call_adapter" not in Receiver.generate.__dict__


def test_copied_client_cannot_reuse_the_original_clients_bound_adapter():
    from copy import copy

    class Client:
        name = "original"

        def __call__(self, text):
            return self.name

    original = Client()
    assert _call_llm(original, "hello", None, False)[0] == "original"
    copied = copy(original)
    copied.name = "copied"
    assert _call_llm(copied, "hello", None, False)[0] == "copied"
