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

"""Regression: ``_call_llm`` must drop ``tools=`` on TypeError, not retry the same call."""

import numpy as np

from emet.agent.loop import _call_llm


class _NoToolsClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, text: str, verbose: bool = False, **kwargs):
        self.calls.append({"text": text, "verbose": verbose, **kwargs})
        if "tools" in kwargs:
            raise TypeError("this client does not accept tools")
        return "ok"


def test_call_llm_drops_tools_after_typeerror():
    c = _NoToolsClient()
    tools = [{"type": "function", "function": {"name": "x"}}]
    raw, _elapsed = _call_llm(c, "hi", tools, debug=False, image=None)
    assert raw == "ok"
    assert len(c.calls) == 2
    assert "tools" in c.calls[0]
    assert "tools" not in c.calls[1]


def test_call_llm_drops_tools_then_uses_image():
    """After tools+image and tools-only fail, an image-only client still gets the frame."""

    class _Client:
        def __init__(self) -> None:
            self.last_kw: dict | None = None

        def __call__(self, text: str, verbose: bool = False, **kwargs):
            self.last_kw = dict(kwargs)
            if kwargs.get("tools") and kwargs.get("image") is not None:
                raise TypeError("no tools+image")
            if kwargs.get("tools"):
                raise TypeError("no tools")
            if kwargs.get("image") is not None:
                return "with_image"
            return "plain"

    c = _Client()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    tools = [{"type": "function", "function": {"name": "x"}}]
    raw, _ = _call_llm(c, "q", tools, False, image=img)
    assert raw == "with_image"
    assert c.last_kw.get("image") is img
    assert "tools" not in c.last_kw
