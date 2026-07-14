# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Pause/resume ZMQ observation decode loops around GPU-heavy work.

Continuous JPEG/JP2 decode + depth unprojection on the ZMQ recv threads contends with
Hugging Face weight load and ``model.generate`` on the same process (GIL + host memory
bandwidth). Loading the chat LLM while those threads run can take ~100s and leave the
first generate hung until the streams are paused once. Pause around LLM load/generate;
CONFLATE sockets still deliver the latest frame on resume.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class ZmqStreamPauseMixin:
    """Mixin for ZMQ robot clients with background obs/state/servo threads."""

    def _init_stream_pause(self) -> None:
        # Set = streams may decode; clear = pause (threads sleep, sockets keep CONFLATE).
        self._streams_active = threading.Event()
        self._streams_active.set()

    def pause_streams(self) -> None:
        """Stop decoding ZMQ frames until :meth:`resume_streams` (threads keep running)."""
        gate = getattr(self, "_streams_active", None)
        if gate is not None:
            gate.clear()

    def resume_streams(self) -> None:
        """Allow ZMQ decode loops to process frames again."""
        gate = getattr(self, "_streams_active", None)
        if gate is not None:
            gate.set()

    def streams_paused(self) -> bool:
        gate = getattr(self, "_streams_active", None)
        return gate is not None and not gate.is_set()

    def _wait_if_streams_paused(self, poll_s: float = 0.05) -> bool:
        """Block while paused. Return False if ``_finish`` was set (caller should exit)."""
        gate = getattr(self, "_streams_active", None)
        if gate is None:
            return not bool(getattr(self, "_finish", False))
        while not getattr(self, "_finish", False):
            if gate.wait(timeout=poll_s):
                return True
        return False


@contextmanager
def paused_robot_streams(robot: Any) -> Iterator[None]:
    """Pause ZMQ decode on *robot* for the duration of the block (no-op if unsupported)."""
    pause = getattr(robot, "pause_streams", None)
    resume = getattr(robot, "resume_streams", None)
    if not callable(pause):
        yield
        return
    pause()
    try:
        yield
    finally:
        if callable(resume):
            resume()
