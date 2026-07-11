# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for ZMQ stream pause/resume (no robot required)."""

from __future__ import annotations

import threading
import time

from emet.controller.zmq_stream_control import ZmqStreamPauseMixin, paused_robot_streams


class _FakeClient(ZmqStreamPauseMixin):
    def __init__(self) -> None:
        self._finish = False
        self._init_stream_pause()
        self.decoded = 0

    def spin_once(self) -> None:
        if not self._wait_if_streams_paused(poll_s=0.01):
            return
        self.decoded += 1


def test_pause_blocks_wait_until_resume():
    c = _FakeClient()
    c.pause_streams()
    assert c.streams_paused()

    done = threading.Event()

    def worker() -> None:
        assert c._wait_if_streams_paused(poll_s=0.02)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05)
    assert not done.is_set()
    c.resume_streams()
    assert done.wait(timeout=1.0)
    assert not c.streams_paused()


def test_paused_robot_streams_contextmanager():
    c = _FakeClient()
    with paused_robot_streams(c):
        assert c.streams_paused()
    assert not c.streams_paused()


def test_paused_robot_streams_noop_without_api():
    with paused_robot_streams(object()):
        pass
