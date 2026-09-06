# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.core.command_client import command_receipt, send_command, wait_navigation
from emet.core.command_runtime import CommandRuntime
from emet.core.command_tracker import CommandTracker


class Robot(CommandRuntime):
    def __init__(self):
        self.initialize_commands()
        self._last_step = -1
        self.starts = 0
        self.outcome = None
        self.stopped = True

    def start_navigation_command(self, action):
        self.starts += 1
        return {"resolved_goal": action["xyt"]}

    def navigation_command_result(self, context):
        return self.outcome

    def cancel_navigation_command(self):
        return self.stopped

    def handle_action(self, action):
        pass


def connection(robot):
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=robot.command_message({}))

    def send(action):
        robot.dispatch_command(action)
        client._state = robot.command_message({})

    client.send_message = send
    return client


def test_end_to_end_identity_busy_measured_completion_and_release():
    robot = Robot()
    client = connection(robot)
    action = send_command(client, {"xyt": np.array([0, 0, 1])})
    for _ in range(10):
        client.send_message(action)
    assert robot.starts == 1
    with pytest.raises(RuntimeError, match="busy"):
        send_command(client, {"xyt": [0, 0, 2]})
    with pytest.raises(RuntimeError, match="busy"):
        send_command(client, {"posture": "navigation"})
    robot.outcome = ("succeeded", {"xy_error": 0})
    robot.poll_navigation_command()
    client._state = robot.command_message({})
    assert wait_navigation(client, action, 1)
    send_command(client, {"release_control": True})
    other = connection(robot)
    send_command(other, {"xyt": [0, 0, 2]})
    client.send_message(action)
    assert robot.starts == 2


@pytest.mark.parametrize("stopped", [True, False])
def test_deadline_cancels_and_unconfirmed_stop_locks_navigation(stopped):
    robot = Robot()
    now = [0]
    robot.command_tracker = CommandTracker(clock=lambda: now[0])
    robot.stopped = stopped
    client = connection(robot)
    action = send_command(client, {"xyt": [0, 0, 1], "nav_timeout_s": 1})
    now[0] = 2
    robot.poll_navigation_command()
    client._state = robot.command_message({})
    assert command_receipt(client, action)["status"] == "failed"
    assert robot._navigation_fault is not stopped
    if not stopped:
        with pytest.raises(RuntimeError, match="busy"):
            send_command(client, {"xyt": [0, 0, 2]})
        client.send_message(action)
        assert command_receipt(client, action)["status"] == "failed"


def test_old_bridge_or_restart_never_sends_motion():
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state={}, send_message=Mock())
    with pytest.raises(RuntimeError, match="deploy"):
        send_command(client, {"xyt": [0, 0, 1]})
    client.send_message.assert_not_called()
    robot = Robot()
    client = connection(robot)
    action = send_command(client, {"xyt": [0, 0, 1]})
    client._state = Robot().command_message({})
    with pytest.raises(RuntimeError, match="boot changed"):
        command_receipt(client, action)


def test_command_counter_is_independent_of_telemetry_counter():
    robot = Robot()
    client = connection(robot)
    client._iter = -1
    first = send_command(client, {"say": "one"})
    client._iter = 900
    second = send_command(client, {"say": "two"})
    assert first["command"]["sequence"] == 0
    assert second["command"]["sequence"] == 1
    assert first["step"] == 0 and second["step"] == 900


@pytest.mark.parametrize(
    "action", [None, [], {"command": None}, {"command": {}}, {"xyt": [0, 0, float("nan")], "command": {}}]
)
def test_malformed_input_cannot_crash_or_start_motion(action):
    robot = Robot()
    assert not robot.dispatch_command(action)
    assert robot.starts == 0
