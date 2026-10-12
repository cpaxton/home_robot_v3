# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.core.command_client import cancel_navigation, command_receipt, send_command, wait_navigation
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


def test_policy_rejected_before_client_sends_motion():
    robot = Robot()
    client = connection(robot)
    with pytest.raises(RuntimeError, match="does not advertise"):
        send_command(client, {"xyt": [0, 0, 0], "nav_policy": "precision"})
    assert robot.starts == 0


def test_corrections_preserve_episode_frame():
    action = Robot().navigation_correction_action(
        {"resolved_goal": [1, 2, 3], "frame": "episode"}, remaining=4, policy="precision"
    )
    assert action["nav_world"] is False
    assert action["xyt"] == [1, 2, 3]
    with pytest.raises(ValueError, match="resolved goal frame"):
        Robot().navigation_correction_action(
            {"resolved_goal": [1, 2, 3], "frame": "unknown"}, remaining=4, policy="precision"
        )


def test_policy_correction_keeps_identity_absolute_goal_and_deadline(monkeypatch):
    from emet.core import command_runtime

    clock = [0.0]
    monkeypatch.setattr(command_runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class PolicyRobot(Robot):
        def navigation_policy_names(self):
            return ("precision",)

        def start_navigation_command(self, action):
            self.starts += 1
            self.last_action = action
            return {"resolved_goal": [2, 0, 0], "frame": "world"}

        def navigation_policy_measurement(self):
            return {"pose": self.pose, "timestamp": clock[0], "stopped": True}

    robot = PolicyRobot()
    robot.pose = [1, 0, 0]
    client = connection(robot)
    action = send_command(client, {"xyt": [1, 0, 0], "nav_relative": True, "nav_policy": "precision"})
    for now in (0.25, 0.5, 0.75):
        clock[0] = now
        robot.poll_navigation_command()
    assert robot.starts == 2
    assert robot.last_action["xyt"] == [2, 0, 0]
    assert robot.last_action["nav_world"] is True
    assert "nav_relative" not in robot.last_action
    assert robot.last_action["nav_timeout_s"] == pytest.approx(29.25)
    robot.pose = [2, 0, 0]
    for now in (1.0, 1.25, 1.5):
        clock[0] = now
        robot.poll_navigation_command()
    client._state = robot.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "succeeded"
    assert receipt["result"]["corrections"] == 1
    assert len(robot.command_tracker.snapshot()) == 1


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


def test_busy_rejection_keeps_original_cancellation_target():
    robot = Robot()
    client = connection(robot)
    original = send_command(client, {"xyt": [0, 0, 1]})
    with pytest.raises(RuntimeError, match="busy"):
        send_command(client, {"xyt": [0, 0, 2]})
    assert client._last_navigation_command == original
    assert cancel_navigation(client)
    assert command_receipt(client, original)["status"] == "cancelled"


def test_failure_preserves_measured_error_and_frozen_goal():
    robot = Robot()
    client = connection(robot)
    action = send_command(client, {"xyt": [0, 0, 1]})
    robot.outcome = ("failed", {"yaw_error": 0.8, "yaw_tolerance": 0.15})
    robot.poll_navigation_command()
    client._state = robot.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "failed"
    assert receipt["result"] == {
        "resolved_goal": [0, 0, 1],
        "yaw_error": 0.8,
        "yaw_tolerance": 0.15,
        "stop_confirmed": True,
    }


@pytest.mark.parametrize("client_kind", ["generic", "stretch"])
def test_async_at_goal_cannot_accept_stale_telemetry(client_kind):
    if client_kind == "generic":
        from emet.controller.generic_zmq_client import GenericZmqClient as Client
    else:
        from emet.controller.zmq_client import StretchZmqClient as Client
    robot = Robot()
    client = connection(robot)
    send_command(client, {"xyt": [0, 0, 1]})
    client._state["at_goal"] = True
    assert not Client.at_goal(client)
    robot.outcome = ("succeeded", {})
    robot.poll_navigation_command()
    client._state = robot.command_message({"at_goal": False})
    assert Client.at_goal(client)


@pytest.mark.parametrize(
    "action", [None, [], {"command": None}, {"command": {}}, {"xyt": [0, 0, float("nan")], "command": {}}]
)
def test_malformed_input_cannot_crash_or_start_motion(action):
    robot = Robot()
    assert not robot.dispatch_command(action)
    assert robot.starts == 0


def test_slow_receipt_publication_does_not_block_dispatch_or_mix_revisions(monkeypatch):
    """A large query result must not monopolize the command/receipt locks."""
    from emet.core import command_tracker

    robot = Robot()
    identity = {**robot.command_tracker.metadata(), "client_session_id": "client", "sequence": 0}
    robot.command_tracker.accept(identity, {})
    robot.command_tracker.transition("client", 0, "running", result={"clear": [False]})
    copying, release, dispatched = threading.Event(), threading.Event(), threading.Event()
    messages, errors = [], []
    original_copy = command_tracker.deepcopy

    def slow_copy(value):
        if threading.current_thread() is publisher:
            copying.set()
            if not release.wait(3):
                raise AssertionError("publisher was not released")
        return original_copy(value)

    def publish():
        try:
            messages.append(robot.command_message({}))
        except BaseException as exc:
            errors.append(exc)

    def dispatch():
        try:
            robot.command_tracker.transition("client", 0, "succeeded", result={"clear": [True]})
            robot.dispatch_command({"command": {**identity, "sequence": 1}, "joint": [0.]})
            dispatched.set()
        except BaseException as exc:
            errors.append(exc)

    publisher = threading.Thread(target=publish)
    writer = threading.Thread(target=dispatch)
    monkeypatch.setattr(command_tracker, "deepcopy", slow_copy)
    publisher.start()
    try:
        assert copying.wait(1)
        writer.start()
        assert dispatched.wait(1), "publishing receipt history blocks new command dispatch"
    finally:
        release.set()
        publisher.join(3)
        if writer.ident is not None:
            writer.join(3)
    assert not publisher.is_alive() and not writer.is_alive()
    assert not errors
    old = messages[0]["command_receipts"][0]
    assert old["status"] == "running" and old["result"] == {"clear": [False]}
    latest = robot.command_tracker.snapshot()
    assert latest[0]["status"] == "succeeded" and latest[0]["result"] == {"clear": [True]}
    assert latest[1]["status"] == "succeeded"
    latest[0]["result"]["clear"][0] = False
    assert robot.command_tracker.snapshot()[0]["result"] == {"clear": [True]}
