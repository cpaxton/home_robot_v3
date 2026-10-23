import threading
from types import SimpleNamespace

import pytest

from emet.core.command_client import (
    cancel_reactive_goal,
    close_command_session,
    command_receipt,
    send_command,
    send_reactive_goal,
    update_reactive_goal,
)
from emet.core.command_runtime import CommandRuntime


class Robot(CommandRuntime):
    def __init__(self):
        self.initialize_commands()
        self._last_step = -1
        self.starts = self.updates = self.stops = 0
        self.stopped = True
        self.outcome = None

    def supports_reactive_control(self):
        return True

    def validate_reactive_goal(self, goal):
        if not isinstance(goal, dict) or not goal:
            raise ValueError("empty goal")

    def start_reactive_command(self, goal):
        self.starts += 1
        return {"goal": goal}

    def update_reactive_command(self, goal):
        self.updates += 1
        return {"goal": goal}

    def reactive_command_result(self, context):
        return self.outcome

    def cancel_reactive_command(self):
        self.stops += 1
        return self.stopped


def connect(robot):
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=robot.command_message({}))

    def send(action):
        robot.dispatch_command(action)
        client._state = robot.command_message({})

    client.send_message = send
    return client


def test_updates_preserve_identity_deadline_and_retry_deduplication():
    robot = Robot()
    client = connect(robot)
    action = send_reactive_goal(client, base=[0.1, 0, 0], arm={"joint1": 0.2})
    original = command_receipt(client, action)
    update = update_reactive_goal(client, action, {"base": [0.2, 0, 0]})
    for _ in range(3):
        client.send_message(action)
        client.send_message(update)
    current = command_receipt(client, action)
    assert current["deadline"] == original["deadline"]
    assert current["sequence"] == original["sequence"]
    assert current["status"] == "running"
    assert robot.starts == robot.updates == 1
    for payload in ({"xyt": [1, 0, 0]}, {"joint": [0]}, {"base_velocity": {"v": 1, "w": 0}}):
        with pytest.raises(RuntimeError, match="busy"):
            send_command(client, payload)
    assert cancel_reactive_goal(client, action)
    assert command_receipt(client, action)["status"] == "cancelled"


def test_other_owner_wrong_target_and_mixed_payload_cannot_update():
    robot = Robot()
    client, other = connect(robot), connect(robot)
    action = send_reactive_goal(client, base=[0, 0, 0])
    with pytest.raises(RuntimeError, match="another client"):
        update_reactive_goal(other, action, {"base": [1, 0, 0]})
    with pytest.raises(RuntimeError, match="target"):
        send_command(
            client,
            {
                "update_reactive_goal": {
                    "target": {"client_session_id": "bad", "sequence": 0},
                    "goal": {"base": [1, 0, 0]},
                }
            },
        )
    with pytest.raises(RuntimeError, match="standalone"):
        send_command(client, {"update_reactive_goal": {}, "joint": [0]})
    assert robot.updates == 0


def test_expiry_and_unconfirmed_stop_retain_ownership():
    robot = Robot()
    clock = [0.0]
    robot.command_tracker._clock = lambda: clock[0]
    client = connect(robot)
    action = send_reactive_goal(client, base=[1, 0, 0], timeout=1)
    robot.stopped = False
    clock[0] = 2
    robot.poll_navigation_command()
    client._state = robot.command_message({})
    assert command_receipt(client, action)["status"] == "failed"
    assert robot._reactive_fault
    with pytest.raises(RuntimeError, match="busy"):
        send_command(client, {"release_control": True})
    with pytest.raises(RuntimeError, match="unconfirmed"):
        update_reactive_goal(client, action, {"base": [0, 0, 0]})
    assert robot.updates == 0


def test_success_requires_stop_and_shutdown_cancels_active_goal():
    robot = Robot()
    client = connect(robot)
    action = send_reactive_goal(client, base=[0, 0, 0])
    robot.outcome = ("succeeded", {"xy_error": 0})
    robot.poll_navigation_command()
    client._state = robot.command_message({})
    assert command_receipt(client, action)["result"]["stop_confirmed"]
    send_reactive_goal(client, base=[1, 0, 0])
    close_command_session(client)
    assert robot._reactive_command is None
    assert robot.stops == 2
    send_reactive_goal(connect(robot), base=[0, 0, 0])


def test_legacy_peer_rejected_before_motion():
    robot = Robot()
    robot.supports_reactive_control = lambda: False
    with pytest.raises(RuntimeError, match="does not advertise"):
        send_reactive_goal(connect(robot), base=[1, 0, 0])
    assert robot.starts == 0


def test_shutdown_does_not_cleanup_navigation_or_release_when_reactive_stop_is_unconfirmed(monkeypatch):
    import emet.core.command_client as commands

    robot = Robot()
    client = connect(robot)
    send_reactive_goal(client, base=[1, 0, 0])
    robot.stopped = False
    client._last_navigation_command = {"client_session_id": "other", "sequence": 999}
    cleanup = []
    monkeypatch.setattr(commands, "cancel_navigation", lambda *args: cleanup.append("navigation"))
    close_command_session(client)
    assert not cleanup
    assert robot._reactive_fault
    assert robot._reactive_command is not None
    with pytest.raises(RuntimeError, match="another client"):
        send_reactive_goal(connect(robot), base=[0, 0, 0])
