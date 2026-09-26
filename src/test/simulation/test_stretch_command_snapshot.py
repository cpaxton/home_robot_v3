"""An in-flight command acknowledgement must not erase a newer arm target."""

import copy
import threading
from types import SimpleNamespace

from emet.simulation.stretch_mujoco.datamodels.status_command import CommandMove, StatusCommand
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer, MujocoServerProxies


def test_producer_waits_for_atomic_command_consumption():
    proxy = MujocoServerProxies(
        command_lock=threading.RLock(), _command={"val": StatusCommand()},
        _status={}, _cameras={}, _sensors={}, _joint_limits={},
    )
    read = threading.Event()
    producer_attempted = threading.Event()
    stored = threading.Event()

    def consume(command):
        # Manager proxies deserialize snapshots, unlike an ordinary local dict.
        command = copy.deepcopy(command)
        read.set()
        assert producer_attempted.wait(2)
        assert not stored.wait(.05)
        proxy.set_command(command)

    def produce():
        assert read.wait(2)
        producer_attempted.set()
        with proxy.command_lock:
            command = copy.deepcopy(proxy.get_command())
            command.set_move_to(CommandMove("wrist_pitch", True, -.5))
            proxy.set_command(command)
        stored.set()

    writer = threading.Thread(target=produce)
    writer.start()
    try:
        MujocoServer._consume_commands(SimpleNamespace(data_proxies=proxy, push_command=consume))
    finally:
        writer.join(timeout=2)
    assert not writer.is_alive()
    assert stored.is_set()
    target = proxy.get_command().move_to["wrist_pitch"]
    assert target.trigger and target.pos == -.5
