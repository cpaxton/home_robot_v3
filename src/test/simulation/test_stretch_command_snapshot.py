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


def test_stop_allows_physics_consumer_to_acknowledge():
    from emet.simulation.stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

    lock = threading.RLock()
    submitted = threading.Event()
    command = StatusCommand()
    stamp = 0.

    def status():
        nonlocal stamp
        stamp += .01
        return SimpleNamespace(time=stamp, base=SimpleNamespace(x_vel=0., theta_vel=0.))

    def publish(value):
        nonlocal command
        command = value
        submitted.set()

    def physics():
        assert submitted.wait(1)
        with lock:
            command.base_velocity.trigger = False

    worker = threading.Thread(target=physics)
    worker.start()
    simulator = SimpleNamespace(
        is_running=lambda: True, _command_lock=lock,
        data_proxies=SimpleNamespace(get_command=lambda: command, set_command=publish, get_status=status),
    )
    try:
        assert StretchMujocoSimulator.cancel_base_motion(simulator, timeout=.5)
    finally:
        worker.join(timeout=1)
    assert not worker.is_alive()
