# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from emet.simulation.stretch_mujoco.datamodels.status_command import CommandMove, StatusCommand
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer
from emet.simulation.stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class CommandProxy:
    """Copy-on-read/write, like the multiprocessing manager's serialized value."""

    def __init__(self):
        self.command_lock = threading.Lock()
        self.command = StatusCommand()
        self.status_time = 0

    def get_command(self):
        return copy.deepcopy(self.command)

    def set_command(self, value):
        self.command = copy.deepcopy(value)

    def get_status(self):
        assert not self.command_lock.locked(), "stop wait must release the consumer's lock"
        self.command.base_velocity.trigger = False
        self.status_time += 1
        return SimpleNamespace(time=self.status_time, base=SimpleNamespace(x_vel=0, theta_vel=0))


def simulator(proxy):
    sim = object.__new__(StretchMujocoSimulator)
    sim.is_running = lambda: True
    sim._command_lock = proxy.command_lock
    sim.data_proxies = proxy
    return sim


def test_physics_ack_cannot_overwrite_a_new_joint_command():
    proxy = CommandProxy()
    proxy.command.set_move_to(CommandMove("lift", True, 0.6))
    sim = simulator(proxy)
    consuming = threading.Event()
    release_consumer = threading.Event()
    writer_started = threading.Event()
    written = threading.Event()
    errors = []
    server = object.__new__(MujocoServer)
    server.data_proxies = proxy
    server.physics_fps_counter = Mock()
    server.pull_status = Mock()

    def consume(snapshot):
        consuming.set()
        assert release_consumer.wait(2)
        snapshot.move_to["lift"].trigger = False
        proxy.set_command(snapshot)

    server.push_command = consume

    def tick():
        try:
            server._ctrl_callback(SimpleNamespace(), SimpleNamespace(time=1))
        except Exception as exc:
            errors.append(exc)

    def write():
        writer_started.set()
        sim.move_to("lift", 0.643)
        written.set()

    consumer = threading.Thread(target=tick)
    writer = threading.Thread(target=write)
    consumer.start()
    try:
        assert consuming.wait(1)
        writer.start()
        assert writer_started.wait(1)
        assert not written.wait(0.05)
    finally:
        release_consumer.set()
        consumer.join(2)
        if writer.ident is not None:
            writer.join(2)
    assert not consumer.is_alive() and not writer.is_alive()
    assert not errors
    assert written.is_set()
    pending = proxy.get_command().move_to["lift"]
    assert pending.pos == 0.643 and pending.trigger


def test_cancel_wait_does_not_block_physics_consumption():
    assert simulator(CommandProxy()).cancel_base_motion(timeout=0.5)
