"""Actual server threads and ZMQ receipts, using inproc endpoints (no robot ports)."""

import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("mujoco")
zmq = pytest.importorskip("zmq")

from emet.core.command_client import command_receipt, send_reactive_goal, wait_reactive_goal
from emet.core.server import BaseZmqServer
from emet.rl.model import build_model
from emet.robots.innate_mars import InnateMarsBackend
from emet.simulation.robosuite_server import RobosuiteZmqServer

pytestmark = pytest.mark.sim


def test_actual_zmq_server_receives_and_completes_reactive_goal(monkeypatch):
    prefix = f"inproc://rl-{uuid4().hex}"
    monkeypatch.setattr(BaseZmqServer, "_get_ip_address", lambda self, port, remote: f"{prefix}-{port}")
    import emet.core.server as core_server

    monkeypatch.setattr(core_server, "imported_text_to_speech", False)
    for key in ("EMET_ZMQ_FULL_HZ", "EMET_ZMQ_STATE_HZ", "EMET_ZMQ_SERVO_HZ"):
        monkeypatch.setenv(key, "40")
    server = RobosuiteZmqServer(
        InnateMarsBackend().get_spec(), scene_model=build_model("table"), controller="baseline", no_cameras=True
    )
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0)
    publisher = server.context.socket(zmq.PUB)
    subscriber = server.context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.RCVTIMEO, 100)
    publisher.connect(f"{prefix}-4402")
    subscriber.connect(f"{prefix}-4403")
    client.send_message = publisher.send_pyobj
    done = threading.Event()

    def receive():
        while not done.is_set():
            try:
                client._state = subscriber.recv_pyobj()
            except zmq.Again:
                pass

    reader = threading.Thread(target=receive)
    runner = threading.Thread(target=lambda: server.start(headless=True))
    runner.start()
    reader.start()
    try:
        deadline = time.monotonic() + 10
        while not hasattr(client, "_state") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hasattr(client, "_state"), "server did not publish command protocol"
        action = send_reactive_goal(client, base=[0.10, 0, 0], arm={"joint1": 0.10}, timeout=8)
        success = wait_reactive_goal(client, action, timeout=8)
        assert success, (
            command_receipt(client, action),
            server._reactive_controller.control.observe()[1],
            server._mjdata.time,
            server._mjdata.qvel.tolist(),
        )
        receipt = command_receipt(client, action)
        assert receipt["result"]["stop_confirmed"]
        assert receipt["result"]["xy_error"] <= 0.05
    finally:
        server.stop()
        done.set()
        runner.join(timeout=3)
        reader.join(timeout=3)
        for name in ("_sim_thread", "_send_thread", "_send_state_thread", "_send_servo_thread", "_recv_thread"):
            thread = getattr(server, name, None)
            if thread is not None:
                thread.join(timeout=3)
        publisher.close(0)
        subscriber.close(0)
        server.close_zmq_resources()
