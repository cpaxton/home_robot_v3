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
# Shared robot ZMQ client construction for `emet run` CLIs (`--robot`, `--robot-ip`, `--port-offset`).

from __future__ import annotations

import importlib
import os
import time
import timeit

import click
import zmq

from emet.controller.zmq_client import StretchZmqClient
from emet.core.robot import AbstractRobotClient
from emet.core.zmq_protocol import read_emet_robot_id_from_message_or_session
from emet.robots import ROBOT_REGISTRY
from emet.utils.memory import lookup_address


def discover_zmq_server_robot_id(
    robot_ip: str,
    *,
    port_offset: int = 0,
    timeout: float = 60.0,
    use_remote_computer: bool = True,
) -> str | None:
    """Subscribe briefly to obs + state ports; return first ``emet_robot_id`` (or session) seen."""
    recv_port = 4401 + int(port_offset)
    state_port = 4403 + int(port_offset)
    ip = lookup_address(robot_ip, use_remote_computer)
    if ip is None:
        return None
    ctx = zmq.Context()
    sockets: list[zmq.Socket] = []
    try:
        for port in (recv_port, state_port):
            s = ctx.socket(zmq.SUB)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.setsockopt(zmq.CONFLATE, 1)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f"{ip}:{port}")
            sockets.append(s)
        t0 = timeit.default_timer()
        while timeit.default_timer() - t0 < float(timeout):
            for s in sockets:
                try:
                    msg = s.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                if not isinstance(msg, dict):
                    continue
                rid = read_emet_robot_id_from_message_or_session(msg)
                if rid:
                    return rid
            time.sleep(0.05)
        return None
    finally:
        for s in sockets:
            s.close(linger=0)
        ctx.term()


def create_robot_client_from_cli(
    robot: str,
    robot_ip: str,
    *,
    port_offset: int = 0,
    enable_rerun_server: bool = False,
    rerun_headless: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    start_immediately: bool = True,
    zmq_startup_timeout: float | None = None,
    allow_missing_depth: bool = False,
) -> AbstractRobotClient:
    """
    Resolve ``--robot`` the same way as ``emet serve mujoco --robot`` / ``run_dynamem``:
    Stretch uses ``StretchZmqClient``; other names use ``ROBOT_REGISTRY`` backends.

    Rerun is **off** unless *enable_rerun_server* is true (``run_dynamem`` / ``run_scene_graph`` pass
    ``enable_rerun_server=not no_rerun``; ``run_graph_eqa`` passes true explicitly).
    """
    robot_key = robot.lower().replace("-", "_")
    if robot_key == "stretch":
        return StretchZmqClient(
            robot_ip=robot_ip,
            enable_rerun_server=enable_rerun_server,
            rerun_headless=rerun_headless,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            port_offset=port_offset,
            start_immediately=start_immediately,
            allow_missing_depth=allow_missing_depth,
        )
    if robot_key in ROBOT_REGISTRY:
        mod = importlib.import_module(ROBOT_REGISTRY[robot_key])
        backend_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, "get_spec") and attr_name != "RobotBackend":
                backend_cls = attr
                break
        if backend_cls is None:
            raise RuntimeError(f"No RobotBackend found in {ROBOT_REGISTRY[robot_key]}")
        backend = backend_cls()
        kwargs: dict = {
            "robot_ip": robot_ip,
            "port_offset": port_offset,
            "allow_missing_depth": allow_missing_depth,
            "enable_rerun_server": enable_rerun_server,
            "rerun_headless": rerun_headless,
            "rerun_show_panels": rerun_show_panels,
            "rerun_debug": rerun_debug,
        }
        if zmq_startup_timeout is not None:
            kwargs["zmq_startup_timeout"] = float(zmq_startup_timeout)
        elif os.environ.get("EMET_ZMQ_STARTUP_TIMEOUT", "").strip():
            kwargs["zmq_startup_timeout"] = float(os.environ["EMET_ZMQ_STARTUP_TIMEOUT"].strip())
        return backend.create_client(**kwargs)
    raise click.UsageError(
        f"Unknown robot '{robot}'. Known: stretch, {list(ROBOT_REGISTRY.keys())}. "
        "Start the server with the same robot: emet serve mujoco --robot <name>"
    )
