# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Shared robot ZMQ client construction for `emet run` CLIs (`--robot`, `--robot-ip`, `--port-offset`).

from __future__ import annotations

import importlib
import os

import click

from emet.controller.zmq_client import StretchZmqClient
from emet.core.robot import AbstractRobotClient
from emet.robots import ROBOT_REGISTRY


def create_robot_client_from_cli(
    robot: str,
    robot_ip: str,
    *,
    port_offset: int = 0,
    enable_rerun_server: bool = True,
    rerun_headless: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    start_immediately: bool = True,
    zmq_startup_timeout: float | None = None,
) -> AbstractRobotClient:
    """
    Resolve ``--robot`` the same way as ``emet serve mujoco --robot`` / ``run_dynamem``:
    Stretch uses ``StretchZmqClient``; other names use ``ROBOT_REGISTRY`` backends.
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
        kwargs: dict = {"robot_ip": robot_ip, "port_offset": port_offset}
        if zmq_startup_timeout is not None:
            kwargs["zmq_startup_timeout"] = float(zmq_startup_timeout)
        elif os.environ.get("EMET_ZMQ_STARTUP_TIMEOUT", "").strip():
            kwargs["zmq_startup_timeout"] = float(os.environ["EMET_ZMQ_STARTUP_TIMEOUT"].strip())
        return backend.create_client(**kwargs)
    raise click.UsageError(
        f"Unknown robot '{robot}'. Known: stretch, {list(ROBOT_REGISTRY.keys())}. "
        "Start the server with the same robot: emet serve mujoco --robot <name>"
    )
