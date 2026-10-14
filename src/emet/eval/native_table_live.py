# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live ZMQ facade for the explicitly privileged stationary-table evaluator."""
import time
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from emet.controller.manipulation.native_scene import model_signature, read_snapshot


class LiveTableRobot:
    """Read measured snapshots into private scoring data; forward ordinary commands."""

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.data = mujoco.MjData(model)
        self.signature = model_signature(model)
        self.commands = []  # Live commands are recorded by the server actuation audit.
        self.boot = None
        self.refresh()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def refresh(self):
        boot, snapshot = read_snapshot(self.client, self.signature)
        if self.boot is not None and boot != self.boot:
            raise RuntimeError('live simulator restarted')
        q, v = np.asarray(snapshot['qpos']), np.asarray(snapshot['qvel'])
        stamp = float(snapshot['sim_time'])
        if q.shape != (self.model.nq,) or v.shape != (self.model.nv,) or not np.isfinite(stamp):
            raise ValueError('invalid live scoring snapshot')
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite live scoring snapshot')
        self.boot = boot
        self.data.qpos[:], self.data.qvel[:], self.data.time = q, v, stamp
        mujoco.mj_kinematics(self.model, self.data)

    def step(self, seconds):
        """Wait for physics progress; never advance or pause the remote simulator."""
        self.refresh()
        start = float(self.data.time)
        deadline = time.monotonic() + 30
        while self.data.time - start < seconds:
            if time.monotonic() >= deadline:
                raise TimeoutError('live physics stopped advancing')
            time.sleep(.02)
            self.refresh()


@contextmanager
def live_table_robot(sim_path, output, model, *, initialize=True):
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.eval.sim_eval_session import (
        connect_benchmark_robot,
        launch_benchmark_sim_server,
        terminate_benchmark_sim_server,
    )

    repo = Path(__file__).resolve().parents[3]
    config = load_sim_launch_config_from_path(str(sim_path))
    if not initialize and not config.no_cameras:
        raise ValueError("Read-only scoring connection requires no_cameras")
    client = None
    with (output / 'server.log').open('w') as log:
        server = launch_benchmark_sim_server(config, repo=repo, cpu_only=False, server_stderr=log)
        try:
            if config.no_cameras:
                from emet.controller.zmq_client import StretchZmqClient
                client = StretchZmqClient(robot_ip='127.0.0.1', port_offset=server.port_offset,
                                          enable_rerun_server=False, start_immediately=False, state_only=True)
                if not client.start():
                    raise RuntimeError('camera-free simulator did not publish valid state')
                if initialize:
                    client.move_to_nav_posture()
                    client.set_velocity(v=30., w=15.)
            else:
                client = connect_benchmark_robot(config, server.port_offset, enable_rerun_server=False)
            yield LiveTableRobot(client, model)
        finally:
            try:
                if client is not None:
                    client.stop()
            finally:
                terminate_benchmark_sim_server(server)
