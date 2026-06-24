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

"""Replay MolmoBot joint trajectories against a running emet ZMQ sim server."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.datasets.molmobot.reader import MolmoBotBatchReader
from emet.robots import get_robot_spec


@dataclass
class ReplayMetrics:
    steps_sent: int
    qpos_mse_mean: float | None
    notes: str = ""


def _joint_vector(spec, joint_dict: dict[str, float]) -> np.ndarray:
    out = np.zeros(spec.dof, dtype=np.float64)
    for i, name in enumerate(spec.joint_names):
        if name in joint_dict:
            out[i] = float(joint_dict[name])
    return out


def replay_episode_open_loop(
    h5_path: Path | str,
    traj_key: str,
    *,
    robot: str,
    robot_ip: str = "127.0.0.1",
    port_offset: int = 0,
    hz: float = 10.0,
    max_steps: int | None = None,
) -> ReplayMetrics:
    """Send ``actions/joint_pos`` from H5 to ZMQ ``joint`` pins (open loop).

    Requires ``emet serve mujoco`` running with matching ``--robot``.
    """
    from emet.app.robot_cli import create_robot_client_from_cli

    spec = get_robot_spec(robot)
    if spec is None:
        raise ValueError(f"Unknown robot {robot!r}")

    ep = MolmoBotBatchReader(h5_path).read_episode(traj_key)
    steps = ep.trimmed_actions()
    if max_steps is not None:
        steps = steps[: max(0, int(max_steps))]

    client = create_robot_client_from_cli(robot=robot, robot_ip=robot_ip, port_offset=port_offset)

    mse_vals: list[float] = []
    sent = 0
    dt = 1.0 / max(hz, 1e-3)
    for step in steps:
        target = step.action_joint_pos or step.qpos
        if not target:
            continue
        vec = _joint_vector(spec, target)
        if hasattr(client, "send_action"):
            client.send_action({"joint": vec, "step": sent}, reliable=False)
        sent += 1
        obs_q = client.get_joint_positions() if hasattr(client, "get_joint_positions") else None
        if obs_q is not None and len(obs_q) == len(vec):
            mse_vals.append(float(np.mean((np.asarray(obs_q) - vec) ** 2)))
        time.sleep(dt)

    mse_mean = float(np.mean(mse_vals)) if mse_vals else None
    return ReplayMetrics(steps_sent=sent, qpos_mse_mean=mse_mean)


def write_replay_report(path: Path | str, metrics: ReplayMetrics, *, extra: dict | None = None) -> None:
    payload = {
        "steps_sent": metrics.steps_sent,
        "qpos_mse_mean": metrics.qpos_mse_mean,
        "notes": metrics.notes,
    }
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2))
