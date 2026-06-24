#!/usr/bin/env python3
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

"""Thin wrapper: replay one MolmoBot trajectory against a running ZMQ sim."""

from __future__ import annotations

import click

from emet.datasets.molmobot.replay import replay_episode_open_loop, write_replay_report


@click.command()
@click.option("--h5", required=True, type=click.Path(exists=True))
@click.option("--traj-key", required=True)
@click.option("--robot", required=True)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option("--hz", default=10.0, type=float, show_default=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--report", type=click.Path(), default=None)
def main(
    h5: str,
    traj_key: str,
    robot: str,
    robot_ip: str,
    port_offset: int,
    hz: float,
    max_steps: int | None,
    report: str | None,
) -> None:
    metrics = replay_episode_open_loop(
        h5,
        traj_key,
        robot=robot,
        robot_ip=robot_ip,
        port_offset=port_offset,
        hz=hz,
        max_steps=max_steps,
    )
    click.echo(f"steps={metrics.steps_sent} qpos_mse_mean={metrics.qpos_mse_mean}")
    if report:
        write_replay_report(report, metrics, extra={"h5": h5, "traj_key": traj_key})


if __name__ == "__main__":
    main()
