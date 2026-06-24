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

"""CLI: ``emet dataset molmobot …``."""

from __future__ import annotations

import json

import click

from emet.datasets.molmobot.lerobot_export import export_lerobot_tree
from emet.datasets.molmobot.reader import MolmoBotBatchReader, summarize_dataset
from emet.datasets.molmobot.replay import replay_episode_open_loop, write_replay_report


@click.group("molmobot")
def molmobot_dataset_group() -> None:
    """MolmoBot-Data (H5) inspect, export, and sim replay."""


@molmobot_dataset_group.command("inspect")
@click.argument("path", type=click.Path(exists=True))
@click.option("--json-out", type=click.Path(), default=None, help="Write summary JSON to this path.")
def inspect_cmd(path: str, json_out: str | None) -> None:
    """Print episode counts, lengths, cameras, and qpos dimensions."""
    summary = summarize_dataset(path)
    text = json.dumps(summary, indent=2)
    click.echo(text)
    if json_out:
        with open(json_out, "w") as f:
            f.write(text + "\n")


@molmobot_dataset_group.command("list-trajs")
@click.argument("h5_file", type=click.Path(exists=True))
def list_trajs_cmd(h5_file: str) -> None:
    reader = MolmoBotBatchReader(h5_file)
    for key in reader.traj_keys():
        click.echo(key)


@molmobot_dataset_group.command("export-lerobot")
@click.option("--src", required=True, type=click.Path(exists=True), help="H5 file or dataset root.")
@click.option("--out", required=True, type=click.Path(), help="Output directory.")
@click.option("--task", default="molmobot", show_default=True)
@click.option("--max-episodes", type=int, default=None)
def export_lerobot_cmd(src: str, out: str, task: str, max_episodes: int | None) -> None:
    """Export JSONL episodes compatible with hello-robot LeRobot preprocessing."""
    n = export_lerobot_tree(src, out, task=task, max_episodes=max_episodes)
    click.echo(f"Wrote {n} episode(s) under {out}")


@molmobot_dataset_group.command("replay")
@click.option("--h5", required=True, type=click.Path(exists=True))
@click.option("--traj-key", required=True)
@click.option("--robot", required=True)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option("--hz", default=10.0, type=float, show_default=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--report", type=click.Path(), default=None, help="Write replay metrics JSON.")
def replay_cmd(
    h5: str,
    traj_key: str,
    robot: str,
    robot_ip: str,
    port_offset: int,
    hz: float,
    max_steps: int | None,
    report: str | None,
) -> None:
    """Open-loop replay against a running ``emet serve mujoco`` server."""
    metrics = replay_episode_open_loop(
        h5,
        traj_key,
        robot=robot,
        robot_ip=robot_ip,
        port_offset=port_offset,
        hz=hz,
        max_steps=max_steps,
    )
    click.echo(f"Replay sent {metrics.steps_sent} steps; qpos MSE mean={metrics.qpos_mse_mean}")
    if report:
        write_replay_report(report, metrics, extra={"h5": h5, "traj_key": traj_key, "robot": robot})
