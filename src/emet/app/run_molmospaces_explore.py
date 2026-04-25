# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Record exploration trajectories from a running MuJoCo ZMQ server (MolmoSpaces homes, etc.).

from __future__ import annotations

from pathlib import Path

import click

from emet.app.robot_cli import create_robot_client_from_cli
from emet.core.parameters import get_parameters
from emet.molmospaces.episode_writer import (
    MolmoEpisodeWriter,
    export_nerfstudio_transforms,
    write_episode_rgb_mp4,
)
from emet.molmospaces.exploration import MolmoExploreSession, build_graph_sidecar


@click.command()
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", show_default=True)
@click.option(
    "--robot",
    "robot_backend",
    default="rby1",
    show_default=True,
    help="Robot backend; must match ``emet serve mujoco --robot``.",
)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option(
    "--molmospaces-scene",
    default="ithor",
    show_default=True,
    help="Scene id for episode metadata (must match the running server).",
)
@click.option("--molmospaces-split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--molmospaces-index", default=0, type=int, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    required=True,
    help="Episode root: images/, metadata.jsonl, episode.json",
)
@click.option("--steps", default=60, type=int, show_default=True, help="Number of capture iterations.")
@click.option(
    "--capture-hz",
    default=2.0,
    type=float,
    show_default=True,
    help="Target capture rate (sleep between frames after work).",
)
@click.option("--navigate-every", default=5, type=int, show_default=True, help="Issue a random base goal every N steps.")
@click.option("--nav-timeout", default=90.0, type=float, show_default=True)
@click.option(
    "--goal-x-min",
    type=float,
    default=-4.0,
    show_default=True,
)
@click.option("--goal-x-max", type=float, default=4.0, show_default=True)
@click.option("--goal-y-min", type=float, default=-4.0, show_default=True)
@click.option("--goal-y-max", type=float, default=4.0, show_default=True)
@click.option("--no-depth", is_flag=True, help="Do not save depth .npy files.")
@click.option(
    "--export-transforms",
    is_flag=True,
    help="After run, write transforms.json (NERFStudio-style) from metadata.jsonl.",
)
@click.option(
    "--with-graph-report",
    is_flag=True,
    help="Maintain GraphEQAMemory + SensorGraphBuilder; write graph_report.txt on exit.",
)
@click.option("--cpu-only", is_flag=True, help="CPU-only VLM labels for graph sidecar.")
@click.option(
    "--no-mp4",
    is_flag=True,
    help="Skip encoding episode_rgb.mp4 from captured PNGs (OpenCV mp4v).",
)
@click.option("--mp4-fps", default=10.0, type=float, show_default=True, help="Frame rate for episode_rgb.mp4.")
@click.option(
    "--zmq-startup-timeout",
    default=None,
    type=float,
    help="Seconds to wait for first ZMQ obs+state (default 60, or EMET_ZMQ_STARTUP_TIMEOUT). Molmo loads can be slow.",
)
def main(
    robot_ip: str,
    robot_backend: str,
    port_offset: int,
    molmospaces_scene: str,
    molmospaces_split: str,
    molmospaces_index: int,
    output_dir: str,
    steps: int,
    capture_hz: float,
    navigate_every: int,
    nav_timeout: float,
    goal_x_min: float,
    goal_x_max: float,
    goal_y_min: float,
    goal_y_max: float,
    no_depth: bool,
    export_transforms: bool,
    with_graph_report: bool,
    cpu_only: bool,
    no_mp4: bool,
    mp4_fps: float,
    zmq_startup_timeout: float | None,
) -> None:
    """Explore and record posed RGB for NeRF-style datasets.

    Start the sim in another terminal, e.g.::

        emet serve mujoco --molmospaces-scene ithor --molmospaces-split train \\
          --molmospaces-index 0 --robot rby1 --headless

    Then run this command with matching ``--robot`` / scene metadata.
    """
    click.echo(
        "MolmoSpaces explore: connecting to ZMQ MuJoCo server… "
        "(ensure `emet serve mujoco` is already running with the same --robot and --port-offset.)"
    )
    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=False,
        start_immediately=True,
        zmq_startup_timeout=zmq_startup_timeout,
    )

    parameters = get_parameters("dynav_config.yaml")
    episode_fields = {
        "molmospaces_scene": molmospaces_scene,
        "molmospaces_split": molmospaces_split,
        "molmospaces_index": int(molmospaces_index),
        "robot": robot_backend,
        "steps": int(steps),
        "capture_hz": float(capture_hz),
        "navigate_every": int(navigate_every),
    }
    writer = MolmoEpisodeWriter(
        output_dir,
        episode_fields=episode_fields,
        save_depth=not no_depth,
    )

    graph_memory = None
    sensor_builder = None
    if with_graph_report:
        graph_memory, sensor_builder = build_graph_sidecar(
            parameters,
            cpu_only=cpu_only,
            device="cpu" if cpu_only else "cuda",
        )

    session = MolmoExploreSession(
        robot,
        writer,
        goal_xy_bounds=(goal_x_min, goal_x_max, goal_y_min, goal_y_max),
        navigate_every=navigate_every,
        nav_timeout=nav_timeout,
        graph_memory=graph_memory,
        sensor_builder=sensor_builder,
    )

    try:
        session.run(steps=steps, capture_hz=capture_hz)
    finally:
        extra: dict[str, str] = {}
        if with_graph_report and graph_memory is not None:
            rp = Path(output_dir) / "graph_report.txt"
            session.save_graph_report(rp)
            extra["graph_report"] = str(rp)
        writer.finalize(extra=extra or None)
        click.echo(f"Wrote episode to {output_dir} ({writer.frame_count} frames).")
        if getattr(session, "navigation_goal_timeouts", 0) > 0:
            n = session.navigation_goal_timeouts
            click.echo(
                f"Note: {n} random navigation goal(s) timed out (sim did not report at_goal in time); "
                "capture continued. Tighten --goal-* bounds, increase --nav-timeout, or use --navigate-every.",
                err=True,
            )

    if not no_mp4 and writer.frame_count > 0:
        try:
            mp4_path = write_episode_rgb_mp4(output_dir, fps=mp4_fps)
            click.echo(f"Wrote exploration video {mp4_path}")
        except Exception as e:
            click.echo(f"Warning: could not write MP4 ({e})", err=True)

    if export_transforms:
        out = export_nerfstudio_transforms(output_dir)
        click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
