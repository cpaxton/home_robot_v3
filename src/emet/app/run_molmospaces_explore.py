# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Record exploration trajectories from a running MuJoCo ZMQ server (MolmoSpaces homes, etc.).

from __future__ import annotations

import os
from pathlib import Path

import click

from emet.app.robot_cli import create_robot_client_from_cli, discover_zmq_server_robot_id
from emet.core.parameters import get_parameters
from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY, EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY, robot_ids_match
from emet.molmospaces.episode_writer import (
    MolmoEpisodeWriter,
    export_nerfstudio_transforms,
    write_episode_rgb_mp4,
)
from emet.molmospaces.exploration import MolmoExploreSession, build_graph_sidecar


def _json_safe_episode_session(sess: dict[str, object]) -> dict[str, object]:
    """Subset of ``emet_session`` safe to embed in ``episode.json``."""
    out: dict[str, object] = {}
    for k in (
        EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
        "runtime_kind",
        "is_simulation",
        EMET_ZMQ_ROBOT_ID_KEY,
        "mjcf_model_name",
        "scene_source_basename",
    ):
        if k not in sess:
            continue
        v = sess[k]
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
    env = sess.get("environment")
    if isinstance(env, dict):
        safe_env: dict[str, object] = {}
        for ek, ev in env.items():
            if isinstance(ev, (str, int, float, bool, type(None))):
                safe_env[str(ek)] = ev
        out["environment"] = safe_env
    caps = sess.get("capabilities")
    if isinstance(caps, dict):
        safe_c: dict[str, object] = {}
        for ck, cv in caps.items():
            if isinstance(cv, (str, int, float, bool, type(None))):
                safe_c[str(ck)] = cv
        out["capabilities"] = safe_c
    return out


def sync_episode_metadata_from_zmq_session(
    robot: object,
    episode_fields: dict[str, object],
    *,
    cli_robot: str | None,
    cli_scene: str,
    cli_split: str,
    cli_index: int,
) -> None:
    """Fill ``episode_fields`` from ``robot.get_emet_session()`` (robot + scene + capabilities)."""
    get_sess = getattr(robot, "get_emet_session", None)
    if not callable(get_sess):
        click.echo(
            "Note: client has no get_emet_session(); episode metadata uses CLI / discovered values only.",
            err=True,
        )
        return
    sess = get_sess()
    if not sess:
        click.echo(
            "Note: no ``emet_session`` on ZMQ messages (restart ``emet serve mujoco`` with current emet for "
            "full metadata). Episode uses CLI / discovered robot and scene flags.",
            err=True,
        )
        return

    episode_fields["emet_session"] = _json_safe_episode_session(sess)
    srv_robot = sess.get(EMET_ZMQ_ROBOT_ID_KEY)
    if isinstance(srv_robot, str) and srv_robot.strip():
        episode_fields["robot"] = str(srv_robot).strip()
    if cli_robot and isinstance(srv_robot, str) and not robot_ids_match(str(srv_robot), cli_robot):
        click.echo(
            f"Warning: server ``emet_session`` robot {srv_robot!r} differs from CLI --robot {cli_robot!r}; "
            "episode.json follows the server.",
            err=True,
        )

    env = sess.get("environment")
    if not isinstance(env, dict) or env.get("kind") != "molmospaces":
        return
    srv_scene = str(env.get("scene", cli_scene))
    srv_split = str(env.get("split", cli_split))
    srv_index = int(env.get("index", cli_index))
    warn: list[str] = []
    if srv_scene != cli_scene:
        warn.append(f"scene (server={srv_scene!r} vs CLI={cli_scene!r})")
    if srv_split != cli_split:
        warn.append(f"split (server={srv_split!r} vs CLI={cli_split!r})")
    if srv_index != cli_index:
        warn.append(f"index (server={srv_index} vs CLI={cli_index})")
    if warn:
        click.echo(
            "Warning: ``emet_session`` MolmoSpaces environment differs from CLI: "
            + "; ".join(warn)
            + ". Episode JSON will follow the server.",
            err=True,
        )
    episode_fields["molmospaces_scene"] = srv_scene
    episode_fields["molmospaces_split"] = srv_split
    episode_fields["molmospaces_index"] = srv_index


@click.command()
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", show_default=True)
@click.option(
    "--robot",
    "robot_backend",
    default=None,
    show_default=False,
    help=(
        "Robot backend (e.g. rby1). If omitted, discover ``emet_robot_id`` from the running ZMQ server "
        "(server must already be publishing). Otherwise must match ``emet serve mujoco --robot``."
    ),
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
    default=-1.5,
    show_default=True,
    help="Random nav goals are spawn-relative (ZMQ); keep range modest so goals do not map to world origin/porch.",
)
@click.option(
    "--goal-x-max",
    type=float,
    default=1.5,
    show_default=True,
    help="Upper X bound for spawn-relative random goals (see --goal-x-min).",
)
@click.option(
    "--goal-y-min",
    type=float,
    default=-1.5,
    show_default=True,
    help="Lower Y bound for spawn-relative random goals.",
)
@click.option(
    "--goal-y-max",
    type=float,
    default=1.5,
    show_default=True,
    help="Upper Y bound for spawn-relative random goals.",
)
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
    robot_backend: str | None,
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

    Then run this command; ``--robot`` is optional if the server publishes ``emet_robot_id`` / ``emet_session``.
    """
    zmq_to = zmq_startup_timeout
    if zmq_to is None:
        env = os.environ.get("EMET_ZMQ_STARTUP_TIMEOUT", "").strip()
        zmq_to = float(env) if env else 60.0

    resolved_robot = (robot_backend or "").strip() or None
    if not resolved_robot:
        click.echo("MolmoSpaces explore: discovering robot from ZMQ (no --robot)…", err=True)
        discovered = discover_zmq_server_robot_id(
            robot_ip,
            port_offset=port_offset,
            timeout=float(zmq_to),
            use_remote_computer=True,
        )
        if not discovered:
            raise click.UsageError(
                "Could not read robot id from ZMQ (timeout). Start ``emet serve mujoco`` first, or pass "
                "``--robot <name>`` explicitly (same as the server)."
            )
        resolved_robot = discovered
        click.echo(f"Using robot from ZMQ server: {resolved_robot!r} (pass --robot to override).")

    click.echo(
        "MolmoSpaces explore: connecting to ZMQ MuJoCo server… "
        "(ensure `emet serve mujoco` is already running with the same --robot and --port-offset.)"
    )
    robot = create_robot_client_from_cli(
        resolved_robot,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=False,
        start_immediately=True,
        zmq_startup_timeout=zmq_startup_timeout,
    )

    parameters = get_parameters("dynav_config.yaml")
    episode_fields: dict[str, object] = {
        "molmospaces_scene": molmospaces_scene,
        "molmospaces_split": molmospaces_split,
        "molmospaces_index": int(molmospaces_index),
        "robot": resolved_robot,
        "steps": int(steps),
        "capture_hz": float(capture_hz),
        "navigate_every": int(navigate_every),
    }
    sync_episode_metadata_from_zmq_session(
        robot,
        episode_fields,
        cli_robot=robot_backend,
        cli_scene=molmospaces_scene,
        cli_split=molmospaces_split,
        cli_index=int(molmospaces_index),
    )

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
        progress_echo=click.echo,
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
        click.echo(f"Progress log: {Path(output_dir) / 'explore_progress.txt'}")
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
