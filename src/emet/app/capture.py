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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""One-shot ZMQ capture: camera montage, metadata JSON, optional single-frame DynaMem map."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from click.core import ParameterSource

import emet.utils.compression as compression
from emet.app.preview_robot_cameras import (
    _decode_obs_message,
    _recv_zmq_obs,
    _save_montage_rgb,
    build_montage,
)
from emet.app.stream_agent_factory import create_dynamem_agent, resolve_stream_dynav_config, stream_stats
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML, get_robot_spec
from emet.utils.connection import get_connection, get_host_from_connection


def _resolve_host(robot_ip: str, connection_name: str | None, *, ip_from_default: bool) -> str:
    if not ip_from_default and robot_ip.strip():
        return robot_ip.strip()
    if connection_name:
        host = get_host_from_connection(connection_name)
        if host:
            return host.strip()
    if ip_from_default:
        host = get_host_from_connection()
        if host:
            return host.strip()
    return robot_ip.strip() or "127.0.0.1"


def _resolve_robot(robot: str, connection_name: str | None, *, robot_from_default: bool) -> str:
    if not robot_from_default:
        return robot.lower().replace("-", "_")
    conn = get_connection(connection_name) if connection_name else get_connection()
    if conn and conn.get("robot"):
        return str(conn["robot"]).lower().replace("-", "_")
    return robot.lower().replace("-", "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _rgb_channel_means(rgb: np.ndarray) -> dict[str, float]:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {}
    return {
        "r_mean": float(arr[..., 0].mean()),
        "g_mean": float(arr[..., 1].mean()),
        "b_mean": float(arr[..., 2].mean()),
    }


def _obs_metadata(raw: dict[str, Any], *, robot: str, host: str, recv_port: int) -> dict[str, Any]:
    skip_keys = {"rgb", "rgb_right", "rgb_tertiary", "depth", "sensor_depth", "segmentation"}
    meta: dict[str, Any] = {
        "robot": robot,
        "host": host,
        "recv_port": recv_port,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in raw.items():
        if key in skip_keys:
            continue
        meta[key] = _json_safe(value)
    if raw.get("rgb") is not None:
        try:
            meta["rgb_channel_means"] = _rgb_channel_means(compression.from_jpg(raw["rgb"]))
        except Exception:
            pass
    return meta


def _save_capture_images(
    raw: dict[str, Any],
    spec_names: list[str],
    out_dir: Path,
) -> tuple[list[np.ndarray], list[str], dict[str, Path]]:
    imgs, labels = _decode_obs_message(raw, spec_names)
    if not imgs:
        raise click.ClickException("Observation had no decodable RGB.")
    saved: dict[str, Path] = {}
    image_keys = ["rgb", "rgb_right", "rgb_tertiary"]
    for idx, (img, label) in enumerate(zip(imgs, labels, strict=False)):
        key = image_keys[idx] if idx < len(image_keys) else f"cam_{idx}"
        stem = label.replace(" ", "_").replace("/", "_")
        path = out_dir / f"{key}_{stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved[key] = path
    montage_path = out_dir / "montage.png"
    _save_montage_rgb(montage_path, build_montage(imgs, labels))
    saved["montage"] = montage_path
    return imgs, labels, saved


def _run_map_update(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    no_rerun: bool,
    headless: bool,
    rerun_hold_s: float,
    cpu_only: bool,
    map_only: bool = False,
    dynav_from_default: bool = True,
) -> dict[str, Any]:
    dynav_resolved = resolve_stream_dynav_config(robot, host, dynav_config, dynav_from_default=dynav_from_default)
    if dynav_resolved != dynav_config and dynav_from_default:
        click.echo(f"Dynav: using {dynav_resolved!r} for {robot} @ {host} (hardware ZMQ has no depth → DA3/auto).")
    agent, dynav_resolved = create_dynamem_agent(
        robot=robot,
        host=host,
        port_offset=port_offset,
        dynav_config=dynav_resolved,
        enable_rerun=not no_rerun,
        headless=headless,
        cpu_only=cpu_only,
        map_only=map_only,
    )
    click.echo("Running one DynaMem update…")
    agent.update()
    map_stats = stream_stats(agent, "dynamem", dynav_resolved=dynav_resolved)

    if not no_rerun and rerun_hold_s > 0:
        url = "http://localhost:9090?url=ws://localhost:9877"
        click.echo(f"Rerun viewer: {url}  (holding {rerun_hold_s:.0f}s)")
        time.sleep(float(rerun_hold_s))
    return map_stats


@click.command("capture")
@click.option(
    "--ip",
    "--robot-ip",
    "robot_ip",
    default="127.0.0.1",
    show_default=True,
    help="ZMQ host (sim on localhost, or robot hostname/IP)",
)
@click.option("--connection", "-c", "connection_name", default=None, help="Saved connection profile (host/robot)")
@click.option(
    "--robot",
    default="stretch",
    show_default=True,
    help="Robot backend (stretch, innate_mars, rby1, galaxea_r1, …)",
)
@click.option("--port-offset", default=0, type=int, show_default=True, help="Add to default ZMQ ports (4401+)")
@click.option("--recv-port", default=None, type=int, help="Observation SUB port (default 4401 + port-offset)")
@click.option("--timeout-ms", default=9000, type=int, show_default=True, help="ZMQ receive timeout")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: runs/capture/<timestamp>)",
)
@click.option(
    "--map", "run_map", is_flag=True, help="After capture, run one DynaMem update (+ Rerun unless --no-rerun)"
)
@click.option(
    "--dynav-config",
    "--dynav_config",
    default=DEFAULT_DYNAV_CONFIG_YAML,
    show_default=True,
    help="DynaMem YAML when --map (use dynav_innate_mars.yaml for hardware Mars without ZMQ depth)",
)
@click.option("--no-rerun", is_flag=True, help="Disable Rerun when --map is set")
@click.option("--headless", is_flag=True, help="Rerun without opening a browser (when --map)")
@click.option("--rerun-hold-s", default=30.0, show_default=True, help="Seconds to keep Rerun open after --map")
@click.option("--cpu-only", is_flag=True, help="CPU-only models for --map (skip heavy GPU VLMs)")
@click.option(
    "--map-only",
    is_flag=True,
    help="With --map: voxel/obstacle map only (no SigLIP/YoloE/VLM; DA3 when depth missing)",
)
@click.pass_context
def main(
    ctx: click.Context,
    robot_ip: str,
    connection_name: str | None,
    robot: str,
    port_offset: int,
    recv_port: int | None,
    timeout_ms: int,
    out_dir: Path | None,
    run_map: bool,
    dynav_config: str,
    no_rerun: bool,
    headless: bool,
    rerun_hold_s: float,
    cpu_only: bool,
    map_only: bool,
) -> None:
    """Grab one ZMQ observation, save camera montage + metadata, optionally build a one-frame map.

    Works with any robot that exposes the standard ZMQ observation port (sim on localhost or hardware bridge).

    Examples:
      emet capture
      emet capture --robot innate_mars --ip herman
      emet capture --connection herman --map --dynav-config dynav_innate_mars.yaml
      emet capture --robot stretch --map --no-rerun
    """
    ip_from_default = ctx.get_parameter_source("robot_ip") == ParameterSource.DEFAULT
    robot_from_default = ctx.get_parameter_source("robot") == ParameterSource.DEFAULT
    dynav_from_default = ctx.get_parameter_source("dynav_config") == ParameterSource.DEFAULT
    host = _resolve_host(robot_ip, connection_name, ip_from_default=ip_from_default)
    robot_key = _resolve_robot(robot, connection_name, robot_from_default=robot_from_default)

    spec = get_robot_spec(robot_key)
    if spec is None:
        raise click.ClickException(f"Unknown robot {robot_key!r}.")

    obs_port = int(recv_port if recv_port is not None else 4401 + int(port_offset))
    spec_names = list(spec.camera_names)

    click.echo(f"Capturing from tcp://{host}:{obs_port} ({robot_key})…")
    raw = _recv_zmq_obs(host, obs_port, timeout_ms)
    if raw is None:
        raise click.ClickException(
            f"No observation on tcp://{host}:{obs_port} within {timeout_ms} ms (start sim or bridge first)."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir or Path("runs") / "capture" / f"{robot_key}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    _, _, saved_paths = _save_capture_images(raw, spec_names, dest)
    meta = _obs_metadata(raw, robot=robot_key, host=host, recv_port=obs_port)
    meta_path = dest / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    click.echo(f"Wrote {dest.resolve()}/")
    for name, path in saved_paths.items():
        click.echo(f"  {name}: {path.name}")

    if run_map:
        map_stats = _run_map_update(
            robot=robot_key,
            host=host,
            port_offset=port_offset,
            dynav_config=dynav_config,
            no_rerun=no_rerun,
            headless=headless,
            rerun_hold_s=rerun_hold_s,
            cpu_only=cpu_only,
            map_only=map_only,
            dynav_from_default=dynav_from_default,
        )
        meta["map"] = map_stats
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        click.echo(
            f"Map: {map_stats.get('n_voxel_observations', 0)} observations, "
            f"{map_stats.get('n_voxel_explored_cells', 0)} explored cells"
        )
