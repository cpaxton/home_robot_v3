# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Fast ZMQ + DA3 depth / point-cloud debug stream (Rerun). Matches DynamemController depth resolution."""

from __future__ import annotations

import os
import time
from typing import Any

import click
import cv2
import numpy as np

from emet.app.robot_cli import create_robot_client_from_cli
from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.perception.depth.da3_estimator import (
    DA3DepthEstimator,
    apply_da3_sky_row_mask,
    resolve_depth_map,
)


def _depth_colorize(depth: np.ndarray, near: float = 0.15, far: float = 6.0) -> np.ndarray:
    """HxW float depth → HxWx3 uint8 RGB colormap for Rerun."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    finite = d[d > 1e-6]
    if finite.size > 50:
        near = float(np.percentile(finite, 5))
        far = float(np.percentile(finite, 95))
        far = max(far, near + 0.5)
    u8 = (np.clip((d - near) / (far - near), 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _depth_to_world_points(
    depth: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    rgb: np.ndarray | None,
    *,
    stride: int,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subsampled camera-frame unprojection → world (cam_to_world is 4x4)."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Twc = np.asarray(cam_to_world, dtype=np.float64).reshape(4, 4)
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for v in range(0, H, stride):
        for u in range(0, W, stride):
            z = float(depth[v, u])
            if not np.isfinite(z) or z < z_min or z > z_max:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            p_c = np.array([x, y, z, 1.0], dtype=np.float64)
            p_w = (Twc @ p_c)[:3]
            pts.append(p_w.astype(np.float32))
            if rgb is not None and rgb.ndim == 3:
                cols.append(np.asarray(rgb[v, u], dtype=np.uint8))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32), None
    pc = np.stack(pts, axis=0)
    rgb_out = np.stack(cols, axis=0) if cols else None
    return pc, rgb_out


@click.command(
    "debug-da3-depth",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot", default="innate_mars", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option(
    "--model-id",
    default="depth-anything/DA3-SMALL",
    show_default=True,
    help="Hugging Face repo id. DA3-SMALL is faster for stereo; DA3METRIC-LARGE is heavier.",
)
@click.option("--process-res", default=378, type=int, show_default=True, help="Internal DA3 resolution (smaller = faster).")
@click.option("--hz", default=5.0, type=float, show_default=True, help="Poll rate (frames per second cap).")
@click.option("--stride", default=10, type=int, show_default=True, help="Point-cloud pixel stride (larger = fewer points).")
@click.option("--max-frames", default=0, type=int, help="Stop after N frames (0 = until Ctrl+C).")
@click.option("--cpu-only", is_flag=True, help="Force DA3 on CPU.")
@click.option(
    "--spawn/--no-spawn",
    default=False,
    show_default=True,
    help="Native Rerun window (spawn) vs web viewer only (serve on ws://…:9877).",
)
@click.option(
    "--meshes/--no-meshes",
    default=True,
    show_default=True,
    help="Log robot visual meshes (MJCF) in world frame for alignment with the point cloud. Requires mujoco.",
)
@click.option(
    "--depth-source",
    type=click.Choice(["da3", "sensor", "auto"]),
    default="da3",
    show_default=True,
    help="Same as dynav depth_source: sensor uses sim depth if present; da3 runs DA3.",
)
@click.option(
    "--sky-fraction-top",
    default=0.0,
    type=float,
    show_default=True,
    help="Zero top fraction of rows in DA3 depth (matches dynav da3_ignore_sky_fraction_top; e.g. 0.16).",
)
@click.option(
    "--clip-depth-max-m",
    default=4.0,
    type=float,
    show_default=True,
    help="Clip DA3 output depth to this max (meters); aligns with dynav da3_clip_max_m.",
)
def main(
    robot_ip: str,
    robot: str,
    port_offset: int,
    model_id: str,
    process_res: int,
    hz: float,
    stride: int,
    max_frames: int,
    cpu_only: bool,
    spawn: bool,
    depth_source: str,
    show_meshes: bool,
    sky_fraction_top: float,
    clip_depth_max_m: float,
) -> None:
    """Stream head camera(s) from the ZMQ server through DA3 and log RGB, depth, and a light point cloud to Rerun.

    Start the sim first, e.g. ``emet serve mujoco --robot innate_mars --headless``.

    Examples:

        emet debug-da3-depth --robot innate_mars

        emet debug-da3-depth --model-id depth-anything/DA3METRIC-LARGE --process-res 504

        emet debug-da3-depth --depth-source sensor   # MuJoCo rendered depth (no DA3)
    """
    try:
        import rerun as rr
    except ImportError as e:
        click.echo("rerun-sdk required. Install project deps (uv sync).", err=True)
        raise SystemExit(1) from e

    mesh_logger: object | None = None
    if show_meshes:
        try:
            from emet.robots import get_robot_spec
            from emet.visualization.mjcf_rerun_robot import MjcfVisualMeshLogger

            rspec = get_robot_spec(robot)
            if rspec is not None and rspec.mjcf_path:
                mesh_logger = MjcfVisualMeshLogger(
                    rspec.mjcf_path,
                    rspec.joint_names,
                    rspec.dof,
                    rspec.base_link_name,
                )
                click.echo(f"Mesh overlay: {rspec.mjcf_path} (turn off with --no-meshes).")
        except Exception as e:
            click.echo(f"Mesh overlay disabled ({e!r}). Use --no-meshes to silence.", err=True)
            mesh_logger = None

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if depth_source in ("da3", "auto") and cpu_only is False and torch is not None and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    rr.init("da3_depth_debug", spawn=spawn)
    if not spawn:
        rr.serve(open_browser=not os.environ.get("RERUN_HEADLESS", "").strip())
        host = os.environ.get("RERUN_SERVE_HOST", "127.0.0.1")
        click.echo(
            f"Rerun web: http://{host}:9090?url=ws://{host}:9877 "
            "(set RERUN_HEADLESS=1 to skip opening a browser)"
        )

    est: DA3DepthEstimator | None = None
    if depth_source in ("da3", "auto"):
        est = DA3DepthEstimator(
            model_id=model_id,
            device=device,
            process_res=process_res,
            clip_output_max_m=float(clip_depth_max_m),
        )

    robot_client = create_robot_client_from_cli(
        robot,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=False,
        start_immediately=True,
        allow_missing_depth=True,
    )

    dt = 1.0 / max(float(hz), 0.25)
    frame = 0
    t_next = time.monotonic()
    click.echo(
        f"DA3 debug: depth_source={depth_source} model={model_id!r} process_res={process_res} device={device}. "
        "Ctrl+C to stop."
    )

    try:
        while True:
            now = time.monotonic()
            if now < t_next:
                time.sleep(min(0.05, t_next - now))
                continue
            t_next = time.monotonic() + dt

            obs = robot_client.get_observation()
            if obs is None:
                click.echo("No observation yet (waiting for ZMQ)...", err=True)
                time.sleep(0.2)
                continue

            rgb = obs.rgb
            K = obs.camera_K
            pose = obs.camera_pose
            depth = resolve_depth_map(
                depth_source,
                est,
                rgb,
                obs.depth,
                K,
                pose,
                getattr(obs, "head_rgb_right", None),
                getattr(obs, "head_camera_K_right", None),
                getattr(obs, "head_camera_pose_right", None),
            )
            if depth is not None and sky_fraction_top > 0:
                depth = apply_da3_sky_row_mask(np.asarray(depth, dtype=np.float32), sky_fraction_top)

            rr.set_time_sequence("frame", frame)

            if mesh_logger is not None:
                pose_d: dict[str, Any] = {}
                if getattr(obs, "joint", None) is not None:
                    pose_d["joint"] = obs.joint
                if getattr(obs, "gps", None) is not None:
                    pose_d["gps"] = obs.gps
                if getattr(obs, "compass", None) is not None:
                    pose_d["compass"] = obs.compass
                if getattr(obs, "emet_session", None) is not None:
                    pose_d[EMET_ZMQ_SESSION_KEY] = obs.emet_session
                if pose_d:
                    mesh_logger.log_meshes_world(rr, pose_d, entity_prefix="da3/robot/mesh")

            if rgb is not None:
                rr.log("da3/left_rgb", rr.Image(rgb))

            if depth is None:
                click.echo(f"frame {frame}: no depth map (depth_source={depth_source!r})", err=True)
            else:
                dvis = _depth_colorize(depth)
                rr.log("da3/depth_colormap", rr.Image(dvis))
                finite = depth[np.isfinite(depth) & (depth > 1e-6)]
                stats = (
                    float(np.min(finite)),
                    float(np.max(finite)),
                    float(np.mean(finite)),
                    float(np.median(finite)),
                ) if finite.size else (0.0, 0.0, 0.0, 0.0)
                click.echo(
                    f"frame {frame}: depth min/max/mean/median = {stats[0]:.3f} / {stats[1]:.3f} / "
                    f"{stats[2]:.3f} / {stats[3]:.3f} m | valid px {int(finite.size)}/{depth.size}"
                )

                if K is not None and pose is not None and np.asarray(K).shape == (3, 3):
                    pc, c_rgb = _depth_to_world_points(
                        depth,
                        np.asarray(K, dtype=np.float64),
                        np.asarray(pose, dtype=np.float64),
                        rgb,
                        stride=stride,
                        z_min=0.05,
                        z_max=12.0,
                    )
                    if pc.shape[0] > 0:
                        if c_rgb is not None:
                            rr.log(
                                "da3/points_world",
                                rr.Points3D(positions=pc, colors=c_rgb, radii=0.012),
                            )
                        else:
                            rr.log("da3/points_world", rr.Points3D(positions=pc, radii=0.012))

            frame += 1
            if max_frames and frame >= max_frames:
                break
    except KeyboardInterrupt:
        click.echo("Stopped.")
    finally:
        try:
            robot_client.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
