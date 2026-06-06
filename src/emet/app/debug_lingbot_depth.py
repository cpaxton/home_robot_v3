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

"""ZMQ + LingBot-Map depth/pose debug stream (Rerun). Uses subprocess batch infer in lingbot venv."""

from __future__ import annotations

import os
import time
from typing import Any

import click
import cv2
import numpy as np

from emet.app.robot_cli import create_robot_client_from_cli
from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.perception.depth.da3_estimator import resolve_depth_map, resolve_depth_map_uses_observation_sensor_only
from emet.perception.depth.lingbot_estimator import LingBotDepthEstimator, create_lingbot_estimator_from_parameters


def _depth_colorize(depth: np.ndarray, near: float = 0.15, far: float = 6.0) -> np.ndarray:
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
    "debug-lingbot-depth",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot", default="innate_mars", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option("--hz", default=2.0, type=float, show_default=True, help="Poll rate (LingBot infer is heavy).")
@click.option("--infer-every-n", default=2, type=int, show_default=True)
@click.option("--max-frames", default=0, type=int, help="Stop after N frames (0 = until Ctrl+C).")
@click.option("--stride", default=10, type=int, show_default=True)
@click.option(
    "--depth-source",
    type=click.Choice(["lingbot", "sensor"]),
    default="lingbot",
    show_default=True,
)
@click.option("--use-lingbot-pose/--sim-pose", default=True, show_default=True)
@click.option("--spawn/--no-spawn", default=False, show_default=True)
@click.option("--meshes/--no-meshes", default=True, show_default=True)
def main(
    robot_ip: str,
    robot: str,
    port_offset: int,
    hz: float,
    infer_every_n: int,
    max_frames: int,
    stride: int,
    depth_source: str,
    use_lingbot_pose: bool,
    spawn: bool,
    meshes: bool,
) -> None:
    """Stream Mars sim RGB through LingBot-Map (subprocess) and log depth/pose to Rerun."""
    try:
        import rerun as rr
    except ImportError as e:
        click.echo("rerun-sdk required.", err=True)
        raise SystemExit(1) from e

    params = {
        "lingbot_infer_every_n": infer_every_n,
        "lingbot_keyframe_interval": 2,
        "lingbot_use_pose": use_lingbot_pose,
        "lingbot_use_sdpa": True,
    }
    ckpt = os.environ.get("LINGBOT_MAP_CHECKPOINT", "")
    if ckpt:
        params["lingbot_checkpoint"] = ckpt

    lingbot_est: LingBotDepthEstimator | None = None
    if depth_source == "lingbot":
        lingbot_est = LingBotDepthEstimator(
            create_lingbot_estimator_from_parameters(params),
            use_lingbot_pose=use_lingbot_pose,
        )

    mesh_logger: object | None = None
    if meshes:
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
        except Exception as e:
            click.echo(f"Mesh overlay disabled: {e!r}", err=True)

    rr.init("lingbot_depth_debug", spawn=spawn)
    if not spawn:
        rr.serve(open_browser=not os.environ.get("RERUN_HEADLESS", "").strip())

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
        f"LingBot debug: depth_source={depth_source} infer_every_n={infer_every_n} "
        f"use_lingbot_pose={use_lingbot_pose}. Ctrl+C to stop."
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
                time.sleep(0.2)
                continue

            rgb = obs.rgb
            K = obs.camera_K
            pose = obs.camera_pose
            depth = None
            if depth_source == "sensor":
                depth = resolve_depth_map("sensor", None, rgb, obs.depth, K, pose)
            elif lingbot_est is not None:
                depth = lingbot_est.infer(
                    rgb,
                    camera_K=K,
                    camera_pose=pose,
                    force=(frame == 0),
                )
                if use_lingbot_pose and lingbot_est.last_camera_pose is not None:
                    pose = lingbot_est.last_camera_pose
                if lingbot_est.last_camera_K is not None:
                    K = lingbot_est.last_camera_K

            rr.set_time_sequence("frame", frame)
            if rgb is not None:
                rr.log("lingbot/left_rgb", rr.Image(rgb))

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
                    mesh_logger.log_meshes_world(rr, pose_d, entity_prefix="lingbot/robot/mesh")

            if depth is None:
                click.echo(f"frame {frame}: no depth", err=True)
            else:
                rr.log("lingbot/depth_colormap", rr.Image(_depth_colorize(depth)))
                if K is not None and pose is not None:
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
                        ent = "lingbot/points_world"
                        if c_rgb is not None:
                            rr.log(ent, rr.Points3D(positions=pc, colors=c_rgb, radii=0.012))
                        else:
                            rr.log(ent, rr.Points3D(positions=pc, radii=0.012))

            if (
                depth_source == "sensor"
                and obs.depth is not None
                and not resolve_depth_map_uses_observation_sensor_only("sensor", obs.depth)
            ):
                pass

            frame += 1
            if max_frames and frame >= max_frames:
                break
    except KeyboardInterrupt:
        click.echo("Stopped.")
    finally:
        if lingbot_est is not None:
            lingbot_est.close()
        try:
            robot_client.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
