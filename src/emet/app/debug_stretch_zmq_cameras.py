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

"""Passive ZMQ snapshot: full-obs head RGB/depth + servo head/EE RGB (no robot client, no posture).

Use while ``emet serve mujoco --robot stretch`` (or real bridge) is running.

Examples:

  uv run python -m emet.app.debug_stretch_zmq_cameras --out /tmp/cam_dbg
  uv run emet debug-stretch-zmq-cameras --out ./cam_dbg --port-offset 100
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
import zmq

import emet.utils.compression as compression
from emet.utils.logger import Logger
from emet.utils.memory import lookup_address

logger = Logger(__name__)


def _sub_socket(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt(zmq.RCVHWM, 1)
    s.setsockopt(zmq.SNDHWM, 1)
    s.connect(endpoint)
    return s


def _recv_latest(sock: zmq.Socket, poll_timeout_ms: int) -> dict | None:
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    msg = None
    while True:
        ev = dict(poller.poll(poll_timeout_ms))
        if sock not in ev:
            break
        msg = sock.recv_pyobj(flags=zmq.NOBLOCK)
    return msg


def _save_bgr(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"cv2.imwrite failed: {path}")


def _save_depth_vis(path: Path, depth_m: np.ndarray) -> None:
    d = np.asarray(depth_m, dtype=np.float32)
    mask = np.isfinite(d) & (d > 1e-6)
    if not np.any(mask):
        vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    else:
        lo, hi = np.percentile(d[mask], [5, 95])
        hi = max(hi, lo + 1e-3)
        g = ((d - lo) / (hi - lo)).clip(0, 1)
        g = np.where(mask, g, 0.0)
        vis = (np.stack([g, g, g], axis=-1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), vis):
        raise OSError(f"cv2.imwrite failed: {path}")


def _print_K(label: str, K: np.ndarray | None) -> None:
    if K is None:
        print(f"{label}: (missing)")
        return
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    print(f"{label} K (3x3):\n{K}")
    try:
        Kin = np.linalg.inv(K)
        print(f"{label} K^{-1} row0: {Kin[0]}")
    except np.linalg.LinAlgError as e:
        print(f"{label} K is singular: {e}")


@click.command()
@click.option("--robot-ip", "--robot_ip", default="", help="Robot host (default: saved connection or 127.0.0.1)")
@click.option(
    "--local/--remote",
    default=True,
    help="Treat --robot-ip as this machine (local) vs remote lookup (default: local for sim).",
)
@click.option("--port-offset", default=0, type=int, help="Added to default ports 4401 (obs) and 4404 (servo)")
@click.option("--out", "out_dir", type=click.Path(), default="stretch_zmq_cam_debug", help="Output directory")
@click.option("--wait", default=5.0, type=float, help="Seconds to wait per socket for first message")
def main(robot_ip: str, local: bool, port_offset: int, out_dir: str, wait: float) -> None:
    use_remote = not local
    ip = lookup_address(robot_ip, use_remote_computer=use_remote, update=False)
    if ip is None:
        host = (robot_ip or "").strip() or "127.0.0.1"
        ip = f"tcp://{host}"
    obs_port = 4401 + port_offset
    servo_port = 4404 + port_offset
    obs_ep = f"{ip}:{obs_port}"
    servo_ep = f"{ip}:{servo_port}"
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Connecting OBS  {obs_ep}")
    print(f"Connecting SERVO {servo_ep}")
    print(f"Writing to {out}")

    ctx = zmq.Context()
    poll_ms = max(100, int(wait * 1000))
    try:
        obs_sock = _sub_socket(ctx, obs_ep)
        servo_sock = _sub_socket(ctx, servo_ep)

        t_deadline = time.monotonic() + wait
        obs_msg = None
        while time.monotonic() < t_deadline and obs_msg is None:
            obs_msg = _recv_latest(obs_sock, poll_ms)
            if obs_msg is None:
                time.sleep(0.05)

        t_deadline = time.monotonic() + wait
        servo_msg = None
        while time.monotonic() < t_deadline and servo_msg is None:
            servo_msg = _recv_latest(servo_sock, poll_ms)
            if servo_msg is None:
                time.sleep(0.05)

        if obs_msg is None:
            logger.error("No full observation on %s within %.1fs.", obs_ep, wait)
            print("Tip: start the sim first, e.g. `emet serve mujoco --robot stretch --headless`", file=sys.stderr)
            sys.exit(2)

        rgb = compression.from_jpg(obs_msg["rgb"])
        _save_bgr(out / "full_head_rgb.png", rgb)
        print("full_head_rgb:", rgb.shape, rgb.dtype)

        raw_d = obs_msg.get("depth")
        if raw_d is not None:
            depth = compression.from_jp2(raw_d) / 1000.0
            _save_depth_vis(out / "full_head_depth_vis.png", depth)
            print("full_head_depth:", depth.shape, depth.dtype, "finite_frac", float(np.mean(np.isfinite(depth))))
        else:
            depth = None
            print("full_head_depth: (missing)")

        _print_K("full_head", obs_msg.get("camera_K"))
        if obs_msg.get("rgb_width") is not None:
            print("rgb_width / rgb_height:", obs_msg.get("rgb_width"), obs_msg.get("rgb_height"))
        print("control_mode (full obs):", obs_msg.get("control_mode"))

        meta = [
            f"obs_endpoint={obs_ep}",
            f"servo_endpoint={servo_ep}",
            f"full_rgb_shape={getattr(rgb, 'shape', None)}",
            f"full_depth_shape={getattr(depth, 'shape', None)}",
        ]
        (out / "README.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")

        if servo_msg is None:
            logger.warning("No servo message on %s within %.1fs (EE / low-res head skipped).", servo_ep, wait)
            sys.exit(0)

        if servo_msg.get("head_cam/color_image") is not None:
            h_rgb = compression.from_jpg(servo_msg["head_cam/color_image"])
            _save_bgr(out / "servo_head_rgb.png", h_rgb)
            print("servo_head_rgb:", h_rgb.shape, h_rgb.dtype)
            raw_hd = servo_msg.get("head_cam/depth_image")
            if raw_hd is not None:
                hd = compression.from_jp2(raw_hd) / 1000.0
                _save_depth_vis(out / "servo_head_depth_vis.png", hd)
                print("servo_head_depth:", hd.shape)
            _print_K("servo_head", servo_msg.get("head_cam/depth_camera_K"))
        elif servo_msg.get("head_color_image") is not None:
            h_rgb = compression.from_jpg(servo_msg["head_color_image"])
            _save_bgr(out / "servo_head_rgb.png", h_rgb)
            print("servo_head_rgb (legacy keys):", h_rgb.shape, h_rgb.dtype)
            _print_K("servo_head", servo_msg.get("head_camera_K"))
        else:
            print("servo: no head_cam/* or head_color_image keys; keys:", sorted(servo_msg.keys())[:40])

        if servo_msg.get("ee_cam/color_image") is not None:
            ee_rgb = compression.from_jpg(servo_msg["ee_cam/color_image"])
            _save_bgr(out / "servo_ee_rgb.png", ee_rgb)
            print("servo_ee_rgb:", ee_rgb.shape, ee_rgb.dtype)
            raw_ed = servo_msg.get("ee_cam/depth_image")
            if raw_ed is not None:
                ed = compression.from_jp2(raw_ed) / 1000.0
                _save_depth_vis(out / "servo_ee_depth_vis.png", ed)
                print("servo_ee_depth:", ed.shape)
            _print_K("servo_ee", servo_msg.get("ee_cam/depth_camera_K"))
        else:
            print("servo: (no ee_cam/color_image)")

        print("Done.")
    finally:
        ctx.term()


if __name__ == "__main__":
    main()
