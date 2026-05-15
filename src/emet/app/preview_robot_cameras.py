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

"""Horizontal montage of robot cameras (default: head left, head right, arm) for MJCF vs ZMQ checks."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import click
import cv2
import mujoco
import numpy as np
import zmq

import emet.utils.compression as compression
from emet.robots import get_robot_spec
from emet.simulation.scene_resolution import load_default_scene_with_robot
from emet.utils.connection import get_host_from_connection
from emet.utils.discord_bot import read_discord_token_from_env
from emet.utils.memory import lookup_address
from emet.utils.pinhole_intrinsics import apply_pinhole_pixel_ops

_PREVIEW_RW, _PREVIEW_RH = 640, 480


def _robot_host(robot_ip: str) -> str:
    if robot_ip:
        return robot_ip.strip()
    addr = lookup_address("", use_remote_computer=True, update=False)
    if addr and addr.startswith("tcp://"):
        return addr[6:].strip()
    host = get_host_from_connection()
    return host or "127.0.0.1"


def _postprocess_preview_rgb(spec, rgb: np.ndarray) -> np.ndarray:
    """Match :class:`RobosuiteZmqServer` RGB postprocess (ops or ``EMET_ROBOSUITE_RENDER_FLIPUD``)."""
    ops = getattr(spec, "robosuite_rgb_depth_ops", ()) or ()
    if ops:
        return apply_pinhole_pixel_ops(rgb, ops)
    v = os.environ.get("EMET_ROBOSUITE_RENDER_FLIPUD")
    if v and str(v).strip().lower() in ("1", "true", "yes", "on"):
        return np.flipud(rgb).copy()
    return rgb


def _camera_id(model: mujoco.MjModel, name: str) -> int:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    return int(cid) if cid >= 0 else -1


def _render_local_rgb(
    model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, spec, cam_name: str
) -> np.ndarray:
    cam = _camera_id(model, cam_name)
    renderer.update_scene(data, camera=cam)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
    return _postprocess_preview_rgb(spec, rgb)


def _label_strip(img_rgb: np.ndarray, label: str, banner_h: int = 28) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    strip = np.zeros((banner_h, w, 3), dtype=np.uint8)
    strip[:] = (36, 36, 36)
    cv2.putText(strip, label[:80], (6, banner_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return np.vstack([strip, img_rgb])


def _resize_to_height(img_rgb: np.ndarray, row_height: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((row_height, row_height // 2, 3), dtype=np.uint8)
    scale = row_height / float(h)
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img_rgb, (nw, row_height), interpolation=cv2.INTER_AREA)


def build_montage(images: list[np.ndarray], labels: list[str], row_height: int = _PREVIEW_RH) -> np.ndarray:
    parts = [_label_strip(_resize_to_height(im, row_height), lb) for im, lb in zip(images, labels, strict=True)]
    return np.hstack(parts) if parts else np.zeros((row_height + 28, 1, 3), dtype=np.uint8)


def _save_montage_rgb(path: Path, montage_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(montage_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _discord_send_file(path: Path, caption: str) -> None:
    import discord as discord_pkg

    token = read_discord_token_from_env()
    channel_name = os.environ.get("EMET_DISCORD_CHANNEL", "talk-to-stretch")
    intents = discord_pkg.Intents.default()
    client = discord_pkg.Client(intents=intents)
    caption_s = caption[:1900]

    async def _run() -> None:
        @client.event
        async def on_ready() -> None:
            try:
                for guild in client.guilds:
                    for channel in getattr(guild, "text_channels", []) or []:
                        if getattr(channel, "name", None) == channel_name:
                            await channel.send(content=caption_s, file=discord_pkg.File(path))
                            return
                raise RuntimeError(
                    f"No Discord text channel named {channel_name!r} "
                    "(set EMET_DISCORD_CHANNEL to match your server channel)."
                )
            finally:
                await client.close()

        await client.start(token)

    asyncio.run(_run())


def _normalized_robot_key(robot_key: str) -> str:
    return robot_key.lower().replace("-", "_")


def _load_local_merged_preview(robot_key: str, max_cams: int):
    rk = _normalized_robot_key(robot_key)
    spec = get_robot_spec(rk)
    if spec is None:
        raise ValueError(f"Unknown robot {rk!r}. Try: innate_mars, rby1, stretch, …")
    cam_names = list(spec.camera_names)[: max(1, max_cams)]
    if not cam_names:
        raise ValueError(f"No cameras in spec for {rk!r}.")
        model = load_default_scene_with_robot(rk)
    if model is None:
        raise ValueError(f"Could not load default scene + robot MJCF for {rk!r} (missing assets?).")
    return model, spec, cam_names


def _set_hinge_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, radians: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name!r}")
    if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError(f"Joint {joint_name!r}: expected mjJNT_HINGE for this helper")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = float(radians)


def record_head_nod_montage_sequence(
    robot_key: str,
    out_dir: str | Path,
    *,
    joint_name: str = "joint_head",
    angle_low_rad: float = -0.1745,
    angle_high_rad: float = 0.5236,
    n_frames: int = 21,
    max_cams: int = 3,
    row_height: int = _PREVIEW_RH,
    mp4_out: Path | str | None = None,
    video_fps: float = 12.0,
    motion: str = "bounce",
) -> list[Path]:
    """Sweep a head hinge angle, save one labeled camera montage per frame (PNG). Optional mp4 stitch.

    ``motion``:
    - ``once``: monotonic ``angle_low_rad`` → ``angle_high_rad``
    - ``bounce``: same as once on the way up, then back to ``angle_low_rad`` (one full nod)

    Returns paths to PNGs in capture order."""
    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2 for a nod sweep, got {n_frames}")
    if motion not in ("once", "bounce"):
        raise ValueError("motion must be 'once' or 'bounce'")

    model, spec, cam_names = _load_local_merged_preview(robot_key, max_cams)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(
            f"Joint {joint_name!r} not in merged MJCF; nod capture needs a kinematic head (e.g. innate_mars)."
        )

    rng = model.jnt_range[jid].copy()
    lo, hi = float(rng[0]), float(rng[1])
    if hi > lo:
        clip_lo = max(angle_low_rad, lo)
        clip_hi = min(angle_high_rad, hi)
        if clip_lo >= clip_hi:
            raise ValueError(
                f"Nod angles [{angle_low_rad}, {angle_high_rad}] rad do not overlap joint limits ({lo}, {hi})."
            )
    else:
        clip_lo, clip_hi = float(angle_low_rad), float(angle_high_rad)

    if motion == "bounce":
        if int(n_frames) < 3:
            raise ValueError("motion='bounce' requires n_frames >= 3")
        n_up = max(2, int(np.ceil(float(n_frames) / 2.0)))
        n_down = max(2, int(n_frames) - n_up + 1)
        angles_up = np.linspace(clip_lo, clip_hi, n_up, dtype=np.float64)
        angles_dn = np.linspace(clip_hi, clip_lo, n_down, dtype=np.float64)[1:]
        angles = np.concatenate([angles_up, angles_dn])
        assert angles.size == int(n_frames), f"bounce nod length {angles.size} != {int(n_frames)}"
    else:
        angles = np.linspace(clip_lo, clip_hi, max(2, int(n_frames)), dtype=np.float64)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=_PREVIEW_RH, width=_PREVIEW_RW)
    written: list[Path] = []
    bgr_frames: list[np.ndarray] = []
    try:
        for i, ang in enumerate(angles):
            _set_hinge_joint_qpos(model, data, joint_name, float(ang))
            mujoco.mj_forward(model, data)
            imgs = [_render_local_rgb(model, data, renderer, spec, n) for n in cam_names]
            labels = [f"{n}  {joint_name}={ang:.3f}rad" for n in cam_names]
            montage_rgb = build_montage(imgs, labels, row_height=row_height)
            png_path = out_p / f"montage_{i:04d}.png"
            _save_montage_rgb(png_path, montage_rgb)
            written.append(png_path.resolve())
            if mp4_out is not None:
                bgr_frames.append(cv2.cvtColor(montage_rgb, cv2.COLOR_RGB2BGR))
    finally:
        renderer.close()

    if mp4_out is not None and bgr_frames:
        outp = Path(mp4_out).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        mh, mw = bgr_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), float(video_fps), (mw, mh))
        if not vw.isOpened():
            raise ValueError(f"Could not open VideoWriter for {outp}")
        try:
            for fr in bgr_frames:
                vw.write(fr)
        finally:
            vw.release()

    return written


def _local_frames(robot_key: str, max_cams: int):
    model, spec, names = _load_local_merged_preview(robot_key, max_cams)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=_PREVIEW_RH, width=_PREVIEW_RW)
    imgs: list[np.ndarray] = []
    try:
        for n in names:
            imgs.append(_render_local_rgb(model, data, renderer, spec, n))
    finally:
        renderer.close()
    return imgs, names


def _recv_zmq_obs(host: str, recv_port: int, timeout_ms: int) -> dict | None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{recv_port}")
    time.sleep(0.25)
    try:
        raw = sock.recv_pyobj(flags=0)
    except zmq.Again:
        return None
    finally:
        sock.close(0)
        ctx.term()
    if not isinstance(raw, dict):
        return None
    return raw


def _decode_obs_message(raw: dict, spec_names: list[str]) -> tuple[list[np.ndarray], list[str]]:
    imgs: list[np.ndarray] = []
    labels: list[str] = []
    try:
        if raw.get("rgb") is None:
            return imgs, labels
        imgs.append(compression.from_jpg(raw["rgb"]))
        labels.append(spec_names[0] if spec_names else "rgb")

        if raw.get("rgb_right") is not None:
            imgs.append(compression.from_jpg(raw["rgb_right"]))
            labels.append(spec_names[1] if len(spec_names) > 1 else "rgb_right")

        tertiary_buf = raw.get("rgb_tertiary")
        tertiary_name = raw.get("camera_name_tertiary")
        if tertiary_buf is not None:
            imgs.append(compression.from_jpg(tertiary_buf))
            labels.append(str(tertiary_name or (spec_names[2] if len(spec_names) > 2 else "tertiary")))
        elif len(spec_names) >= 3:
            missing = np.zeros((_PREVIEW_RH, _PREVIEW_RW // 2, 3), dtype=np.uint8)
            missing[:] = (32, 32, 32)
            imgs.append(missing)
            labels.append(f"{spec_names[2]} (no rgb_tertiary; restart bridge sim)")
    except Exception:
        return [], []
    return imgs, labels


def _zmq_frames(robot_key: str, host: str, recv_port: int, timeout_ms: int, max_cams: int):
    spec = get_robot_spec(robot_key)
    if spec is None:
        raise click.ClickException(f"Unknown robot {robot_key!r}.")
    spec_names = list(spec.camera_names)
    raw = _recv_zmq_obs(host, recv_port, timeout_ms)
    if raw is None:
        raise click.ClickException(
            f"No observation on tcp://{host}:{recv_port} within {timeout_ms} ms "
            "(is the ZMQ bridge running on the observation port?)."
        )
    imgs, labels = _decode_obs_message(raw, spec_names)
    if not imgs:
        raise click.ClickException("Observation had no decodable RGB.")
    while len(labels) < len(imgs):
        labels.append(f"cam_{len(labels)}")
    if len(imgs) > max_cams:
        imgs, labels = imgs[:max_cams], labels[:max_cams]
    return imgs, labels


@click.command()
@click.option("--robot", "robot_key", default="innate_mars", help="Robot key (same as serve mujoco --robot)")
@click.option(
    "--source",
    type=click.Choice(["local", "zmq"]),
    default="local",
    help="Render merged default scene MJCF locally, or grab one observation over ZMQ",
)
@click.option("--robot-ip", "--robot_ip", "robot_ip", default="", help="ZMQ host (default: active connection)")
@click.option(
    "--recv-port",
    "recv_port",
    default=4401,
    type=int,
    help="ZMQ SUB port for full observations (default 4401, same as GenericZmqClient)",
)
@click.option("--timeout-ms", default=9000, type=int, help="ZMQ receive timeout")
@click.option("--max-cams", default=3, type=int, help="Montage up to this many cameras (default 3)")
@click.option("--row-height", default=_PREVIEW_RH, type=int, help="Resized RGB row height inside montage")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="PNG path (default: ./robot_cam_preview_<robot>.png)",
)
@click.option("--discord/--no-discord", default=False, help="Post PNG to Discord (DISCORD_TOKEN, EMET_DISCORD_CHANNEL)")
@click.option("--caption", default="", help="Discord message text (short)")
@click.option("--nod", is_flag=True, help="Local-only: sweep a head hinge and save one montage PNG per pose")
@click.option(
    "--nod-out-dir",
    "nod_out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for --nod PNGs (default: ./robot_cam_nod_<robot>)",
)
@click.option("--nod-joint", "nod_joint", default="joint_head", show_default=True)
@click.option("--nod-low", "nod_low", type=float, default=-0.1745, show_default=True, help="Start angle (rad)")
@click.option("--nod-high", "nod_high", type=float, default=0.5236, show_default=True, help="End angle (rad)")
@click.option("--nod-frames", "nod_frames", type=int, default=21, show_default=True)
@click.option(
    "--nod-video",
    "nod_video",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional montage mp4 path (opencv mp4v); same frames as PNGs",
)
@click.option(
    "--nod-motion",
    "nod_motion",
    type=click.Choice(["bounce", "once"]),
    default="bounce",
    show_default=True,
    help="bounce: low→high→low; once: low→high only",
)
@click.option("--nod-fps", "nod_fps", type=float, default=12.0, show_default=True)
def main(
    robot_key: str,
    source: str,
    robot_ip: str,
    recv_port: int,
    timeout_ms: int,
    max_cams: int,
    row_height: int,
    out_path: Path | None,
    discord: bool,
    caption: str,
    nod: bool,
    nod_out_dir: Path | None,
    nod_joint: str,
    nod_low: float,
    nod_high: float,
    nod_frames: int,
    nod_video: Path | None,
    nod_fps: float,
    nod_motion: str,
) -> None:
    """Save a labeled horizontal montage of robot cameras (MJCF preview or live ZMQ frame).

    For ZMQ, uses the observation stream (default port 4401). After upgrading, the simulation server
    also sends rgb_tertiary for the third listed camera when present.

    Examples:

        emet preview-cameras

        emet preview-cameras --source zmq --robot-ip 192.168.1.43

        emet preview-cameras --discord --caption "mars cams smoke test"

        emet preview-cameras --nod --nod-out-dir ./nod_run --nod-frames 31 --nod-video ./nod.mp4
    """
    robot_key_norm = _normalized_robot_key(robot_key)

    try:
        if nod:
            if source != "local":
                raise click.UsageError("--nod requires --source local.")
            if discord:
                raise click.UsageError("--discord is not supported with --nod.")

            dest_dir = (
                nod_out_dir.resolve()
                if nod_out_dir is not None
                else (Path.cwd() / f"robot_cam_nod_{robot_key_norm}").resolve()
            )
            vid = nod_video.resolve() if nod_video is not None else None
            seq = record_head_nod_montage_sequence(
                robot_key_norm,
                dest_dir,
                joint_name=nod_joint,
                angle_low_rad=nod_low,
                angle_high_rad=nod_high,
                n_frames=nod_frames,
                max_cams=max_cams,
                row_height=row_height,
                mp4_out=vid,
                video_fps=nod_fps,
                motion=nod_motion,
            )
            click.echo(f"Wrote {len(seq)} montage PNGs under {dest_dir}")
            if vid is not None:
                click.echo(f"Video: {vid}")

            prv = os.environ.get("EMET_PREVIEW_CAMERAS_OPENCV") or os.environ.get("EMET_OPENCV_PREVIEW")
            if prv and str(prv).strip().lower() in ("1", "true", "yes", "on") and seq:
                last = cv2.imread(str(seq[-1]))
                if last is not None:
                    cv2.imshow("preview_robot_cameras_last_nod_frame", last)
                    click.echo("OpenCV preview: press a key to close.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            return

        if source == "local":
            imgs, labels = _local_frames(robot_key_norm, max_cams)
        else:
            host = _robot_host(robot_ip)
            imgs, labels = _zmq_frames(robot_key_norm, host, recv_port, timeout_ms, max_cams)

        montage = build_montage(imgs, labels, row_height=row_height)

        dest = out_path if out_path is not None else Path.cwd() / f"robot_cam_preview_{robot_key_norm}.png"
        _save_montage_rgb(dest, montage)
        click.echo(f"Wrote {dest.resolve()} ({montage.shape[1]}×{montage.shape[0]})")

        if discord:
            cap = caption.strip() if caption.strip() else f"Robot camera montage ({robot_key_norm}, source={source})"
            _discord_send_file(dest, cap)
            click.echo("Posted to Discord.")

        preview = os.environ.get("EMET_PREVIEW_CAMERAS_OPENCV") or os.environ.get("EMET_OPENCV_PREVIEW")
        if preview and str(preview).strip().lower() in ("1", "true", "yes", "on"):
            cv2.imshow("preview_robot_cameras", cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            click.echo("OpenCV preview: press a key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except ValueError as e:
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
