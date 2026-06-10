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

"""Record Innate Mars sim ZMQ stream + rotate_in_place for LingBot-Map offline eval."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
from PIL import Image

from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.core.parameters import get_parameters


def _np_to_jsonable(x: np.ndarray | None):
    if x is None:
        return None
    return np.asarray(x, dtype=float).tolist()


@click.command()
@click.option("--output", type=click.Path(), required=True, help="Episode output directory")
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot", default="innate_mars", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option(
    "--motion",
    type=click.Choice(["base_spin", "rotate_in_place", "poll_only"]),
    default="base_spin",
    show_default=True,
    help="base_spin: relative base rotation + poll (fast, no Dynamem). "
    "rotate_in_place: full DynamemTaskExecutor. poll_only: ZMQ poll only.",
)
@click.option(
    "--spin-steps", default=16, type=int, show_default=True, help="Steps for base_spin (360/spin_steps per step)."
)
@click.option(
    "--spin-settle-s", default=1.5, type=float, show_default=True, help="Seconds to poll after each spin step."
)
@click.option("--poll-hz", default=10.0, type=float, show_default=True, help="Recording rate when poll_only.")
@click.option("--max-frames", default=0, type=int, show_default=True, help="Stop after N frames (0 = motion default).")
def main(
    output: str,
    robot_ip: str,
    robot: str,
    port_offset: int,
    motion: str,
    poll_hz: float,
    max_frames: int,
    spin_steps: int,
    spin_settle_s: float,
) -> None:
    """Record RGB + sensor depth + poses from Mars sim for LingBot offline inference.

    Start sim first: ``uv run emet serve mujoco --robot innate_mars --headless``
    """
    out_root = Path(output)
    images_dir = out_root / "images"
    depths_dir = out_root / "depths"
    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.jsonl"

    robot_client = create_robot_client_from_cli(
        robot,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=False,
        start_immediately=True,
        allow_missing_depth=True,
    )

    frame_count = 0
    meta_fp = meta_path.open("w", encoding="utf-8")

    def write_obs(obs) -> None:
        nonlocal frame_count
        if obs is None or obs.rgb is None:
            return
        idx = frame_count
        img_name = f"frame_{idx:06d}.png"
        rel_img = f"images/{img_name}"
        rgb = np.asarray(obs.rgb)
        Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").save(images_dir / img_name)

        depth_rel = None
        if obs.depth is not None and np.asarray(obs.depth).size > 0:
            d = np.asarray(obs.depth, dtype=np.float32)
            stem = depths_dir / f"frame_{idx:06d}"
            np.save(stem, d)
            depth_rel = f"depths/{stem.name}.npy"

        row = {
            "frame_idx": idx,
            "image": rel_img,
            "depth": depth_rel,
            "camera_pose": _np_to_jsonable(obs.camera_pose),
            "camera_K": _np_to_jsonable(obs.camera_K),
            "gps": obs.gps.tolist() if obs.gps is not None else None,
            "compass": obs.compass.tolist() if obs.compass is not None else None,
            "seq_id": int(obs.seq_id),
            "timestamp": time.time(),
        }
        meta_fp.write(json.dumps(row) + "\n")
        meta_fp.flush()
        frame_count += 1

    _orig_get_observation = robot_client.get_observation

    def _recording_get_observation():
        obs = _orig_get_observation()
        write_obs(obs)
        return obs

    robot_client.get_observation = _recording_get_observation  # type: ignore[method-assign]

    def _run_base_spin() -> None:
        robot_client.move_to_nav_posture()
        look_front = getattr(robot_client, "look_front", None)
        if callable(look_front):
            look_front(blocking=True)
        time.sleep(0.5)
        write_obs(robot_client.get_observation())
        step_rad = 2.0 * np.pi / max(spin_steps, 1)
        poll_dt = 1.0 / max(poll_hz, 0.5)
        limit = max_frames if max_frames > 0 else 400
        for _ in range(max(spin_steps, 1)):
            if frame_count >= limit:
                break
            robot_client.move_base_to([0.0, 0.0, step_rad], relative=True, blocking=False)
            t_end = time.monotonic() + max(spin_settle_s, poll_dt)
            while time.monotonic() < t_end and frame_count < limit:
                write_obs(robot_client.get_observation())
                time.sleep(poll_dt)

    try:
        if motion == "base_spin":
            _run_base_spin()
        elif motion == "rotate_in_place":
            parameters = get_parameters("dynav_config.yaml")
            executor = DynamemTaskExecutor(
                robot_client,
                parameters,
                skip_confirmations=True,
                cpu_only=True,
            )
            executor([("rotate_in_place", "")])
        else:
            dt = 1.0 / max(poll_hz, 0.5)
            t_next = time.monotonic()
            limit = max_frames if max_frames > 0 else 400
            while frame_count < limit:
                now = time.monotonic()
                if now < t_next:
                    time.sleep(min(0.05, t_next - now))
                    continue
                t_next = time.monotonic() + dt
                write_obs(robot_client.get_observation())
    finally:
        meta_fp.close()
        robot_client.stop()

    episode_meta = {
        "robot": robot,
        "scene": "default_table",
        "motion": motion,
        "num_frames": frame_count,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "episode.json").write_text(json.dumps(episode_meta, indent=2), encoding="utf-8")
    click.echo(f"Recorded {frame_count} frames to {out_root}")


if __name__ == "__main__":
    main()
