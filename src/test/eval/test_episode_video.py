# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from emet.eval.episode_video import (
    build_motion_paced_rgb_frames,
    normalize_yaw_delta,
    pose_motion_repeat_count,
    write_episode_mp4_from_metadata,
    write_rgb_sequence_mp4,
)


def test_write_rgb_sequence_mp4_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    frames = [
        np.full((32, 48, 3), fill, dtype=np.uint8)
        for fill in ([40, 80, 120], [50, 90, 130], [60, 100, 140])
    ]
    out = write_rgb_sequence_mp4(frames, tmp_path / "clip.mp4", fps=4.0)
    assert out.is_file()
    assert out.stat().st_size > 64


def test_normalize_yaw_delta_shortest_path() -> None:
    assert abs(normalize_yaw_delta(0.0, math.pi / 2) - math.pi / 2) < 1e-6
    delta = normalize_yaw_delta(0.05, 2 * math.pi - 0.05)
    assert abs(delta - (-0.1)) < 1e-5


def test_pose_motion_repeat_count_scales_with_motion() -> None:
    still = pose_motion_repeat_count((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    moved = pose_motion_repeat_count((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), meters_per_repeat=0.25)
    assert still == 1
    assert moved == 4


def test_build_motion_paced_rgb_frames_repeats_and_crossfades(tmp_path: Path) -> None:
    from PIL import Image

    img_a = np.full((8, 8, 3), 10, dtype=np.uint8)
    img_b = np.full((8, 8, 3), 200, dtype=np.uint8)
    Image.fromarray(img_a).save(tmp_path / "a.png")
    Image.fromarray(img_b).save(tmp_path / "b.png")
    rows = [
        {"frame_idx": 0, "image": "a.png", "pose_xyt": [0.0, 0.0, 0.0]},
        {"frame_idx": 1, "image": "b.png", "pose_xyt": [2.0, 0.0, 0.0]},
    ]
    frames = build_motion_paced_rgb_frames(
        rows,
        tmp_path,
        meters_per_repeat=0.5,
        crossfade_teleport_m=1.5,
        crossfade_steps=2,
    )
    assert len(frames) > 2
    assert frames[0].mean() < 50
    assert frames[-1].mean() > 150


def test_write_episode_mp4_from_metadata_motion_paced(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from PIL import Image

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i, color in enumerate([30, 60, 90]):
        Image.fromarray(np.full((16, 16, 3), color, dtype=np.uint8)).save(
            frames_dir / f"rgb_{i:04d}.png"
        )
    meta = tmp_path / "metadata.jsonl"
    with meta.open("w") as fh:
        for i, (x, color) in enumerate([(0.0, 30), (0.5, 60), (1.0, 90)]):
            fh.write(
                json.dumps(
                    {
                        "frame_idx": i,
                        "image": f"frames/rgb_{i:04d}.png",
                        "pose_xyt": [x, 0.0, 0.0],
                    }
                )
                + "\n"
            )
    out = write_episode_mp4_from_metadata(tmp_path, fps=4.0, motion_paced=True)
    assert out.is_file()
    assert out.stat().st_size > 64
