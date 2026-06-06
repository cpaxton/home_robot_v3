# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Load emet-recorded episodes (metadata.jsonl + images + depths)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EpisodeFrame:
    frame_idx: int
    rgb_path: Path
    depth_path: Path | None
    camera_K: np.ndarray | None
    camera_pose: np.ndarray | None  # 4x4 cam-to-world
    gps: np.ndarray | None
    compass: np.ndarray | None


@dataclass
class Episode:
    root: Path
    frames: list[EpisodeFrame]

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def load_episode(episode_dir: Path | str) -> Episode:
    root = Path(episode_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")

    frames: list[EpisodeFrame] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row.get("frame_idx", len(frames)))
        rel_img = row.get("image")
        if not rel_img:
            continue
        rgb_path = root / str(rel_img)
        depth_rel = row.get("depth")
        depth_path = root / str(depth_rel) if depth_rel else None
        K = row.get("camera_K")
        pose = row.get("camera_pose")
        gps = row.get("gps")
        compass = row.get("compass")
        frames.append(
            EpisodeFrame(
                frame_idx=idx,
                rgb_path=rgb_path,
                depth_path=depth_path if depth_path and depth_path.is_file() else None,
                camera_K=np.asarray(K, dtype=np.float64).reshape(3, 3) if K is not None else None,
                camera_pose=np.asarray(pose, dtype=np.float64).reshape(4, 4) if pose is not None else None,
                gps=np.asarray(gps, dtype=np.float64) if gps is not None else None,
                compass=np.asarray(compass, dtype=np.float64) if compass is not None else None,
            )
        )
    frames.sort(key=lambda f: f.frame_idx)
    if not frames:
        raise ValueError(f"No frames in {meta_path}")
    return Episode(root=root, frames=frames)


def load_rgb(frame: EpisodeFrame) -> np.ndarray:
    """HxWx3 uint8 RGB."""
    with Image.open(frame.rgb_path) as im:
        return np.asarray(im.convert("RGB"))


def load_depth_meters(frame: EpisodeFrame) -> np.ndarray | None:
    if frame.depth_path is None or not frame.depth_path.is_file():
        return None
    d = np.load(frame.depth_path)
    d = np.asarray(d, dtype=np.float32)
    if d.ndim != 2:
        return None
    # uint16 mm from sim JP2 decode path is already float meters in episode writer;
    # if values look like mm, convert.
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size and float(np.median(finite)) > 100.0:
        d = d / 1000.0
    return d
