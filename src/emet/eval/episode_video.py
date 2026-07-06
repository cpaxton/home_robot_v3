# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Encode diagnostics RGB / top-down map frame sequences to MP4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def normalize_yaw_delta(prev: float, nxt: float) -> float:
    """Shortest signed yaw delta in radians."""
    return (float(nxt) - float(prev) + math.pi) % (2 * math.pi) - math.pi


def pose_motion_repeat_count(
    prev: tuple[float, float, float] | None,
    nxt: tuple[float, float, float] | None,
    *,
    meters_per_repeat: float = 0.25,
    radians_per_repeat: float = math.radians(10.0),
    max_repeats: int = 12,
) -> int:
    """How many times to hold *nxt* so playback pacing tracks sim motion."""
    if nxt is None:
        return 1
    if prev is None:
        return 1
    dx = float(nxt[0]) - float(prev[0])
    dz = float(nxt[1]) - float(prev[1])
    dist = math.hypot(dx, dz)
    dtheta = abs(normalize_yaw_delta(float(prev[2]) if len(prev) >= 3 else 0.0, float(nxt[2]) if len(nxt) >= 3 else 0.0))
    mpr = max(0.05, float(meters_per_repeat))
    rpr = max(0.05, float(radians_per_repeat))
    by_dist = int(math.ceil(dist / mpr)) if dist > 1e-4 else 1
    by_rot = int(math.ceil(dtheta / rpr)) if dtheta > 1e-4 else 1
    return max(1, min(int(max_repeats), max(by_dist, by_rot)))


def _load_metadata_rows(meta_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return rows


def _pose_from_row(row: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = row.get("pose_xyt")
    if not pose or len(pose) < 2:
        return None
    theta = float(pose[2]) if len(pose) >= 3 else 0.0
    return (float(pose[0]), float(pose[1]), theta)


def build_motion_paced_rgb_frames(
    rows: list[dict[str, Any]],
    root: Path,
    *,
    meters_per_repeat: float = 0.25,
    radians_per_repeat: float = math.radians(10.0),
    max_repeats: int = 12,
    crossfade_teleport_m: float = 1.5,
    crossfade_steps: int = 3,
) -> list[np.ndarray]:
    """Expand sparse keyframes using motion-weighted repeats and optional crossfade."""
    if not rows:
        return []

    out: list[np.ndarray] = []
    prev_rgb: np.ndarray | None = None
    prev_pose: tuple[float, float, float] | None = None

    for row in rows:
        rel = row.get("image")
        if not rel:
            continue
        path = root / str(rel)
        if not path.is_file():
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pose = _pose_from_row(row)

        if prev_rgb is not None and prev_pose is not None and pose is not None:
            dist = math.hypot(pose[0] - prev_pose[0], pose[1] - prev_pose[1])
            if dist >= float(crossfade_teleport_m) and crossfade_steps > 0:
                for step in range(1, crossfade_steps + 1):
                    alpha = step / float(crossfade_steps + 1)
                    blend = (
                        (1.0 - alpha) * prev_rgb.astype(np.float32) + alpha * rgb.astype(np.float32)
                    ).astype(np.uint8)
                    out.append(blend)

        repeats = pose_motion_repeat_count(
            prev_pose,
            pose,
            meters_per_repeat=meters_per_repeat,
            radians_per_repeat=radians_per_repeat,
            max_repeats=max_repeats,
        )
        out.extend([rgb] * repeats)
        prev_rgb = rgb
        prev_pose = pose

    return out


def write_rgb_sequence_mp4(
    frames: list[np.ndarray],
    out_path: Path | str,
    *,
    fps: float = 6.0,
) -> Path:
    """Encode a list of HxWx3 uint8 RGB arrays to MP4 (``mp4v`` via OpenCV)."""
    import cv2

    if not frames:
        raise ValueError("no frames to encode")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0], dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] < 3:
        raise ValueError(f"expected HxWx3 rgb, got {first.shape}")
    h, w = int(first.shape[0]), int(first.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, max(0.5, float(fps)), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                rgb = np.asarray(Image.fromarray(rgb[:, :, :3]).resize((w, h), Image.Resampling.NEAREST))
            bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
            vw.write(bgr)
    finally:
        vw.release()
    if not out.is_file() or out.stat().st_size < 32:
        raise RuntimeError(f"MP4 missing or too small: {out}")
    return out


def write_png_sequence_mp4(
    png_paths: list[Path],
    out_path: Path | str,
    *,
    fps: float = 6.0,
) -> Path:
    """Encode ordered PNG paths to MP4."""
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in png_paths if p.is_file()]
    if not frames:
        raise ValueError("no readable PNG frames")
    return write_rgb_sequence_mp4(frames, out_path, fps=fps)


def write_episode_mp4_from_metadata(
    episode_dir: Path | str,
    *,
    fps: float = 6.0,
    filename: str = "episode_rgb.mp4",
    out_path: Path | str | None = None,
    motion_paced: bool = True,
    meters_per_repeat: float = 0.25,
    radians_per_repeat: float = math.radians(10.0),
    max_repeats: int = 12,
    crossfade_teleport_m: float = 1.5,
) -> Path:
    """Build MP4 from ``metadata.jsonl`` (+ ``images/`` or ``frames/`` PNGs)."""
    root = Path(episode_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    rows = _load_metadata_rows(meta_path)
    if not rows:
        raise ValueError("metadata.jsonl has no frames")

    if motion_paced:
        frames = build_motion_paced_rgb_frames(
            rows,
            root,
            meters_per_repeat=meters_per_repeat,
            radians_per_repeat=radians_per_repeat,
            max_repeats=max_repeats,
            crossfade_teleport_m=crossfade_teleport_m,
        )
    else:
        frames = []
        for row in rows:
            rel = row.get("image")
            if not rel:
                continue
            path = root / str(rel)
            if path.is_file():
                frames.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))

    if not frames:
        raise FileNotFoundError(f"No readable images for {root}")
    dest = Path(out_path) if out_path is not None else root / filename
    return write_rgb_sequence_mp4(frames, dest, fps=fps)
