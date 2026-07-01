# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Encode diagnostics RGB / top-down map frame sequences to MP4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


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
