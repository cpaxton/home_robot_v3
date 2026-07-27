# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for manip video overlays (no sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emet.visualization.manip_video import overlay_manip_frame


def test_overlay_manip_frame_banners(tmp_path: Path):
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :] = (40, 60, 80)
    out = overlay_manip_frame(
        rgb,
        action="grasp",
        goal="object2 → object1",
        detail="grasp_index=2",
        title="tamp pick-place",
    )
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    # Top banner darkened / text drawn — not identical to flat fill.
    assert not np.array_equal(out[10, 10], rgb[10, 10]) or out[10, 160, 0] != 40


def test_write_overlay_mp4(tmp_path: Path):
    from emet.eval.episode_video import write_rgb_sequence_mp4
    from emet.visualization.manip_video import overlay_manip_frame

    frames = []
    for i, act in enumerate(("approach", "grasp", "place", "done")):
        rgb = np.full((120, 160, 3), 30 + 10 * i, dtype=np.uint8)
        frames.append(overlay_manip_frame(rgb, action=act, goal="obj→recep", detail=f"frame={i}"))
    out = write_rgb_sequence_mp4(frames, tmp_path / "t.mp4", fps=4.0)
    assert out.is_file()
    assert out.stat().st_size > 64
