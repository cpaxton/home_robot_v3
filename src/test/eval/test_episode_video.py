# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emet.eval.episode_video import write_rgb_sequence_mp4


def test_write_rgb_sequence_mp4_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    frames = [
        np.full((32, 48, 3), fill, dtype=np.uint8)
        for fill in ([40, 80, 120], [50, 90, 130], [60, 100, 140])
    ]
    out = write_rgb_sequence_mp4(frames, tmp_path / "clip.mp4", fps=4.0)
    assert out.is_file()
    assert out.stat().st_size > 64
